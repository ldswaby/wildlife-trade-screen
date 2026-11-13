"""Exercise task boundaries with invented adverts and no external calls."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from wildlife_screening.cli import main
from wildlife_screening.data import inspect_data, load_data
from wildlife_screening.prompts import Prompt
from wildlife_screening.report import create_report
from wildlife_screening.tasks import TASKS, label_review
from wildlife_screening.utils import file_hash

ROOT = Path(__file__).resolve().parents[1]


def test_published_baselines_follow_table_2() -> None:
    """Keep all task baselines aligned with the paper's reported model rows."""
    expected = {
        "sharks": ((0.956, 0.901, 0.884, 0.893), (0.956, 0.908, 0.873, 0.89)),
        "animals": ((0.83, 0.51, 0.718, 0.596), (0.921, 0.8, 0.732, 0.765)),
        "leather": ((0.872, 0.863, 0.839, 0.851), (0.946, 0.913, 0.968, 0.94)),
    }
    arms = ["ACTL text", "LTS-text", "ACTL text-image", "LTS text-image", "GPT-4 zero-shot"]
    for task, image_values in expected.items():
        path = ROOT / f"baselines/published_{task}.yaml"
        assert TASKS[task].baseline == Path(f"baselines/published_{task}.yaml")
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))["rows"]
        assert [row["arm"] for row in rows] == arms
        assert all(row["task"] == task for row in rows)
        for row, values in zip(rows[2:4], image_values, strict=True):
            assert row["input_modality"] == "text-image"
            assert (
                tuple(
                    row[field]
                    for field in (
                        "accuracy",
                        "published_precision",
                        "published_recall",
                        "published_f1",
                    )
                )
                == values
            )
        assert "published_precision" not in rows[4] and "published_recall" not in rows[4]


def invented_csv(tmp_path: Path) -> Path:
    path = tmp_path / "invented.csv"
    path.write_text(
        "id,title,label,GPT_label,description\n"
        "a,Invented leather wallet,1,0,metadata must stay local\n"
        "b,Invented plastic wallet,0,1,metadata must stay local\n",
        encoding="utf-8",
    )
    return path


def review_file(tmp_path: Path, data: Path, task: str, basis: str) -> Path:
    path = tmp_path / f"{task}-review.json"
    path.write_text(
        json.dumps(
            {
                "task": task,
                "data_sha256": file_hash(data),
                "positive_label": "1",
                "reviewer": "Fixture reviewer",
                "basis": basis,
                "evidence": "Independent fixture annotation with declared positive class.",
                "audit_scope": "Both invented rows were reviewed.",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("task", ["animals", "leather"])
def test_additional_tasks_preserve_optional_source_bound_review(tmp_path: Path, task: str) -> None:
    data = invented_csv(tmp_path)
    base = [
        "run",
        "--task",
        task,
        "--data",
        str(data),
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--cache",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "output"),
    ]
    assert main(base + ["--dry-run"]) == 0
    review = review_file(tmp_path, data, task, "independent_adjudication")
    assert main(base + ["--label-review", str(review), "--zero-shot"]) == 0
    output = tmp_path / "output" / task / "zero-shot"
    rows = list(csv.DictReader((output / "table1.csv").open()))
    assert len(rows) == 1 and rows[0]["task"] == task
    assert rows[0]["positive_label"] == "1"
    assert rows[0]["data_sha256"] == file_hash(data)
    assert rows[0]["label_review_sha256"] == file_hash(review)
    assert rows[0]["published_f1"] == "unknown"
    manifest = json.loads(next(output.rglob("manifest-*.json")).read_text())
    assert manifest["task"] == task
    assert manifest["label_review_sha256"] == file_hash(review)
    assert manifest["positive_label"] == "1"
    assert manifest["allowlisted_fields"] == ["title"]
    adverts = load_data(data, positive_label="1").adverts
    assert all("GPT_label" not in advert.text() for advert in adverts)
    assert all("metadata must stay local" not in advert.text() for advert in adverts)


def test_review_mismatch_and_leather_basis_fail_closed(tmp_path: Path) -> None:
    data = invented_csv(tmp_path)
    review = review_file(tmp_path, data, "leather", "independent_adjudication")
    payload = json.loads(review.read_text())
    payload["data_sha256"] = "not this file"
    review.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        label_review(TASKS["leather"], data, review)
    payload["data_sha256"] = file_hash(data)
    payload["audit_scope"] = ""
    review.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="audit scope"):
        label_review(TASKS["leather"], data, review)


def test_task_prompts_and_baselines_are_distinct(tmp_path: Path) -> None:
    for task in ("animals", "leather"):
        for shot, condition, count in (("zeroshot", "zero-shot", 0), ("fewshot", "few-shot", 5)):
            llm = Prompt.load(ROOT / f"prompts/{task}/llm_{shot}_v0.txt")
            decision = Prompt.load(ROOT / f"prompts/{task}/decision_{shot}_v0.yaml")
            assert llm.condition == decision.condition == condition
            assert len(llm.examples) == len(decision.examples) == count
            assert llm.sha256 != decision.sha256
        path = ROOT / f"baselines/published_{task}.yaml"
        rows = create_report([0, 1], {}, tmp_path / task, path, task=task)
        assert rows and all(row["task"] == task for row in rows)
        with pytest.raises(ValueError, match="baseline task"):
            create_report([0, 1], {}, tmp_path / "wrong", path, task="sharks")


def test_nonshark_inspection_does_not_read_shark_pool(tmp_path: Path) -> None:
    data = invented_csv(tmp_path)
    report = inspect_data(data)
    assert "pool" not in report
    assert report["gpt_label_values"] == {"0": 1, "1": 1}
    assert report["description_present"] == 2
    assert report["mapping_error"] is not None
    output = tmp_path / "inspection.json"
    assert main(["inspect", "--data", str(data), "--output", str(output)]) == 1
    assert "pool" not in json.loads(output.read_text())


@pytest.mark.parametrize("task", ["animals", "leather"])
def test_manual_review_workflow_records_polarity_without_json(tmp_path: Path, task: str) -> None:
    from wildlife_screening.snapshots import _load_pinned

    data = invented_csv(tmp_path)
    output = tmp_path / "runs"
    args = [
        "run",
        "--task",
        task,
        "--data",
        str(data),
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--output",
        str(output),
        "--cache",
        str(tmp_path / "cache"),
    ]
    assert main(args) == 0
    pin = next(output.rglob("manifest-*.json"))
    manifest = json.loads(pin.read_text())
    assert manifest["positive_label"] == "1"
    assert manifest["data_file_hash"] == file_hash(data)
    assert manifest["label_review_sha256"] is None
    assert manifest["allowlisted_fields"] == ["title"]
    settings = {"data": str(data)}
    selection = {
        "pins": [str(pin)],
        "datasets": {f"{task}-{file_hash(data)[:12]}": settings},
    }
    row, _ = _load_pinned(pin, selection)
    assert row["dataset"].startswith(task + "-")
    # Historical review paths in a combined selection do not gate new manifests.
    review = review_file(tmp_path, data, task, "source_documentation")
    settings["label_review"] = str(review)
    _load_pinned(pin, selection)
    # Polarity is checked against task configuration, not trusted from the manifest.
    manifest["positive_label"] = "0"
    pin.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="polarity"):
        _load_pinned(pin, selection)


@pytest.mark.parametrize("saved_positive", ["inferred from relevant", "0"])
def test_report_supports_historical_shark_inference(tmp_path: Path, saved_positive: str) -> None:
    from wildlife_screening.snapshots import _dataset_data

    data = tmp_path / "sharks.csv"
    data.write_text(
        "id,title,label,relevant\na,Invented animal,0,animal origin\n"
        "b,Invented plastic,1,not animal origin\n"
    )
    manifest = {"data_file_hash": file_hash(data), "positive_label": saved_positive}
    selection = {"datasets": {f"sharks-{file_hash(data)[:12]}": {"data": str(data)}}}
    loaded, _ = _dataset_data("sharks", manifest, selection)
    assert loaded.labels == [1, 0]
    assert loaded.positive_label == "0"
