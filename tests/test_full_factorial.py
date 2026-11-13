"""Exercise campaign auditing and publication gates with invented adverts."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from wildlife_screening import full_factorial as ff
from wildlife_screening.config import ModelConfig
from wildlife_screening.scorers.api import OpenAIDecisionsScorer, SystemOneScorer
from wildlife_screening.tasks import TASKS, Task
from wildlife_screening.types import ScoreResult
from wildlife_screening.utils import file_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def invented_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Create a small source and independent synthetic prompt files."""
    data = tmp_path / "invented.csv"
    data.write_text(
        "id,title,description,label,relevant\n"
        "one,Invented shark item,Invented details,1,animal origin\n"
        "two,Invented plastic item,,0,not animal origin\n"
        "three,Invented ray item,More invented details,1,animal origin\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("task: sharks\nrows: []\n", encoding="utf-8")
    monkeypatch.setitem(TASKS, "sharks", Task("sharks", data, baseline))
    monkeypatch.setitem(ff.DATA_HASHES, "sharks", file_hash(data))
    monkeypatch.setattr(
        ff, "MODEL_PATHS", {name: ROOT / path for name, path in ff.MODEL_PATHS.items()}
    )
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for stem in ("fewshot", "zeroshot"):
        chat = (
            "Label 1: relevant - Invented real item. "
            "Label 2: not relevant - Invented other item. "
            "Return only one of the two labels: relevant or not relevant, "
            "no explanation is necessary."
        )
        examples: list[dict[str, str]] = []
        if stem == "fewshot":
            chat += "\nExamples:\n1. Advertisement: Invented example A\nLabel: relevant\n"
            chat += "2. Advertisement: Invented example B\nLabel: not relevant\n"
            examples = [
                {"text": "Invented example A", "label": "relevant", "explanation": "Invented."},
                {"text": "Invented example B", "label": "not relevant", "explanation": "Invented."},
            ]
        (prompt_dir / f"llm_{stem}_v0.txt").write_text(chat, encoding="utf-8")
        definition = {
            "version": f"decision-{stem}-v0",
            "family": "decision",
            "condition": "few-shot" if stem == "fewshot" else "zero-shot",
            "instructions": "Classify invented products.",
            "criteria": {"relevant": "Invented real item", "not relevant": "Invented other"},
            "examples": examples,
        }
        (prompt_dir / f"decision_{stem}_v0.yaml").write_text(
            yaml.safe_dump(definition), encoding="utf-8"
        )

    def paths(_task: str, version: str, shot: str) -> dict[str, Path]:
        stem = "fewshot" if shot == "few-shot" else "zeroshot"
        return {
            "llm": prompt_dir / f"llm_{stem}_{version}.txt",
            "decision": prompt_dir / f"decision_{stem}_{version}.yaml",
        }

    monkeypatch.setattr(ff, "prompt_paths", paths)
    return {"data": data, "audit": ff.build_audit("sharks", "v0", None)}


def test_audit_checks_all_rendered_evidence_and_explicit_prompts(
    invented_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = invented_campaign["audit"]
    assert len(audit["inputs"]) == 4
    assert {item["adverts"] for item in audit["inputs"].values()} == {3}
    args = ff.CampaignCondition.from_audit(
        audit, "title_and_desc", "zero-shot", Path("out"), 2.0, False
    ).command()
    assert "--include-description" in args and "--zero-shot" in args
    assert args[args.index("--llm-prompt") + 1].endswith("llm_zeroshot_v0.txt")
    assert args[args.index("--decision-prompt") + 1].endswith("decision_zeroshot_v0.yaml")
    assert args[args.index("--cache") + 1] == "outputs/cache"
    assert args.count("--model") == 4
    assert "Invented shark item" not in ff.audit_text(audit)

    original = SystemOneScorer.payload

    def corrupted(self: SystemOneScorer, advert: Any) -> dict[str, Any]:
        result = original(self, advert)
        result["state"] += " altered"
        return result

    monkeypatch.setattr(SystemOneScorer, "payload", corrupted)
    with pytest.raises(ValueError, match="request differs"):
        ff.build_audit("sharks", "v0", None)


def test_audit_rejects_reordered_native_choices(
    invented_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = OpenAIDecisionsScorer.payload

    def reordered(self: OpenAIDecisionsScorer, advert: Any) -> dict[str, Any]:
        result = original(self, advert)
        result["questions"][0]["choices"].reverse()
        return result

    monkeypatch.setattr(OpenAIDecisionsScorer, "payload", reordered)
    with pytest.raises(ValueError, match="luna request differs"):
        ff.build_audit("sharks", "v0", None)


@pytest.mark.parametrize("task", ["animals", "leather"])
@pytest.mark.parametrize("with_review", [False, True])
def test_additional_task_uses_title_only(
    task: str,
    with_review: bool,
    invented_campaign: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "leather.csv"
    data.write_text(
        "id,title,label\none,Invented leather item,1\ntwo,Invented other item,0\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        TASKS,
        task,
        Task(task, data, tmp_path / "baseline.yaml", positive_label="1", input_modes=("title",)),
    )
    monkeypatch.setitem(ff.DATA_HASHES, task, file_hash(data))
    assert ff.input_modes(task) == ("title",)
    assert ff.build_audit(task, "v0", None)["positive_label"] == "1"
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "task": task,
                "data_sha256": file_hash(data),
                "positive_label": "1",
                "reviewer": "Invented tester",
                "basis": "source_documentation",
                "evidence": "Invented fixture provenance",
            }
        ),
        encoding="utf-8",
    )
    audit = ff.build_audit(task, "v0", review if with_review else None)
    assert len(audit["inputs"]) == 2
    assert len(ff.input_modes(task)) * len(ff.SHOTS) * len(ff.MODEL_PATHS) == 8
    (tmp_path / "baseline.yaml").write_text(f"task: {task}\nrows: []\n", encoding="utf-8")
    root = tmp_path / "leather-run"
    root.mkdir()
    for shot in ff.SHOTS:
        for model, item in audit["model_files"].items():
            config = ModelConfig.load(Path(item["path"]))
            family = "llm" if model == "chat" else "decision"
            folder = root / "title" / shot / model
            folder.mkdir(parents=True)
            manifest = {
                "arm": config.arm,
                "task": task,
                "data_file_hash": audit["data_sha256"],
                "label_review_sha256": audit["label_review_sha256"],
                "positive_label": "1",
                "dry_run": False,
                "status": "complete",
                "condition": shot,
                "example_count": int(shot == "few-shot"),
                "prompt_family": family,
                "prompt_hash": audit["prompt_files"][shot][family]["sha256"],
                "model_config": asdict(config),
                "model_config_hash": audit["inputs"][f"title/{shot}"]["config_hashes"][model],
                "allowlisted_fields": ["title"],
                "git_sha": audit["git_sha"],
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "cached_results": 2,
                "estimate": {"pending_adverts": 0},
            }
            (folder / "manifest-invented.json").write_text(json.dumps(manifest), encoding="utf-8")
            scores = [
                asdict(
                    ScoreResult(
                        advert_id,
                        float(advert_id == "one"),
                        "binary" if model == "chat" else "native_prob",
                        "",
                        0,
                        0,
                        0.0,
                        0.1,
                    )
                )
                for advert_id in ("one", "two")
            ]
            (folder / "scores-invented.json").write_text(json.dumps(scores), encoding="utf-8")
    ff.finalize(root, audit)
    candidate = yaml.safe_load((root / "selection-candidate.yaml").read_text())
    assert len(candidate["pins"]) == 8
    if with_review:
        assert candidate["datasets"][f"{task}-{file_hash(data)[:12]}"]["label_review"] == str(
            review
        )
    else:
        assert "datasets" not in candidate
    assert not (root / "report-preview.html").exists()


def test_arm_inventory_and_offline_preflight(
    invented_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = invented_campaign["audit"]
    assert len(ff.input_modes("sharks")) * len(ff.SHOTS) * len(ff.MODEL_PATHS) == 16
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "report").exists()
    monkeypatch.setattr(ff, "build_audit", lambda *_args: audit)
    monkeypatch.setattr(ff, "_preflight", lambda *_args: (tmp_path / "checks", 0.1))

    def no_server() -> None:
        raise AssertionError("Offline preflight must not query the d3 server")

    monkeypatch.setattr(ff, "_check_d3", no_server)
    assert ff.main(["--task", "sharks", "--prompt-version", "v0", "--preflight-only"]) == 0


def test_preflight_enforces_cumulative_ceiling(
    invented_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = invented_campaign["audit"]

    def fake_cli(condition: ff.CampaignCondition) -> None:
        args = condition.command()
        output = Path(args[args.index("--output") + 1])
        shot = "zero-shot" if "--zero-shot" in args else "few-shot"
        input_name = output.name
        fields = ["title", "description"] if input_name == "title_and_desc" else ["title"]
        for name, item in audit["model_files"].items():
            config = ModelConfig.load(Path(item["path"]))
            family = "llm" if name == "chat" else "decision"
            manifest = {
                "status": "dry_run",
                "task": "sharks",
                "data_file_hash": audit["data_sha256"],
                "arm": config.arm,
                "condition": shot,
                "allowlisted_fields": fields,
                "prompt_hash": audit["prompt_files"][shot][family]["sha256"],
                "model_config_hash": audit["inputs"][f"{input_name}/{shot}"]["config_hashes"][name],
                "label_review_sha256": None,
                "positive_label": "1",
                "model_config": asdict(config),
                "estimate": {
                    "pending_adverts": 3,
                    "cached_adverts": 0,
                    "cost_upper_estimate": 0.01 if config.billable else None,
                },
            }
            path = output / shot / name / "manifest-invented.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(ff, "_call_cli", fake_cli)
    monkeypatch.setattr(ff, "cache_spend", lambda _path: 0.03)
    with pytest.raises(ValueError, match="supplied ceiling"):
        ff._preflight(audit, 0.14, tmp_path / "low")
    _, total = ff._preflight(audit, 0.16, tmp_path / "high")
    assert total == pytest.approx(0.15)


def test_archive_preserves_original_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs/comparison").mkdir(parents=True)
    (tmp_path / "outputs/report").mkdir()
    (tmp_path / "outputs/comparison/old.txt").write_text("old", encoding="utf-8")
    (tmp_path / "outputs/report/selection.yaml").write_text("pins: []\n", encoding="utf-8")
    archive = ff._archive_current("invented")
    assert (archive / "comparison/old.txt").read_text() == "old"
    assert not (archive / "report").exists()
    assert (tmp_path / "outputs/comparison/old.txt").exists()


def test_candidate_requires_all_complete_snapshots(
    invented_campaign: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = invented_campaign["audit"]
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ValueError, match="Campaign incomplete"):
        ff.finalize(root, audit)
    assert (root / "manifest-index.txt").exists()
    assert not (root / "selection-candidate.yaml").exists()

    for input_name in ff.input_modes("sharks"):
        fields = ["title", "description"] if input_name == "title_and_desc" else ["title"]
        for shot in ff.SHOTS:
            for model, item in audit["model_files"].items():
                config = ModelConfig.load(Path(item["path"]))
                folder = root / input_name / shot / model
                folder.mkdir(parents=True)
                family = "llm" if model == "chat" else "decision"
                manifest = {
                    "arm": config.arm,
                    "run_id": f"{input_name}-{shot}-{model}",
                    "task": "sharks",
                    "data_file_hash": audit["data_sha256"],
                    "label_review_sha256": None,
                    "positive_label": "1",
                    "dry_run": False,
                    "status": "complete",
                    "condition": shot,
                    "example_count": 2 if shot == "few-shot" else 0,
                    "prompt_family": family,
                    "prompt_hash": audit["prompt_files"][shot][family]["sha256"],
                    "model_config": asdict(config),
                    "model_config_hash": audit["inputs"][f"{input_name}/{shot}"]["config_hashes"][
                        model
                    ],
                    "allowlisted_fields": fields,
                    "git_sha": audit["git_sha"],
                    "started_at": "2026-01-01T00:00:00Z",
                    "finished_at": "2026-01-01T00:00:01Z",
                    "cached_results": 3,
                    "estimate": {"pending_adverts": 0},
                }
                (folder / "manifest-invented.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                scores = [
                    asdict(
                        ScoreResult(
                            index,
                            float(index != "two"),
                            "binary" if model == "chat" else "native_prob",
                            "",
                            0,
                            0,
                            0.0,
                            0.1,
                        )
                    )
                    for index in ("one", "two", "three")
                ]
                (folder / "scores-invented.json").write_text(json.dumps(scores), encoding="utf-8")

    # The index and selected pins must describe the same scan of the campaign directory.
    inventory = ff._manifest_rows
    scans = 0

    def scan(directory: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal scans
        scans += 1
        return inventory(directory, settings)

    monkeypatch.setattr(ff, "_manifest_rows", scan)
    ff.finalize(root, audit)
    assert scans == 1
    selection = yaml.safe_load((root / "selection-candidate.yaml").read_text())
    assert len(selection["pins"]) == 16
    assert not (root / "report-preview.html").exists()
    assert "new calls 0" in (root / "manifest-index.txt").read_text()


def test_resume_requires_same_frozen_audit(
    invented_campaign: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = invented_campaign["audit"]
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "report").exists()
    run = tmp_path / "outputs/comparison/reviewed/sharks/invented"
    run.mkdir(parents=True)
    (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(ff, "build_audit", lambda *_args: audit)
    monkeypatch.setattr(ff, "_require_clean_commit", lambda: None)
    monkeypatch.setattr(ff, "_check_d3", lambda: None)
    monkeypatch.setattr(ff, "_preflight", lambda *_args: (tmp_path / "checks", 0.1))
    monkeypatch.setattr(ff, "write_index", lambda *_args: ([], []))
    monkeypatch.setattr(ff, "finalize", lambda *_args: None)
    calls: list[ff.CampaignCondition] = []
    monkeypatch.setattr(ff, "_call_cli", lambda args: calls.append(args))
    monkeypatch.setenv("OPENAI_API_KEY", "invented-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "invented-test-key")
    args = [
        "--task",
        "sharks",
        "--prompt-version",
        "v0",
        "--spend-ceiling",
        "2",
        "--resume",
        str(run),
    ]
    assert ff.main(args) == 0
    assert len(calls) == 4
    # Adding explicit polarity to the audit must not silently resume an older campaign.
    historical = {key: value for key, value in audit.items() if key != "positive_label"}
    (run / "audit.json").write_text(json.dumps(historical), encoding="utf-8")
    assert ff.main(args) == 1
    assert len(calls) == 4
    changed = {**audit, "data_sha256": "different"}
    (run / "audit.json").write_text(json.dumps(changed), encoding="utf-8")
    assert ff.main(args) == 1
    assert len(calls) == 4


def test_handled_failures_continue_conditions_without_replaying(
    invented_campaign: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from dataclasses import replace

    from wildlife_screening import cli, runner
    from wildlife_screening.scorers.local import FakeScorer

    audit = invented_campaign["audit"]
    root = tmp_path / "campaign"
    # Source fixtures and real request configs, but an exclusively local invented scorer.
    calls: list[str] = []

    class RefusingFake(FakeScorer):
        def score(self, adverts: list[Any]) -> list[ScoreResult]:
            calls.extend(a.advert_id for a in adverts)
            return [
                replace(r, p_relevant=None, error="invented refusal") if r.advert_id == "one" else r
                for r in super().score(adverts)
            ]

    monkeypatch.setattr(runner, "make_scorer", lambda config, prompt: RefusingFake())
    # Ensure the subprocess simulation reads the fixture dataset, never private data.
    data = invented_campaign["data"]
    assert TASKS["sharks"].data == data

    def subprocess_cli(args: list[str], **kwargs: Any) -> Any:
        start = args.index("run")
        supplied = [*args[start:], "--data", str(data)]
        supplied[supplied.index("--cache") + 1] = str(tmp_path / "cache")
        return subprocess.CompletedProcess(args, cli.main(supplied), stdout="invented CLI output")

    from types import SimpleNamespace

    monkeypatch.setattr(ff, "subprocess", SimpleNamespace(run=subprocess_cli, PIPE=subprocess.PIPE))
    for shot in ff.SHOTS:
        args = ff.CampaignCondition.from_audit(audit, "title", shot, root, 5, False)
        ff._call_cli(args)
    assert len(calls) == 3 * 4 * 2
    for shot in ff.SHOTS:
        ff._call_cli(ff.CampaignCondition.from_audit(audit, "title", shot, root, 5, False))
    assert len(calls) == 3 * 4 * 2  # Including refusals: nothing is replayed.
    for path in root.rglob("manifest-*.json"):
        m = json.loads(path.read_text())
        assert m["status"] == "incomplete" and m["failures"] == 1
        assert m["valid_predictions"] == 2
    # Missing declared description conditions still block finalization.
    assert ff._selected_rows(ff._manifest_rows(root, audit), audit)[1]
    for shot in ff.SHOTS:
        ff._call_cli(ff.CampaignCondition.from_audit(audit, "title_and_desc", shot, root, 5, False))
    ff.finalize(root, audit)
    candidate = yaml.safe_load((root / "selection-candidate.yaml").read_text())
    assert len(candidate["pins"]) == 16
    assert candidate["evaluation_policy"] == "review_on_failure"
    assert not (root / "report-preview.html").exists()
    assert "coverage 2/3" in (root / "manifest-index.txt").read_text()


@pytest.mark.parametrize("code", [1, 2, -2])
def test_subprocess_failure_without_valid_snapshots_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, invented_campaign: dict[str, Any]
) -> None:
    import subprocess
    from types import SimpleNamespace

    def failed(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args, code, stdout="invented failure")

    monkeypatch.setattr(ff, "subprocess", SimpleNamespace(run=failed, PIPE=subprocess.PIPE))
    with pytest.raises((RuntimeError, ValueError)):
        ff._call_cli(
            ff.CampaignCondition.from_audit(
                invented_campaign["audit"], "title", "few-shot", tmp_path, 0, False
            )
        )
