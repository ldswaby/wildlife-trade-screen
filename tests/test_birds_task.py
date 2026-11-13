"""Verify bird campaign/report integration without inference or private data."""

import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from report_support import render_evidence

from wildlife_screening import full_factorial as ff
from wildlife_screening import runner
from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.data import input_fields, input_mode
from wildlife_screening.prompts import Prompt
from wildlife_screening.scorers import make_scorer
from wildlife_screening.scorers.base import Scorer
from wildlife_screening.snapshots import _load_pinned, discover_runs
from wildlife_screening.tasks import TASKS
from wildlife_screening.types import Advert, ScoreResult
from wildlife_screening.utils import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


class InventedScorer(Scorer):
    """Return fixed invented predictions, including an optional cached refusal."""

    def __init__(self, binary: bool, fail: bool, calls: list[str]) -> None:
        self.binary = binary
        self.fail = fail
        self.calls = calls

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        results = []
        for advert in adverts:
            assert advert.title == "" and advert.description is None
            assert "DO_NOT_SEND" not in advert.text()
            self.calls.append(advert.advert_id)
            failed = self.fail and advert.advert_id == "three"
            value = None if failed else float(advert.advert_id == "one")
            results.append(
                ScoreResult(
                    advert.advert_id,
                    value,
                    "binary" if self.binary else "native_prob",
                    "Invented response",
                    10,
                    0,
                    0.000001,
                    0.1,
                    "Invented refusal" if failed else None,
                )
            )
        return results


@pytest.fixture
def bird_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data = tmp_path / "birds.csv"
    data.write_text(
        "advert_id,title_and_desc,label,seller\n"
        "one,Invented budgies for sale,1,DO_NOT_SEND\n"
        "two,Invented empty cage,0,DO_NOT_SEND\n"
        "three,Invented wanted ducks,0,DO_NOT_SEND\n"
    )
    monkeypatch.setitem(TASKS, "birds", replace(TASKS["birds"], data=data))
    monkeypatch.setitem(ff.DATA_HASHES, "birds", file_hash(data))
    monkeypatch.setattr(ff, "MODEL_PATHS", {k: ROOT / p for k, p in ff.MODEL_PATHS.items()})
    original_paths = ff.prompt_paths
    monkeypatch.setattr(
        ff,
        "prompt_paths",
        lambda task, version, shot: {
            k: ROOT / p for k, p in original_paths(task, version, shot).items()
        },
    )
    audit = ff.build_audit("birds", "v0", None)
    monkeypatch.setattr(runner, "git_sha", lambda: audit["git_sha"])
    monkeypatch.setattr(ff, "git_sha", lambda: audit["git_sha"])
    monkeypatch.chdir(tmp_path)
    return {"data": data, "audit": audit}


def call_in_process(condition: ff.CampaignCondition) -> None:
    """Exercise the real CLI without losing test task definitions in a subprocess."""
    args = condition.command()
    output = condition.output
    before = set(output.rglob("manifest-*.json"))
    code = main(args[args.index("run") :])
    if code == 2:
        ff._validate_handled_condition(condition, set(output.rglob("manifest-*.json")) - before)
    else:
        assert code == 0


def test_initial_manifest_survives_interruption(
    bird_campaign: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit = bird_campaign["audit"]
    snapshots: list[dict[str, Any]] = []
    original_write = write_json

    def capture(path: Path, value: Any) -> None:
        if path.name.startswith("manifest-"):
            snapshots.append(json.loads(json.dumps(value)))
        original_write(path, value)

    class InterruptedScorer(InventedScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "write_json", capture)
    monkeypatch.setattr(runner, "make_scorer", lambda *_: InterruptedScorer(False, False, []))
    args = ff.CampaignCondition.from_audit(
        audit, "combined_text", "zero-shot", tmp_path / "run", 1, False
    ).command()
    with pytest.raises(KeyboardInterrupt):
        main(args[args.index("run") :])
    assert snapshots[0]["status"] == "started"
    assert (
        snapshots[0]["model_config_hash"]
        == audit["inputs"]["combined_text/zero-shot"]["config_hashes"]["d3"]
    )
    assert snapshots[-1]["status"] == "aborted"
    # A hard kill can leave only the initial snapshot; indexing must accept it.
    interrupted = tmp_path / "hard-kill"
    interrupted.mkdir()
    original_write(interrupted / "manifest-invented.json", snapshots[0])
    assert ff._manifest_rows(interrupted, audit)[0]["manifest"]["status"] == "started"
    # Neither an interrupt nor the manifest fix authorises replaying an uncertain call.
    assert main(args[args.index("run") :]) == 1
    assert "Uncertain prior call" in capsys.readouterr().out


@pytest.mark.parametrize("interrupted", [False, True])
def test_campaign_index_error_preserves_primary_failure(
    bird_campaign: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interrupted: bool,
) -> None:
    monkeypatch.setattr(ff, "build_audit", lambda *_: bird_campaign["audit"])
    monkeypatch.setattr(ff, "_require_clean_commit", lambda: None)
    monkeypatch.setattr(ff, "_check_d3", lambda: None)
    monkeypatch.setattr(ModelConfig, "validate_live", lambda _: None)
    monkeypatch.setenv("OPENAI_API_KEY", "invented")
    monkeypatch.setenv("OPENROUTER_API_KEY", "invented")
    monkeypatch.setattr(ff, "_preflight", lambda *_: (tmp_path, 0))
    monkeypatch.setattr(ff, "_archive_current", lambda _: tmp_path)
    monkeypatch.setattr(ff, "_run_root", lambda _: tmp_path / "campaign")

    def fail_scoring(_: ff.CampaignCondition) -> None:
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError("Invented primary scoring failure")

    def fail_index(*_: Any) -> None:
        raise ValueError("Invented secondary index failure")

    monkeypatch.setattr(ff, "_call_cli", fail_scoring)
    monkeypatch.setattr(ff, "write_index", fail_index)
    args = ["--task", "birds", "--prompt-version", "v0", "--spend-ceiling", "1"]
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            ff.main(args)
    else:
        assert ff.main(args) == 1
    error = capsys.readouterr().err
    assert "WARNING: Could not update campaign index" in error
    if not interrupted:
        assert "ERROR: Invented primary scoring failure" in error


def test_bird_preflight_and_legacy_defaults(
    bird_campaign: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = bird_campaign["audit"]
    assert set(audit["prompt_files"]) == {"zero-shot"}
    assert set(audit["inputs"]) == {"combined_text/zero-shot"}
    assert audit["positive_label"] == "1"
    assert "shot_conditions" not in audit  # No new field required in historical audits.
    assert TASKS["birds"].baseline is None
    tuned = {**audit, "runtime_concurrency": {"d3": 2, "chat": 8, "luna": 16, "microsoft": 4}}
    tuned_args = ff.CampaignCondition.from_audit(
        tuned, "combined_text", "zero-shot", tmp_path, 1, True
    ).command()
    overrides = {
        tuned_args[index + 1]: int(tuned_args[index + 2])
        for index, value in enumerate(tuned_args)
        if value == "--model-concurrency"
    }
    assert overrides == {
        ModelConfig.load(Path(tuned["model_files"][name]["path"])).arm: count
        for name, count in tuned["runtime_concurrency"].items()
    }
    for task, count in [("sharks", 16), ("animals", 8), ("leather", 8)]:
        assert ff.shot_conditions(task) == ff.SHOTS
        assert len(ff.input_modes(task)) * len(ff.shot_conditions(task)) * 4 == count
    for mode in ("title", "title_and_desc", "combined_text"):
        assert input_mode(input_fields(mode)) == mode
    with pytest.raises(ValueError, match="Unsupported input"):
        input_fields("seller")
    monkeypatch.setattr(ff, "_call_cli", call_in_process)
    root, upper = ff._preflight(audit, None, tmp_path / "preflight")
    manifests = [json.loads(p.read_text()) for p in root.rglob("manifest-*.json")]
    assert len(manifests) == 4 and upper > 0
    assert all(m["allowlisted_fields"] == ["title_and_desc"] for m in manifests)
    assert all(m["estimate"]["pending_adverts"] == 3 for m in manifests)
    assert all(m["estimate"]["cached_adverts"] == 0 for m in manifests)
    assert not list((tmp_path / "outputs/cache").rglob("*.json"))
    with pytest.raises(ValueError, match="differs from its declared"):
        bird_campaign["data"].write_text("changed fixture")
        ff.build_audit("birds", "v0", None)


@pytest.mark.parametrize("fail", [False, True])
def test_four_bird_arms_cache_and_mixed_report(
    bird_campaign: dict[str, Any],
    fail: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = bird_campaign["audit"]
    calls: list[str] = []

    def scorer(config: ModelConfig, _prompt: Prompt) -> Scorer:
        return InventedScorer(config.adapter == "chat", fail, calls)

    monkeypatch.setattr(runner, "make_scorer", scorer)
    root = tmp_path / "campaign"
    args = ff.CampaignCondition.from_audit(audit, "combined_text", "zero-shot", root, 1.0, False)
    assert "--combined-text" in args.command() and "--include-description" not in args.command()
    call_in_process(args)
    assert len(calls) == 12
    call_in_process(args)
    assert len(calls) == 12  # Includes refusal reuse without replay.
    ff.finalize(root, audit)
    selection_path = root / "selection-candidate.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    assert len(selection["pins"]) == 4
    for pin in selection["pins"]:
        manifest = json.loads(Path(pin).read_text())
        assert manifest["cached_results"] == 3
        assert manifest["input_condition"] == "combined_text"
        artifact = json.loads(
            Path(pin).with_name(Path(pin).name.replace("manifest-", "metrics-")).read_text()
        )
        row, _ = _load_pinned(Path(pin), selection)
        assert artifact["metrics"] == row["metrics"]
        assert row["metrics"]["positive_recall"] == 1.0
        assert row["metrics"]["workload"] == pytest.approx((2 if fail else 1) / 3)
        assert row["failures"] == int(fail)
    with (next(root.rglob("table1.csv"))).open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 4 and all(r["task"] == "birds" for r in table)
    assert next(root.rglob("figure1.png")).stat().st_size > 0
    assert next(root.rglob("figure1.svg")).stat().st_size > 0
    assert len(discover_runs(root)) == 8
    # Existing task snapshots still combine with birds in the same report.
    for task in ("sharks", "animals"):
        data = tmp_path / f"{task}.csv"
        data.write_text("id,title,label\na,Invented item,1\nb,Invented plastic item,0\n")
        old_args = [
            "run",
            "--task",
            task,
            "--data",
            str(data),
            "--positive-label",
            "1",
            "--zero-shot",
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--decision-prompt",
            str(ROOT / f"prompts/{task}/decision_zeroshot_v0.yaml"),
            "--output",
            str(tmp_path / task),
            "--cache",
            str(tmp_path / "fake-cache"),
        ]
        # Fake adapter is offline and bypasses the real-config scorer stub.
        with monkeypatch.context() as local:
            local.setattr(runner, "make_scorer", make_scorer)
            assert main(old_args) == 0
        pin = next((tmp_path / task).rglob("manifest-*.json"))
        selection["pins"].append(str(pin))
        selection.setdefault("datasets", {})[f"{task}-{file_hash(data)[:12]}"] = {
            "data": str(data),
            "positive_label": "1",
        }
    selection_path.write_text(yaml.safe_dump(selection))
    output = tmp_path / "mixed.html"
    render_evidence(selection_path, output)
    page = output.read_text()
    embedded = re.search(r'<script id="study-data" type="application/json">(.*?)</script>', page)
    assert embedded is not None
    datasets = json.loads(embedded.group(1))
    assert len(datasets) == 3
    bird = datasets[f"birds-{file_hash(bird_campaign['data'])[:12]}"]
    assert len(bird["rows"]) == 4 and all(r["kind"] == "measured" for r in bird["rows"])
    assert len(bird["resources"]) == 4
    assert sum(r["point"] is not None for r in bird["rows"]) == 1
    assert "Invented budgies for sale" not in page and "DO_NOT_SEND" not in page
    assert "title + desc" in page


def test_bird_cli_rejects_wrong_conditions_before_dispatch(
    bird_campaign: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = ["run", "--task", "birds", "--model", str(ROOT / "configs/models/fake.yaml")]
    assert main(base + ["--combined-text"]) == 1
    assert "--zero-shot" in capsys.readouterr().out
    assert main(base + ["--zero-shot"]) == 1
    assert "--combined-text" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()
    output = tmp_path / "inspection.json"
    assert main(["inspect", "--task", "birds", "--output", str(output)]) == 0
    inspected = json.loads(output.read_text())
    assert inspected["task"] == "birds" and inspected["positives"] == 1
