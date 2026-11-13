"""Progress is terminal output only; saved experiment evidence remains authoritative."""

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.progress import TerminalProgress
from wildlife_screening.prompts import Prompt
from wildlife_screening.runner import run_arm
from wildlife_screening.scorers.local import FakeScorer
from wildlife_screening.types import Advert, EvaluationData, ScoreResult

ROOT = Path(__file__).resolve().parents[1]


class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _run(
    data: EvaluationData,
    prompt: Prompt,
    tmp_path: Path,
    progress: TerminalProgress | None = None,
    scorer: FakeScorer | None = None,
    *,
    dry_run: bool = False,
) -> tuple[list[ScoreResult], dict[str, object]]:
    results, path = run_arm(
        data,
        ModelConfig.load(ROOT / "configs/models/fake.yaml"),
        prompt,
        tmp_path / "runs",
        tmp_path / "cache",
        scorer=scorer or FakeScorer(7),
        progress=progress,
        dry_run=dry_run,
    )
    return results, json.loads(path.read_text())


def test_progress_preserves_results_cache_and_resume(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    baseline, baseline_manifest = _run(data, prompt, tmp_path / "baseline")
    stream = TerminalBuffer()
    observed, observed_manifest = _run(
        data, prompt, tmp_path / "observed", TerminalProgress("Fake arm", stream)
    )
    assert observed == baseline
    for key in ("status", "model_config_hash", "prompt_hash", "data_file_hash", "totals"):
        assert observed_manifest[key] == baseline_manifest[key]
    baseline_entries = {
        json.loads(path.read_text())["advert_id"]: json.loads(path.read_text())
        for path in (tmp_path / "baseline/cache").glob("*/*/*.json")
    }
    observed_entries = {
        json.loads(path.read_text())["advert_id"]: json.loads(path.read_text())
        for path in (tmp_path / "observed/cache").glob("*/*/*.json")
    }
    assert baseline_entries.keys() == observed_entries.keys()
    for advert_id in baseline_entries:
        for key in ("result", "status", "input_hash", "model_config_hash", "prompt_hash"):
            assert baseline_entries[advert_id][key] == observed_entries[advert_id][key]
    display = stream.getvalue()
    assert "Fake arm" in display
    assert "20/20 processed" in display
    assert "0 pending" in display
    assert "0 cached" in display
    assert "0 errors" in display
    assert "ETA" in display
    assert "complete" in display

    cached_stream = TerminalBuffer()
    cached, cached_manifest = _run(
        data, prompt, tmp_path / "observed", TerminalProgress("Fake arm", cached_stream)
    )
    assert cached == observed
    assert cached_manifest["cached_results"] == 20
    assert "20 cached" in cached_stream.getvalue()
    assert "0 pending" in cached_stream.getvalue()


def test_progress_is_quiet_for_dry_runs_and_non_tty(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    tty = TerminalBuffer()
    _run(data, prompt, tmp_path / "dry", TerminalProgress("Fake", tty), dry_run=True)
    assert tty.getvalue() == ""
    ordinary = io.StringIO()
    _run(data, prompt, tmp_path / "live", TerminalProgress("Fake", ordinary))
    assert ordinary.getvalue() == ""


def test_partly_cached_progress_uses_only_new_results_for_eta(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    subset = EvaluationData(data.adverts[:5], data.labels[:5], data.sha256)
    _run(subset, prompt, tmp_path)
    stream = TerminalBuffer()
    results, manifest = _run(data, prompt, tmp_path, TerminalProgress("Fake", stream))
    assert len(results) == 20
    assert manifest["cached_results"] == 5
    display = stream.getvalue()
    assert "5/20 processed" in display
    assert "15 pending" in display
    assert "5 cached" in display
    assert "20/20 processed" in display


def test_cli_keeps_json_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stream = TerminalBuffer()
    monkeypatch.setattr("sys.stderr", stream)
    status = main(
        [
            "run",
            "--data",
            str(ROOT / "tests/fixtures/synthetic20.csv"),
            "--model",
            str(ROOT / "configs/models/fake.yaml"),
            "--output",
            str(tmp_path / "runs"),
            "--cache",
            str(tmp_path / "cache"),
            "--score-only",
        ]
    )
    assert status == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert "20/20 processed" in stream.getvalue()


def test_error_result_is_processed_but_not_successful(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    class InvalidFake(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            return [
                replace(result, p_relevant=None, error="invalid")
                for result in super().score(adverts)
            ]

    stream = TerminalBuffer()
    results, manifest = _run(
        data, prompt, tmp_path, TerminalProgress("Fake", stream), InvalidFake()
    )
    assert len(results) == 20
    assert manifest["status"] == "incomplete"
    assert "20 errors" in stream.getvalue()
    assert "incomplete" in stream.getvalue()


def test_interrupted_run_stops_at_saved_count(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    class InterruptedFake(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            if adverts[0].advert_id == data.adverts[1].advert_id:
                raise RuntimeError("connection lost")
            return super().score(adverts)

    stream = TerminalBuffer()
    with pytest.raises(RuntimeError, match="uncertain"):
        _run(data, prompt, tmp_path, TerminalProgress("Fake", stream), InterruptedFake())
    assert "1/20 processed" in stream.getvalue()
    assert "stopped" in stream.getvalue()
    entries = [json.loads(path.read_text()) for path in (tmp_path / "cache").glob("*/*/*.json")]
    assert sorted(entry["status"] for entry in entries) == ["complete", "in_flight"]


def test_terminal_failure_cannot_stop_scoring(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    class BrokenTerminal(TerminalBuffer):
        def write(self, value: str) -> int:
            raise OSError("terminal closed")

    results, manifest = _run(data, prompt, tmp_path, TerminalProgress("Fake", BrokenTerminal()))
    assert len(results) == 20
    assert manifest["status"] == "complete"
