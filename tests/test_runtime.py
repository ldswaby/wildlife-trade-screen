"""Verify throughput changes without private data, credentials or paid calls."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from wildlife_screening import full_factorial as ff
from wildlife_screening import runner
from wildlife_screening import runtime_benchmark as bench
from wildlife_screening.cache import cache_spend
from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.prompts import Prompt
from wildlife_screening.scorers.base import Scorer
from wildlife_screening.scorers.local import FakeScorer
from wildlife_screening.types import Advert, EvaluationData, ScoreResult

ROOT = Path(__file__).resolve().parents[1]


def test_refill_without_waiting_for_slow_sibling(
    data: EvaluationData, prompt: Prompt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    third_started = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    scans = 0
    scan = cache_spend

    def counted(cache: Path) -> float:
        nonlocal scans
        scans += 1
        return scan(cache)

    class UnevenScorer(Scorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            advert = adverts[0]
            if advert.advert_id == data.adverts[0].advert_id:
                assert third_started.wait(5), "Workers stalled behind the slow first advert"
            if advert.advert_id == data.adverts[2].advert_id:
                third_started.set()
            with lock:
                active -= 1
            return [ScoreResult(advert.advert_id, 0.5, "native_prob", "invented", 0, 0, 0, 0)]

    # run_arm makes both ledger scans through its own module-level name.
    monkeypatch.setattr(runner, "cache_spend", counted)
    results, path = runner.run_arm(
        data,
        ModelConfig.load(ROOT / "configs/models/fake.yaml"),
        prompt,
        tmp_path / "run",
        tmp_path / "cache",
        concurrency=2,
        scorer=UnevenScorer(),
    )
    assert peak == 2 and active == 0
    assert scans == 2  # independent of the number of adverts
    assert [r.advert_id for r in results] == [a.advert_id for a in data.adverts]
    manifest = json.loads(path.read_text())
    assert manifest["scoring_wall_time_s"] > 0
    assert manifest["cache_external_spend_after_usd"] == 0


def test_per_arm_cli_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    original = runner.run_arm

    def capture(*args: Any, **kwargs: Any) -> tuple[list[ScoreResult], Path]:
        seen.append(args[6])
        return original(*args, **kwargs)

    monkeypatch.setattr("wildlife_screening.cli.run_arm", capture)
    config = ModelConfig.load(ROOT / "configs/models/fake.yaml")
    assert (
        main(
            [
                "run",
                "--data",
                str(ROOT / "tests/fixtures/synthetic20.csv"),
                "--model",
                str(ROOT / "configs/models/fake.yaml"),
                "--concurrency",
                "2",
                "--model-concurrency",
                config.arm,
                "4",
                "--score-only",
                "--output",
                str(tmp_path / "out"),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert seen == [4]


def test_d3_checks_requested_slots_and_per_slot_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: Any) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps(props).encode()

    props = {
        "model_path": "/invented/vllm-d3-Q4_K_M.gguf",
        "total_slots": 2,
        "default_generation_settings": {"n_ctx": 8192},
    }
    monkeypatch.setattr(ff, "urlopen", lambda *args, **kwargs: Response())
    ff._check_d3(2)
    with pytest.raises(ValueError, match="1 slot"):
        ff._check_d3()
    props["default_generation_settings"]["n_ctx"] = 4096  # type: ignore[index]
    with pytest.raises(ValueError, match="per-slot context"):
        ff._check_d3(2)


def test_invented_benchmark_and_selection() -> None:
    first = bench.invented_adverts(64, "one")
    second = bench.invented_adverts(64, "two")
    assert [a.text() for a in first] == [a.text() for a in second]
    assert {len(a.evidence()) for a in first} == {128, 512, 2048, 6000}
    assert not ({a.advert_id for a in first} & {a.advert_id for a in second})
    rows = [
        {"arm": "chat", "status": "complete", "errors": 0, "concurrency": c, "adverts_per_s": speed}
        for c, speed in [(1, 10), (4, 20), (8, 21), (16, 25)]
    ]
    assert bench.select_concurrency(rows, "chat") == 16
    assert bench.select_concurrency(rows[:-1], "chat") == 4
    assert bench.select_concurrency(rows, "d3") is None


def test_benchmark_preparation_is_offline_and_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "CACHE", tmp_path / "cache")
    root = tmp_path / "benchmark"
    plan = bench.prepare(root)
    assert 0 < plan["hosted_screen_and_two_confirmations_upper_usd"] < 1
    assert plan["cumulative_ceiling_usd"] == 1
    assert bench.prepare(root) == plan
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_benchmark_records_fresh_round_and_reuses_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "CACHE", tmp_path / "cache")
    plan = bench.prepare(tmp_path / "benchmark")
    scorer = FakeScorer()
    monkeypatch.setattr(runner, "make_scorer", lambda *_: scorer)
    row = bench.time_setting(tmp_path / "benchmark", plan, "luna", 4, "screen")
    assert row["cached_results"] == 0 and row["status"] == "complete"
    assert scorer.calls == 64 and row["adverts_per_s"] > 0
    assert bench.time_setting(tmp_path / "benchmark", plan, "luna", 4, "screen") == row
    assert scorer.calls == 64
