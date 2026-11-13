from __future__ import annotations

import bdb
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from matplotlib.axes import Axes

from wildlife_screening.cache import cache_spend, reservation
from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.plots import plot_roc
from wildlife_screening.prompts import Prompt
from wildlife_screening.report import create_report
from wildlife_screening.runner import BudgetExceeded, run_arm
from wildlife_screening.scorers.api import PromptedLLMScorer
from wildlife_screening.scorers.base import Scorer
from wildlife_screening.scorers.local import FakeScorer
from wildlife_screening.types import Advert, EvaluationData, ScoreResult
from wildlife_screening.utils import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["local", "paid", "remote", "missing_authorization"])
def test_explicit_local_retry_preserves_uncertain_attempt(
    data: EvaluationData, prompt: Prompt, tmp_path: Path, case: str
) -> None:
    config = ModelConfig(
        "invented local",
        "invented",
        "system_one",
        endpoint="https://example.test/api"
        if case == "remote"
        else "http://127.0.0.1:8080/v1/systemone",
        billable=case == "paid",
        price_input_per_million=0,
        price_output_per_million=0,
    )
    one = replace(data, adverts=data.adverts[:1], labels=data.labels[:1])
    cache = tmp_path / "cache"

    class LostResponse(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            raise RuntimeError("Invented lost response")

    with pytest.raises(RuntimeError, match="uncertain"):
        run_arm(one, config, prompt, tmp_path / "lost", cache, scorer=LostResponse())
    path = next(cache.glob("*/*/*.json"))
    original = json.loads(path.read_text())
    entry = {**original, "status": "retry_authorized_local"}
    if case != "missing_authorization":
        entry["retry_authorization"] = {
            "authorized_at": "2026-10-21T00:00:00+00:00",
            "reason": "Invented explicit user authorization for local retry",
            "previous_entry": original,
        }
    write_json(path, entry)
    scorer = FakeScorer()
    if case != "local":
        with pytest.raises(RuntimeError, match="Invalid local retry authorization"):
            run_arm(one, config, prompt, tmp_path / "retry", cache, scorer=scorer)
        assert scorer.calls == 0
        assert json.loads(path.read_text()) == entry
        return
    run_arm(one, config, prompt, tmp_path / "dry", cache, scorer=scorer, dry_run=True)
    assert scorer.calls == 0 and json.loads(path.read_text()) == entry
    run_arm(one, config, prompt, tmp_path / "retry", cache, scorer=scorer)
    saved = json.loads(path.read_text())
    assert scorer.calls == 1 and saved["status"] == "complete"
    assert saved["retry_authorization"]["previous_entry"] == original
    run_arm(one, config, prompt, tmp_path / "cached", cache, scorer=scorer)
    assert scorer.calls == 1


def test_static_published_markers_and_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Differentiate the two published modalities without plotting F1-only GPT-4."""
    rows = yaml.safe_load((ROOT / "baselines/published_sharks.yaml").read_text())["rows"]
    seen: list[tuple[str, str, str]] = []
    original = Axes.scatter

    def capture(self: Axes, *args: Any, **kwargs: Any) -> Any:
        seen.append((str(kwargs["label"]), str(kwargs["marker"]), str(kwargs["color"])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture)
    plot_roc([1, 0], {}, rows, tmp_path)
    assert [(marker, color) for _, marker, color in seen] == [
        ("x", "#2367b1"),
        ("x", "#e27422"),
        ("s", "#2367b1"),
        ("s", "#e27422"),
    ]
    assert all("GPT-4" not in label for label, _, _ in seen)


def test_static_published_fpr_is_derived_for_tables(tmp_path: Path) -> None:
    rows = create_report([1, 0], {}, tmp_path, ROOT / "baselines/published_sharks.yaml")
    assert rows[0]["false_positive_rate"] == pytest.approx(0.892 * (1 / 0.958 - 1))
    assert rows[4].get("false_positive_rate") is None
    csv_rows = list(csv.DictReader((tmp_path / "table1.csv").open()))
    assert "false_positive_rate" in csv_rows[0]
    assert "workload" not in csv_rows[0]
    assert csv_rows[0]["false_positive_rate"] != "unknown"
    assert csv_rows[4]["false_positive_rate"] == "unknown"
    markdown = (tmp_path / "table1.md").read_text()
    assert "False-positive rate (FPR) %" in markdown
    assert "Review %" not in markdown


def fake_config() -> ModelConfig:
    return ModelConfig.load(ROOT / "configs/models/fake.yaml")


def test_resume_no_repay(data: EvaluationData, prompt: Prompt, tmp_path: Path) -> None:
    scorer = FakeScorer(7)
    config = fake_config()
    first, _ = run_arm(
        data, config, prompt, tmp_path / "first", tmp_path / "cache", scorer=scorer, concurrency=3
    )
    assert scorer.calls == 20
    second, manifest = run_arm(
        data, config, prompt, tmp_path / "second", tmp_path / "cache", scorer=scorer
    )
    assert scorer.calls == 20 and second == first
    m = json.loads(manifest.read_text())
    assert m["cached_results"] == 20
    assert m["new_call_totals"]["cost"] == 0
    assert m["data_file_hash"] == data.sha256
    assert m["prompt_hash"] == prompt.sha256
    assert m["model_config_hash"]
    assert m["finished_at"] >= m["started_at"]


def test_dry_run_no_calls_and_unknown_prices(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    scorer = FakeScorer()
    config = ModelConfig("unknown", "model", "chat", endpoint="https://example.test/api")
    results, manifest = run_arm(
        data, config, prompt, tmp_path / "dry", tmp_path / "cache", dry_run=True, scorer=scorer
    )
    assert not results and scorer.calls == 0
    assert json.loads(manifest.read_text())["estimate"]["cost_upper_estimate"] is None
    assert not list((tmp_path / "cache").glob("*/*/*.json"))
    with pytest.raises(ValueError, match="Unknown price"):
        run_arm(data, config, prompt, tmp_path / "live", tmp_path / "cache", scorer=scorer)
    assert scorer.calls == 0


def test_budget_abort_then_resume(data: EvaluationData, prompt: Prompt, tmp_path: Path) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    config = replace(
        fake_config(), billable=True, price_input_per_million=1, price_output_per_million=1
    )
    # Fake calls have no real cost; use a request-priced stub to exercise reservation accounting.
    config = replace(config, adapter="chat", endpoint="https://example.test/api")

    class ChargedFake(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            return [replace(r, cost=0.004) for r in super().score(adverts)]

    scorer = ChargedFake()
    bound = reservation(config, prompt, data.adverts[0])["external_spend_reserved"]
    with pytest.raises(BudgetExceeded):
        run_arm(
            data,
            config,
            prompt,
            tmp_path / "run",
            tmp_path / "cache",
            scorer=scorer,
            spend_ceiling=bound + 0.0001,
        )
    assert scorer.calls == 1
    aborted = json.loads(next((tmp_path / "run").glob("manifest-*.json")).read_text())
    assert aborted["status"] == "aborted"
    assert aborted["totals"]["cost"] == 0.004
    results, _ = run_arm(
        data,
        config,
        prompt,
        tmp_path / "resume",
        tmp_path / "cache",
        scorer=scorer,
        spend_ceiling=1,
    )
    assert len(results) == 20 and scorer.calls == 20


def test_uncertain_calls_never_automatically_replayed(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    config = fake_config()

    class BrokenScorer(Scorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            raise RuntimeError("response was lost after sending")

    with pytest.raises(RuntimeError):
        run_arm(data, config, prompt, tmp_path / "run", tmp_path / "cache", scorer=BrokenScorer())
    scorer = FakeScorer()
    with pytest.raises(RuntimeError, match="Uncertain prior"):
        run_arm(data, config, prompt, tmp_path / "resume", tmp_path / "cache", scorer=scorer)
    assert scorer.calls == 0


def test_payload_debug_exit_is_reusable_without_a_provider_call(tmp_path: Path) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    config = ModelConfig(
        "offline-chat",
        "offline-model",
        "chat",
        endpoint="https://example.test/api",
        price_input_per_million=1,
        price_output_per_million=1,
    )
    data = EvaluationData([Advert("invented", "Invented research item")], [0], "invented")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "not relevant"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    scorer = PromptedLLMScorer(config, prompt, client)
    original_payload = scorer.payload

    def stopped_payload(advert: Advert) -> dict[str, object]:
        raise bdb.BdbQuit

    scorer.payload = stopped_payload  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="before dispatch"):
        run_arm(
            data,
            config,
            prompt,
            tmp_path / "stopped",
            tmp_path / "cache",
            scorer=scorer,
            spend_ceiling=1,
        )
    entry_path = next((tmp_path / "cache").glob("*/*/*.json"))
    entry = json.loads(entry_path.read_text())
    assert entry["status"] == "cancelled_before_send"
    assert "sent_at" not in entry
    assert cache_spend(tmp_path / "cache") == 0
    assert not requests

    scorer.payload = original_payload  # type: ignore[method-assign]
    results, _ = run_arm(
        data,
        config,
        prompt,
        tmp_path / "resumed",
        tmp_path / "cache",
        scorer=scorer,
        spend_ceiling=1,
    )
    assert len(requests) == 1
    assert results[0].error is None
    assert results[0].score_type == "binary" and results[0].p_relevant == 0.0
    assert json.loads(entry_path.read_text())["status"] == "complete"


def test_changed_content_fails_closed(data: EvaluationData, prompt: Prompt, tmp_path: Path) -> None:
    config = fake_config()
    run_arm(data, config, prompt, tmp_path / "first", tmp_path / "cache")
    changed = replace(data, adverts=[replace(data.adverts[0], title="changed"), *data.adverts[1:]])
    with pytest.raises(ValueError, match="text changed"):
        run_arm(changed, config, prompt, tmp_path / "second", tmp_path / "cache")


def test_prompt_and_config_changes_invalidate_cache(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    scorer = FakeScorer()
    config = fake_config()
    run_arm(data, config, prompt, tmp_path / "a", tmp_path / "cache", scorer=scorer)
    run_arm(
        data,
        replace(config, model_id="other"),
        prompt,
        tmp_path / "b",
        tmp_path / "cache",
        scorer=scorer,
    )
    run_arm(
        data,
        config,
        replace(prompt, sha256="new-prompt"),
        tmp_path / "c",
        tmp_path / "cache",
        scorer=scorer,
    )
    assert scorer.calls == 60


def test_refusals_cached_without_repay(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    class Refused(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            self.calls += len(adverts)
            return [
                ScoreResult(
                    a.advert_id, None, "native_prob", '{"refusal":true}', 0, 0, 0, 0, "refusal"
                )
                for a in adverts
            ]

    scorer = Refused()
    results, _ = run_arm(
        data, fake_config(), prompt, tmp_path / "run", tmp_path / "cache", scorer=scorer
    )
    run_arm(data, fake_config(), prompt, tmp_path / "resume", tmp_path / "cache", scorer=scorer)
    assert scorer.calls == 20
    rows = create_report(
        data.labels,
        {"refused": results},
        tmp_path / "report",
        ROOT / "baselines/published_sharks.yaml",
    )
    assert rows[0]["status"] == "finished with failures"
    assert rows[0]["coverage"] == 0
    assert rows[0]["workload"] == 1
    assert rows[0]["false_positive_rate"] == 1
    assert rows[0]["positive_recall"] == 1
    assert rows[0]["fp"] == len(data.labels) - sum(data.labels)
    assert all(result.p_relevant is None for result in results)


def test_end_to_end_cli(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    assert (
        main(
            [
                "run",
                "--zero-shot",
                "--data",
                str(ROOT / "tests/fixtures/synthetic20.csv"),
                "--model",
                str(ROOT / "configs/models/fake.yaml"),
                "--model",
                str(ROOT / "configs/models/fake_binary.yaml"),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    for filename in ("table1.csv", "table1.md", "figure1.png", "figure1.svg"):
        assert (output / "zero-shot" / filename).stat().st_size > 100
    rows = list(csv.DictReader((output / "zero-shot" / "table1.csv").open()))
    assert len(rows) == 2
    assert "false_positive_rate" in rows[0]
    assert "workload" not in rows[0]
    assert rows[0]["cutoff"] == "0.500000"
    assert rows[0]["n"] == "20"
    assert rows[1]["cutoff"] == "binary decision"
    assert (rows[1]["tp"], rows[1]["fp"], rows[1]["fn"], rows[1]["tn"]) == ("2", "10", "3", "5")
    assert all(row["provenance"].startswith("synthetic/imported") for row in rows)
    assert all(row["published_f1"] == "unknown" for row in rows)
    assert "binary point" in (output / "zero-shot" / "figure1.svg").read_text()


def test_chat_binary_metrics_use_chosen_labels(tmp_path: Path) -> None:
    results = [
        ScoreResult("invented-a", 1.0, "binary", "", 10, 1, 0.01, 0.1),
        ScoreResult("invented-b", 0.0, "binary", "", 10, 1, 0.01, 0.1),
        ScoreResult("invented-c", 1.0, "binary", "", 10, 1, 0.01, 0.1),
    ]
    row = create_report(
        [1, 0, 0],
        {"chat": results},
        tmp_path,
        None,
        {"chat": {"prompt_family": "llm", "condition": "few-shot"}},
    )[0]
    assert row["cutoff"] == "binary decision"
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 0, 1)
    assert row["positive_recall"] == 1.0
    assert row["false_positive_rate"] == 0.5
    assert row["workload"] == 2 / 3


def test_score_only_saves_single_raw_result_without_report(tmp_path: Path) -> None:
    sample = tmp_path / "single.csv"
    sample.write_text(
        "advert_id,title,label,relevant\n"
        "mock-001,Lot of three modern shark teeth,1,animal origin\n",
        encoding="utf-8",
    )
    output = tmp_path / "outputs"
    assert (
        main(
            [
                "run",
                "--data",
                str(sample),
                "--model",
                str(ROOT / "configs/models/fake.yaml"),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
                "--score-only",
            ]
        )
        == 0
    )
    scores_path = next(output.rglob("scores-*.json"))
    rows = json.loads(scores_path.read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["advert_id"] == "mock-001"
    assert rows[0]["raw_output"]
    assert not list(output.rglob("table1.csv"))


def test_check_and_dry_run_route_outputs_without_changing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep check records separate while reusing the ordinary cache namespace."""
    sample = tmp_path / "single.csv"
    sample.write_text(
        "advert_id,title,label,relevant\n"
        "mock-001,Invented modern shark tooth lot,1,animal origin\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    base = [
        "run",
        "--data",
        str(sample),
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--decision-prompt",
        str(ROOT / "prompts/sharks/decision_fewshot_v0.yaml"),
        "--spend-ceiling",
        "1.25",
    ]

    assert main(base + ["--dry-run"]) == 0
    checks = tmp_path / "outputs/checks"
    dry_manifest = json.loads(next(checks.rglob("manifest-*.json")).read_text())
    assert dry_manifest["status"] == "dry_run"
    assert not list(checks.rglob("scores-*.json"))

    assert main(base + ["--check", "--score-only"]) == 0
    check_scores = next(checks.rglob("scores-*.json"))
    assert json.loads(check_scores.read_text())[0]["advert_id"] == "mock-001"
    check_manifests = [json.loads(path.read_text()) for path in checks.rglob("manifest-*.json")]
    check_manifest = next(item for item in check_manifests if item["status"] == "complete")
    assert check_manifest["spend_ceiling_usd"] == 1.25
    cache_entries = list((tmp_path / "outputs/cache").rglob("*.json"))
    assert len(cache_entries) == 1
    cache_bytes = cache_entries[0].read_bytes()

    assert main(base + ["--score-only"]) == 0
    comparison = tmp_path / "outputs/comparison"
    live_manifest = json.loads(next(comparison.rglob("manifest-*.json")).read_text())
    assert live_manifest["estimate"]["cached_adverts"] == 1
    assert live_manifest["estimate"]["pending_adverts"] == 0
    assert live_manifest["spend_ceiling_usd"] == check_manifest["spend_ceiling_usd"]
    assert next(comparison.rglob("scores-*.json")).is_file()
    assert cache_entries[0].read_bytes() == cache_bytes

    custom = tmp_path / "custom-output"
    assert main(base + ["--check", "--dry-run", "--output", str(custom)]) == 0
    assert next(custom.rglob("manifest-*.json")).is_file()
    assert len(list(checks.rglob("manifest-*.json"))) == 2


def test_precomputed_source_hash_changes(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("advert_id,p_relevant\nx,0.2\n")
    config = ModelConfig(
        "import", "import", "precomputed", billable=False, options={"path": str(path)}
    )
    a = config.fingerprint(("title",))
    path.write_text("advert_id,p_relevant\nx,0.7\n")
    assert a != config.fingerprint(("title",))


def test_concurrent_batch_budget_reserved_before_any_call(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    config = replace(
        fake_config(),
        adapter="chat",
        endpoint="https://example.test/api",
        billable=True,
        price_input_per_million=1,
        price_output_per_million=1,
    )
    scorer = FakeScorer()
    one = reservation(config, prompt, data.adverts[0])["external_spend_reserved"]
    with pytest.raises(BudgetExceeded):
        run_arm(
            data,
            config,
            prompt,
            tmp_path / "run",
            tmp_path / "cache",
            scorer=scorer,
            concurrency=4,
            spend_ceiling=one * 2,
        )
    assert scorer.calls == 0
    assert not list((tmp_path / "cache").glob("*/*/*.json"))


def test_overlong_request_blocked_before_call(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    changed = replace(data, adverts=[replace(a, title="x" * 20000) for a in data.adverts])
    scorer = FakeScorer()
    config = replace(fake_config(), adapter="chat", endpoint="https://example.test/api")
    with pytest.raises(ValueError, match="short text input bound"):
        run_arm(
            changed,
            config,
            prompt,
            tmp_path / "dry",
            tmp_path / "cache",
            dry_run=True,
            scorer=scorer,
        )
    assert scorer.calls == 0
    with pytest.raises(ValueError, match="short text input bound"):
        run_arm(changed, config, prompt, tmp_path / "run", tmp_path / "cache", scorer=scorer)
    assert scorer.calls == 0


def test_model_specific_input_bound_allows_longer_request(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    changed = replace(data, adverts=[replace(a, title="x" * 20000) for a in data.adverts])
    config = replace(
        fake_config(),
        adapter="chat",
        endpoint="https://example.test/api",
        max_input_bound_tokens=32768,
    )
    scorer = FakeScorer()
    results, _ = run_arm(
        changed, config, prompt, tmp_path / "run", tmp_path / "cache", scorer=scorer
    )
    assert len(results) == len(data.adverts)
    assert scorer.calls == len(data.adverts)


def test_missing_precomputed_file_still_has_abort_manifest(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    config = ModelConfig(
        "import",
        "import",
        "precomputed",
        billable=False,
        options={"path": str(tmp_path / "missing.csv")},
    )
    with pytest.raises(FileNotFoundError):
        run_arm(data, config, prompt, tmp_path / "run", tmp_path / "cache")
    manifest = json.loads(next((tmp_path / "run").glob("manifest-*.json")).read_text())
    assert manifest["status"] == "aborted"


def test_unknown_billed_usage_stops_dispatch_and_reserves(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    config = replace(
        fake_config(),
        adapter="chat",
        endpoint="https://example.test/api",
        billable=True,
        price_input_per_million=1,
        price_output_per_million=1,
    )

    class UnknownUsage(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            return [replace(r, tokens_in=None, cost=None) for r in super().score(adverts)]

    scorer = UnknownUsage()
    with pytest.raises(RuntimeError, match="Missing billed usage"):
        run_arm(
            data,
            config,
            prompt,
            tmp_path / "run",
            tmp_path / "cache",
            scorer=scorer,
            spend_ceiling=1,
        )
    assert scorer.calls == 1
    entry = json.loads(next((tmp_path / "cache").glob("*/*/*.json")).read_text())
    assert entry["status"] == "complete" and entry["result"]["cost"] is None
    assert entry["reservation"]["external_spend_reserved"] > 0
    # A fully cached arm with missing billing must also stop, even with no pending work.
    one = replace(data, adverts=data.adverts[:1], labels=data.labels[:1])
    with pytest.raises(RuntimeError, match="Missing billed usage in cache"):
        run_arm(
            one,
            config,
            prompt,
            tmp_path / "cached",
            tmp_path / "cache",
            scorer=scorer,
            spend_ceiling=1,
        )
    assert scorer.calls == 1


def test_concurrent_success_saved_when_sibling_fails(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    class PartialFailure(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            if adverts[0].advert_id == data.adverts[0].advert_id:
                raise RuntimeError("Lost response")
            return super().score(adverts)

    with pytest.raises(RuntimeError, match="uncertain"):
        run_arm(
            data,
            fake_config(),
            prompt,
            tmp_path / "run",
            tmp_path / "cache",
            scorer=PartialFailure(),
            concurrency=2,
        )
    entries = [json.loads(p.read_text()) for p in (tmp_path / "cache").glob("*/*/*.json")]
    assert len(entries) == 2
    assert sorted(e["status"] for e in entries) == ["complete", "in_flight"]
    complete = next(e for e in entries if e["status"] == "complete")
    assert complete["result"]["advert_id"] == data.adverts[1].advert_id


def test_budget_is_shared_across_arm_configs(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    prompt = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    config = replace(
        fake_config(),
        adapter="chat",
        endpoint="https://example.test/api",
        billable=True,
        price_input_per_million=1,
        price_output_per_million=1,
    )

    class ChargedFake(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            return [replace(r, cost=0.0001) for r in super().score(adverts)]

    run_arm(
        data,
        config,
        prompt,
        tmp_path / "first",
        tmp_path / "cache",
        scorer=ChargedFake(),
        spend_ceiling=1,
    )
    second = ChargedFake()
    with pytest.raises(BudgetExceeded):
        run_arm(
            data,
            replace(config, model_id="second"),
            prompt,
            tmp_path / "second",
            tmp_path / "cache",
            scorer=second,
            spend_ceiling=0.003,
        )
    assert second.calls == 0
