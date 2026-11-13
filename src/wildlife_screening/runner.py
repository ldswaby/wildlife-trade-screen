"""Reserve spend before dispatch and never automatically replay an uncertain request."""

import fcntl
import math
import subprocess
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .cache import PendingCall, cache_spend, cached_charge, prepare_cache
from .config import ModelConfig
from .data import input_mode
from .metrics import EVALUATION_POLICY, screening_metrics
from .prompts import Prompt
from .scorers import Scorer, make_scorer
from .scorers.api import RequestNotSent
from .types import EvaluationData, ScoreResult
from .utils import file_hash, hash_json, write_json


class BudgetExceeded(RuntimeError):
    """Signal that no further requests may be dispatched within the selected ceiling."""

    pass


def now() -> str:
    """Use UTC ISO timestamps so manifests are unambiguous across machines."""
    return datetime.now(UTC).isoformat()


def git_sha() -> str | None:
    """Record the current commit, returning None instead of inventing repository history."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def totals(results: list[ScoreResult]) -> dict[str, Any]:
    """Keep incomplete resource accounting visibly unknown.

    Args:
        results: Cached or newly scored adverts to aggregate.

    Returns:
        Token/cost totals, known cost subtotal, unknown-cost count and error count.
    """

    def complete_sum(values: list[int | float | None]) -> int | float | None:
        """Avoid presenting a partial sum as a complete resource total."""
        return None if any(v is None for v in values) else sum(v for v in values if v is not None)

    return {
        "tokens_in": complete_sum([r.tokens_in for r in results]),
        "tokens_out": complete_sum([r.tokens_out for r in results]),
        "cost": complete_sum([r.cost for r in results]),
        "latency_s_total": sum(r.latency_s for r in results),
        "cost_known_subtotal": sum(r.cost for r in results if r.cost is not None),
        "unknown_cost_count": sum(r.cost is None for r in results),
        "errors": sum(r.error is not None for r in results),
    }


class ProgressSink(Protocol):
    """Receive display-only events after cache inspection and durable result writes."""

    def begin(self, cached: int, pending: int, cached_errors: int) -> None:
        """Display the initial work count."""

    def saved_result(self, result: ScoreResult) -> None:
        """Display one newly persisted result."""

    def finish(self, status: str) -> None:
        """Close the display with the manifest status."""


def notify_progress(progress: ProgressSink | None, method: str, *args: Any) -> None:
    """Never let optional terminal output alter an experiment or its cache."""
    if progress is not None:
        try:
            getattr(progress, method)(*args)
        except Exception:
            pass


def score_batch(
    batch: list[PendingCall],
    active: Scorer,
    concurrency: int,
    results: dict[str, ScoreResult],
    new_results: list[ScoreResult],
    progress: ProgressSink | None = None,
    *,
    existing: float,
    spend_ceiling: float,
    billable: bool,
    stop_on_error: bool = False,
) -> float:
    """Refill bounded workers and persist completions under one cache lock.

    Reservations and accounting are owned by the calling thread. Once a failure
    is observed, drain already submitted work without admitting another request.
    The caller reconciles the incremental total against disk before unlocking.
    """

    def score_one(item: PendingCall) -> ScoreResult:
        returned = active.score([item.advert])
        if len(returned) != 1 or returned[0].advert_id != item.advert.advert_id:
            raise ValueError("Scorer violated advert identity/cardinality contract")
        return returned[0]

    failures: list[BaseException] = []
    pending = iter(batch)
    exhausted = False
    running: dict[Future[ScoreResult], tuple[PendingCall, float]] = {}
    # Preserve the existing fail-before-dispatch guarantee when the initial
    # worker window cannot fit under the authorised cumulative ceiling.
    initial = 0.0
    for item in batch[:concurrency]:
        charge = item.reservation["external_spend_reserved"]
        if charge is None:
            raise ValueError("Unknown external price; cannot enforce spend ceiling")
        initial += float(charge) - item.prior_charge()
    if existing + initial > spend_ceiling + 1e-12:
        raise BudgetExceeded("Initial request reservations would exceed the cache spend ceiling")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while running or (not exhausted and not failures):
            while not failures and not exhausted and len(running) < concurrency:
                try:
                    item = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                charge = item.reservation["external_spend_reserved"]
                if charge is None:
                    failures.append(
                        ValueError("Unknown external price; cannot enforce spend ceiling")
                    )
                    break
                # An authorised retry may already contribute a reservation to
                # the initial scan; replace rather than double-count that amount.
                prior_charge = item.prior_charge()
                amount = float(charge)
                if existing + amount - prior_charge > spend_ceiling + 1e-12:
                    failures.append(
                        BudgetExceeded(
                            "Next request reservation would exceed the cache spend ceiling"
                        )
                    )
                    break
                item.entry["sent_at"] = now()
                try:
                    write_json(item.path, item.entry)
                    existing += amount - prior_charge
                    running[executor.submit(score_one, item)] = (item, amount)
                except BaseException as exc:
                    failures.append(exc)
                    break
            if not running:
                break
            try:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
            except BaseException as exc:
                failures.append(exc)
                continue
            # Process the whole observed completion set before refilling, so a
            # sibling's missing usage/error cannot be hidden by a fast success.
            for future in done:
                item, reserved = running.pop(future)
                try:
                    result = future.result()
                except RequestNotSent as exc:
                    item.entry["status"] = "cancelled_before_send"
                    item.entry["reserved_at"] = item.entry.pop("sent_at", None)
                    item.entry["cancelled_before_send_at"] = now()
                    write_json(item.path, item.entry)
                    existing -= reserved
                    failures.append(
                        RuntimeError(
                            "Scorer stopped before dispatch; no call was sent for cancelled adverts"
                        )
                    )
                    failures[-1].__cause__ = exc
                    continue
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        failure = RuntimeError(
                            "Scorer failed; uncertain calls remain reserved in cache"
                        )
                        failure.__cause__ = exc
                        failures.append(failure)
                    else:
                        failures.append(exc)
                    continue
                item.entry["status"] = "complete"
                item.entry["finished_at"] = now()
                item.entry["result"] = asdict(result)
                write_json(item.path, item.entry)
                existing += cached_charge(item.entry) - reserved
                results[item.advert.advert_id] = result
                new_results.append(result)
                notify_progress(progress, "saved_result", result)
                if existing > spend_ceiling + 1e-12:
                    failures.append(
                        BudgetExceeded(
                            "Provider usage exceeded reservation; stopped before further calls"
                        )
                    )
                if billable and result.cost is None:
                    failures.append(
                        RuntimeError("Missing billed usage; reservation retained, no further calls")
                    )
                if stop_on_error and result.error is not None:
                    failures.append(
                        RuntimeError("Benchmark inference failed; no further calls dispatched")
                    )
    if failures:
        raise failures[0]
    return existing


def _write_results(
    output: Path, manifest: dict[str, Any], data: EvaluationData, results: dict[str, ScoreResult]
) -> list[ScoreResult]:
    """Write full-denominator metrics and scores in the original advert order."""
    run_id = manifest["run_id"]
    manifest["status"] = (
        "complete" if all(r.error is None for r in results.values()) else "incomplete"
    )
    ordered = [results[a.advert_id] for a in data.adverts]
    coverage = screening_metrics(data.labels, ordered)
    manifest.update({key: coverage[key] for key in ("valid_predictions", "failures", "coverage")})
    score_path = output / f"scores-{run_id}.json"
    write_json(score_path, [asdict(r) for r in ordered])
    # Snapshot loading verifies this checksum, so edits to the predictions are detected.
    manifest["scores_sha256"] = file_hash(score_path)
    write_json(
        output / f"metrics-{run_id}.json",
        {
            "run_id": run_id,
            "arm": manifest["arm"],
            "task": manifest["task"],
            "data_file_hash": data.sha256,
            "prompt_hash": manifest["prompt_hash"],
            "model_config_hash": manifest["model_config_hash"],
            "score_type": "+".join(sorted({r.score_type for r in ordered})),
            "decision_rule": "binary decision"
            if any(r.score_type == "binary" for r in ordered)
            else "score >= 0.5",
            "evaluation_policy": EVALUATION_POLICY,
            "metrics": coverage,
        },
    )
    return ordered


def run_arm(
    data: EvaluationData,
    config: ModelConfig,
    prompt: Prompt,
    output: Path,
    cache: Path,
    fields: tuple[str, ...] = ("title",),
    concurrency: int = 1,
    spend_ceiling: float = 0,
    dry_run: bool = False,
    scorer: Scorer | None = None,
    task: str = "sharks",
    label_review_sha256: str | None = None,
    positive_label: str | None = None,
    progress: ProgressSink | None = None,
    stop_on_error: bool = False,
) -> tuple[list[ScoreResult], Path]:
    """Run one arm with durable caching and a ceiling shared across the cache.

    Args:
        data: Aligned evaluation data; only adverts are passed to the scorer.
        config: Model/API/pricing configuration, with explicit unknowns.
        prompt: Fixed versioned criteria; never tuned within this function.
        output: Private directory for unique manifests, scores and aggregate metrics JSON.
        cache: Persistent cache; prior charges count toward the ceiling.
        fields: Enabled text fields included in configuration identity.
        concurrency: Maximum calls reserved and dispatched together.
        spend_ceiling: Lifetime external API ceiling in USD for this cache.
        dry_run: Estimate pending calls without constructing a scorer or sending requests.
        scorer: Optional injected scorer for offline tests; otherwise use the config factory.
        task: Research task; non-shark tasks use a separate cache namespace.
        label_review_sha256: Hash of the private label review declaration, if required.
        positive_label: Reviewed or inferred numeric positive-label provenance.
        progress: Optional display observer; its failures never change scoring.
        stop_on_error: Stop admitting work after a benchmark response fails.

    Returns:
        Results in input order and the manifest path. Dry runs return cached results only.

    Raises:
        ValueError: Settings, prices, request bounds or cache reuse are invalid.
        BudgetExceeded: The next batch cannot be reserved, or unexpected provider
            usage already exceeded its reserve; further calls stop.
        RuntimeError: Cache is locked, usage is unresolved or a scorer fails.
    """
    if concurrency < 1 or not math.isfinite(spend_ceiling) or spend_ceiling < 0:
        raise ValueError("Concurrency must be positive; spend ceiling finite and nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    manifest_path = output / f"manifest-{run_id}.json"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "evaluation_policy": EVALUATION_POLICY,
        "arm": config.arm,
        "prompt_used": config.adapter not in {"fake", "precomputed"},
        "git_sha": git_sha(),
        "data_file_hash": data.sha256,
        "task": task,
        "label_review_sha256": label_review_sha256,
        "positive_label": positive_label,
        **prompt.metadata(),
        "example_overlap": [
            {
                "example_index": index,
                "advert_ids": [
                    advert.advert_id
                    for advert in data.adverts
                    if " ".join(advert.evidence().casefold().split())
                    == " ".join(example["text"].casefold().split())
                ],
            }
            for index, example in enumerate(prompt.examples, 1)
        ],
        "model_config": asdict(config),
        "model_config_hash": None,
        "allowlisted_fields": fields,
        "input_condition": input_mode(fields),
        "started_at": now(),
        "status": "started",
        "spend_ceiling_usd": spend_ceiling,
        "dry_run": dry_run,
        "concurrency": concurrency,
        "cache_key": "model_config_hash,prompt_hash,advert_id",
    }
    results: dict[str, ScoreResult] = {}
    new_results: list[ScoreResult] = []
    try:
        # Persist the cache's final identity before scoring. If fingerprinting
        # itself fails (e.g. a missing import), finally still saves an abort
        # record with an explicitly unknown hash rather than a temporary one.
        config_hash = config.fingerprint(fields)
        if task != "sharks":
            config_hash = hash_json({"model_config_hash": config_hash, "task": task})
        manifest["model_config_hash"] = config_hash
        write_json(manifest_path, manifest)
        # A process-wide namespace lock prevents two runners from paying for the same key.
        with (cache / ".lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another process is using this cache") from exc
            results, pending = prepare_cache(data, config, prompt, cache, config_hash)
            existing = cache_spend(cache)
            estimates = [item.reservation for item in pending]
            known = [e["cost_upper_estimate"] for e in estimates]
            manifest["estimate"] = {
                "pending_adverts": len(pending),
                "cached_adverts": len(results),
                "tokens_in_upper_estimate": sum(e["tokens_in_upper_estimate"] for e in estimates),
                "tokens_out_upper_estimate": sum(e["tokens_out_upper_estimate"] for e in estimates),
                "cost_upper_estimate": (None if any(v is None for v in known) else sum(known)),
                "cache_external_spend_usd": existing,
                "estimator": "UTF-8 bytes + 2048 input tokens; bounded/structural output reserve",
            }
            oversized = [
                estimate["tokens_in_upper_estimate"]
                for estimate in estimates
                if estimate["tokens_in_upper_estimate"] > config.max_input_bound_tokens
            ]
            if oversized:
                raise ValueError(
                    f"{len(oversized)} request(s) exceed the configured short text input bound "
                    f"of {config.max_input_bound_tokens} (largest estimate: {max(oversized)}); "
                    "no calls sent"
                )
            if dry_run:
                manifest["status"] = "dry_run"
                return list(results.values()), manifest_path
            # Even a fully cached arm must not let unresolved billing unlock later arms.
            if config.billable and any(r.cost is None for r in results.values()):
                raise RuntimeError("Missing billed usage in cache; reconcile before more calls")
            notify_progress(
                progress,
                "begin",
                len(results),
                len(pending),
                sum(result.error is not None for result in results.values()),
            )
            if pending:
                config.validate_live()
                if config.adapter == "perplexity_local" and concurrency != 1:
                    raise ValueError("Local Perplexity model requires concurrency=1")
                active = scorer or make_scorer(config, prompt)
                if hasattr(active, "headers"):
                    active.headers()
                scoring_started = time.perf_counter()
                expected_spend = None
                try:
                    expected_spend = score_batch(
                        pending,
                        active,
                        concurrency,
                        results,
                        new_results,
                        progress,
                        existing=existing,
                        spend_ceiling=spend_ceiling,
                        billable=config.billable,
                        stop_on_error=stop_on_error,
                    )
                finally:
                    manifest["scoring_wall_time_s"] = time.perf_counter() - scoring_started
                    # The lock excludes concurrent writers. Scan once after all
                    # submitted work is drained, including aborted/uncertain calls.
                    manifest["cache_external_spend_after_usd"] = cache_spend(cache)
                    if expected_spend is not None and not math.isclose(
                        expected_spend,
                        manifest["cache_external_spend_after_usd"],
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError("Incremental spending differs from the durable cache")
            ordered = _write_results(output, manifest, data, results)
            return ordered, manifest_path
    except BaseException as exc:
        manifest["status"] = "aborted"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at"] = now()
        manifest["wall_time_s"] = time.perf_counter() - started
        manifest["totals"] = totals(list(results.values()))
        manifest["new_call_totals"] = totals(new_results)
        manifest["cached_results"] = len(results) - len(new_results)
        try:
            write_json(manifest_path, manifest)
        finally:
            notify_progress(progress, "finish", manifest["status"])
