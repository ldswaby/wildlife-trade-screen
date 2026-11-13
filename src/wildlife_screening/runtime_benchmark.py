"""Time frozen invented adverts without using wildlife evaluation predictions."""

import argparse
import json
import statistics
import uuid
from pathlib import Path
from typing import Any

from .cache import cache_spend, reservation
from .config import ModelConfig
from .full_factorial import MODEL_PATHS, _check_d3, _require_clean_commit
from .prompts import Prompt
from .runner import git_sha, now, run_arm
from .types import Advert, EvaluationData
from .utils import file_hash, hash_json, write_json

CACHE = Path("outputs/cache")
HOSTED = ("chat", "luna", "microsoft")
GRID = (1, 4, 8, 16, 32)


def invented_adverts(count: int, namespace: str) -> list[Advert]:
    """Create balanced length strata with no real adverts or seller information."""
    templates = (
        "Invented listing: two adult budgies offered for sale. ",
        "Invented listing: empty bird cage and spare perches for sale. ",
        "Invented listing: wanted, a pair of canaries. ",
        "Invented listing: domestic chickens offered for sale. ",
    )
    lengths = (128, 512, 2048, 6000)
    result = []
    for index in range(count):
        length = lengths[index % 4]
        text = templates[(index // 4) % 4]
        filler = " This fictional advert describes housing, care, collection and availability."
        text = (text + filler * (length // len(filler) + 1))[:length]
        result.append(Advert(f"runtime:{namespace}:{index}", title_and_desc=text))
    return result


def select_concurrency(rows: list[dict[str, Any]], arm: str) -> int | None:
    """Prefer fewer workers unless the next setting improves throughput by 10%."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["arm"] == arm and row["status"] == "complete" and not row["errors"]:
            grouped.setdefault(row["concurrency"], []).append(row["adverts_per_s"])
    selected = None
    best = 0.0
    for concurrency in sorted(grouped):
        speed = statistics.median(grouped[concurrency])
        if selected is None or speed >= best * 1.1:
            selected, best = concurrency, speed
    return selected


def prepare(root: Path) -> dict[str, Any]:
    """Freeze costs, prompts and model settings; make no provider requests."""
    plan_path = root / "benchmark-plan.json"
    evidence = {
        "models": {name: file_hash(path) for name, path in MODEL_PATHS.items()},
        "prompts": {
            family: file_hash(Path(f"prompts/birds/{filename}"))
            for family, filename in (
                ("llm", "llm_zeroshot_v0.txt"),
                ("decision", "decision_zeroshot_v0.yaml"),
            )
        },
        "fixture_sha256": hash_json([a.text() for a in invented_adverts(128, "fixed")]),
    }
    if plan_path.exists():
        plan: dict[str, Any] = json.loads(plan_path.read_text())
        if plan["evidence"] != evidence:
            raise ValueError("Benchmark fixtures/configs/prompts changed; preserve the prior plan")
        return plan
    start_spend = cache_spend(CACHE)
    per_round = {}
    for name, path in MODEL_PATHS.items():
        config = ModelConfig.load(path)
        family = "llm" if name == "chat" else "decision"
        filename = "llm_zeroshot_v0.txt" if name == "chat" else "decision_zeroshot_v0.yaml"
        prompt = Prompt.load(Path("prompts/birds") / filename)
        per_round[name] = sum(
            float(reservation(config, prompt, advert)["external_spend_reserved"])
            for advert in invented_adverts(64, family)
        )
    plan = {
        "id": uuid.uuid4().hex,
        "prepared_at": now(),
        "git_sha_at_prepare": git_sha(),
        "evidence": evidence,
        "start_external_spend_usd": start_spend,
        "additional_ceiling_usd": 1.0,
        "cumulative_ceiling_usd": min(15.0, start_spend + 1.0),
        "hosted_screen_and_two_confirmations_upper_usd": sum(
            per_round[name] * (len(GRID) + 2) for name in HOSTED
        ),
        "per_64_adverts_upper_usd": per_round,
        "note": "Invented inputs; no model-performance evidence. No automatic retries.",
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    return plan


def time_setting(
    root: Path, plan: dict[str, Any], arm: str, concurrency: int, phase: str
) -> dict[str, Any]:
    """Execute exactly one fresh timing round and retain all accounting evidence."""
    key = f"{arm}-{concurrency}-{phase}"
    record_path = root / f"timing-{key}.json"
    if record_path.exists():
        record: dict[str, Any] = json.loads(record_path.read_text())
        if record["status"] != "complete":
            raise RuntimeError("Previous benchmark setting failed; inspect it without replay")
        return record
    count = 32 if arm == "d3" else (128 if concurrency == 64 else 64)
    adverts = invented_adverts(count, f"{plan['id']}:{key}")
    data = EvaluationData(adverts, [0] * count, hash_json([a.text() for a in adverts]), "1")
    config = ModelConfig.load(MODEL_PATHS[arm])
    filename = "llm_zeroshot_v0.txt" if arm == "chat" else "decision_zeroshot_v0.yaml"
    prompt = Prompt.load(Path("prompts/birds") / filename)
    upper = sum(float(reservation(config, prompt, a)["external_spend_reserved"]) for a in adverts)
    if cache_spend(CACHE) + upper > plan["cumulative_ceiling_usd"]:
        raise ValueError("Benchmark allowance cannot cover this setting's conservative estimate")
    output = root / key
    # A started manifest without a timing record requires inspection, not another
    # fresh run that would accidentally hide cached or uncertain responses.
    if list(output.glob("manifest-*.json")):
        raise RuntimeError("Unfinished timing round exists; inspect before continuing")
    error: BaseException | None = None
    try:
        run_arm(
            data,
            config,
            prompt,
            output,
            CACHE,
            ("title_and_desc",),
            concurrency,
            plan["cumulative_ceiling_usd"],
            task="birds",
            positive_label="1",
            stop_on_error=True,
        )
    except BaseException as exc:
        error = exc
    manifest_path = next(output.glob("manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    wall = manifest.get("scoring_wall_time_s")
    cached = manifest.get("cached_results", 0)
    complete = manifest["status"] == "complete" and not cached
    record = {
        "arm": arm,
        "concurrency": concurrency,
        "phase": phase,
        "adverts": count,
        "status": "complete" if complete else "failed",
        "errors": manifest["totals"]["errors"],
        "scoring_wall_time_s": wall,
        "invocation_wall_time_s": manifest["wall_time_s"],
        "adverts_per_s": count / wall if complete and wall else None,
        "wall_s_per_advert": wall / count if complete and wall else None,
        "mean_request_latency_s": manifest["totals"]["latency_s_total"] / count
        if complete
        else None,
        "cost_usd": manifest["new_call_totals"]["cost"],
        "cached_results": cached,
        "manifest": str(manifest_path),
        "git_sha": manifest["git_sha"],
    }
    write_json(record_path, record)
    print(json.dumps(record), flush=True)
    if error is not None:
        raise error
    if not complete:
        raise RuntimeError("Timing round incomplete or cached; do not treat it as throughput")
    return record


def main(argv: list[str] | None = None) -> int:
    """Prepare offline, or explicitly run the authorised small timing experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--arm", choices=(*HOSTED, "d3"))
    parser.add_argument("--d3-slots", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--confirm", action="store_true", help="Repeat one d3 slot setting")
    args = parser.parse_args(argv)
    plan = prepare(args.root)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.live:
        return 0
    _require_clean_commit()
    arms = (args.arm,) if args.arm else HOSTED
    for arm in arms:
        if arm == "d3":
            _check_d3(args.d3_slots)
            time_setting(
                args.root, plan, arm, args.d3_slots, "confirm" if args.confirm else "screen"
            )
            continue
        rows = []
        for concurrency in GRID:
            rows.append(time_setting(args.root, plan, arm, concurrency, "screen"))
        if rows[-1]["adverts_per_s"] >= rows[-2]["adverts_per_s"] * 1.1:
            # Extension is optional and cannot consume the allowance reserved for
            # confirmation of already measured settings.
            reserve = plan["per_64_adverts_upper_usd"][arm] * 4
            if cache_spend(CACHE) + reserve <= plan["cumulative_ceiling_usd"]:
                rows.append(time_setting(args.root, plan, arm, 64, "screen"))
        best = sorted(rows, key=lambda r: r["adverts_per_s"], reverse=True)[:2]
        for row in best:
            time_setting(args.root, plan, arm, row["concurrency"], "confirm")
    all_rows = [json.loads(p.read_text()) for p in args.root.glob("timing-*.json")]
    selected = {name: select_concurrency(all_rows, name) for name in MODEL_PATHS}
    write_json(args.root / "recommendations.json", selected)
    print("Measured concurrency candidates:", json.dumps(selected), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
