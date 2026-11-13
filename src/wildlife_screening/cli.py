"""A small CLI is sufficient for inspection, paid runs and offline reproduction."""

import argparse
import json
import re
from pathlib import Path

from .config import ModelConfig
from .data import input_mode, inspect_data, load_data
from .metrics import policy_evaluate
from .progress import TerminalProgress
from .prompts import Prompt
from .runner import run_arm
from .scorers.api import PromptedLLMScorer
from .tasks import TASKS, Task, label_review
from .types import EvaluationData
from .utils import hash_json, write_json


def parser() -> argparse.ArgumentParser:
    """Build the CLI with explicit privacy opt-ins and a zero default spend ceiling."""
    result = argparse.ArgumentParser(description="Wildlife advert screening comparison")
    commands = result.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect private data locally")
    inspect.add_argument("--task", choices=TASKS, default="sharks")
    inspect.add_argument("--data", type=Path)
    inspect.add_argument("--pool", type=Path)
    inspect.add_argument("--output", type=Path, default=Path("outputs/inspection.json"))
    inspect.add_argument("--positive-label", choices=["0", "1"])
    run = commands.add_parser("run", help="Score one or more arms and generate Table 1 / Figure 1")
    run.add_argument("--task", choices=TASKS, default="sharks")
    run.add_argument("--data", type=Path)
    run.add_argument("--model", type=Path, action="append", required=True)
    run.add_argument("--llm-prompt", type=Path, help="Override the versioned LLM text file")
    run.add_argument(
        "--decision-prompt",
        type=Path,
        help="Override the versioned native decision YAML file",
    )
    run.add_argument("--zero-shot", action="store_true", help="Omit paper demonstrations")
    run.add_argument("--baselines", type=Path)
    run.add_argument(
        "--output",
        type=Path,
        help="Output root; defaults to outputs/checks for --check/--dry-run, "
        "otherwise outputs/comparison",
    )
    run.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    run.add_argument(
        "--check",
        action="store_true",
        help="Write to outputs/checks by default; live calls and spend rules still apply",
    )
    inputs = run.add_mutually_exclusive_group()
    inputs.add_argument("--include-description", action="store_true")
    inputs.add_argument(
        "--combined-text", action="store_true", help="Use only the title_and_desc column"
    )
    run.add_argument(
        "--preview-request",
        action="store_true",
        help="Save the exact one-advert Chat request offline, without using the cache or API",
    )
    run.add_argument(
        "--score-only",
        action="store_true",
        help="Save scores and raw provider responses without building comparison reports",
    )
    run.add_argument("--positive-label", choices=["0", "1"])
    run.add_argument(
        "--label-review", type=Path, help="Optional historical source-bound label review JSON"
    )
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument(
        "--model-concurrency",
        nargs=2,
        action="append",
        default=[],
        metavar=("ARM", "N"),
        help="Override concurrency for one selected arm",
    )
    run.add_argument(
        "--stop-on-error", action="store_true", help="Stop dispatch on a benchmark inference error"
    )
    run.add_argument(
        "--spend-ceiling",
        type=float,
        default=0,
        help="USD ceiling across this cache namespace, including prior calls",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate pending calls without sending requests; writes to outputs/checks by default",
    )
    return result


def _inspect(args: argparse.Namespace, task: Task, data_path: Path) -> int:
    """Write the inspection details locally and print only aggregate counts."""
    pool = args.pool
    if pool is None and args.data is None and task.pool is not None and task.pool.exists():
        pool = task.pool
    inspection = inspect_data(
        data_path,
        pool,
        args.positive_label or (task.positive_label if task.name == "birds" else None),
    )
    inspection["task"] = task.name
    write_json(args.output, inspection)
    print(
        json.dumps(
            {
                "rows": inspection["rows"],
                "positives": inspection.get("positives"),
                "missing_relevant": inspection["missing_relevant"],
                "disagreements": len(inspection["disagreements"]),
                "mapping_error": inspection["mapping_error"],
                # The default pool is optional, so say explicitly when it was skipped.
                "pool": f"included: {pool}"
                if pool is not None
                else "not inspected (pass --pool to include)",
                "private_report": str(args.output),
            },
            indent=2,
        )
    )
    if inspection["mapping_error"]:
        raise ValueError(inspection["mapping_error"])
    return 0


def _load_inputs(
    args: argparse.Namespace, task: Task, data_path: Path
) -> tuple[EvaluationData, tuple[str, ...], str, str | None]:
    """Verify the requested condition and resolve label polarity before loading models."""
    fields = ("title", "description") if args.include_description else ("title",)
    if args.combined_text:
        fields = ("title_and_desc",)
    condition = "zero-shot" if args.zero_shot else "few-shot"
    if condition not in task.shot_conditions:
        raise ValueError(f"{task.name} supports zero-shot only; supply --zero-shot")
    if task.name == "birds" and not args.combined_text:
        raise ValueError("Birds requires --combined-text; separate title fields are unavailable")
    review = label_review(task, data_path, args.label_review)
    positive_label = review.get("positive_label", args.positive_label or task.positive_label)
    if review and args.positive_label is not None and args.positive_label != positive_label:
        raise ValueError("--positive-label conflicts with the source-bound review")
    data = load_data(data_path, fields, positive_label)
    return data, fields, condition, review.get("label_review_sha256")


def _models(args: argparse.Namespace) -> tuple[list[ModelConfig], dict[str, int]]:
    """Load the selected arms and validate their concurrency overrides."""
    configs = [ModelConfig.load(path) for path in args.model]
    names = {config.arm for config in configs}
    if len(names) != len(configs):
        raise ValueError("Arm names must be unique")
    concurrency_by_arm = {}
    for arm, value in args.model_concurrency:
        count = int(value)
        if count < 1 or arm not in names or arm in concurrency_by_arm:
            raise ValueError(
                "Model concurrency requires a selected unique arm and a positive integer"
            )
        concurrency_by_arm[arm] = count
    return configs, concurrency_by_arm


def _prompts(
    args: argparse.Namespace, task: Task, configs: list[ModelConfig], condition: str
) -> dict[str, Prompt]:
    """Resolve the fixed prompt for each selected model family."""
    shot = "zeroshot" if args.zero_shot else "fewshot"
    prompt_paths = {
        "llm": args.llm_prompt or Path(f"prompts/{task.name}/llm_{shot}_v0.txt"),
        "decision": args.decision_prompt or Path(f"prompts/{task.name}/decision_{shot}_v0.yaml"),
    }
    families = {"llm" if c.adapter == "chat" else "decision" for c in configs}
    if task.name != "sharks" and any(
        prompt_paths[family].parent.name != task.name for family in families
    ):
        raise ValueError("Additional task prompts must come from their task folder")
    prompts = {family: Prompt.load(prompt_paths[family]) for family in families}
    for family, prompt in prompts.items():
        if prompt.family != family or prompt.condition != condition:
            raise ValueError(f"{family} prompt family or shot condition does not match the run")
    if args.llm_prompt is not None and "llm" not in families:
        raise ValueError("--llm-prompt requires a chat arm")
    if args.decision_prompt is not None and "decision" not in families:
        raise ValueError("--decision-prompt requires a native decision arm")
    return prompts


def _output_directory(
    args: argparse.Namespace,
    task: Task,
    data: EvaluationData,
    fields: tuple[str, ...],
    condition: str,
    prompts: dict[str, Prompt],
) -> Path:
    """Locate the run while retaining historical layouts for explicit output roots."""
    condition_dir = condition
    if args.llm_prompt is not None or args.decision_prompt is not None:
        condition_dir += (
            "-"
            + hash_json({family: prompt.sha256 for family, prompt in sorted(prompts.items())})[:12]
        )
    output_root = args.output or Path(
        "outputs/checks"
        if args.check or args.dry_run or args.preview_request
        else "outputs/comparison"
    )
    if args.output is None:
        input_dir = input_mode(fields)
        output = output_root / task.name / data.sha256 / input_dir / condition_dir
    else:
        # Explicit roots retain their established layout for older scripts.
        output = output_root / (
            condition_dir if task.name == "sharks" else f"{task.name}/{condition_dir}"
        )
    return output


def _preview(
    configs: list[ModelConfig], prompts: dict[str, Prompt], data: EvaluationData, output: Path
) -> int:
    """Save the single-advert Chat payload without touching the inference cache."""
    if len(configs) != 1 or configs[0].adapter != "chat" or len(data.adverts) != 1:
        raise ValueError("--preview-request requires one Chat model and one advert")
    config = configs[0]
    payload = PromptedLLMScorer(config, prompts["llm"]).payload(data.adverts[0])
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", config.arm)
    preview_path = output / slug / "request-preview.json"
    write_json(preview_path, payload)
    print(f"Private request preview: {preview_path}")
    return 0


def _print_manifest(arm: str, manifest_path: Path, dry_run: bool) -> None:
    """Print the arm status and the paths a caller can inspect."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_path = manifest_path.with_name(manifest_path.name.replace("manifest-", "scores-", 1))
    print(
        json.dumps(
            {
                "arm": arm,
                "status": manifest["status"],
                "estimate": manifest["estimate"],
                "manifest": str(manifest_path),
                "scores": None if dry_run else str(score_path),
                "metrics": None
                if dry_run
                else str(
                    manifest_path.with_name(manifest_path.name.replace("manifest-", "metrics-", 1))
                ),
            },
            indent=2,
        )
    )


def _run(args: argparse.Namespace, task: Task, data_path: Path) -> int:
    """Prepare one condition, score its arms, then write the requested report."""
    data, fields, condition, review_hash = _load_inputs(args, task, data_path)
    positive_label = data.positive_label or "inferred from relevant"
    configs, concurrency_by_arm = _models(args)
    prompts = _prompts(args, task, configs, condition)
    output = _output_directory(args, task, data, fields, condition, prompts)
    if args.preview_request:
        return _preview(configs, prompts, data, output)
    arms = {}
    arm_metadata = {}
    for config in configs:
        prompt = prompts["llm" if config.adapter == "chat" else "decision"]
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", config.arm)
        results, manifest_path = run_arm(
            data,
            config,
            prompt,
            output / slug,
            args.cache,
            fields,
            concurrency_by_arm.get(config.arm, args.concurrency),
            args.spend_ceiling,
            args.dry_run,
            task=task.name,
            label_review_sha256=review_hash,
            positive_label=positive_label,
            progress=TerminalProgress(config.arm),
            stop_on_error=args.stop_on_error,
        )
        if not args.dry_run:
            if [r.advert_id for r in results] != [a.advert_id for a in data.adverts]:
                raise ValueError("Score IDs/order differ from the source CSV")
            policy_evaluate(data.labels, results)
        _print_manifest(config.arm, manifest_path, args.dry_run)
        name = f"{config.arm} ({prompt.condition})"
        arms[name] = results
        arm_metadata[name] = {
            **prompt.metadata(),
            "prompt_used": config.adapter not in {"fake", "precomputed"},
            "task": task.name,
            "input_modality": "text",
            "input_condition": input_mode(fields),
            "allowlisted_fields": list(fields),
            "data_sha256": data.sha256,
            "positive_label": positive_label,
            "label_review_sha256": review_hash,
        }
    if not args.score_only and not args.dry_run:
        from .report import create_report

        baseline = args.baselines
        if baseline is None and args.data is None:
            baseline = task.baseline
        create_report(data.labels, arms, output, baseline, arm_metadata, task.name)
        print(f"Table 1 and Figure 1: {output}")
    if args.score_only or not args.dry_run:
        return 2 if any(r.error is not None for results in arms.values() for r in results) else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute inspection or comparison without exposing advert text on the console.

    Args:
        argv: Command arguments, or None to read the process arguments.

    Returns:
        Zero on success, two for finished scoring with advert failures, one on errors.
    """
    args = parser().parse_args(argv)
    try:
        task = TASKS[args.task]
        data_path = args.data or task.data
        if args.command == "inspect":
            return _inspect(args, task, data_path)
        return _run(args, task, data_path)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
