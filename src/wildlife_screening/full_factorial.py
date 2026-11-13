"""Audit and orchestrate private, task-specific factorial runs without tuning scores."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import httpx
import yaml

from .cache import cache_spend
from .campaign_condition import CampaignCondition as CampaignCondition
from .config import ModelConfig
from .data import input_fields, input_mode, load_data
from .metrics import EVALUATION_POLICY
from .prompts import Prompt
from .runner import git_sha
from .scorers.api import OpenAIDecisionsScorer, PromptedLLMScorer, SystemOneScorer
from .snapshots import _load_pinned
from .tasks import TASKS, label_review
from .utils import file_hash, hash_json

DATA_HASHES = {
    "birds": "5dc76882aa0b9ab67b078ac9493fd004cbf6325b2dc030727ed65e839f03af77",
    "sharks": "4ad4c1a4b4ea2f7adc48c236715ff43f13938a98e16a58f0b584a1e0facb69e8",
    "animals": "92f9858d45c3fb432fdbfb49e2d32e8db240dac8b0d1bdb4016f5f986ea2ef52",
    "leather": "a6725811f488aad5e9f1ff1e8ecce9c0451d3aa796be893361b0b9681344c51b",
}
MODEL_PATHS = {
    "d3": Path("configs/models/d3_gguf_q4.yaml"),
    "chat": Path("configs/models/gpt6_luna_chat_anchor.yaml"),
    "luna": Path("configs/models/gpt6_luna_decisions.yaml"),
    "microsoft": Path("configs/models/microsoft_decision1_openrouter.yaml"),
}
SHOTS = ("few-shot", "zero-shot")


def input_modes(task: str) -> tuple[str, ...]:
    """Read the declared campaign conditions from the task definition."""
    return TASKS[task].input_modes


def shot_conditions(task: str) -> tuple[str, ...]:
    """Resolve supported shots without changing historical campaign audit fields."""
    return TASKS[task].shot_conditions


def prompt_paths(task: str, version: str, shot: str) -> dict[str, Path]:
    """Resolve both explicit prompt families for one declared condition."""
    stem = "fewshot" if shot == "few-shot" else "zeroshot"
    return {
        "llm": Path(f"prompts/{task}/llm_{stem}_{version}.txt"),
        "decision": Path(f"prompts/{task}/decision_{stem}_{version}.yaml"),
    }


def _digest_evidence(values: list[str]) -> str:
    """Hash ordered private evidence without persisting the text in the audit."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_audit(task: str, version: str, review_path: Path | None) -> dict[str, Any]:
    """Verify every adapter's rendered evidence and shared native question offline."""
    if task not in TASKS or re.fullmatch(r"v[0-9]+", version) is None:
        raise ValueError("Use a registered --task and --prompt-version vN")
    source = TASKS[task]
    if file_hash(source.data) != DATA_HASHES[task]:
        raise ValueError(f"{task} validation CSV differs from its declared dataset hash")
    review = label_review(source, source.data, review_path)
    positive = review.get("positive_label", source.positive_label)
    configs = {name: ModelConfig.load(path) for name, path in MODEL_PATHS.items()}
    if len({config.arm for config in configs.values()}) != 4:
        raise ValueError("The four model arms must have distinct names")
    prompts: dict[str, dict[str, Any]] = {}
    for shot in shot_conditions(task):
        paths = prompt_paths(task, version, shot)
        selected = {family: Prompt.load(path) for family, path in paths.items()}
        for family, prompt in selected.items():
            if prompt.family != family or prompt.condition != shot:
                raise ValueError(f"Wrong prompt family or condition: {paths[family]}")
            expected_version = f"decision-{shot.replace('-', '')}-{version}"
            if family == "decision" and prompt.definition["version"] != expected_version:
                raise ValueError(f"Wrong decision prompt version: {paths[family]}")
        prompts[shot] = {
            family: {"path": str(paths[family]), "sha256": prompt.sha256}
            for family, prompt in selected.items()
        }
    inputs: dict[str, Any] = {}
    with httpx.Client() as client:
        for input_name in input_modes(task):
            fields = input_fields(input_name)
            data = load_data(source.data, fields, positive)
            if data.sha256 != DATA_HASHES[task]:
                raise ValueError("Validation data changed during audit")
            evidence = [advert.text() for advert in data.adverts]
            for shot in shot_conditions(task):
                paths = prompt_paths(task, version, shot)
                llm = Prompt.load(paths["llm"])
                decision = Prompt.load(paths["decision"])
                chat = PromptedLLMScorer(configs["chat"], llm, client)
                native = {
                    "luna": OpenAIDecisionsScorer(configs["luna"], decision, client),
                    "microsoft": SystemOneScorer(configs["microsoft"], decision, client),
                    "d3": SystemOneScorer(configs["d3"], decision, client),
                }
                question = decision.question()
                for advert, expected in zip(data.adverts, evidence, strict=True):
                    chat_payload = chat.payload(advert)
                    messages = chat_payload["messages"]
                    if messages != [
                        {"role": "developer", "content": llm.template},
                        {"role": "user", "content": expected},
                    ]:
                        raise ValueError("Chat request does not match the selected prompt/evidence")
                    for name, scorer in native.items():
                        payload = scorer.payload(advert)
                        if name == "luna":
                            expected_payload: dict[str, Any] = {
                                "model": configs[name].model_id,
                                "input": expected,
                                "questions": [
                                    {
                                        "name": "relevance",
                                        "type": question["type"],
                                        "instructions": question["instructions"],
                                        "choices": [
                                            {"value": value, "description": description}
                                            for value, description in question["criteria"].items()
                                        ],
                                    }
                                ],
                            }
                        else:
                            expected_payload = {
                                "model": configs[name].model_id,
                                "state": expected,
                                "questions": {"relevance": question},
                            }
                        if payload != expected_payload:
                            raise ValueError(f"{name} request differs from shared decision input")
                inputs[f"{input_name}/{shot}"] = {
                    "adverts": len(data.adverts),
                    "evidence_sha256": _digest_evidence(evidence),
                    "decision_question_sha256": hash_json(question),
                    "chat_developer_sha256": hashlib.sha256(
                        (llm.template or "").encode("utf-8")
                    ).hexdigest(),
                    "config_hashes": {
                        name: (
                            hash_json(
                                {"model_config_hash": config.fingerprint(fields), "task": task}
                            )
                            if task != "sharks"
                            else config.fingerprint(fields)
                        )
                        for name, config in configs.items()
                    },
                }
    return {
        "task": task,
        "positive_label": data.positive_label,
        "prompt_version": version,
        "evaluation_policy": EVALUATION_POLICY,
        "git_sha": git_sha(),
        "data_sha256": DATA_HASHES[task],
        "label_review": str(review_path) if review_path else None,
        "label_review_sha256": review.get("label_review_sha256"),
        "prompt_files": prompts,
        "model_files": {
            name: {"path": str(path), "sha256": file_hash(path)}
            for name, path in MODEL_PATHS.items()
        },
        "inputs": inputs,
    }


def audit_text(audit: dict[str, Any]) -> str:
    """Render a private, text-free account of what the adapters will receive."""
    lines = [
        f"Task: {audit['task']}",
        f"Prompt version: {audit['prompt_version']}",
        f"Git commit: {audit['git_sha']}",
        f"Dataset SHA-256: {audit['data_sha256']}",
        f"Positive numeric label: {audit['positive_label']}",
        f"Evaluation policy: {audit['evaluation_policy']} (recorded failures always reviewed)",
        "Label meaning and prompt alignment require manual review; keep private notes.",
        f"Historical label review: {audit['label_review'] or 'none supplied'}",
        f"Label-review SHA-256: {audit['label_review_sha256'] or 'not required'}",
        "",
        "Exact prompt files:",
    ]
    for shot, families in audit["prompt_files"].items():
        for family, item in families.items():
            lines.append(f"  {shot} {family}: {item['path']}  SHA-256 {item['sha256']}")
    lines.extend(["", "Exact model configuration files:"])
    for model, item in audit["model_files"].items():
        lines.append(f"  {model}: {item['path']}  SHA-256 {item['sha256']}")
    lines.extend(["", "Request-content checks:"])
    for setting, item in audit["inputs"].items():
        lines.append(
            f"  {setting}: {item['adverts']} adverts; evidence SHA-256 "
            f"{item['evidence_sha256']}; decision question SHA-256 "
            f"{item['decision_question_sha256']}"
        )
    lines.extend(
        [
            "",
            "All three decision adapters received identical rendered instructions, "
            "choices, examples and evidence for each setting.",
            "Chat received the selected LLM developer text and identical advert evidence.",
            "Provider envelopes differ. Semantic equivalence of Chat and decision "
            "prompts requires the researcher's manual review.",
        ]
    )
    if "runtime_concurrency" in audit:
        lines.extend(["", f"Per-model request concurrency: {audit['runtime_concurrency']}"])
    return "\n".join(lines) + "\n"


def _conditions(
    audit: dict[str, Any], root: Path, ceiling: float, *, dry_run: bool
) -> list[CampaignCondition]:
    """Expand one task's declared inputs and shots in the historical campaign order."""
    return [
        CampaignCondition.from_audit(audit, mode, shot, root, ceiling, dry_run)
        for mode in input_modes(audit["task"])
        for shot in shot_conditions(audit["task"])
    ]


def _call_cli(condition: CampaignCondition) -> None:
    """Continue only after full result snapshots certify handled advert failures."""
    before = set(condition.output.rglob("manifest-*.json"))
    result = subprocess.run(
        condition.command(),
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 2 and not condition.dry_run:
        paths = set(condition.output.rglob("manifest-*.json")) - before
        _validate_handled_condition(condition, paths)
        print(result.stdout)
        print("Condition finished with failures; continuing with review-on-failure reporting.")
        return
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"Scoring command failed with exit code {result.returncode}")


def _validate_handled_condition(condition: CampaignCondition, paths: set[Path]) -> None:
    """Reject fatal or incomplete subprocess output disguised as handled failures."""
    task = condition.task
    loaded = [ModelConfig.load(path) for path in condition.models.values()]
    configs = {config.arm: config for config in loaded}
    if len(paths) != len(configs):
        raise ValueError("Handled-failure condition lacks every expected new manifest")
    seen: set[str] = set()
    failures = 0
    fields = list(input_fields(condition.input_mode))
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        arm = manifest["arm"]
        if arm not in configs or arm in seen:
            raise ValueError("Handled-failure condition has unexpected or duplicate arms")
        seen.add(arm)
        config = configs[arm]
        family = "llm" if config.adapter == "chat" else "decision"
        prompt_path = condition.prompts[family]
        if (
            manifest["task"] != task
            or manifest["condition"] != condition.shot
            or manifest["allowlisted_fields"] != fields
            or manifest["prompt_hash"] != file_hash(prompt_path)
            or ModelConfig(**manifest["model_config"]) != config
            or manifest["git_sha"] != git_sha()
        ):
            raise ValueError("Handled-failure condition provenance mismatch")
        dataset_id = f"{task}-{manifest['data_file_hash'][:12]}"
        settings = {"label_review": condition.label_review} if condition.label_review else {}
        row, _ = _load_pinned(path, {"datasets": {dataset_id: settings}}, include_curve=False)
        if config.billable and row["cost"] is None:
            raise ValueError("Missing billed usage; reconcile before continuing")
        failures += row["failures"]
    if not failures:
        raise ValueError("Handled-failure exit status has no recorded inference failures")


def _preflight(
    audit: dict[str, Any], ceiling: float | None, root: Path | None = None
) -> tuple[Path, float]:
    """Dry-run every arm and sum pending billable reservations once."""
    root = root or Path("outputs/checks/full-factorial") / uuid.uuid4().hex
    root.mkdir(parents=True)
    (root / "prompt-audit.txt").write_text(audit_text(audit), encoding="utf-8")
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    conditions = _conditions(audit, root, ceiling or 0.0, dry_run=True)
    for condition in conditions:
        _call_cli(condition)
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in root.rglob("manifest-*.json")]
    expected = sum(len(condition.models) for condition in conditions)
    if len(manifests) != expected or any(m["status"] != "dry_run" for m in manifests):
        raise ValueError("Preflight did not produce every expected dry-run manifest")
    declared = {(condition.input_mode, condition.shot) for condition in conditions}
    model_names = {
        ModelConfig.load(Path(info["path"])).arm: name
        for name, info in audit["model_files"].items()
    }
    seen: set[tuple[str, str, str]] = set()
    pending = 0.0
    for manifest in manifests:
        fields = manifest["allowlisted_fields"]
        input_name = input_mode(tuple(fields))
        shot = manifest["condition"]
        model = model_names.get(manifest["arm"])
        if model is None or (input_name, shot) not in declared:
            raise ValueError("Preflight contains an undeclared arm or condition")
        key = (input_name, shot, model)
        if key in seen:
            raise ValueError("Preflight contains a duplicate arm")
        seen.add(key)
        prompt_family = "llm" if model == "chat" else "decision"
        expected_n = audit["inputs"][f"{input_name}/{shot}"]["adverts"]
        if any(
            (
                manifest["task"] != audit["task"],
                manifest["data_file_hash"] != audit["data_sha256"],
                manifest["prompt_hash"] != audit["prompt_files"][shot][prompt_family]["sha256"],
                manifest["model_config_hash"]
                != audit["inputs"][f"{input_name}/{shot}"]["config_hashes"][model],
                manifest["label_review_sha256"] != audit["label_review_sha256"],
                manifest["positive_label"] != audit["positive_label"],
                manifest["estimate"]["pending_adverts"] + manifest["estimate"]["cached_adverts"]
                != expected_n,
            )
        ):
            raise ValueError("Preflight manifest differs from the frozen audit")
        if manifest["model_config"]["billable"]:
            estimate = manifest["estimate"]["cost_upper_estimate"]
            if estimate is None:
                raise ValueError("A billable arm has unknown estimated cost")
            pending += float(estimate)
    total_upper = cache_spend(Path("outputs/cache")) + pending
    if ceiling is not None and total_upper > ceiling + 1e-12:
        raise ValueError(
            f"Cumulative spend plus pending upper estimates is ${total_upper:.6f}; "
            f"the supplied ceiling is ${ceiling:.6f}"
        )
    return root, total_upper


def _check_d3(expected_slots: int = 1) -> None:
    """Require the pinned checkpoint, requested slots and adequate per-slot context."""
    with urlopen("http://127.0.0.1:8080/health", timeout=5) as response:
        if response.status != 200:
            raise ValueError("d3 server health check failed")
    with urlopen("http://127.0.0.1:8080/props", timeout=5) as response:
        props = json.load(response)
    model = props.get("model_path", "")
    context = props.get("default_generation_settings", {}).get("n_ctx")
    if (
        not isinstance(model, str)
        or not model.endswith("vllm-d3-Q4_K_M.gguf")
        or props.get("total_slots") != expected_slots
        or type(context) is not int
        or context < 8192
    ):
        raise ValueError(
            f"d3 must load the pinned Q4_K_M file with {expected_slots} slot(s) "
            "and 8192 per-slot context"
        )


def _require_clean_commit() -> None:
    """Make the manifest Git SHA identify the reviewed files."""
    if (
        git_sha() is None
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        ).strip()
    ):
        raise ValueError("Commit reviewed prompts, configs and code before a live run")


def _run_root(audit: dict[str, Any]) -> Path:
    """Isolate one immutable dataset/prompt bundle from historical snapshots."""
    bundle = hash_json(audit["prompt_files"])[:12]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    return (
        Path("outputs/comparison/reviewed")
        / str(audit["task"])
        / str(audit["data_sha256"])
        / f"{audit['prompt_version']}-{bundle}"
        / run_id
    )


def _archive_current(run_id: str) -> Path:
    """Copy private historical outputs without changing their original paths."""
    target = Path("outputs/archive") / f"before-{run_id}"
    target.mkdir(parents=True, exist_ok=False)
    for name in ("comparison",):
        source = Path("outputs") / name
        if source.exists():
            shutil.copytree(source, target / name)
    return target


def _manifest_rows(root: Path, audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Find completed snapshots that match the frozen campaign identity."""
    rows = []
    for path in root.rglob("manifest-*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dry_run"):
            continue
        fields = manifest.get("allowlisted_fields")
        input_name = input_mode(tuple(fields))
        shot = manifest.get("condition")
        if input_name not in input_modes(audit["task"]) or shot not in shot_conditions(
            audit["task"]
        ):
            raise ValueError(f"Unexpected condition in campaign: {path}")
        config_name = next(
            (
                name
                for name, item in audit["model_files"].items()
                if item["path"] and manifest.get("arm") == ModelConfig.load(Path(item["path"])).arm
            ),
            None,
        )
        if config_name is None:
            raise ValueError(f"Unexpected model arm in campaign: {path}")
        expected_prompt = audit["prompt_files"][shot][
            "llm" if config_name == "chat" else "decision"
        ]["sha256"]
        expected_config = audit["inputs"][f"{input_name}/{shot}"]["config_hashes"][config_name]
        if manifest.get("status") != "aborted" and any(
            (
                manifest.get("task") != audit["task"],
                manifest.get("data_file_hash") != audit["data_sha256"],
                manifest.get("prompt_hash") != expected_prompt,
                manifest.get("model_config_hash") != expected_config,
                manifest.get("label_review_sha256") != audit["label_review_sha256"],
                manifest.get("positive_label") != audit["positive_label"],
                manifest.get("git_sha") != audit["git_sha"],
            )
        ):
            raise ValueError(f"Campaign provenance mismatch: {path}")
        rows.append(
            {
                "path": path,
                "manifest": manifest,
                "input": input_name,
                "shot": shot,
                "model": config_name,
            }
        )
    return rows


def _selected_rows(
    rows: list[dict[str, Any]], audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select finished full-coverage snapshots, preserving all recorded failures."""
    selected = []
    missing = []
    for input_name in input_modes(audit["task"]):
        for shot in shot_conditions(audit["task"]):
            for model in audit["model_files"]:
                candidates = [
                    row
                    for row in rows
                    if (row["input"], row["shot"], row["model"]) == (input_name, shot, model)
                ]
                finished = [
                    row
                    for row in candidates
                    if row["manifest"]["status"] in {"complete", "incomplete"}
                ]
                if finished:
                    chosen = max(finished, key=lambda row: row["manifest"]["finished_at"])
                    dataset_key = f"{audit['task']}-{audit['data_sha256'][:12]}"
                    settings = (
                        {"label_review": audit["label_review"]} if audit["label_review"] else {}
                    )
                    evaluated, _ = _load_pinned(
                        chosen["path"],
                        {"datasets": {dataset_key: settings}},
                        include_curve=False,
                    )
                    if chosen["manifest"]["model_config"]["billable"] and evaluated["cost"] is None:
                        raise ValueError("Missing billed usage; reconcile before continuing")
                    chosen["evaluation"] = evaluated
                    selected.append(chosen)
                else:
                    missing.append(f"{input_name} | {shot} | {model}")
    return selected, missing


def write_index(root: Path, audit: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep a human-readable manifest inventory even after a failed campaign."""
    attempted = _manifest_rows(root, audit)
    selected, missing = _selected_rows(attempted, audit)
    by_key = {(row["input"], row["shot"], row["model"]): row for row in selected}
    lines = [f"Task: {audit['task']}", f"Prompt version: {audit['prompt_version']}", ""]
    for input_name in input_modes(audit["task"]):
        for shot in shot_conditions(audit["task"]):
            lines.append(f"{input_name} | {shot}")
            for model in audit["model_files"]:
                row = by_key.get((input_name, shot, model))
                if row is None:
                    attempts = [
                        item
                        for item in attempted
                        if (item["input"], item["shot"], item["model"]) == (input_name, shot, model)
                    ]
                    if attempts:
                        latest = max(attempts, key=lambda item: item["manifest"]["started_at"])
                        lines.append(
                            f"  {model}: {latest['manifest']['status']} | {latest['path']}"
                        )
                    else:
                        lines.append(f"  {model}: NOT STARTED")
                else:
                    manifest = row["manifest"]
                    lines.append(
                        f"  {model}: {row['evaluation']['status']} | {row['path']} | "
                        f"coverage {row['evaluation']['valid_predictions']}/"
                        f"{row['evaluation']['n']} | "
                        f"failures {row['evaluation']['failures']} | "
                        f"cached {manifest['cached_results']} | "
                        f"new calls {manifest['estimate']['pending_adverts']}"
                    )
            lines.append("")
    if missing:
        lines.append("Missing finished arms: " + ", ".join(missing))
    (root / "manifest-index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected, missing


def finalize(root: Path, audit: dict[str, Any]) -> None:
    """Stage all finished configurations, including failures, after full validation."""
    selected, missing = write_index(root, audit)
    if missing:
        raise ValueError("Campaign incomplete; selection candidate was not created")
    dataset_key = f"{audit['task']}-{audit['data_sha256'][:12]}"
    settings = (
        {"datasets": {dataset_key: {"label_review": audit["label_review"]}}}
        if audit["label_review"]
        else {}
    )
    selection = {
        "evaluation_policy": EVALUATION_POLICY,
        "pins": [str(row["path"]) for row in selected],
        **settings,
    }
    staged = root / "selection-candidate.staged.yaml"
    staged.write_text(
        "# Review-on-failure campaign candidate. Inspect coverage and merge approved pins.\n"
        + yaml.safe_dump(selection, sort_keys=False),
        encoding="utf-8",
    )
    for row in selected:
        _load_pinned(row["path"], selection)
    staged.replace(root / "selection-candidate.yaml")


def parser() -> argparse.ArgumentParser:
    """Expose the root script's task, prompt and budget controls."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", choices=TASKS, required=True)
    result.add_argument("--prompt-version", required=True)
    result.add_argument("--label-review", type=Path, help="Optional historical review JSON")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--spend-ceiling", type=float)
    result.add_argument("--resume", type=Path, metavar="RUN_DIR")
    for model in MODEL_PATHS:
        result.add_argument(f"--{model}-concurrency", type=int, default=1)
    return result


def main(argv: list[str] | None = None) -> int:
    """Preflight, score and prepare a private candidate without activating pins."""
    args = parser().parse_args(argv)
    try:
        if args.spend_ceiling is not None and (
            not math.isfinite(args.spend_ceiling) or args.spend_ceiling < 0
        ):
            raise ValueError("Spend ceiling must be a finite nonnegative USD amount")
        if not args.preflight_only and args.spend_ceiling is None:
            raise ValueError("Live runs require --spend-ceiling USD")
        audit = build_audit(args.task, args.prompt_version, args.label_review)
        concurrency = {name: getattr(args, f"{name}_concurrency") for name in MODEL_PATHS}
        if any(value < 1 for value in concurrency.values()):
            raise ValueError("All model concurrency values must be positive")
        if any(value != 1 for value in concurrency.values()):
            audit["runtime_concurrency"] = concurrency
        if not args.preflight_only:
            _require_clean_commit()
            for item in audit["model_files"].values():
                ModelConfig.load(Path(item["path"])).validate_live()
            for env_name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
                if not os.environ.get(env_name):
                    raise ValueError(f"Set {env_name} before a live run")
            if args.d3_concurrency == 1:
                _check_d3()
            else:
                _check_d3(args.d3_concurrency)
        preflight_root, total_upper = _preflight(audit, args.spend_ceiling)
        print(f"Preflight audit: {preflight_root / 'prompt-audit.txt'}")
        print(f"Historical spend plus pending upper estimates: ${total_upper:.6f}")
        if args.preflight_only:
            return 0
        if args.resume:
            root = args.resume
            reviewed_root = Path("outputs/comparison/reviewed").resolve()
            if not root.resolve().is_relative_to(reviewed_root):
                raise ValueError("Resume directory must be within outputs/comparison/reviewed")
            prior = json.loads((root / "audit.json").read_text(encoding="utf-8"))
            if prior != audit:
                raise ValueError("Resume request differs from the frozen campaign audit")
        else:
            root = _run_root(audit)
            archived = _archive_current(root.name)
            root.mkdir(parents=True)
            (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            (root / "prompt-audit.txt").write_text(audit_text(audit), encoding="utf-8")
            print(f"Historical outputs archived at: {archived}")
        print(f"Campaign directory: {root}")
        try:
            for condition in _conditions(audit, root, args.spend_ceiling, dry_run=False):
                _call_cli(condition)
        except BaseException:
            # Indexing is diagnostic: never replace the scoring/interruption error
            # with a secondary error while inspecting partially written output.
            try:
                write_index(root, audit)
            except Exception as exc:
                print(f"WARNING: Could not update campaign index: {exc}", file=sys.stderr)
            raise
        finalize(root, audit)
        print(f"Manifest index: {root / 'manifest-index.txt'}")
        print(f"Selection candidate: {root / 'selection-candidate.yaml'}")
        return 0
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
