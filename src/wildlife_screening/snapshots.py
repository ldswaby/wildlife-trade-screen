"""Validate immutable experiment snapshots and derive numerical evidence offline."""

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .config import ModelConfig
from .data import input_mode, load_data
from .metrics import (
    EVALUATION_POLICY,
    FINISHED_WITH_FAILURES,
    policy_roc_curve,
    published_rates,
    screening_metrics,
)
from .tasks import TASKS, label_review
from .types import EvaluationData, ScoreResult
from .utils import digest, hash_json


def _slug(value: str) -> str:
    """Make stable, readable names from model identifiers."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _model_slug(config: dict[str, Any]) -> str:
    """Distinguish APIs that expose the same named model."""
    model_id = str(config["model_id"])
    adapter = str(config["adapter"])
    if model_id == "gpt-6-luna":
        return "gpt_6_luna_chat" if adapter == "chat" else "gpt_6_luna_decisions"
    if model_id == "microsoft/microsoft-decision-1":
        return "microsoft_decision_1"
    if "d3-GGUF" in model_id:
        return "d3_27b_q4_k_m"
    return f"{_slug(model_id.split('/')[-1])}_{_slug(adapter)}"


def _identity(manifest: dict[str, Any]) -> dict[str, str]:
    """Derive dataset and series identities from saved provenance."""
    task = str(manifest["task"])
    digest = str(manifest["data_file_hash"])
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Manifest dataset hash must be SHA-256")
    fields = manifest["allowlisted_fields"]
    input_name = input_mode(tuple(fields))
    condition = str(manifest["condition"])
    if condition not in {"few-shot", "zero-shot"}:
        raise ValueError("Unsupported prompt condition in manifest")
    model = _model_slug(manifest["model_config"])
    return {
        "dataset": f"{task}-{digest[:12]}",
        "model": model,
        "input": input_name,
        "prompt": condition,
        "series": f"{model}_{input_name}_{condition.replace('-', '_')}",
    }


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """List manifests without reading or exposing individual advert responses."""
    found = []
    for path in sorted(root.rglob("manifest-*.json")) if root.exists() else []:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            identity = _identity(manifest)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        found.append(
            {
                "manifest": str(path),
                "dataset": identity["dataset"],
                "series": identity["series"],
                "status": manifest.get("status"),
                "prompt_hash": manifest.get("prompt_hash"),
                "model_config_hash": manifest.get("model_config_hash"),
            }
        )
    return found


def _read_selection(path: Path) -> dict[str, Any]:
    """Validate the small, user-edited private selection file."""
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("Report selection must be a YAML mapping")
    if not isinstance(selection.get("pins"), list):
        raise ValueError("Report selection needs a pins list")
    if not all(isinstance(item, str) for item in selection["pins"]):
        raise ValueError("Every report pin must be a manifest path")
    datasets = selection.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("datasets must be a mapping")
    return selection


def _dataset_data(
    task_name: str, manifest: dict[str, Any], selection: dict[str, Any]
) -> tuple[EvaluationData, Path]:
    """Load gold labels and validate saved polarity and optional historical review."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task in pinned manifest: {task_name}")
    task = TASKS[task_name]
    dataset_id = f"{task_name}-{str(manifest['data_file_hash'])[:12]}"
    settings = selection.get("datasets", {}).get(dataset_id, {})
    if not isinstance(settings, dict):
        raise ValueError(f"Dataset settings for {task_name} must be a mapping")
    path = Path(settings.get("data", task.data))
    # A selection can contain historical review-backed and new manually reviewed runs
    # for the same CSV. Only review-backed manifests depend on the historical file.
    review_path = settings.get("label_review") if manifest.get("label_review_sha256") else None
    review = label_review(task, path, Path(review_path) if review_path else None)
    if review.get("label_review_sha256") != manifest.get("label_review_sha256"):
        raise ValueError("Pinned run's label-review hash does not match current review")
    saved_positive = manifest.get("positive_label")
    positive = review.get("positive_label") or settings.get("positive_label") or task.positive_label
    fields = (
        ("title_and_desc",)
        if manifest.get("allowlisted_fields") == ["title_and_desc"]
        else ("title",)
    )
    data = load_data(path, fields, positive)
    if data.sha256 != manifest["data_file_hash"]:
        raise ValueError("Pinned run's dataset hash does not match the selected CSV")
    # Historical shark runs recorded the inference method instead of its numeric result.
    expected_positive = (
        "inferred from relevant"
        if saved_positive == "inferred from relevant"
        else data.positive_label
    )
    if saved_positive != expected_positive:
        raise ValueError("Pinned run's label polarity does not match the selected CSV")
    return data, path


def _score_path(manifest_path: Path) -> Path:
    """Find the score snapshot paired with one immutable run ID."""
    return manifest_path.with_name(manifest_path.name.replace("manifest-", "scores-", 1))


def _load_pinned(
    path: Path,
    selection: dict[str, Any],
    *,
    include_curve: bool = True,
) -> tuple[dict[str, Any], EvaluationData]:
    """Validate a snapshot, optionally omitting ROC coordinates for aggregate-only exports.

    Validation and screening metrics are identical in both modes. The default
    retains the complete report payload; exporters can avoid an unused ROC sweep.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = _identity(manifest)
    if manifest.get("dry_run") or manifest.get("status") not in {"complete", "incomplete"}:
        raise ValueError(f"Pin is not a completed scoring snapshot: {path}")
    if not isinstance(manifest.get("prompt_hash"), str) or not isinstance(
        manifest.get("model_config_hash"), str
    ):
        raise ValueError(f"Missing prompt/config provenance: {path}")
    config = ModelConfig(**manifest["model_config"])
    fields = tuple(manifest["allowlisted_fields"])
    expected_config_hash = config.fingerprint(fields)
    if manifest["task"] != "sharks":
        expected_config_hash = hash_json(
            {"model_config_hash": expected_config_hash, "task": manifest["task"]}
        )
    if config.adapter != "precomputed" and expected_config_hash != manifest["model_config_hash"]:
        raise ValueError(f"Manifest model configuration hash is inconsistent: {path}")
    if manifest.get("prompt_family") != ("llm" if config.adapter == "chat" else "decision"):
        raise ValueError(f"Manifest prompt family does not match its adapter: {path}")
    if (manifest["condition"] == "zero-shot") != (manifest.get("example_count") == 0):
        raise ValueError(f"Manifest shot condition and example count disagree: {path}")
    data, data_path = _dataset_data(str(manifest["task"]), manifest, selection)
    score_path = _score_path(path)
    score_bytes = score_path.read_bytes()
    if "scores_sha256" in manifest and digest(score_bytes) != manifest["scores_sha256"]:
        raise ValueError(f"Score snapshot checksum mismatch: {score_path}")
    raw_scores = json.loads(score_bytes)
    if not isinstance(raw_scores, list) or len(raw_scores) != len(data.adverts):
        raise ValueError(f"Score coverage differs from the source CSV: {score_path}")
    scores = [ScoreResult(**item) for item in raw_scores]
    if [score.advert_id for score in scores] != [advert.advert_id for advert in data.adverts]:
        raise ValueError(f"Score IDs/order differ from the source CSV: {score_path}")
    complete = all(score.error is None and score.p_relevant is not None for score in scores)
    if complete != (manifest["status"] == "complete"):
        raise ValueError(f"Manifest status and score coverage disagree: {path}")
    metrics = screening_metrics(data.labels, scores)
    values = [
        float(score.p_relevant)
        for score in scores
        if score.error is None and score.p_relevant is not None
    ]
    single_point = any(score.score_type == "binary" for score in scores) or not values
    has_both_classes = 0 < sum(data.labels) < len(data.labels)
    points = (
        policy_roc_curve(data.labels, scores)
        if include_curve and has_both_classes and not single_point
        else []
    )
    point = (
        [metrics["false_positive_rate"], metrics["positive_recall"]]
        if single_point
        and metrics["positive_recall"] is not None
        and metrics["false_positive_rate"] is not None
        else None
    )
    thresholds = [
        "failures only" if metrics["failures"] else "no review",
        *[f"{value:.4f}" for value in sorted(set(values), reverse=True)],
    ]
    row = {
        **identity,
        "kind": "measured",
        "status": "complete" if complete else FINISHED_WITH_FAILURES,
        "snapshot_status": manifest["status"],
        "evaluation_policy": EVALUATION_POLICY,
        "valid_predictions": metrics["valid_predictions"],
        "failures": metrics["failures"],
        "coverage": metrics["coverage"],
        "arm": manifest["arm"],
        "metrics": metrics,
        "points": points,
        "point": point,
        "thresholds": thresholds if points else [],
        "score_type": "+".join(sorted({score.score_type for score in scores})),
        "prompt_hash": manifest["prompt_hash"],
        "prompt_version": manifest.get("prompt_version"),
        "model_config_hash": manifest["model_config_hash"],
        "manifest": str(path),
        "data_sha256": data.sha256,
        "data_path": str(data_path),
        "n": len(data.labels),
        "positives": sum(data.labels),
        "local_api_free": (
            config.adapter == "system_one"
            and not config.billable
            and urlsplit(config.endpoint or "").hostname in {"localhost", "127.0.0.1", "::1"}
        ),
        "resource_eligible": config.adapter not in {"fake", "precomputed"},
        "cost": None
        if any(score.cost is None for score in scores)
        else sum(score.cost or 0 for score in scores),
        # Fake and imported records use a zero placeholder, not measured inference time.
        "latency_s_total": (
            None
            if config.adapter in {"fake", "precomputed"}
            else sum(score.latency_s for score in scores)
        ),
    }
    return row, data


def _published_rows(
    task_name: str, data: EvaluationData, default_data: bool
) -> list[dict[str, Any]]:
    """Preserve published aggregates as conditional points or unplottable context."""
    baseline = TASKS[task_name].baseline
    if not default_data or baseline is None:
        return []
    source = yaml.safe_load(baseline.read_text(encoding="utf-8"))
    if source.get("task") != task_name:
        raise ValueError("Published baseline task mismatch")
    published = []
    positives = sum(data.labels)
    for item in source["rows"]:
        recall = item.get("published_recall")
        false_positive_rate, accuracy = published_rates(
            item.get("published_precision"), recall, positives, len(data.labels)
        )
        point = [false_positive_rate, recall] if false_positive_rate is not None else None
        published.append(
            {
                "series": f"published_{_slug(item['arm'])}",
                "model": item["arm"],
                "input": (
                    "published_text_image"
                    if item.get("input_modality") == "text-image"
                    else "published_text"
                ),
                "input_modality": item.get("input_modality", "text"),
                "prompt": "unreported",
                "kind": "published",
                "arm": item["arm"],
                "status": "published conditional point" if point else "published; no plot point",
                "point": point,
                "precision": item.get("published_precision"),
                "recall": item.get("published_recall"),
                "f1": item.get("published_f1"),
                "false_positive_rate": false_positive_rate,
                "balanced_accuracy": accuracy,
                "note": item.get("note", ""),
            }
        )
    return published
