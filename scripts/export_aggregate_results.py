"""Export allowlisted measured aggregates from validated private snapshots offline."""

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

from wildlife_screening.metrics import EVALUATION_POLICY
from wildlife_screening.snapshots import _load_pinned, _read_selection

COLUMNS = (
    "schema_version",
    "task",
    "dataset_sha256",
    "model",
    "model_id",
    "adapter",
    "input_condition",
    "prompt_condition",
    "prompt_sha256",
    "model_config_sha256",
    "score_type",
    "cutoff",
    "evaluation_policy",
    "status",
    "n",
    "positives",
    "negatives",
    "tp",
    "fp",
    "tn",
    "fn",
    "valid_predictions",
    "failures",
    "coverage",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "false_positive_rate",
    "fpr_at_95_tpr",
    "api_cost_usd",
    "summed_request_seconds",
    "api_cost_per_1000_adverts",
    "summed_request_seconds_per_1000_adverts",
    "api_cost_scope",
)
Cell = str | int | float | None


def aggregate_rows(selection_path: Path) -> list[dict[str, Cell]]:
    """Validate every selected snapshot and project only the fixed public columns.

    Paths in selections retain the report's repository-root-relative convention.
    Snapshot loading reads private data locally; none of its free-text fields or
    per-advert records are copied into the public projection.
    """
    selection = _read_selection(selection_path)
    if selection.get("evaluation_policy", EVALUATION_POLICY) != EVALUATION_POLICY:
        raise ValueError("Unsupported evaluation policy")
    paths = [Path(item) for item in selection["pins"]]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Selection needs nonempty, unique manifest pins")
    identities: set[tuple[str, ...]] = set()
    rows: list[dict[str, Cell]] = []
    for path in paths:
        evidence, _ = _load_pinned(path, selection, include_curve=False)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        config = manifest["model_config"]
        identity = (
            str(manifest["task"]),
            evidence["data_sha256"],
            evidence["model"],
            evidence["input"],
            evidence["prompt"],
        )
        if identity in identities:
            raise ValueError("Selection repeats a measured configuration")
        identities.add(identity)
        metrics = evidence["metrics"]
        n = evidence["n"]
        # Match the report's distinction between zero local API spend and unknown
        # infrastructure costs. Fake/imported resource placeholders are not costs.
        if evidence["local_api_free"]:
            cost, scope = 0.0, "external_api_only_local_infrastructure_unknown"
        elif not evidence["resource_eligible"]:
            cost, scope = None, "not_measured"
        else:
            cost = evidence["cost"]
            scope = "usage_at_recorded_prices" if cost is not None else "unknown"
        seconds = evidence["latency_s_total"]
        binary = "binary" in evidence["score_type"].split("+")
        row: dict[str, Cell] = {
            "schema_version": "1",
            "task": manifest["task"],
            "dataset_sha256": evidence["data_sha256"],
            "model": evidence["model"],
            "model_id": config["model_id"],
            "adapter": config["adapter"],
            "input_condition": evidence["input"],
            "prompt_condition": evidence["prompt"],
            "prompt_sha256": evidence["prompt_hash"],
            "model_config_sha256": evidence["model_config_hash"],
            "score_type": evidence["score_type"],
            "cutoff": None if binary else 0.5,
            "evaluation_policy": evidence["evaluation_policy"],
            "status": evidence["status"],
            "n": n,
            "positives": evidence["positives"],
            "negatives": n - evidence["positives"],
            **{key: metrics[key] for key in ("tp", "fp", "tn", "fn")},
            **{key: evidence[key] for key in ("valid_predictions", "failures", "coverage")},
            "precision": metrics["positive_precision"],
            "recall": metrics["positive_recall"],
            "f1": metrics["positive_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "false_positive_rate": metrics["false_positive_rate"],
            "fpr_at_95_tpr": None if binary else metrics["fpr_at_95_tpr"],
            "api_cost_usd": cost,
            "summed_request_seconds": seconds,
            "api_cost_per_1000_adverts": None if cost is None else cost * 1000 / n,
            "summed_request_seconds_per_1000_adverts": (
                None if seconds is None else seconds * 1000 / n
            ),
            "api_cost_scope": scope,
        }
        for key in ("dataset_sha256", "prompt_sha256", "model_config_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row[key])) is None:
                raise ValueError("Invalid exported provenance hash")
        if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite aggregate")
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: tuple(
            str(row[key])
            for key in ("task", "model", "input_condition", "prompt_condition", "dataset_sha256")
        ),
    )


def export_aggregates(selection: Path, output: Path) -> int:
    """Validate a complete selection before atomically replacing its CSV destination."""
    rows = aggregate_rows(selection)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone exporter without invoking the experiment CLI or models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("report/selection.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/aggregate-results.csv"))
    args = parser.parse_args(argv)
    try:
        count = export_aggregates(args.selection, args.output)
    except (OSError, ValueError, KeyError, TypeError):
        # Upstream errors can contain source paths or private record details.
        print(
            "Export failed: selected evidence could not be validated or written.", file=sys.stderr
        )
        return 1
    print(f"Exported {count} measured configurations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
