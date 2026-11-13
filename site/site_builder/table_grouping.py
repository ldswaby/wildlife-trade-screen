"""Summarize declared configurations without pooling predictions or changing evidence."""

import math
import statistics
from dataclasses import dataclass
from typing import Any

from wildlife_screening.metrics import FINISHED_WITH_FAILURES, balanced_accuracy


@dataclass(frozen=True)
class GroupStats:
    """Describe one metric with strict availability across the group's configurations."""

    mean: float | None
    sd: float | None
    available: int
    total: int
    reason: str


def table_numbers(row: dict[str, Any]) -> dict[str, Any]:
    """Read original, unrounded aggregate metrics using the existing table conventions."""
    metrics = row.get("metrics") or {}
    recall = metrics.get("positive_recall", row.get("recall"))
    fpr = metrics.get("false_positive_rate", row.get("false_positive_rate"))
    return {
        "precision": metrics.get("positive_precision", row.get("precision")),
        "recall": recall,
        "f1": metrics.get("positive_f1", row.get("f1")),
        "balanced_accuracy": balanced_accuracy(recall, fpr),
        "fpr_at_95_tpr": metrics.get("fpr_at_95_tpr"),
        "false_positive_rate": fpr,
        "missed": metrics.get("fn"),
    }


def _stats(values: list[Any], incomplete: int = 0) -> GroupStats:
    """Require every value; population SD describes configurations, not sampling error."""
    known = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    ]
    n = len(values)
    reason = (
        f"Performance withheld: {incomplete}/{n} configurations incomplete."
        if incomplete
        else f"Metric available for {len(known)}/{n} configurations."
    )
    complete = not incomplete and len(known) == n
    return GroupStats(
        statistics.mean(known) if complete else None,
        statistics.pstdev(known) if complete else None,
        len(known),
        n,
        reason,
    )


def group_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Collapse checked dimensions, retaining models, other conditions and published rows."""
    if not fields or not set(fields) <= {"input", "prompt"}:
        raise ValueError("Grouping requires Input, Prompt, or both")
    if any("_group_stats" in row for row in rows):
        raise ValueError("Group original configurations, never existing summaries")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    unchanged = []
    for row in rows:
        if row["kind"] != "measured":
            unchanged.append(row)
            continue
        key = (row.get("model") or row["arm"],) + tuple(
            row[field] for field in ("input", "prompt") if field not in fields
        )
        groups.setdefault(key, []).append(row)
    grouped = []
    for members in groups.values():
        incomplete = sum(
            row["status"] not in {"complete", FINISHED_WITH_FAILURES} for row in members
        )
        stats = {
            metric: _stats([table_numbers(row)[metric] for row in members], incomplete)
            for metric in table_numbers(members[0])
        }
        # Coverage and failures describe availability, even if performance is withheld.
        stats.update(
            {
                field: _stats([row.get(field) for row in members])
                for field in ("coverage", "failures")
            }
        )
        grouped.append(
            {
                **members[0],
                "status": "incomplete; metrics withheld"
                if incomplete
                else FINISHED_WITH_FAILURES
                if any(row["status"] == FINISHED_WITH_FAILURES for row in members)
                else "complete",
                "_group_stats": stats,
                "_group_size": len(members),
                "_group_inputs": list(dict.fromkeys(row["input"] for row in members)),
                "_group_prompts": list(dict.fromkeys(row["prompt"] for row in members)),
            }
        )
    return grouped + unchanged


def format_stats(stats: GroupStats, *, percent: bool = False) -> str:
    """Use percentage-point SD for rates, allowing fractional mean counts."""
    if stats.mean is None or stats.sd is None:
        return "N/A"
    if percent:
        deviation = "0" if stats.sd == 0 else f"{stats.sd * 100:.1f}"
        return f"{stats.mean * 100:.1f}% (±{deviation})"
    mean = f"{stats.mean:.1f}".removesuffix(".0")
    deviation = f"{stats.sd:.1f}".removesuffix(".0")
    return f"{mean} (±{deviation})"
