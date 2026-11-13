"""Missing aggregate metrics stay missing rather than acquiring invented denominators."""

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .metrics import (
    EVALUATION_POLICY,
    FINISHED_WITH_FAILURES,
    published_rates,
    screening_metrics,
)
from .plots import plot_roc
from .runner import totals
from .types import ScoreResult

COLUMNS = [
    "arm",
    "task",
    "data_sha256",
    "positive_label",
    "provenance",
    "input_modality",
    "input_condition",
    "label_review_sha256",
    "condition",
    "prompt_family",
    "prompt_used",
    "example_count",
    "prompt_version",
    "prompt_hash",
    "prompt_rendering",
    "status",
    "score_type",
    "cutoff",
    "n",
    "positives",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "missed",
    "recall_ci_low",
    "recall_ci_high",
    "false_positive_rate",
    "published_precision",
    "published_recall",
    "published_f1",
    "tokens_in",
    "tokens_out",
    "cost",
    "latency_s_total",
    "errors",
    "evaluation_policy",
    "valid_predictions",
    "failures",
    "coverage",
    "note",
    "balanced_accuracy",
    "fpr_at_95_tpr",
]


def render_value(value: Any) -> str:
    """Render missing values as unknown and escape Markdown-sensitive text."""
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


PERFORMANCE_HEADERS = [
    "Model",
    "Input",
    "Prompt",
    "Balanced accuracy",
    "F1",
    "TPR [95% CI]",
    "Missed",
    "FPR",
    "FPR @ 95% TPR",
    "Precision",
    "Coverage / failure count",
]


def percentage(value: float | None) -> str:
    """Format an available rate without converting unknowns into zeros."""
    return f"{100 * value:.1f}%" if value is not None else "N/A"


def recall_display(metrics: Mapping[str, Any], recall: float | None) -> str:
    """Preserve a known point estimate even when its interval is unavailable."""
    if recall is None:
        return "N/A"
    low, high = metrics.get("recall_ci_low"), metrics.get("recall_ci_high")
    interval = (
        f"{percentage(low)}, {percentage(high)}" if low is not None and high is not None else "N/A"
    )
    return f"{percentage(recall)} [{interval}]"


def coverage_display(coverage: float | None, failures: int | None) -> str:
    """Show prediction coverage and the explicit count of failed predictions."""
    return (
        f"{percentage(coverage)} / {failures}"
        if coverage is not None and failures is not None
        else "N/A"
    )


def performance_summary(rows: list[dict[str, Any]]) -> str:
    """Add the requested human-readable summary while retaining legacy tables."""
    lines = [
        "## Performance summary",
        "",
        "| " + " | ".join(PERFORMANCE_HEADERS) + " |",
        "| " + " | ".join("---" for _ in PERFORMANCE_HEADERS) + " |",
    ]
    for row in rows:
        published = row.get("status") == "published aggregate"
        prefix = "published" if published else "positive"
        recall = row.get(f"{prefix}_recall")
        ba = percentage(row.get("balanced_accuracy"))
        if published and row.get("balanced_accuracy") is not None:
            ba += " (est.)"
        values = [
            row["arm"],
            row.get("input_condition", row.get("input_modality", "N/A")),
            row.get("condition", "N/A"),
            ba,
            percentage(row.get(f"{prefix}_f1")),
            recall_display(row, recall),
            row.get("missed", "N/A"),
            percentage(row.get("false_positive_rate")),
            percentage(row.get("fpr_at_95_tpr")),
            percentage(row.get(f"{prefix}_precision")),
            coverage_display(row.get("coverage"), row.get("failures")),
        ]
        lines.append("| " + " | ".join(render_value(value) for value in values) + " |")
    return "\n".join(lines) + "\n\n"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Group related metrics to keep the comparison table readable.

    Args:
        rows: Measured and published report rows; absent fields remain unknown.

    Returns:
        Markdown with comparison metrics, coverage and resource accounting.
    """

    def triple(row: dict[str, Any], prefix: str) -> str:
        """Keep precision, recall and F1 together in one compact table cell."""
        return " / ".join(
            render_value(row.get(f"{prefix}_{metric}")) for metric in ("precision", "recall", "f1")
        )

    text = "# Table 1 — measured arms at fixed 0.5 cutoff\n\n"
    text += (
        "Measured cutoff is 0.5; recorded inference failures are always sent for review "
        f"({EVALUATION_POLICY}). Coverage counts valid model predictions. "
        "Published thresholds and failure handling are unreported. Costs are USD.\n\n"
    )
    text += performance_summary(rows)
    headers = [
        "Arm",
        "Provenance",
        "Input",
        "Coverage",
        "Positive P / R / F1",
        "Weighted P / R / F1",
        "Published P / R / F1",
        "Accuracy",
        "TP / FP / FN / TN",
        "Missed",
        "95% recall CI",
        "False-positive rate (FPR) %",
    ]
    text += "| " + " | ".join(headers) + " |\n"
    text += "| " + " | ".join("---" for _ in headers) + " |\n"
    for row in rows:
        ci = " to ".join(render_value(row.get(k)) for k in ("recall_ci_low", "recall_ci_high"))
        counts = " / ".join(render_value(row.get(k)) for k in ("tp", "fp", "fn", "tn"))
        false_positive_rate = (
            None if row.get("false_positive_rate") is None else 100 * row["false_positive_rate"]
        )
        values = [
            render_value(row["arm"]),
            render_value(row["provenance"]),
            (
                "combined text (title_and_desc column)"
                if row.get("input_condition") == "combined_text"
                else render_value(row.get("input_modality"))
            ),
            (
                f"{row['valid_predictions']}/{row['n']} ({100 * row['coverage']:.1f}%)"
                if row.get("coverage") is not None
                else "unknown"
            ),
            triple(row, "positive"),
            triple(row, "weighted"),
            triple(row, "published"),
            render_value(row.get("accuracy")),
            counts,
            render_value(row.get("missed")),
            ci,
            render_value(false_positive_rate),
        ]
        text += "| " + " | ".join(values) + " |\n"
    text += "\n## Resource accounting and coverage\n\n"
    columns = [
        "arm",
        "status",
        "score_type",
        "n",
        "positives",
        "tokens_in",
        "tokens_out",
        "cost",
        "latency_s_total",
        "errors",
        "valid_predictions",
        "coverage",
        "evaluation_policy",
    ]
    text += "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    text += "\n".join(
        "| " + " | ".join(render_value(row.get(c)) for c in columns) + " |" for row in rows
    )
    notes = [
        f"- {render_value(row['arm'])}: {render_value(row['note'])}"
        for row in rows
        if row.get("note")
    ]
    if notes:
        text += "\n\n" + "\n".join(notes)
    return text


def create_report(
    labels: list[int],
    arms: Mapping[str, list[ScoreResult]],
    output: Path,
    baseline_path: Path | None,
    prompt_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    task: str = "sharks",
) -> list[dict[str, Any]]:
    """Report review-on-failure metrics without changing the test denominator.

    Args:
        labels: Gold labels in the same order as each arm's results.
        arms: Names mapped to scored evaluation adverts.
        output: Private directory for Table 1 CSV/Markdown and Figure 1 PNG/SVG.
        baseline_path: Optional task-matched published LTS YAML.
        prompt_metadata: Per-arm condition and prompt identity; absent means unreported.
        task: Research task used to validate published baseline identity.

    Returns:
        Report rows including published aggregates, with unsupported fields absent.
    """
    output.mkdir(parents=True, exist_ok=True)
    all_metadata = dict(prompt_metadata or {})
    rows: list[dict[str, Any]] = []
    for name, results in arms.items():
        metadata = dict(all_metadata.get(name, {}))
        prompt_used = metadata.get("prompt_used", True)
        condition = metadata.get("condition", "unreported condition")
        provenance = (
            f"measured; {condition}; fixed prompt"
            if prompt_used
            else f"synthetic/imported; {condition}; no prompt sent"
        )
        row: dict[str, Any] = {
            "arm": name,
            "task": task,
            "input_modality": "text",
            "provenance": provenance,
            **metadata,
            "score_type": "+".join(sorted({r.score_type for r in results})),
            "cutoff": 0.5,
            "n": len(labels),
            "positives": sum(labels),
            **totals(results),
        }
        metrics = screening_metrics(labels, results)
        row.update(metrics)
        row["status"] = FINISHED_WITH_FAILURES if metrics["failures"] else "complete"
        row["note"] = (
            "Review on failure: recorded inference failures are always flagged; "
            "probabilities remain unchanged."
        )
        if any(r.score_type == "binary" for r in results):
            row["cutoff"] = "binary decision"
            row["note"] += " Binary labels encoded as 1/0; no probability cutoff."
        rows.append(row)
    published: list[dict[str, Any]] = []
    if baseline_path is not None:
        baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("task", "sharks") != task:
            raise ValueError("Published baseline task does not match measured task")
        positives = sum(labels)
        for baseline_row in baseline["rows"]:
            row = dict(baseline_row)
            fpr, accuracy = published_rates(
                row.get("published_precision"),
                row.get("published_recall"),
                positives,
                len(labels),
                reported_fpr=row.get("false_positive_rate"),
            )
            if fpr is not None:
                row["false_positive_rate"] = fpr
            row["balanced_accuracy"] = accuracy
            published.append(row)
    rows.extend(
        {**r, "status": "published aggregate", "score_type": "unreported"} for r in published
    )
    with (output / "table1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                c: (
                    "N/A"
                    if c in {"balanced_accuracy", "fpr_at_95_tpr"} and row.get(c) is None
                    else render_value(row.get(c))
                )
                for c in COLUMNS
            }
            for row in rows
        )
    markdown = markdown_table(rows)
    if all_metadata:
        markdown += "\n\nPrompts by arm:\n" + "\n".join(
            "- "
            + render_value(name)
            + ": "
            + "; ".join(f"{key}={render_value(value)}" for key, value in values.items())
            for name, values in all_metadata.items()
        )
        markdown += "\n"
    markdown += (
        "\n\nRecall intervals: 95% Wilson, conditional on independent adverts; duplicate/group "
        "dependence is not modelled. Undefined recall is unknown. Zero-division precision/F1 "
        "use 0, matching sklearn. Costs are USD list-price estimates on measured usage. "
        "Summed request latency is not parallel-run wall time. "
        "This comparison performs no threshold or prompt tuning.\n"
    )
    if published:
        markdown += (
            "\nPublished P/R/F1 use separate columns because averaging is not established for "
            "every row. No confusion counts, missed counts or CIs are inferred from rounded "
            "aggregates. Published trained models used validation in development; GPT-4 tuning "
            "is unestablished. Text-image rows have a different input modality from title-only "
            "measured arms. Figure 1 and Table 1 published FPR values are conditional on "
            "positive-class P/R and compatible prevalence; F1-only rows cannot be plotted. "
            "A similar anchor "
            "F1 is not proof of identical data, metrics or prompting. Figure 3 and section 3.4 "
            "describe few-shot pseudo-labelling; section 4.3 calls Table 5 zero-shot, while "
            "section 3.4 also calls Figure 3 zero-shot. The baseline prompting is ambiguous.\n"
        )
    (output / "table1.md").write_text(markdown, encoding="utf-8")
    plot_roc(labels, arms, published, output)
    return rows
