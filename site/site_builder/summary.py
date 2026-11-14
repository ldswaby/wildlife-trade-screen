"""Describe cross-dataset summaries without changing experimental evidence."""

import html
import json
import math
import statistics
from typing import Any

from wildlife_screening.metrics import balanced_accuracy

from .content import DATASETS
from .descriptions import ResultDescriptions
from .evidence import INPUT_LABELS, DatasetEvidence

SUMMARY_MODELS = (
    ("d3_27b_q4_k_m", "d3"),
    ("microsoft_decision_1", "Microsoft-Decision-1"),
    ("gpt_6_luna_decisions", "GPT-6 Luna Decisions"),
    ("gpt_6_luna_chat", "GPT-6 Luna Chat"),
)
DECISION_MODELS = SUMMARY_MODELS[:3]
SUMMARY_INPUTS = {"sharks": "title_and_desc", "animals": "title", "birds": "combined_text"}
SUMMARY_BASELINES = {"sharks": "ACTL text", "animals": "LTS-text"}
# Best balanced accuracy per named policy in Stringham et al. Appendix S2.
# Published fold-mean rates are rounded to three decimals; these are not
# predictions for our composite-policy dataset and never enter measured means.
BIRD_SPLIT_REFERENCES = (
    ("Domestic poultry", "Naive Bayes", 16475, 0.979, 0.982),
    ("Junk", "Naive Bayes", 16446, 0.849, 0.977),
    ("Wanted adverts", "Logistic regression", 13751, 0.764, 0.993),
)
BIRD_SPLIT_SOURCE = "https://doi.org/10.1371/journal.pone.0254007.s002"
BIRD_SPLIT_POLICIES = {"Domestic poultry", "Junk", "Wanted adverts"}
BIRD_SPLIT_CAVEAT = (
    "Derived as 100 × (mean recall + mean specificity) / 2 from rounded published rates. "
    "Ten-fold cross-validation on separate exclusion policies with different eligible subsets; "
    "not our combined-policy performance or pooled predictions. "
    "Source: Stringham et al. (2021), Appendix S2."
)
BIRD_POLICY_SUMMARY_CAVEAT = (
    "Equal-weight mean and population SD across three separate policy tasks, "
    "not combined-policy performance, confidence intervals or cross-validation-fold variation. "
    "No sample-size weighting or pooled predictions. Source rounding limits precision."
)
BASELINE_CAVEAT = (
    "Estimated from published TPR and reconstructed FPR; conditional on positive-class "
    "metric interpretation and compatible dataset prevalence. "
    "Not directly reported balanced accuracy."
)
SUMMARY_METRICS = (
    ("balanced_accuracy", "Balanced accuracy (%) ↑", 100),
    ("fpr_at_95_tpr", "FPR @ 95% TPR (%) ↓", 100),
    ("cost_per_advert", "API charge / 1,000 adverts (USD) ↓", 1000),
    ("latency_per_advert", "Summed request time / 1,000 adverts (s) ↓", 1000),
)


def summary_data(panels: list[DatasetEvidence], data: dict[str, Any]) -> dict[str, Any]:
    """Weight three declared tasks equally; never select among ambiguous versions."""
    metrics = []
    for field, label, scale in SUMMARY_METRICS:
        cells = []
        for model, name in SUMMARY_MODELS:
            points: list[dict[str, Any]] = []
            reasons = []
            for index, (task, input_name) in enumerate(SUMMARY_INPUTS.items()):
                matching = [panel for panel in panels if panel.task == task]
                reason = ""
                if len(matching) != 1:
                    reason = "ambiguous dataset versions" if matching else "dataset not selected"
                else:
                    dataset = data[matching[0].key]
                    rows = [
                        row
                        for row in dataset["rows"]
                        if row["kind"] == "measured"
                        and row["model"] == model
                        and row["prompt"] == "zero-shot"
                        and row["input"] == input_name
                        and row["status"] in {"complete", "finished with failures"}
                    ]
                    if len(rows) != 1:
                        reason = "required configuration missing or ambiguous"
                    else:
                        row = rows[0]
                        if field in {"cost_per_advert", "latency_per_advert"}:
                            resources = [
                                r for r in dataset["resources"] if r["series"] == row["series"]
                            ]
                            value = resources[0].get(field) if len(resources) == 1 else None
                        else:
                            value = (row.get("metrics") or {}).get(field)
                        # Binary Chat does not supply the native-score threshold metric.
                        if model == "gpt_6_luna_chat" and field == "fpr_at_95_tpr":
                            value = None
                        if value is None or not math.isfinite(value):
                            reason = "metric unavailable"
                        else:
                            points.append(
                                {
                                    "task": task,
                                    "dataset": DATASETS[task],
                                    "input": input_name,
                                    "value": value * scale,
                                    "offset": (index - 1) * 0.16,
                                }
                            )
                if reason:
                    reasons.append(f"{DATASETS[task]}: {reason}")
            values = [point["value"] for point in points]
            complete = len(values) == 3
            cells.append(
                {
                    "model": model,
                    "name": name,
                    "points": points,
                    "n": len(values),
                    "mean": statistics.mean(values) if complete else None,
                    "sd": statistics.pstdev(values) if complete else None,
                    "unavailable": "; ".join(reasons),
                }
            )
        cells.insert(3, _decision_summary(cells))
        metrics.append({"key": field, "label": label, "cells": cells})
    return {
        "metrics": metrics,
        "datasets": [DATASETS[task] for task in SUMMARY_INPUTS],
        "baselines": _baseline_data(panels, data),
    }


def _decision_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep nine-value variation distinct from variation of three dataset means."""
    models = {model for model, _ in DECISION_MODELS}
    selected = [cell for cell in cells if cell["model"] in models]
    components = [
        {"model": cell["model"], "name": cell["name"], **point}
        for cell in selected
        for point in cell["points"]
    ]
    points = []
    for task in SUMMARY_INPUTS:
        members = [component for component in components if component["task"] == task]
        if len(members) == len(DECISION_MODELS):
            points.append(
                {
                    **{
                        field: members[0][field] for field in ("task", "dataset", "input", "offset")
                    },
                    "value": statistics.mean(member["value"] for member in members),
                    "model_count": len(members),
                    # Hover-only dispersion; coloured points retain their existing geometry.
                    "sd": statistics.pstdev(member["value"] for member in members),
                }
            )
    values = [component["value"] for component in components]
    total = len(DECISION_MODELS) * len(SUMMARY_INPUTS)
    complete = len(values) == total
    return {
        "model": "decision_models",
        "name": "Decision Models",
        "models": [name for _, name in DECISION_MODELS],
        "components": components,
        "component_n": len(values),
        "component_total": total,
        "points": points,
        "n": len(points),
        "mean": statistics.mean(values) if complete else None,
        "sd": statistics.pstdev(values) if complete else None,
        "unavailable": "; ".join(
            f"{cell['name']}: {cell['unavailable']}" for cell in selected if cell["unavailable"]
        ),
    }


def _baseline_data(panels: list[DatasetEvidence], data: dict[str, Any]) -> dict[str, Any]:
    """Use the explicitly chosen published references, never substitute missing evidence."""
    points, reasons = [], ["Live Birds: no matching published composite-policy baseline"]
    for index, (task, model) in enumerate(SUMMARY_BASELINES.items()):
        matching = [panel for panel in panels if panel.task == task]
        reason = ""
        if len(matching) != 1:
            reason = "ambiguous dataset versions" if matching else "dataset not selected"
        else:
            rows = [
                row
                for row in data[matching[0].key]["rows"]
                if row["kind"] == "published"
                and row["model"] == model
                and row["input_modality"] == "text"
            ]
            if len(rows) != 1:
                reason = "selected published baseline missing or ambiguous"
            else:
                point = rows[0].get("point")
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
                ):
                    reason = "published operating point unavailable"
                else:
                    value = balanced_accuracy(point[1], point[0])
                    assert value is not None
                    points.append(
                        {
                            "dataset": DATASETS[task],
                            "model": model,
                            "value": value * 100,
                            "offset": (index - 1) * 0.16,
                            "caveat": BASELINE_CAVEAT,
                        }
                    )
        if reason:
            reasons.append(f"{DATASETS[task]}: {reason}")
    birds = [panel for panel in panels if panel.task == "birds"]
    split_points: list[dict[str, Any]] = []
    split_unavailable = ""
    if len(birds) != 1:
        split_unavailable = "Live Birds split-policy references: " + (
            "ambiguous dataset versions" if birds else "dataset not selected"
        )
    else:
        for index, (policy, model, n, recall, specificity) in enumerate(BIRD_SPLIT_REFERENCES):
            rates = [
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0 <= value <= 1
                else None
                for value in (recall, specificity)
            ]
            fold_recall, fold_specificity = rates
            value = (
                100 * (fold_recall + fold_specificity) / 2
                if fold_recall is not None and fold_specificity is not None
                else None
            )
            split_points.append(
                {
                    "dataset": "Live Birds (split policy)",
                    "policy": policy,
                    "model": model,
                    "n": n,
                    "recall": fold_recall,
                    "specificity": fold_specificity,
                    "value": value,
                    "offset": 0.08 * (index + 1),
                    "cv_folds": 10,
                    "source": BIRD_SPLIT_SOURCE,
                    "caveat": BIRD_SPLIT_CAVEAT,
                }
            )
    split_summary = None
    if (
        len(split_points) == 3
        and {point["policy"] for point in split_points} == BIRD_SPLIT_POLICIES
        and all(point["value"] is not None for point in split_points)
    ):
        values = [point["value"] for point in split_points]
        split_summary = {
            "mean": statistics.mean(values),
            "sd": statistics.pstdev(values),
            "n": 3,
            "offset": 0.16,
            "source": BIRD_SPLIT_SOURCE,
            "caveat": BIRD_POLICY_SUMMARY_CAVEAT,
        }
    elif not split_unavailable:
        split_unavailable = (
            "Live Birds split-policy summary unavailable: "
            "all three valid, distinct policy references are required."
        )
    return {
        "points": points,
        "unavailable": "; ".join(reasons),
        "split_policy_points": split_points,
        "split_policy_summary": split_summary,
        "split_policy_unavailable": split_unavailable,
    }


def summary_figure(
    summary: dict[str, Any], descriptions: ResultDescriptions, *, compact: bool = False
) -> str:
    """Render shared summary evidence, optionally linking to the full report caption.

    Args:
        summary: Validated cross-dataset presentation data, without source rows.
        descriptions: Authored figure titles and captions.
        compact: Omit downloads and the values table for the Overview page.
    """
    encoded = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    rows = []
    for metric in () if compact else summary["metrics"]:
        deviation_unit = "pp" if metric["key"] in {"balanced_accuracy", "fpr_at_95_tpr"} else "sd"
        for cell in metric["cells"]:
            value = (
                f"{cell['mean']:.6g} (±{cell['sd']:.6g} {deviation_unit})"
                if cell["mean"] is not None
                else f"N/A ({cell['n']}/3 datasets); {cell['unavailable']}"
            )
            if "component_total" in cell:
                value = (
                    f"{cell['mean']:.6g} (±{cell['sd']:.6g} {deviation_unit})"
                    if cell["mean"] is not None
                    else f"N/A; {cell['unavailable']}"
                )
                value += (
                    f"; {cell['component_n']}/{cell['component_total']} model–dataset values; "
                    f"{cell['n']}/3 complete dataset averages. Population SD across models "
                    "and datasets, not SD of dataset averages. Models: "
                    + ", ".join(cell["models"])
                    + ". Individual contributions: "
                    + "; ".join(
                        f"{p['name']} / {p['dataset']} / zero-shot / "
                        f"{INPUT_LABELS[p['input']]}: {p['value']:.6g}"
                        for p in cell["components"]
                    )
                    + ". Dataset averages: "
                    + "; ".join(f"{p['dataset']}: {p['value']:.6g}" for p in cell["points"])
                    + ". Descriptive average individual model; no ensemble or pooled predictions."
                )
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(v))}</td>"
                    for v in (
                        metric["label"],
                        cell["name"],
                        value,
                    )
                )
                + "</tr>"
            )
        baseline = summary["baselines"]
        values = (
            [
                f"{p['dataset']} — {p['model']}: {p['value']:.6g}% (estimated). {p['caveat']}"
                for p in baseline["points"]
            ]
            if metric["key"] == "balanced_accuracy"
            else []
        )
        if metric["key"] == "balanced_accuracy":
            split_summary = baseline["split_policy_summary"]
            if split_summary is not None:
                values.append(
                    "Live Birds (split policy) — approximately "
                    f"{split_summary['mean']:.6g}% (±{split_summary['sd']:.6g} pp population SD); "
                    f"{split_summary['n']}/3 policies. {split_summary['caveat']}"
                )
            values.extend(
                f"{p['dataset']} — {p['policy']} — {p['model']}: approximately "
                + (f"{p['value']:.6g}%" if p["value"] is not None else "N/A")
                + f" (derived); eligible n={p['n']:,}; "
                f"100 × ({p['recall']} + {p['specificity']}) / 2. {p['caveat']}"
                for p in baseline["split_policy_points"]
            )
        explanation = (
            "; ".join(filter(None, [baseline["unavailable"], baseline["split_policy_unavailable"]]))
            if metric["key"] == "balanced_accuracy"
            else "N/A: no comparable published metric or resource evidence"
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(v)}</td>"
                for v in (
                    metric["label"],
                    "Supervised baselines",
                    "; ".join([*values, explanation]),
                )
            )
            + "</tr>"
        )
    exports = (
        '<div class="summary-exports">'
        '<button type="button" class="summary-png">Download summary PNG</button>'
        '<button type="button" class="summary-svg">Download summary SVG</button></div>'
    )
    values_table = (
        '<details><summary>Summary values and availability</summary><div class="table-wrap">'
        "<table><thead><tr><th>Metric</th><th>Model</th>"
        "<th>Mean ± population SD / published estimate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></details>"
    )
    caption = (
        'See full figure description <a href="report.html#summary-figure-title">here</a>.'
        if compact
        else descriptions.summary.body_html
    )
    return (
        '<figure class="summary-figure" aria-labelledby="summary-figure-title">'
        '<h3 id="summary-figure-title" class="summary-title">'
        f"{descriptions.summary.title_html}</h3>"
        '<div class="summary-panel" tabindex="0" role="region" '
        'aria-label="Scrollable cross-dataset summary figure">'
        '<div class="summary-chart" role="group" '
        'aria-labelledby="summary-figure-title" aria-describedby="summary-guide"></div></div>'
        + ("" if compact else exports)
        + '<figcaption class="result-caption" id="summary-guide">'
        f"{caption}</figcaption>"
        + ("" if compact else values_table)
        + f'<script id="summary-data" type="application/json">{encoded}</script></figure>'
    )
