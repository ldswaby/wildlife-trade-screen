"""Render validated aggregate evidence within the website."""

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import plotly.offline

from wildlife_screening.metrics import FINISHED_WITH_FAILURES
from wildlife_screening.report import PERFORMANCE_HEADERS, coverage_display, recall_display

from .dataset_runs import DatasetRuns, load_dataset_runs
from .descriptions import ResultDescriptions
from .layout import PROJECT_TITLE, website_frame
from .resource_grouping import resource_views
from .table_grouping import GroupStats, format_stats, group_rows, table_numbers

# Storage layout does not define a different kind of model evidence in the report.
INPUT_LABELS = {"title": "title", "title_and_desc": "title + desc", "combined_text": "title + desc"}


MODEL_DISPLAY_ORDER = (
    "gpt_6_luna_chat",
    "gpt_6_luna_decisions",
    "microsoft_decision_1",
    "d3_27b_q4_k_m",
)
CONFIG_DISPLAY_ORDER = (
    ("title", "zero-shot"),
    ("title", "few-shot"),
    ("title_and_desc", "zero-shot"),
    ("title_and_desc", "few-shot"),
    ("combined_text", "zero-shot"),
    ("combined_text", "few-shot"),
)


def _presentation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group measured configurations without changing source or plot ordering."""
    models = list(MODEL_DISPLAY_ORDER)
    for row in rows:
        model = row.get("model", row.get("arm", ""))
        if row["kind"] != "published" and model not in models:
            models.append(model)

    def order(row: dict[str, Any]) -> tuple[int, int, int]:
        if row["kind"] == "published":
            # Stable sorting preserves published rows in their original order.
            return (1, 0, 0)
        model = row.get("model", row.get("arm", ""))
        config = (row["input"], row["prompt"])
        config_rank = (
            CONFIG_DISPLAY_ORDER.index(config)
            if config in CONFIG_DISPLAY_ORDER
            else len(CONFIG_DISPLAY_ORDER)
        )
        return (0, models.index(model), config_rank)

    return sorted(rows, key=order)


def _format(value: Any, percent: bool = False) -> str:
    """Format aggregate values without making unknowns look like zeroes."""
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{100 * value:.1f}%" if percent else f"{value:.4f}"
    return str(value)


def _table_view(rows: list[dict[str, Any]], *, grouped: bool = False) -> str:
    """Render one original or grouped table body, recalculating displayed leaders."""
    rows = _presentation_rows(rows)
    numeric_rows = [
        {key: stats.mean for key, stats in row["_group_stats"].items()}
        if "_group_stats" in row
        else table_numbers(row)
        for row in rows
    ]
    higher_is_better = {
        "balanced_accuracy": True,
        "f1": True,
        "recall": True,
        "missed": False,
        "false_positive_rate": False,
        "fpr_at_95_tpr": False,
        "precision": True,
    }
    leaders: dict[str, float | None] = {}
    for metric, higher in higher_is_better.items():
        values = [
            float(cast(float, numbers[metric]))
            for row, numbers in zip(rows, numeric_rows, strict=True)
            if (row["kind"] == "published" or row["status"] in {"complete", FINISHED_WITH_FAILURES})
            and numbers[metric] is not None
        ]
        leaders[metric] = (max(values) if higher else min(values)) if values else None

    def metric_cell(
        metric: str,
        value: Any,
        display: str | None = None,
        *,
        origin: str | None = None,
        summary: GroupStats | None = None,
    ) -> str:
        """Emphasize a column leader before formatting can create false ties."""
        if summary is not None:
            rendered = format_stats(summary, percent=metric != "missed")
        elif value is None:
            rendered = "N/A"
        elif display is not None:
            rendered = display
        else:
            rendered = _format(
                value,
                metric
                in {
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "false_positive_rate",
                    "fpr_at_95_tpr",
                },
            )
        content = html.escape(rendered)
        leader = leaders[metric]
        if (
            value is not None
            and leader is not None
            and math.isclose(float(value), leader, rel_tol=0, abs_tol=1e-12)
        ):
            content = f"<strong>{content}</strong>"
        if value is not None and origin is not None:
            symbol, description = (
                ("†", "Reported directly in the cited source")
                if origin == "reported"
                else ("‡", "Derived here under the assumptions in Limitations")
            )
            content += (
                f'<sup class="published-origin" title="{description}" '
                f'aria-label="{description}">{symbol}</sup>'
            )
        title = f' title="{html.escape(summary.reason, quote=True)}"' if summary else ""
        return f"<td{title}>{content}</td>"

    body = []
    for row, numbers in zip(rows, numeric_rows, strict=True):
        if row["kind"] == "published":
            model = "GPT-4" if row["arm"] == "GPT-4 zero-shot" else row["arm"]
            modality = row["input_modality"]
            input_name = (
                "text + image (fields N/A)" if modality == "text-image" else "text (fields N/A)"
            )
            prompt = "unverified" if row["arm"] == "GPT-4 zero-shot" else "N/A"
        else:
            model = row["arm"]
            if row.get("model") == "microsoft_decision_1":
                model = "Microsoft-Decision-1"
            if "_group_size" in row:
                n = row["_group_size"]
                model += f" ({n} config{'s' if n != 1 else ''})"
            if row["status"] == FINISHED_WITH_FAILURES:
                model += " (finished with failures)"
            elif row["status"] != "complete":
                model += " (incomplete)"
            input_name = INPUT_LABELS[row["input"]]
            prompt = row["prompt"]
            if "_group_stats" in row:
                input_name = "; ".join(
                    dict.fromkeys(INPUT_LABELS[value] for value in row["_group_inputs"])
                )
                prompt = "; ".join(row["_group_prompts"])
        coverage_text = coverage_display(row.get("coverage"), row.get("failures"))
        coverage_title = ""
        if "_group_stats" in row:
            coverage = row["_group_stats"]["coverage"]
            failures = row["_group_stats"]["failures"]
            coverage_text = format_stats(coverage, percent=True) + " / " + format_stats(failures)
            coverage_title = (
                f' title="{html.escape(coverage.reason + " " + failures.reason, quote=True)}"'
            )
        fields = [model, input_name, prompt]
        kind = ' class="published-row"' if row["kind"] == "published" else ""
        body.append(
            f"<tr{kind}>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in fields)
            + "".join(
                metric_cell(
                    metric,
                    numbers[metric],
                    (row.get("metrics") or {}).get("missed")
                    if metric == "missed"
                    else recall_display(row.get("metrics") or {}, numbers["recall"])
                    if metric == "recall"
                    else None,
                    origin=("reported" if metric in {"precision", "recall", "f1"} else "derived")
                    if row["kind"] == "published"
                    and metric
                    in {"precision", "recall", "f1", "false_positive_rate", "balanced_accuracy"}
                    else None,
                    summary=row.get("_group_stats", {}).get(metric),
                )
                for metric in higher_is_better
            )
            + f"<td{coverage_title}>{html.escape(coverage_text)}</td></tr>"
        )
    headers = [
        "TPR" if grouped and value == "TPR [95% CI]" else value for value in PERFORMANCE_HEADERS
    ]
    return (
        "<thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody>"
    )


def _grouping_controls(rows: list[dict[str, Any]], *, resource: bool = False) -> str:
    """Offer only varying measured dimensions, scoped independently to each display."""
    measured = [row for row in rows if row["kind"] == "measured"]
    dimensions = [
        field for field in ("input", "prompt") if len({row[field] for row in measured}) > 1
    ]
    controls = ""
    if dimensions:
        classes = "table-grouping resource-grouping" if resource else "table-grouping"
        status_class = "resource-grouping-status" if resource else "grouping-status"
        controls = (
            f'<fieldset class="{classes}"><legend>Grouping</legend>'
            + "".join(
                f'<label class="selection-box"><input type="checkbox" value="{field}">'
                f"<span>{field.title()}</span></label>"
                for field in dimensions
            )
            + '<p class="grouping-note">Grouped values: mean ± population SD.</p>'
            "</fieldset>"
            f'<p class="{status_class} visually-hidden" role="status"></p>'
        )
    return controls


def _table(rows: list[dict[str, Any]], index: int = 0, *, descriptions: ResultDescriptions) -> str:
    """Precompute independent grouping views from aggregate evidence only."""
    rows = _presentation_rows(rows)
    views = {"none": _table_view(rows)}
    for fields in (("input",), ("prompt",), ("input", "prompt")):
        views[",".join(fields)] = _table_view(group_rows(rows, fields), grouped=True)
    encoded = json.dumps(views, ensure_ascii=False).replace("<", "\\u003c")
    controls = _grouping_controls(rows)
    guide = (
        f'<div class="table-guide" id="table-guide-{index}">'
        + descriptions.table.body_html
        + "</div>"
    )
    table = (
        f'<div class="table-wrap"><table aria-describedby="table-guide-{index}">'
        f'<caption class="visually-hidden">{html.escape(descriptions.table.title_text)}</caption>'
        + views["none"]
        + "</table></div>"
        + f'<script class="table-grouping-data" type="application/json">{encoded}</script>'
    )
    notes = (
        '<div class="subtle">' + descriptions.table_gpt4_note + "</div>"
        if any(row["kind"] == "published" and row["arm"] == "GPT-4 zero-shot" for row in rows)
        else ""
    )
    return (
        f'<figure class="result-figure results-table" aria-labelledby="table-title-{index}">'
        f'<h3 id="table-title-{index}">{descriptions.table.title_html}</h3>'
        + controls
        + table
        + '<figcaption class="result-caption">'
        + guide
        + notes
        + "</figcaption></figure>"
    )


def _resource_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep comparable, measured per-advert resources for the report figure."""
    eligible = [
        row
        for row in _presentation_rows(rows)
        if row["kind"] == "measured"
        and row["status"] in {"complete", FINISHED_WITH_FAILURES}
        and row.get("resource_eligible")
    ]
    resources = []
    for row in eligible:
        n = row["n"]
        cost = row.get("cost")
        latency = row.get("latency_s_total")
        resources.append(
            {
                "series": row["series"],
                "cost_per_advert": (
                    0.0 if row.get("local_api_free") else cost / n if cost is not None else None
                ),
                "latency_per_advert": latency / n if latency is not None else None,
                "local_api_free": bool(row.get("local_api_free")),
            }
        )
    return resources


@dataclass(frozen=True)
class DatasetEvidence:
    """Keep independently renderable evidence sections tied to one dataset identity."""

    index: int
    key: str
    task: str
    n: int
    positives: int
    sha256: str
    prompts: list[dict[str, str]]
    roc: str
    table: str
    resources: str
    provenance: str


def _chart_payload(rows: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, Any]:
    """Project the selected rows into the browser's ordered chart data."""
    requested = sorted({row["series"] for row in rows if row.get("point") or row.get("points")})
    return {
        # Keep source rows intact: their positions also determine plot colours.
        "display_order": [row["series"] for row in _presentation_rows(rows)],
        "resources": resources,
        "resource_views": resource_views(resources, rows),
        "rows": [
            {
                field: row.get(field)
                for field in (
                    "series",
                    "model",
                    "input",
                    "input_modality",
                    "prompt",
                    "kind",
                    "status",
                    "points",
                    "thresholds",
                    "point",
                    "metrics",
                    "evaluation_policy",
                    "valid_predictions",
                    "failures",
                    "coverage",
                )
            }
            for row in rows
        ],
        "default": requested,
    }


def _render_panel(
    index: int,
    key: str,
    group: DatasetRuns,
    rows: list[dict[str, Any]],
    resource_controls: str,
    descriptions: ResultDescriptions,
) -> DatasetEvidence:
    """Render the dataset's figures, table and provenance from prepared evidence."""
    return DatasetEvidence(
        index=index,
        key=key,
        task=group.task,
        n=len(group.data.labels),
        positives=sum(group.data.labels),
        sha256=group.data.sha256,
        prompts=list(group.prompts.values()),
        roc=(
            f'<figure class="result-figure roc-figure" aria-labelledby="roc-title-{index}">'
            f'<h3 id="roc-title-{index}">{descriptions.roc.title_html}</h3>'
            '<div class="plot-tools" role="group" aria-label="Plot navigation">'
            '<button class="zoom" type="button" aria-pressed="true">Zoom</button>'
            '<button class="pan" type="button" aria-pressed="false">Pan</button>'
            '<button class="reset" type="button">Reset view</button></div>'
            f'<div class="chart" id="chart-{index}" role="group" '
            f'aria-labelledby="roc-title-{index}" '
            f'aria-describedby="roc-guide-{index}"></div>'
            '<div class="plot-legend" aria-label="Visible plot series"></div>'
            '<div class="exports"><button class="png" type="button">Download PNG</button>'
            '<button class="svg" type="button">Download SVG</button></div>'
            '<figcaption class="result-caption">'
            f'<div class="caption-body" id="roc-guide-{index}">'
            f'<span id="roc-key-{index}" aria-hidden="true"></span>'
            f"{descriptions.roc.body_html}</div></figcaption></figure>"
        ),
        resources=(
            '<figure class="resource-panel result-figure" '
            f'aria-labelledby="resource-title-{index}">'
            f'<h3 id="resource-title-{index}">{descriptions.resources.title_html}</h3>'
            f"{resource_controls}"
            '<div class="resource-chart" role="group" '
            f'aria-labelledby="resource-title-{index}" '
            f'aria-describedby="resource-guide-{index}"></div>'
            '<div class="resource-exports"><button class="resource-png" type="button">'
            'Download resource PNG</button><button class="resource-svg" type="button">'
            "Download resource SVG</button></div>"
            '<figcaption class="result-caption">'
            f'<div class="caption-body" id="resource-guide-{index}">'
            f'<span id="resource-notes-{index}" aria-hidden="true"></span>'
            f"{descriptions.resources.body_html}</div></figcaption></figure>"
        ),
        table=_table(rows, index, descriptions=descriptions),
        provenance=(
            "<details><summary>Run provenance</summary><ul>"
            + "".join(
                "<li>"
                + html.escape(row["series"])
                + ": "
                + html.escape(row["manifest"])
                + "; prompt "
                + html.escape(row["prompt_hash"])
                + "; config "
                + html.escape(row["model_config_hash"])
                + "</li>"
                for row in group.rows
            )
            + "</ul></details>"
        ),
    )


def build_evidence(
    selection: dict[str, Any], descriptions: ResultDescriptions
) -> tuple[list[DatasetEvidence], dict[str, Any]]:
    """Build report panels and chart payloads for the explicitly selected datasets."""
    page_data = {}
    panels = []
    for index, (key, group) in enumerate(sorted(load_dataset_runs(selection).items())):
        rows = group.comparison_rows()
        resources = _resource_rows(rows)
        eligible_series = {row["series"] for row in resources}
        resource_controls = _grouping_controls(
            [row for row in rows if row["series"] in eligible_series], resource=True
        )
        page_data[key] = _chart_payload(rows, resources)
        panels.append(_render_panel(index, key, group, rows, resource_controls, descriptions))
    return panels, page_data


def _html_document(content: str, data_json: str) -> str:
    """Bundle Plotly and the website chart controller without remote requests."""
    assets = Path(__file__).resolve().parents[1]
    frame = website_frame(content, "Report")
    plotly_js = plotly.offline.get_plotlyjs()
    controller = (assets / "charts.js").read_text(encoding="utf-8")
    summary_controller = (assets / "summary.js").read_text(encoding="utf-8")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="description" content="Interactive results, methods and limitations '
        'for wildlife-trade advert screening with decision models.">'
        '<link rel="canonical" '
        'href="https://ldswaby.github.io/wildlife-trade-screen/report.html">'
        f"<title>Report · {html.escape(PROJECT_TITLE)}</title>"
        '<link rel="stylesheet" href="styles.css"><script src="site.js" defer></script>'
        '<script src="report.js"></script>'
        f"<script>{plotly_js}</script></head><body>{frame}"
        f'<script id="study-data" type="application/json">{data_json}</script>'
        f"<script>{controller}</script><script>{summary_controller}</script></body></html>"
    )
