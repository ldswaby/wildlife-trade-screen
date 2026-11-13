"""Verify the interactive report with invented adverts and offline scorers."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from report_support import _table, render_evidence
from site_builder.evidence import _presentation_rows, _resource_rows

from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.snapshots import _identity, _load_pinned, _published_rows, discover_runs
from wildlife_screening.types import Advert, EvaluationData
from wildlife_screening.utils import file_hash

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("title_only", [False, True])
def test_presentation_order_groups_models_and_configurations(title_only: bool) -> None:
    models = [
        ("gpt_6_luna_chat", "GPT-6 Luna Chat binary"),
        ("gpt_6_luna_decisions", "GPT-6 Luna Decisions"),
        ("microsoft_decision_1", "Microsoft-Decision-1 via OpenRouter"),
        ("d3_27b_q4_k_m", "d3 GGUF Q4_K_M (local)"),
    ]
    configs = [
        ("title", "zero-shot"),
        ("title", "few-shot"),
        ("title_and_desc", "zero-shot"),
        ("title_and_desc", "few-shot"),
    ]
    expected = [
        {
            "series": f"{model}_{input_name}_{prompt}",
            "model": model,
            "arm": arm,
            "input": input_name,
            "prompt": prompt,
            "kind": "measured",
            "status": "complete",
        }
        for model, arm in models
        for input_name, prompt in configs
        if not title_only or input_name == "title"
    ]
    published = [
        {
            "series": name,
            "arm": name,
            "kind": "published",
            "input_modality": "text",
            "status": "published; no plot point",
        }
        for name in ("Published B", "Published A")
    ]
    shuffled = expected[1::2][::-1] + published + expected[::2][::-1]
    source_order = [row["series"] for row in shuffled]
    ordered = _presentation_rows(shuffled)
    assert ordered == expected + published
    assert [row["series"] for row in shuffled] == source_order
    table = _table(shuffled)
    cells = re.findall(r"<tr[^>]*><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td>", table)
    assert cells[: len(expected)] == [
        (
            "Microsoft-Decision-1" if row["model"] == "microsoft_decision_1" else row["arm"],
            "title" if row["input"] == "title" else "title + desc",
            row["prompt"],
        )
        for row in expected
    ]
    assert [row[0] for row in cells[-2:]] == ["Published B", "Published A"]
    assert "via OpenRouter" not in table
    # Each arm has a distinct charge, so reordering cannot silently misalign its values.
    resource_fields = {
        row["series"]: {
            "n": 4,
            "cost": (index + 1) / 100,
            "latency_s_total": index + 1,
            "resource_eligible": True,
        }
        for index, row in enumerate(expected)
    }
    resources = _resource_rows(
        [{**row, **resource_fields.get(row["series"], {})} for row in shuffled]
    )
    assert [row["series"] for row in resources] == [row["series"] for row in expected]
    for index, resource in enumerate(resources):
        assert resource["cost_per_advert"] == pytest.approx((index + 1) / 400)
        assert resource["latency_per_advert"] == (index + 1) / 4


def test_presentation_order_keeps_additional_models_in_first_seen_order() -> None:
    rows = [
        {
            "series": f"{model}_{prompt}",
            "model": model,
            "kind": "measured",
            "input": "title",
            "prompt": prompt,
            "status": "complete",
            "resource_eligible": True,
            "n": 2,
        }
        for model, prompt in [
            ("other_z", "few-shot"),
            ("d3_27b_q4_k_m", "few-shot"),
            ("other_a", "few-shot"),
            ("other_z", "zero-shot"),
            ("gpt_6_luna_chat", "few-shot"),
        ]
    ]
    assert _presentation_rows(rows) == [rows[i] for i in (4, 1, 3, 0, 2)]
    assert [row["series"] for row in _resource_rows(rows)] == [
        rows[i]["series"] for i in (4, 1, 3, 0, 2)
    ]
    # Filtering an incomplete arm must not change the first-seen model grouping.
    rows[0]["status"] = "incomplete; metrics withheld"
    assert [row["series"] for row in _resource_rows(rows)] == [
        rows[i]["series"] for i in (4, 1, 3, 2)
    ]


def _invented_csv(path: Path, revised: bool = False) -> Path:
    path.write_text(
        "id,title,label,relevant,description\n"
        f"one,Invented shark item{' revised' if revised else ''},1,animal origin,Invented detail\n"
        "two,Invented plastic item,0,not animal origin,Invented detail\n"
        "three,Invented ray item,1,animal origin,Invented detail\n",
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, data: Path, zero_shot: bool = False, checksum: bool = False) -> Path:
    output = tmp_path / "runs" / data.stem / ("zero" if zero_shot else "few")
    args = [
        "run",
        "--data",
        str(data),
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--decision-prompt",
        str(ROOT / f"prompts/sharks/decision_{'zeroshot' if zero_shot else 'fewshot'}_v0.yaml"),
        "--output",
        str(output),
        "--cache",
        str(tmp_path / f"cache-{data.stem}"),
    ]
    if zero_shot:
        args.append("--zero-shot")
    assert main(args) == 0
    pin = next(output.rglob("manifest-*.json"))
    if not checksum:
        # Callers edit the scores to build edge cases; dropping the checksum
        # emulates an older snapshot so the later validation steps are reached.
        manifest = json.loads(pin.read_text())
        del manifest["scores_sha256"]
        pin.write_text(json.dumps(manifest))
    return pin


def test_fresh_run_rejects_edited_scores(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "checked.csv")
    pin = _run(tmp_path, data, checksum=True)
    selection = yaml.safe_load(_selection(tmp_path, [pin], [data]).read_text())
    _load_pinned(pin, selection)
    score_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    scores = json.loads(score_path.read_text())
    scores[0]["p_relevant"] = 1 - scores[0]["p_relevant"]
    score_path.write_text(json.dumps(scores))
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_pinned(pin, selection)


def _selection(tmp_path: Path, pins: list[Path], data: list[Path]) -> Path:
    path = tmp_path / "selection.yaml"
    settings = {f"sharks-{file_hash(item)[:12]}": {"data": str(item)} for item in data}
    path.write_text(
        yaml.safe_dump({"pins": [str(pin) for pin in pins], "datasets": settings}),
        encoding="utf-8",
    )
    return path


def test_report_discovers_versions_and_renders_offline_page(tmp_path: Path) -> None:
    first = _invented_csv(tmp_path / "first.csv")
    second = _invented_csv(tmp_path / "second.csv", revised=True)
    pins = [_run(tmp_path, first), _run(tmp_path, first, True), _run(tmp_path, second)]
    catalog = discover_runs(tmp_path / "runs")
    assert len(catalog) == 3
    assert len({item["dataset"] for item in catalog}) == 2
    assert {item["status"] for item in catalog} == {"complete"}
    selection = _selection(tmp_path, pins, [first, second])
    selected = yaml.safe_load(selection.read_text())
    first_id = f"sharks-{file_hash(first)[:12]}"
    selected["default_visible"] = {first_id: ["fake_v1_fake_title_few_shot"]}
    selection.write_text(yaml.safe_dump(selected))
    output = tmp_path / "report.html"
    render_evidence(selection, output)
    page = output.read_text()
    assert page.count('class="dataset-panel"') == 2
    assert "fake_v1_fake_title_few_shot" in page
    assert "fake_v1_fake_title_zero_shot" in page
    assert "Plotly.react" in page and "Download SVG" in page
    assert "<th>Model</th><th>Input</th><th>Prompt</th>" in page
    assert "<th>Avg request latency / advert (s)</th>" not in page
    assert "Invented resource description" in page
    assert "Invented resource caveat." in page
    assert "Download resource SVG" in page
    assert "<th>Status</th>" not in page
    assert "<th>Configuration</th>" not in page
    assert page.count('class="plot-tools"') == 2
    assert page.count('class="plot-legend"') == 2
    assert 'aria-pressed="true">Zoom' in page
    assert "siteControls.rows()" in page
    assert "uirevision:key" in page and "scrollZoom:false" in page
    assert "margin:{l:95,r:20,t:25,b:90}" in page
    assert "Irrelevant adverts sent for review<br>(false-positive rate, FPR, %)" in page
    assert "Relevant adverts found (TPR, %)" in page
    assert "standoff:12,font:{size:14,color:figureText}" in page
    assert "automargin:true,range:[0,100]" in page
    assert "Plotly.relayout(chart" in page
    assert "layout.margin.b+extraHeight" in page
    assert "Plotly.newPlot(figure" in page and "Plotly.purge(figure)" in page
    assert "row.input_modality === 'text-image' ? 'square' : 'x'" in page
    assert "if (row.model.startsWith('LTS')) return '#ffa965'" in page
    assert "if (row.model.startsWith('ACTL ')) return '#70b7ff'" in page
    assert "Invented shark item" not in page
    assert "Invented detail" not in page
    assert "raw_output" not in page
    embedded = re.search(r'<script id="study-data" type="application/json">(.*?)</script>', page)
    assert embedded is not None
    payload = json.loads(embedded.group(1))
    assert set(payload[first_id]["default"]) == {
        "fake_v1_fake_title_few_shot",
        "fake_v1_fake_title_zero_shot",
    }
    assert len(payload[first_id]["rows"]) == 2
    assert payload[first_id]["resources"] == []  # Fake timing is not measured inference.
    assert payload[first_id]["display_order"] == [
        "fake_v1_fake_title_zero_shot",
        "fake_v1_fake_title_few_shot",
    ]
    # Presentation must preserve the pinned scores' aggregate coordinates and metrics.
    original, _ = _load_pinned(pins[0], selected)
    rendered_row = payload[first_id]["rows"][0]
    for field in ("series", "points", "thresholds", "point", "metrics"):
        assert rendered_row[field] == json.loads(json.dumps(original.get(field)))
    for index in range(2):
        assert f'aria-labelledby="roc-title-{index}"' in page
        assert f'aria-describedby="roc-guide-{index}"' in page
        assert f'aria-labelledby="resource-title-{index}"' in page
        assert f'aria-describedby="resource-guide-{index}"' in page
        assert page.index(f'id="chart-{index}"') < page.index(f'id="roc-guide-{index}"')
        assert page.index(f'id="resource-title-{index}"') < page.index(
            f'id="resource-guide-{index}"'
        )
        assert (
            page.index(f'id="table-title-{index}"')
            < page.index(f'aria-describedby="table-guide-{index}"')
            < page.index(f'id="table-guide-{index}"')
        )
        assert f'id="roc-key-{index}"' in page
        assert "<strong>Figure 1.</strong> Invented ROC description" in page
        assert f'id="resource-notes-{index}"' in page
        assert "<strong>Figure 2.</strong> Invented resource description" in page
    assert page.count('<figcaption class="result-caption">') == 6
    assert page.count("Invented ROC description") == 2
    assert page.count("Invented table description") == 2
    assert page.count("Invented resource description") == 2
    assert "<strong>Table 1.</strong> Invented table description" in page
    assert "Invented ROC caveat." in page

    assert page.count('<caption class="visually-hidden">Table 1. Invented table title') == 2
    assert "Figure 1. Invented ROC title" in page
    assert "Figure 2. Invented resources title" in page
    assert "Invented table caveat." in page
    assert "Invented precision definition." in page
    assert "roc-export-note" not in page
    assert "<dt>Precision</dt>" in page and "<dt>Missed</dt>" in page
    assert "addExportGuide(layout" in page
    assert "panel.querySelector('.roc-figure figcaption')" in page
    assert "WildlifeReport.initializeResources(panel,study[key].resource_views" in page
    assert (
        "panel.querySelector('.resource-panel figcaption')" in (ROOT / "site/report.js").read_text()
    )
    assert "context.measureText(candidate).width" in page


def test_report_accepts_task_namespaced_config_hash(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "animals.csv")
    review = tmp_path / "label-review.json"
    review.write_text(
        json.dumps(
            {
                "task": "animals",
                "data_sha256": file_hash(data),
                "positive_label": "1",
                "reviewer": "Invented reviewer",
                "basis": "source_documentation",
                "evidence": "Invented provenance",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "runs"
    assert (
        main(
            [
                "run",
                "--task",
                "animals",
                "--data",
                str(data),
                "--model",
                str(ROOT / "configs/models/fake.yaml"),
                "--decision-prompt",
                str(ROOT / "prompts/animals/decision_fewshot_v0.yaml"),
                "--label-review",
                str(review),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
                "--score-only",
            ]
        )
        == 0
    )
    pin = next(output.rglob("manifest-*.json"))
    selection = {
        "pins": [str(pin)],
        "datasets": {
            f"animals-{file_hash(data)[:12]}": {
                "data": str(data),
                "label_review": str(review),
            }
        },
    }
    row, _ = _load_pinned(pin, selection)
    assert row["dataset"].startswith("animals-")


def test_resource_panel_uses_completed_measured_aggregates_only() -> None:
    def row(series: str, **changes: Any) -> dict[str, Any]:
        return {
            "series": series,
            "model": series,
            "input": "title",
            "prompt": "few-shot",
            "kind": "measured",
            "status": "complete",
            "resource_eligible": True,
            "n": 4,
            "cost": 0.012,
            "latency_s_total": 10.0,
            "local_api_free": False,
            **changes,
        }

    resources = _resource_rows(
        [
            row("remote"),
            row("local", cost=None, local_api_free=True),
            row("unknown", cost=None, latency_s_total=None),
            row("failed", status="incomplete; metrics withheld"),
            row("synthetic", resource_eligible=False),
            {**row("published"), "kind": "published"},
        ]
    )
    assert resources == [
        {
            "series": "remote",
            "cost_per_advert": 0.003,
            "latency_per_advert": 2.5,
            "local_api_free": False,
        },
        {
            "series": "local",
            "cost_per_advert": 0.0,
            "latency_per_advert": 2.5,
            "local_api_free": True,
        },
        {
            "series": "unknown",
            "cost_per_advert": None,
            "latency_per_advert": None,
            "local_api_free": False,
        },
    ]


def test_report_rejects_duplicate_and_mismatched_pins(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "first.csv")
    pin = _run(tmp_path, data)
    selection = _selection(tmp_path, [pin, pin], [data])
    with pytest.raises(ValueError, match="same manifest twice"):
        render_evidence(selection, tmp_path / "report.html")
    selection = _selection(tmp_path, [pin], [data])
    selected = yaml.safe_load(selection.read_text())
    selected["default_visible"] = {"sharks": ["missing_curve"]}
    selection.write_text(yaml.safe_dump(selected))
    render_evidence(selection, tmp_path / "report.html")
    selection = _selection(tmp_path, [pin], [data])
    _invented_csv(data, revised=True)
    with pytest.raises(ValueError, match="dataset hash"):
        render_evidence(selection, tmp_path / "report.html")


def test_finished_snapshot_with_failure_has_policy_curve(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "first.csv")
    pin = _run(tmp_path, data)
    scores_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    scores = json.loads(scores_path.read_text())
    scores[0]["p_relevant"] = None
    scores[0]["error"] = "invented refusal"
    scores_path.write_text(json.dumps(scores))
    manifest = json.loads(pin.read_text())
    manifest["status"] = "incomplete"
    pin.write_text(json.dumps(manifest))
    selection = _selection(tmp_path, [pin], [data])
    output = tmp_path / "report.html"
    render_evidence(selection, output)
    page = output.read_text()
    assert "finished with failures" in page
    row, _ = _load_pinned(pin, yaml.safe_load(selection.read_text()))
    assert row["valid_predictions"] == 2 and row["failures"] == 1
    assert row["coverage"] == pytest.approx(2 / 3)
    assert row["points"][0] == (0, 0.5)
    assert row["thresholds"][0] == "failures only"
    assert row["metrics"]["n"] == 3
    assert "66.7% / 1" in page


def test_complete_binary_snapshot_has_one_point(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "binary.csv")
    pin = _run(tmp_path, data)
    scores_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    scores = json.loads(scores_path.read_text())
    for score in scores:
        score["p_relevant"] = float(score["p_relevant"] >= 0.5)
        score["score_type"] = "binary"
    scores_path.write_text(json.dumps(scores))
    selection = _selection(tmp_path, [pin], [data])
    output = tmp_path / "report.html"
    render_evidence(selection, output)
    page = output.read_text()
    embedded = re.search(r'<script id="study-data" type="application/json">(.*?)</script>', page)
    assert embedded is not None
    row = json.loads(embedded.group(1))[f"sharks-{file_hash(data)[:12]}"]["rows"][0]
    assert row["point"] == [
        row["metrics"]["false_positive_rate"],
        row["metrics"]["positive_recall"],
    ]
    assert row["points"] == []
    assert "Invented ROC caveat." in page


def test_published_points_require_precision_and_recall() -> None:
    data = EvaluationData(
        [Advert("a", "Invented shark item"), Advert("b", "Invented plastic item")],
        [1, 0],
        "fixture",
    )
    rows = _published_rows("sharks", data, True)
    assert len(rows) == 5
    assert all(row["point"] is not None for row in rows[:4])
    assert rows[2]["input_modality"] == "text-image"
    assert rows[2]["input"] == "published_text_image"
    assert rows[3]["point"] == [pytest.approx(0.873 * (1 / 0.908 - 1)), 0.873]
    assert rows[3]["false_positive_rate"] == rows[3]["point"][0]
    assert rows[4]["point"] is None
    assert rows[4]["prompt"] == "unreported"
    assert _published_rows("sharks", data, False) == []


def test_report_table_displays_conditions_and_published_assumption() -> None:
    data = EvaluationData(
        [Advert("a", "Invented shark item"), Advert("b", "Invented plastic item")],
        [1, 0],
        "fixture",
    )
    measured = {
        "series": "machine_title_and_desc_zero_shot",
        "kind": "measured",
        "arm": "Machine <test>",
        "input": "title_and_desc",
        "prompt": "zero-shot",
        "status": "incomplete; metrics withheld",
    }
    table = _table([measured, *_published_rows("sharks", data, True)])
    assert "<th>Model</th><th>Input</th><th>Prompt</th>" in table
    assert "<th>Status</th>" not in table
    assert "<th>Configuration</th>" not in table
    assert (
        "<td>Machine &lt;test&gt; (incomplete)</td><td>title + desc</td><td>zero-shot</td>"
    ) in table
    assert "<td>ACTL text</td><td>text (fields N/A)</td><td>N/A</td>" in table
    assert "<td>GPT-4</td><td>text (fields N/A)</td><td>unverified</td>" in table
    assert "Invented GPT-4 comparison note." in table
    assert "<strong>Published provenance:</strong>" in table
    assert "<td>N/A</td>" in table
    assert "<td>unknown</td>" not in table
    animal_table = _table(_published_rows("animals", data, True))
    assert "<td>LTS text-image</td><td>text + image (fields N/A)</td>" in animal_table
    assert animal_table.count('class="published-row"') == 5


def test_report_table_displays_title_and_few_shot() -> None:
    table = _table(
        [
            {
                "series": "invented_title_few_shot",
                "kind": "measured",
                "arm": "Invented model",
                "input": "title",
                "prompt": "few-shot",
                "status": "complete",
            }
        ]
    )
    assert "<td>Invented model</td><td>title</td><td>few-shot</td>" in table
    assert "Invented GPT-4 comparison note" not in table


def test_report_table_leaders_and_published_styling() -> None:
    def measured(
        name: str,
        precision: float | None,
        recall: float | None,
        f1: float | None,
        false_positive_rate: float | None,
        fn: int,
        cost: float | None,
        n: int,
        status: str = "complete",
    ) -> dict[str, object]:
        return {
            "kind": "measured",
            "arm": name,
            "input": "title",
            "prompt": "few-shot",
            "status": status,
            "metrics": {
                "positive_precision": precision,
                "positive_recall": recall,
                "positive_f1": f1,
                "false_positive_rate": false_positive_rate,
                "fn": fn,
                "missed": f"missed {fn} of 5 positives",
            },
            "cost": cost,
            "n": n,
        }

    rows = [
        measured("A", 0.9, 0.8, 0.85, 0.2, 2, 0.001, 10),
        measured("B", 0.9, 0.85, 0.86, 0.3, 1, 0.002, 20),
        measured("Incomplete", 1, 1, 1, 0, 0, 0, 10, "incomplete; metrics withheld"),
        measured("Unknown cost", None, None, None, None, 3, None, 10),
        {
            "kind": "published",
            "arm": "Published example",
            "input_modality": "text",
            "status": "published conditional point",
            "precision": 0.8,
            "recall": 0.95,
            "f1": 0.87,
            "point": [0.1, 0.95],
            "false_positive_rate": 0.1,
        },
    ]
    table = _table(rows)
    assert "API cost / advert" not in table
    assert "<th>TPR [95% CI]</th>" in table
    assert table.count("<strong>90.0%</strong>") == 2
    assert "<strong>95.0% [N/A]</strong>" in table
    assert "<strong>87.0%</strong>" in table
    assert "<strong>10.0%</strong>" in table
    assert "<strong>missed 1 of 5 positives</strong>" in table
    assert "<td>Incomplete (incomplete)</td>" in table
    assert "<td>N/A</td>" in table
    assert '<tr class="published-row"><td>Published example</td>' in table
    assert "Invented table caveat." in table


def test_local_api_cost_is_free_without_changing_unknown_costs() -> None:
    rows = [
        {
            "kind": "measured",
            "arm": "Local d3",
            "input": "title",
            "prompt": "few-shot",
            "status": "complete",
            "n": 10,
            "cost": None,
            "local_api_free": True,
        },
        {
            "kind": "measured",
            "arm": "Imported scores",
            "input": "title",
            "prompt": "few-shot",
            "status": "complete",
            "n": 10,
            "cost": None,
            "local_api_free": False,
        },
        {
            "kind": "measured",
            "arm": "Remote API",
            "input": "title",
            "prompt": "few-shot",
            "status": "complete",
            "n": 10,
            "cost": 0.01,
        },
    ]
    table = _table(rows)
    assert "<td>Local d3</td><td>title</td><td>few-shot</td>" in table
    assert "Free" not in table
    assert "<td>Imported scores</td><td>title</td><td>few-shot</td>" in table
    assert "<td>N/A</td>" in table
    assert "0.00100000" not in table
    assert (
        _resource_rows([{**row, "resource_eligible": True, "series": row["arm"]} for row in rows])[
            0
        ]["cost_per_advert"]
        == 0.0
    )
    assert "Invented table caveat." in table


def test_local_free_marker_requires_nonbillable_loopback_config(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "first.csv")
    pin = _run(tmp_path, data)
    selection = yaml.safe_load(_selection(tmp_path, [pin], [data]).read_text())
    ordinary, _ = _load_pinned(pin, selection)
    assert ordinary["local_api_free"] is False
    manifest = json.loads(pin.read_text())
    config = ModelConfig.load(ROOT / "configs/models/d3_gguf_q4.yaml")
    manifest["model_config"] = asdict(config)
    manifest["model_config_hash"] = config.fingerprint(("title",))
    pin.write_text(json.dumps(manifest))
    local, _ = _load_pinned(pin, selection)
    assert local["local_api_free"] is True
    assert local["cost"] == ordinary["cost"]


def test_saved_request_latency_includes_failures_and_excludes_placeholders(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "first.csv")
    pin = _run(tmp_path, data)
    selection = yaml.safe_load(_selection(tmp_path, [pin], [data]).read_text())
    scores_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    scores = json.loads(scores_path.read_text())
    for score, latency in zip(scores, [1.0, 2.0, 3.0], strict=True):
        score["latency_s"] = latency
    scores_path.write_text(json.dumps(scores))
    fake, _ = _load_pinned(pin, selection)
    assert fake["latency_s_total"] is None

    manifest = json.loads(pin.read_text())
    imported_config = ModelConfig(
        arm="Invented import",
        model_id="invented-import",
        adapter="precomputed",
        options={"path": str(data)},
    )
    manifest["model_config"] = asdict(imported_config)
    pin.write_text(json.dumps(manifest))
    imported, _ = _load_pinned(pin, selection)
    assert imported["latency_s_total"] is None

    config = ModelConfig.load(ROOT / "configs/models/d3_gguf_q4.yaml")
    manifest["model_config"] = asdict(config)
    manifest["model_config_hash"] = config.fingerprint(("title",))
    pin.write_text(json.dumps(manifest))
    measured, _ = _load_pinned(pin, selection)
    assert measured["latency_s_total"] == 6.0
    assert _resource_rows([measured])[0]["latency_per_advert"] == 2.0

    scores[1]["error"] = "invented refusal"
    scores[1]["p_relevant"] = None
    scores_path.write_text(json.dumps(scores))
    manifest["status"] = "incomplete"
    pin.write_text(json.dumps(manifest))
    incomplete, _ = _load_pinned(pin, selection)
    assert incomplete["latency_s_total"] == 6.0
    table = _table([incomplete])
    assert "latency / advert" not in table
    assert _resource_rows([incomplete])[0]["latency_per_advert"] == 2.0


def test_table_omits_latency_without_changing_resource_values() -> None:
    def row(name: str, total: float | None, status: str = "complete") -> dict[str, Any]:
        return {
            "kind": "measured",
            "arm": name,
            "input": "title",
            "prompt": "few-shot",
            "status": status,
            "n": 10,
            "latency_s_total": total,
        }

    rows: list[dict[str, Any]] = [
        row("Fast", 10.0001),
        row("Slow", 10.0002),
        row("Incomplete", 0.0, "incomplete; metrics withheld"),
        row("Imported", None),
        {
            "kind": "published",
            "arm": "Published example",
            "input_modality": "text",
            "status": "published conditional point",
        },
    ]
    table = _table(rows)
    assert "latency / advert" not in table
    assert rows[0]["latency_s_total"] == 10.0001
    assert table.count("<td>N/A</td>") >= 2
    assert "Invented table caveat." in table


def test_luna_method_names_do_not_collide() -> None:
    base = {
        "task": "sharks",
        "data_file_hash": "a" * 64,
        "allowlisted_fields": ["title"],
        "condition": "few-shot",
    }
    chat = _identity({**base, "model_config": {"model_id": "gpt-6-luna", "adapter": "chat"}})
    decisions = _identity(
        {**base, "model_config": {"model_id": "gpt-6-luna", "adapter": "openai_decisions"}}
    )
    assert chat["series"] == "gpt_6_luna_chat_title_few_shot"
    assert decisions["series"] == "gpt_6_luna_decisions_title_few_shot"


def test_default_run_path_uses_dataset_hash_and_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _invented_csv(tmp_path / "first.csv")
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "run",
                "--data",
                str(data),
                "--model",
                str(ROOT / "configs/models/fake.yaml"),
                "--decision-prompt",
                str(ROOT / "prompts/sharks/decision_fewshot_v0.yaml"),
                "--include-description",
                "--dry-run",
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    expected = tmp_path / "outputs/checks/sharks" / file_hash(data) / "title_and_desc"
    assert len(list(expected.rglob("manifest-*.json"))) == 1


@pytest.mark.parametrize("pin_count", [0, 1, 2, 3])
def test_report_tabs_follow_only_active_pins(tmp_path: Path, pin_count: int) -> None:
    first = _invented_csv(tmp_path / "first.csv")
    second = _invented_csv(tmp_path / "second.csv", revised=True)
    pins = [_run(tmp_path, first), _run(tmp_path, first, True), _run(tmp_path, second)]
    catalog = discover_runs(tmp_path / "runs")
    selection = _selection(tmp_path, pins[:pin_count], [first, second])
    # Commented pins and dataset settings must not resurrect an unselected tab.
    with selection.open("a") as stream:
        stream.write("\n# Commented pins:\n" + "".join(f"# - {pin}\n" for pin in pins[pin_count:]))
    output = tmp_path / "report.html"
    render_evidence(selection, output)
    page = output.read_text()
    embedded = re.search(r'<script id="study-data" type="application/json">(.*?)</script>', page)
    assert embedded is not None
    tabs = list(json.loads(embedded.group(1)))
    expected = {f"sharks-{file_hash(first)[:12]}"} if pin_count else set()
    if pin_count == 3:
        expected.add(f"sharks-{file_hash(second)[:12]}")
    assert tabs == sorted(expected)
    assert page.count('class="dataset-panel"') == len(expected)
    assert (not tabs) == (pin_count == 0)
    assert discover_runs(tmp_path / "runs") == catalog


def test_empty_selection_builds_editable_report_shell(tmp_path: Path) -> None:
    selection = tmp_path / "selection.yaml"
    selection.write_text("pins: []\n")
    output = tmp_path / "report.html"
    render_evidence(selection, output)
    assert 'type="application/json">{}</script>' in output.read_text()


@pytest.mark.parametrize("all_failed", [False, True])
@pytest.mark.parametrize("no_positives", [False, True])
def test_policy_snapshot_edges_and_published_coverage(
    tmp_path: Path, all_failed: bool, no_positives: bool
) -> None:
    data = _invented_csv(tmp_path / "edge.csv")
    if no_positives:
        data.write_text(data.read_text().replace(",1,animal origin,", ",0,not animal origin,"))
    pin = _run(tmp_path, data)
    manifest = json.loads(pin.read_text())
    score_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    scores = json.loads(score_path.read_text())
    for i, score in enumerate(scores):
        if all_failed or i == 0:
            score.update(p_relevant=None, error="invented refusal")
        else:
            score.update(p_relevant=float(i == 2), score_type="binary")
    score_path.write_text(json.dumps(scores))
    manifest["status"] = "incomplete"
    pin.write_text(json.dumps(manifest))
    selection_path = _selection(tmp_path, [pin], [data])
    row, _ = _load_pinned(pin, yaml.safe_load(selection_path.read_text()))
    assert row["status"] == "finished with failures"
    assert row["points"] == []  # Binary and all-failure arms never invent a ranking.
    if no_positives:
        assert row["point"] is None and row["metrics"]["positive_recall"] is None
    else:
        assert row["point"] == ([1, 1] if all_failed else [0, 1])
    render_evidence(selection_path, tmp_path / "report.html")
    assert "Coverage / failure count</th>" in (tmp_path / "report.html").read_text()
    assert ("0.0% / 3" if all_failed else "66.7% / 1") in _table([row])


def test_corrupt_result_order_and_partial_records_cannot_be_fallbacks(tmp_path: Path) -> None:
    data = _invented_csv(tmp_path / "invalid.csv")
    pin = _run(tmp_path, data)
    selection = yaml.safe_load(_selection(tmp_path, [pin], [data]).read_text())
    score_path = pin.with_name(pin.name.replace("manifest-", "scores-"))
    original = json.loads(score_path.read_text())
    for damaged in (original[:-1], original[::-1], [original[0]] * len(original)):
        score_path.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="coverage|IDs/order"):
            _load_pinned(pin, selection)
