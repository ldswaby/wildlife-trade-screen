"""Verify cross-dataset summaries using invented, unequal-sized datasets offline."""

import copy
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from site_builder.build import initialize
from site_builder.content import report_content
from site_builder.descriptions import load_descriptions
from site_builder.evidence import DatasetEvidence
from site_builder.summary import (
    DECISION_MODELS,
    SUMMARY_INPUTS,
    SUMMARY_MODELS,
    summary_data,
    summary_figure,
)
from test_resource_links import node

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def selected() -> tuple[list[DatasetEvidence], dict[str, Any]]:
    panels = []
    data: dict[str, Any] = {}
    for i, (task, input_name) in enumerate(SUMMARY_INPUTS.items()):
        key = task + "-invented"
        panels.append(DatasetEvidence(i, key, task, 10 ** (i + 1), 1, "a" * 64, [], "", "", "", ""))
        rows, resources = [], []
        for model, _ in SUMMARY_MODELS:
            rows.append(
                {
                    "model": model,
                    "series": model,
                    "kind": "measured",
                    "prompt": "zero-shot",
                    "input": input_name,
                    "status": "complete",
                    "metrics": {
                        "balanced_accuracy": 0.6 + i * 0.2,
                        "fpr_at_95_tpr": 0.01 + i * 0.03,
                    },
                }
            )
            resources.append(
                {
                    "series": model,
                    "cost_per_advert": 0 if model.startswith("d3") else (i + 1) * 1e-5,
                    "latency_per_advert": i + 1,
                }
            )
        data[key] = {"rows": rows, "resources": resources}
    return panels, data


def add_published_references(selected: Any) -> None:
    """Add invented references and distractors, with hand-checkable class rates."""
    panels, data = selected
    for panel, model, point in (
        (panels[0], "ACTL text", [0.1, 0.8]),
        (panels[1], "LTS-text", [0.2, 0.6]),
    ):
        data[panel.key]["rows"].extend(
            [
                {"kind": "published", "model": model, "input_modality": "text", "point": point},
                {
                    "kind": "published",
                    "model": "Other baseline",
                    "input_modality": "text",
                    "point": [0, 1],
                },
                {
                    "kind": "published",
                    "model": model,
                    "input_modality": "text-image",
                    "point": [0, 1],
                },
            ]
        )


def test_published_references_are_separate_estimates_with_method_and_caveat(selected: Any) -> None:
    original = summary_data(*selected)["metrics"]
    add_published_references(selected)
    before = copy.deepcopy(selected)
    summary = summary_data(*selected)
    assert selected == before
    assert summary["metrics"] == original
    points = summary["baselines"]["points"]
    assert [p["model"] for p in points] == ["ACTL text", "LTS-text"]
    assert [p["dataset"] for p in points] == ["Shark Products", "Animal Products"]
    assert [p["value"] for p in points] == pytest.approx([85, 70])
    assert [p["offset"] for p in points] == [-0.16, 0]
    assert all("reconstructed FPR" in p["caveat"] for p in points)
    assert "Live Birds: no matching published" in summary["baselines"]["unavailable"]


@pytest.mark.parametrize(
    "missing", ["row", "duplicate", "point", "nonfinite", "dataset", "version"]
)
def test_published_missing_or_ambiguous_references_never_substitute(
    selected: Any, missing: str
) -> None:
    add_published_references(selected)
    panels, data = selected
    rows = data[panels[0].key]["rows"]
    reference = rows[4]
    if missing == "row":
        rows.remove(reference)
    elif missing == "duplicate":
        rows.append(copy.deepcopy(reference))
    elif missing == "point":
        reference["point"] = None
    elif missing == "nonfinite":
        reference["point"] = [float("nan"), 0.8]
    elif missing == "dataset":
        panels.pop(0)
    else:
        panels.append(copy.deepcopy(panels[0]))
    baseline = summary_data(panels, data)["baselines"]
    assert [p["dataset"] for p in baseline["points"]] == ["Animal Products"]
    assert "Shark Products:" in baseline["unavailable"]


def test_independent_means_population_sd_scaling_and_no_source_mutation(selected: Any) -> None:
    panels, data = selected
    before = copy.deepcopy(data)
    summary = summary_data(panels, data)
    assert data == before
    assert [c["name"] for c in summary["metrics"][0]["cells"]] == [
        *[n for _, n in SUMMARY_MODELS[:3]],
        "Decision Models",
        SUMMARY_MODELS[3][1],
    ]
    for cell in summary["metrics"][0]["cells"]:
        assert cell["n"] == 3 and cell["mean"] == pytest.approx(80)
        assert cell["sd"] == pytest.approx(math.sqrt(800 / 3))
    cells = summary["metrics"][2]["cells"]
    assert cells[0]["mean"] == cells[0]["sd"] == 0
    assert cells[1]["mean"] == pytest.approx(0.02)
    assert cells[1]["sd"] == pytest.approx(math.sqrt(2 / 3) * 0.01)
    assert summary["metrics"][3]["cells"][0]["mean"] == 2000
    assert summary["metrics"][3]["cells"][0]["sd"] == pytest.approx(math.sqrt(2 / 3) * 1000)
    chat = summary["metrics"][1]["cells"][4]
    assert chat["mean"] is None and chat["n"] == 0 and not chat["points"]


@pytest.mark.parametrize("change", ["few-shot", "title", "published", "aborted"])
def test_unselected_conditions_and_baselines_never_enter_summary(
    selected: Any, change: str
) -> None:
    panels, data = selected
    expected = summary_data(panels, data)
    row = copy.deepcopy(data[panels[0].key]["rows"][0])
    if change == "few-shot":
        row["prompt"] = change
    elif change == "title":
        row["input"] = change
    elif change == "published":
        row["kind"] = change
    else:
        row["status"] = change
    row["metrics"]["balanced_accuracy"] = 999
    data[panels[0].key]["rows"].append(row)
    assert summary_data(panels, data) == expected


@pytest.mark.parametrize("missing", ["metric", "resource", "configuration", "dataset", "version"])
def test_partial_coverage_retains_known_points_without_mean(selected: Any, missing: str) -> None:
    panels, data = selected
    first = data[panels[0].key]
    field = 0
    if missing == "metric":
        first["rows"][0]["metrics"]["balanced_accuracy"] = None
    elif missing == "resource":
        first["resources"][0]["latency_per_advert"] = None
        field = 3
    elif missing == "configuration":
        first["rows"].append(copy.deepcopy(first["rows"][0]))
    elif missing == "dataset":
        panels.pop(0)
    else:
        p = panels[0]
        panels.append(
            DatasetEvidence(4, "sharks-other", p.task, 5, 1, "b" * 64, [], "", "", "", "")
        )
    cell = summary_data(panels, data)["metrics"][field]["cells"][0]
    assert cell["n"] == 2 and len(cell["points"]) == 2
    assert cell["mean"] is None and cell["sd"] is None and cell["unavailable"]
    if missing == "version":
        assert "ambiguous dataset versions" in cell["unavailable"]


def test_empty_selection_is_explicitly_unavailable() -> None:
    assert all(
        c["n"] == 0 and c["mean"] is None
        for m in summary_data([], {})["metrics"]
        for c in m["cells"]
    )


def test_caption_and_summary_precede_preserved_prose_and_keep_anchor(
    tmp_path: Path, selected: Any
) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    (directory / "overall-results.md").write_text("Keep **invented summary prose** exactly.")
    panels, data = selected
    add_published_references(selected)
    figure = summary_figure(summary_data(panels, data), load_descriptions(directory))
    page = report_content(directory, panels, (ROOT / "site/report.html").read_text(), figure)
    assert page.index('id="overall-results">Results') < page.index('class="summary-figure"')
    assert page.index('class="summary-figure"') < page.index(
        "<strong>invented summary prose</strong>"
    )
    assert "Shark Products — ACTL text: 85% (estimated)" in page
    assert "Animal Products — LTS-text: 70% (estimated)" in page
    assert "Supervised baselines" in page and "reconstructed FPR" in page
    assert (
        "Live Birds (split policy) — Domestic poultry — Naive Bayes: approximately 98.05%" in page
    )
    assert "eligible n=16,475" in page and "not our combined-policy performance" in page
    assert "92.4% (±4.23615 pp population SD)" in page
    assert "3/3 policies" in page
    assert "Source: Stringham et al. (2021), Appendix S2" in page
    assert 'id="summary-data"' in page and 'id="summary-guide"' in page
    assert "N/A (0/3 datasets)" in page and "metric unavailable" in page


def test_plot_geometry_jitter_and_both_exports_are_identical(selected: Any) -> None:
    add_published_references(selected)
    summary = summary_data(*selected)
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm');
      const api=require(process.argv[1]), input=SUMMARY_JSON;
      const spec=api.plotSpec(input), second=api.plotSpec(input);
      const buttons={},plots=[],downloads=[];
      const host={style:{},clientWidth:1100};
      const document={querySelector:s=>s==='.summary-chart'?host:s==='#summary-guide'?
        {textContent:'Invented guidance'}:buttons[s]??={},
        getElementById:id=>({textContent:id==='summary-data'?
          JSON.stringify(input):'Invented summary'}),
        createElement:()=>({style:{},remove(){}}),body:{append(){}}};
      const Plotly={newPlot:async(el,traces,layout)=>{
          plots.push({traces,layout:JSON.parse(JSON.stringify(layout))});
          delete layout.height;
        },
        downloadImage:async(el,options)=>downloads.push(options),purge(){}};
      const context=vm.createContext({document,Plotly,WildlifeReport:require("./site/report.js"),
        addExportGuide:layout=>{layout.title={text:'Invented guide'};layout.height+=50;return 50}});
      vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
      (async()=>{await buttons['.summary-png'].onclick();await buttons['.summary-svg'].onclick();
        process.stdout.write(JSON.stringify({spec,second,plots,downloads}));})();
    """.replace("SUMMARY_JSON", json.dumps(summary)),
        "site/summary.js",
    )
    assert result["spec"] == result["second"]
    spec = result["spec"]
    assert len(spec["traces"]) == 17
    assert spec["layout"]["yaxis"]["ticktext"] == [
        *[n for _, n in SUMMARY_MODELS[:3]],
        "<b>Decision models</b> (<i>μ</i>)",
        SUMMARY_MODELS[3][1],
        "Supervised baselines",
    ]
    assert spec["layout"]["yaxis"]["tickvals"] == [0, 1, 2, 3, 4, 5]
    assert spec["layout"]["shapes"][:1] == [
        {
            "type": "rect",
            "xref": "paper",
            "yref": "y",
            "x0": 0,
            "x1": 1,
            "y0": 4.5,
            "y1": 5.5,
            "fillcolor": "#303232",
            "opacity": 1,
            "line": {"width": 0},
            "layer": "below",
        }
    ]
    bracket = spec["layout"]["shapes"][1:]
    assert len(bracket) == 4
    assert bracket[0]["y0"] == -0.5 and bracket[0]["y1"] == 3.5
    assert [line["y0"] for line in bracket[1:]] == [-0.5, 2.5, 3.5]
    assert all(line["line"] == {"color": "#707575", "width": 1.1} for line in bracket)
    means = [trace for trace in spec["traces"] if trace["name"] == "Mean ±1 SD"]
    assert all(trace["marker"]["size"] == 12 for trace in means)
    assert all(
        trace["marker"]["symbol"] == ["circle"] * 3 + ["diamond", "circle"] for trace in means
    )
    assert all(p["layout"]["shapes"] == spec["layout"]["shapes"] for p in result["plots"])
    split = spec["traces"][-1]
    assert split["name"] == "Live Birds (split policy)" and split["showlegend"]
    assert split["marker"]["symbol"] == "x-thin"
    assert split["marker"]["color"] == spec["traces"][2]["marker"]["color"] == "#5bd4bc"
    assert split["marker"]["opacity"] == spec["traces"][2]["marker"]["opacity"] == 0.85
    assert all(
        t["marker"]["opacity"] == (1 if t["name"] == "Mean ±1 SD" else 0.85) for t in spec["traces"]
    )
    assert split["marker"]["line"] == {"color": "#5bd4bc", "width": 1}
    assert split["x"] == pytest.approx([92.4])
    assert split["y"] == pytest.approx([5.16])
    assert split["xaxis"] == "x"
    assert split["error_x"]["array"] == pytest.approx([math.sqrt(17.945)])
    assert split["error_x"]["symmetric"] and split["error_x"]["visible"]
    assert split["error_x"]["color"] == "rgba(91,212,188,0.85)"
    split_hover = split["customdata"][0].replace("<br>", " ")
    assert "across 3 policies" in split_hover
    assert "Domestic poultry" in split_hover
    assert "98.05%" in split_hover
    assert "Naive Bayes" in split_hover
    assert "16,475" in split_hover
    assert spec["layout"]["xaxis"]["range"][0] <= 92.4 - math.sqrt(17.945)
    assert spec["layout"]["xaxis"]["range"][1] >= 92.4 + math.sqrt(17.945)
    assert "not combined-policy performance" in split_hover.lower()
    assert "Appendix S2" in split_hover
    assert spec["layout"]["legend"]["itemclick"] is False
    assert spec["layout"]["legend"]["itemdoubleclick"] is False
    for i in range(4):
        axis = spec["layout"]["xaxis" + (str(i + 1) if i else "")]
        assert axis["type"] == "linear"
        assert axis["range"][0] <= (60 if i == 0 else 0)
        expected = [-0.16, 0.84, 1.84, 2.84, 3.84][: 4 if i == 1 else 5]
        assert spec["traces"][i * 4]["y"] == expected + ([4.84] if i == 0 else [])
        assert spec["traces"][i * 4 + 3]["y"] == [0, 1, 2, 3, 4]
        assert axis["tickformat"] == ([".1~f", ".1~f", ".3~f", ".3~s"][i])
    assert "ACTL text" in spec["traces"][0]["customdata"][-1]
    assert "LTS-text" in spec["traces"][1]["customdata"][-1]
    assert "reconstructed FPR" in spec["traces"][0]["customdata"][-1]
    assert spec["traces"][3]["error_x"]["array"][0] == pytest.approx(math.sqrt(800 / 3))
    aggregate_hover = spec["traces"][3]["customdata"][3].replace("<br>", " ")
    assert "9/9 model–dataset values; 3/3 complete dataset averages" in aggregate_hover
    assert "across 9 model–dataset values" in aggregate_hover
    assert "not an ensemble" in aggregate_hover
    assert "<b>Input(s):</b> title + desc, title <b>Prompt(s):</b>" in aggregate_hover
    assert "combined listing text" not in aggregate_hover
    assert "combined title/listing text" not in aggregate_hover
    bird_hover = spec["traces"][2]["customdata"][0]
    assert "<b>Input(s):</b> title + desc<br>" in bird_hover
    for _, name in DECISION_MODELS:
        assert name in aggregate_hover
    for dataset in summary["datasets"]:
        assert dataset in aggregate_hover
    dataset_hover = spec["traces"][0]["customdata"][3].replace("<br>", " ")
    assert "across 3 models" in dataset_hover
    assert "<b>Mean:</b>" in dataset_hover
    for trace in spec["traces"]:
        assert trace["hovertemplate"] == "%{customdata}<extra></extra>"
        for tooltip in filter(None, trace["customdata"]):
            fields = [part.split(":</b>")[0] for part in tooltip.split("<b>")[1:]]
            assert fields[:3] == ["Metric", "Model(s)", "Dataset(s)"]
            expected_fields = ["Metric", "Model(s)", "Dataset(s)"]
            if "Input(s)" in fields:
                expected_fields.extend(["Input(s)", "Prompt(s)"])
                assert "zero-shot" in tooltip and "few-shot" not in tooltip
            expected_fields.append("Mean" if "Mean" in fields else "Value")
            if "Mean" in fields:
                expected_fields.append("Standard deviation")
                deviation = tooltip.split("<b>Standard deviation:</b> ")[1].split("<br>")[0]
                units = (
                    " pp"
                    if trace["xaxis"] in {"x", "x2"}
                    else " USD"
                    if trace["xaxis"] == "x3"
                    else " s"
                )
                assert re.fullmatch(r"[0-9.e+-]+" + re.escape(units), deviation)
                assert "Population SD across" in tooltip.split("<b>Notes:</b> ")[1]
            if "Notes" in fields:
                expected_fields.append("Notes")
            assert fields == expected_fields
    single_hover = spec["traces"][0]["customdata"][0]
    assert "title + desc" in single_hover
    assert "<b>Value:</b> 60%" in single_hover
    assert "Standard deviation" not in single_hover
    baseline_hover = spec["traces"][0]["customdata"][-1]
    assert "<b>Value:</b> 85%" in baseline_hover
    assert "Input(s)" not in baseline_hover and "Prompt(s)" not in baseline_hover
    assert "Standard deviation" not in baseline_hover
    assert "Input(s)" not in split_hover and "Prompt(s)" not in split_hover
    assert "Logistic regression" in split_hover and "Wanted adverts" in split_hover
    assert "confidence interval or fold SD" in split_hover
    resource_hover = spec["traces"][11]["customdata"][0]
    assert "USD" in resource_hover
    time_hover = spec["traces"][15]["customdata"][0]
    assert re.search(r"<b>Standard deviation:</b> [0-9.e+-]+ s<br><b>Notes:</b>", time_hover)
    assert [d["format"] for d in result["downloads"]] == ["png", "svg"]
    assert all(d["width"] >= 1400 and d["height"] == 600 for d in result["downloads"])
    missing_chat = next(
        a for a in spec["layout"]["annotations"] if a.get("y") == 4 and a.get("xref") == "x2 domain"
    )
    assert missing_chat["text"] == "N/A" and "0/3 datasets" in missing_chat["hovertext"]
    assert missing_chat["x"] == 0.55 and missing_chat["yref"] == "y"
    assert all(p["traces"] == spec["traces"] for p in result["plots"])
    assert all(
        a["text"] == "N/A"
        for a in spec["layout"]["annotations"]
        if "N/A" in a["text"] and a["y"] != 5
    )

    assert len([a for a in spec["layout"]["annotations"] if a.get("y") == 5]) == 3
    assert all(p["layout"]["yaxis"] == spec["layout"]["yaxis"] for p in result["plots"])


def test_summary_omits_f1_and_historical_charge_reference(selected: Any) -> None:
    summary = summary_data(*selected)
    assert [m["key"] for m in summary["metrics"]] == [
        "balanced_accuracy",
        "fpr_at_95_tpr",
        "cost_per_advert",
        "latency_per_advert",
    ]
    assert "GPT-4" not in json.dumps(summary)


def test_decision_row_uses_nine_components_not_sd_of_dataset_means(selected: Any) -> None:
    panels, data = selected
    for dataset in data.values():
        for i, row in enumerate(dataset["rows"][:3]):
            row["metrics"]["balanced_accuracy"] = i / 10
            row["metrics"]["fpr_at_95_tpr"] = i / 10
        dataset["rows"][3]["metrics"]["balanced_accuracy"] = 999
    original = copy.deepcopy(selected)
    summary = summary_data(panels, data)
    assert selected == original
    aggregate = summary["metrics"][0]["cells"][3]
    assert aggregate["name"] == "Decision Models"
    assert aggregate["models"] == [name for _, name in DECISION_MODELS]
    assert aggregate["component_n"] == aggregate["component_total"] == 9
    assert aggregate["n"] == 3
    assert [p["value"] for p in aggregate["points"]] == pytest.approx([10, 10, 10])
    assert aggregate["mean"] == pytest.approx(10)
    # Nine equally weighted values: 0/10/20 repeated three times, sum squares=600.
    assert aggregate["sd"] == pytest.approx(math.sqrt(600 / 9))
    assert aggregate["sd"] != 0  # SD of the three dataset averages would be zero.
    # Each coloured mean uses 0, 10, 20: squared deviations total 200.
    assert [p["sd"] for p in aggregate["points"]] == pytest.approx([math.sqrt(200 / 3)] * 3)
    assert {p["model"] for p in aggregate["components"]} == {m for m, _ in DECISION_MODELS}
    assert all(p["input"] == SUMMARY_INPUTS[p["task"]] for p in aggregate["components"])
    figure = summary_figure(summary, load_descriptions(ROOT / "report"))
    assert "9/9 model–dataset values" in figure
    assert "not SD of dataset averages" in figure
    assert "d3 / Shark Products / zero-shot / title + desc" in figure


@pytest.mark.parametrize(
    "missing", ["metric", "resource", "duplicate", "dataset", "version", "incomplete"]
)
def test_decision_row_requires_all_three_per_dataset_and_all_nine_overall(
    selected: Any, missing: str
) -> None:
    panels, data = selected
    first = data[panels[0].key]
    metric = 0
    expected = 8
    if missing == "metric":
        first["rows"][0]["metrics"]["balanced_accuracy"] = None
    elif missing == "resource":
        first["resources"][0]["cost_per_advert"] = None
        metric = 2
    elif missing == "duplicate":
        first["rows"].append(copy.deepcopy(first["rows"][0]))
    elif missing == "dataset":
        panels.pop(0)
        expected = 6
    elif missing == "version":
        panels.append(copy.deepcopy(panels[0]))
        expected = 6
    else:
        first["rows"][0]["status"] = "incomplete"
    aggregate = summary_data(panels, data)["metrics"][metric]["cells"][3]
    assert aggregate["mean"] is aggregate["sd"] is None
    assert aggregate["component_n"] == expected and aggregate["n"] == 2
    assert [p["dataset"] for p in aggregate["points"]] == ["Animal Products", "Live Birds"]
    assert "d3:" in aggregate["unavailable"] and "Shark Products" in aggregate["unavailable"]


def test_decision_resource_scaling_failure_eligibility_and_original_rows_unchanged(
    selected: Any,
) -> None:
    panels, data = selected
    data[panels[0].key]["rows"][0]["status"] = "finished with failures"
    for i, dataset in enumerate(data.values()):
        for j, resource in enumerate(dataset["resources"][:3]):
            resource["cost_per_advert"] = (3 * i + j) / 1000
            resource["latency_per_advert"] = (3 * i + j) / 1000
    summary = summary_data(panels, data)
    for metric in summary["metrics"][2:]:
        aggregate = metric["cells"][3]
        # Values 0..8: mean 4; population variance 60/9.
        assert aggregate["mean"] == pytest.approx(4)
        assert aggregate["sd"] == pytest.approx(math.sqrt(60 / 9))
        assert [p["value"] for p in aggregate["points"]] == pytest.approx([1, 4, 7])
    without_chat = copy.deepcopy(data)
    for dataset in without_chat.values():
        dataset["rows"] = dataset["rows"][:3]
    reduced = summary_data(panels, without_chat)
    assert [m["cells"][3] for m in summary["metrics"]] == [
        m["cells"][3] for m in reduced["metrics"]
    ]


def test_bird_split_references_are_separate_derived_values(selected: Any) -> None:
    before = copy.deepcopy(selected)
    summary = summary_data(*selected)
    assert selected == before
    references = summary["baselines"]["split_policy_points"]
    assert [p["policy"] for p in references] == ["Domestic poultry", "Junk", "Wanted adverts"]
    assert [p["model"] for p in references] == ["Naive Bayes", "Naive Bayes", "Logistic regression"]
    assert [p["n"] for p in references] == [16475, 16446, 13751]
    assert [p["value"] for p in references] == pytest.approx([98.05, 91.30, 87.85])
    assert [p["offset"] for p in references] == pytest.approx([0.08, 0.16, 0.24])
    assert all(p["cv_folds"] == 10 and "s002" in p["source"] for p in references)
    assert all(
        "rounded" in p["caveat"] and "not our combined-policy" in p["caveat"] for p in references
    )
    aggregate = summary["baselines"]["split_policy_summary"]
    # Equal policy weights: deviations from 92.4 are 5.65, -1.10 and -4.55 pp.
    assert aggregate["mean"] == pytest.approx(92.4)
    assert aggregate["sd"] == pytest.approx(math.sqrt((5.65**2 + 1.1**2 + 4.55**2) / 3))
    weighted = sum(p["n"] * p["value"] for p in references) / sum(p["n"] for p in references)
    assert aggregate["mean"] != pytest.approx(weighted)
    assert aggregate["n"] == 3 and aggregate["offset"] == 0.16
    assert "confidence intervals or cross-validation-fold variation" in aggregate["caveat"]
    assert aggregate["source"].endswith("s002")
    # Removing bird source references does not change any measured metric/resource cell.
    panels, data = selected
    other = summary_data([p for p in panels if p.task != "birds"], data)
    assert other["baselines"]["split_policy_points"] == []
    for metric in summary["metrics"]:
        assert all(
            c["n"]
            == (0 if metric["key"] == "fpr_at_95_tpr" and c["model"] == "gpt_6_luna_chat" else 3)
            for c in metric["cells"]
        )


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_bird_references_require_one_unambiguous_dataset(selected: Any, change: str) -> None:
    panels, data = selected
    bird = next(p for p in panels if p.task == "birds")
    if change == "missing":
        panels.remove(bird)
    else:
        panels.append(copy.deepcopy(bird))
    baseline = summary_data(panels, data)["baselines"]
    assert baseline["split_policy_points"] == []
    assert baseline["split_policy_summary"] is None
    assert (
        "dataset not selected" if change == "missing" else "ambiguous dataset versions"
    ) in baseline["split_policy_unavailable"]
    spec = node(
        (
            "const api=require(process.argv[1]); console.log(JSON.stringify(api.plotSpec(INPUT)));"
        ).replace("INPUT", json.dumps(summary_data(panels, data))),
        "site/summary.js",
    )
    assert not any(t["name"] == "Live Birds (split policy)" for t in spec["traces"])


def test_bird_derivation_matches_mean_of_independent_fold_balanced_accuracies(
    selected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Invented folds: mean((60+90)/2, (80+70)/2) = 75%; no pooled confusion counts.
    monkeypatch.setattr(
        "site_builder.summary.BIRD_SPLIT_REFERENCES",
        (("Invented policy", "Invented classifier", 12, 0.7, 0.8),),
    )
    summary = summary_data(*selected)
    reference = summary["baselines"]["split_policy_points"][0]
    expected = ((60 + 90) / 2 + (80 + 70) / 2) / 2
    assert reference["value"] == expected == 75
    # Reference changes do not change measured means or population SDs.
    assert summary["metrics"][0]["cells"][0]["mean"] == pytest.approx(80)
    assert summary["metrics"][0]["cells"][0]["sd"] == pytest.approx(math.sqrt(800 / 3))


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "extra", "none", "nan", "inf", "invalid-rate"]
)
def test_bird_policy_mean_requires_all_three_valid_distinct_references(
    selected: Any, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from site_builder.summary import BIRD_SPLIT_REFERENCES

    expected = summary_data(*selected)["metrics"]
    refs = list(BIRD_SPLIT_REFERENCES)
    if change == "missing":
        refs.pop()
    elif change == "duplicate":
        refs[-1] = refs[0]
    elif change == "extra":
        refs.append(refs[0])
    else:
        policy, model, n, _, specificity = refs[0]
        bad = {"none": None, "nan": float("nan"), "inf": float("inf"), "invalid-rate": 1.2}[change]
        refs[0] = (policy, model, n, bad, specificity)  # type: ignore[assignment]
    monkeypatch.setattr("site_builder.summary.BIRD_SPLIT_REFERENCES", tuple(refs))
    summary = summary_data(*selected)
    assert summary["metrics"] == expected
    assert summary["baselines"]["split_policy_summary"] is None
    assert "all three valid, distinct" in summary["baselines"]["split_policy_unavailable"]
    json.dumps(summary, allow_nan=False)
    page = summary_figure(summary, load_descriptions(ROOT / "report"))
    assert "all three valid, distinct" in page
    spec = node(
        "const api=require(process.argv[1]);console.log(JSON.stringify(api.plotSpec("
        + json.dumps(summary)
        + ")));",
        "site/summary.js",
    )
    assert not any(trace["name"] == "Live Birds (split policy)" for trace in spec["traces"])
