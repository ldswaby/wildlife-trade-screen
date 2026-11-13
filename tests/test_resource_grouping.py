"""Verify resource configuration means, population SDs and independent exports offline."""

import copy
import json
from typing import Any

import pytest
from site_builder.evidence import _grouping_controls, _resource_rows
from site_builder.resource_grouping import resource_views
from test_resource_links import node

from wildlife_screening.metrics import FINISHED_WITH_FAILURES


def row(
    model: str = "gpt_6_luna_chat",
    input_: str = "title",
    prompt: str = "zero-shot",
    value: float | None = 60,
) -> dict[str, Any]:
    return {
        "series": f"{model}-{input_}-{prompt}",
        "model": model,
        "input": input_,
        "prompt": prompt,
        "kind": "measured",
        "status": "complete",
        "resource_eligible": True,
        "n": 1000,
        "cost": value,
        "latency_s_total": 120,
        "local_api_free": False,
    }


def views(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return resource_views(_resource_rows(rows), rows)


@pytest.mark.parametrize("key,n", [("none", 8), ("input", 4), ("prompt", 4), ("input,prompt", 2)])
def test_masks_models_and_source_preservation(key: str, n: int) -> None:
    rows = [
        row(model, input_, prompt)
        for model in ("gpt_6_luna_chat", "microsoft_decision_1")
        for input_ in ("title", "title_and_desc")
        for prompt in ("zero-shot", "few-shot")
    ]
    before = copy.deepcopy(rows)
    selected = views(rows)[key]["rows"]
    assert len(selected) == n and rows == before
    assert {r["model"] for r in selected} == {"gpt_6_luna_chat", "microsoft_decision_1"}
    assert all(r["count"] == 8 // n for r in selected)


def test_original_values_equal_weights_scaling_singletons_and_missing_metrics() -> None:
    a, b = row(), row(input_="title_and_desc", value=160)
    b.update(n=2000, latency_s_total=320, status=FINISHED_WITH_FAILURES)
    original = views([a, b])
    grouped = original["input"]["rows"][0]
    assert grouped["cost_per_advert"] * 1000 == pytest.approx(70)
    assert grouped["cost_sd"] * 1000 == pytest.approx(10)
    assert grouped["latency_per_advert"] * 1000 == pytest.approx(140)
    assert grouped["latency_sd"] * 1000 == pytest.approx(20)
    assert original["prompt"]["rows"][0]["cost_sd"] == 0
    b["cost"] = None
    grouped = views([a, b])["input"]["rows"][0]
    assert grouped["cost_per_advert"] is None and grouped["cost_sd"] is None
    assert "1/2" in grouped["cost_availability"]
    assert grouped["latency_per_advert"] == pytest.approx(0.14)


def test_eligibility_local_zero_and_controls_follow_only_eligible_conditions() -> None:
    a, b, c = row(), row(input_="title_and_desc"), row(prompt="few-shot")
    a.update(local_api_free=True, cost=None)
    b["status"] = "incomplete"
    c["resource_eligible"] = False
    selected = _resource_rows([a, b, c])
    assert len(selected) == 1 and selected[0]["cost_per_advert"] == 0
    assert views([a, b, c])["input,prompt"]["rows"][0]["cost_sd"] == 0
    eligible = {r["series"] for r in selected}
    assert (
        _grouping_controls([r for r in [a, b, c] if r["series"] in eligible], resource=True) == ""
    )
    b["status"] = FINISHED_WITH_FAILURES
    controls = _grouping_controls([a, b], resource=True)
    assert 'value="input"' in controls and 'value="prompt"' not in controls
    assert "resource-grouping-status" in controls and "resource-grouping" in controls


def test_bar_spec_labels_whiskers_unknown_cells_and_unmodified_default() -> None:
    a, b = row(), row(prompt="few-shot", value=80)
    a["latency_s_total"], b["latency_s_total"] = 120.25, 121.75
    selected = views([a, b])
    result = node(
        "const api=require(process.argv[1]);const v=" + json.dumps(selected) + ";"
        "process.stdout.write(JSON.stringify(Object.fromEntries(Object.entries(v).map(([key,view])=>[key,api.resourcePlotSpec(view)]))));",
        "site/report.js",
    )
    grouped = result["prompt"]
    assert grouped["layout"]["yaxis"]["ticktext"] == ["GPT-6 Luna Chat · Title"]
    assert result["input,prompt"]["layout"]["yaxis"]["ticktext"] == ["GPT-6 Luna Chat"]
    assert result["input"]["layout"]["yaxis"]["ticktext"] == [
        "GPT-6 Luna Chat · Zero-shot",
        "GPT-6 Luna Chat · Few-shot",
    ]
    assert grouped["traces"][0]["x"] == pytest.approx([70])
    assert grouped["traces"][0]["error_x"]["array"] == pytest.approx([10])
    assert grouped["traces"][0]["error_x"]["symmetric"] is True
    assert "70.00000 (±10.00000)" in [a["text"] for a in grouped["layout"]["annotations"]]
    assert "121 (±1)" in [a["text"] for a in grouped["layout"]["annotations"]]
    assert grouped["traces"][1]["error_x"]["array"] == pytest.approx([0.75])
    assert result["none"]["traces"][1]["x"] == pytest.approx([120.25, 121.75])
    assert "%{x:.0f}" in grouped["traces"][1]["hovertemplate"]
    assert "configs" in grouped["traces"][0]["hovertemplate"]
    assert "error_x" not in result["none"]["traces"][0]
    assert result["none"]["layout"]["yaxis"]["ticktext"] == [
        "GPT-6 Luna Chat · Zero-shot · Title",
        "GPT-6 Luna Chat · Few-shot · Title",
    ]
    a["cost"] = None
    spec = node(
        "const api=require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(api.resourcePlotSpec("
        + json.dumps(views([a, b])["prompt"])
        + ")));",
        "site/report.js",
    )
    assert spec["traces"][0]["x"] == [None]
    assert "N/A" in [a["text"] for a in spec["layout"]["annotations"]]


def test_symmetric_sd_can_extend_below_zero_without_clipping() -> None:
    rows = [
        row(input_=input_, prompt=prompt, value=value)
        for input_, prompt, value in [
            ("title", "zero-shot", 0),
            ("title", "few-shot", 0),
            ("title_and_desc", "zero-shot", 0),
            ("title_and_desc", "few-shot", 4),
        ]
    ]
    result = node(
        "const api=require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(api.resourcePlotSpec("
        + json.dumps(views(rows)["input,prompt"])
        + ")));",
        "site/report.js",
    )
    assert result["traces"][0]["error_x"]["array"] == pytest.approx([3**0.5])
    assert result["layout"]["xaxis"]["range"][0] < 1 - 3**0.5
    assert result["layout"]["xaxis"]["range"][1] > 1 + 3**0.5


def test_controller_independent_views_restoration_and_fresh_grouped_exports() -> None:
    rows = [
        row(model, input_, prompt, value)
        for model in ("gpt_6_luna_chat", "d3_27b_q4_k_m")
        for input_ in ("title", "title_and_desc")
        for prompt, value in (("zero-shot", 60), ("few-shot", 80))
    ]
    result = node(
        r"""
      const api=require(process.argv[1]),views=VIEWS;
      function panel(id){
        const controls=['input','prompt'].map(value=>({value,checked:false,
          addEventListener(name,fn){this.change=fn}}));
        const nodes={'.resource-chart':{id,style:{},clientWidth:1000},
          '.resource-grouping-status':{textContent:''},
          '[id^="resource-title-"]':{textContent:'Resources per 1,000 adverts'},
          '.resource-panel figcaption':{textContent:'Exact invented caption.'},
          '.resource-png':{},'.resource-svg':{}};
        return {controls,nodes,querySelector:s=>nodes[s],querySelectorAll:()=>controls,
          getAttribute:()=>id};
      }
      const a=panel('a'),b=panel('b'),plots={},exports=[],downloads=[],guides=[];
      const clone=value=>JSON.parse(JSON.stringify(value));
      const Plotly={react:async(chart,traces,layout)=>{
          plots[chart.id]={traces:clone(traces),layout:clone(layout)};
          layout.height=-100;layout.xaxis.range=[999,1000];},
        newPlot:async(chart,traces,layout)=>exports.push({traces:clone(traces),layout:clone(layout)}),
        downloadImage:async(chart,options)=>downloads.push(options),purge(){}};
      const document={createElement:()=>({style:{},remove(){}}),body:{append(){}},
        getElementById:()=>({textContent:'Invented dataset'})};
      const guide=(layout,title,note,dataset)=>{
        guides.push({note:note.textContent,dataset});layout.height+=50;return 50};
      api.initializeResources(a,views,Plotly,guide,'a',document);
      api.initializeResources(b,views,Plotly,guide,'b',document);
      const original=clone(plots.a),states=[];
      for(const flags of [[true,false],[true,true],[false,true],[false,false]]){
        a.controls.forEach((c,i)=>c.checked=flags[i]);a.controls[0].change();
        states.push(plots.a.traces[0].x.length);
      }
      const restored=JSON.stringify(original)===JSON.stringify(plots.a);
      a.controls[1].checked=true;a.controls[1].change();
      b.controls[0].checked=true;b.controls[0].change();
      (async()=>{
        await a.nodes['.resource-png'].onclick();await a.nodes['.resource-svg'].onclick();
        process.stdout.write(JSON.stringify({states,restored,plots,exports,downloads,guides,
          a:a.controls.map(c=>c.checked),b:b.controls.map(c=>c.checked)}));
      })();
    """.replace("VIEWS", json.dumps(views(rows))),
        "site/report.js",
    )
    assert result["states"] == [4, 2, 4, 8] and result["restored"]
    assert result["a"] == [False, True] and result["b"] == [True, False]
    assert [e["format"] for e in result["downloads"]] == ["png", "svg"]
    for exported in result["exports"]:
        assert exported["traces"] == result["plots"]["a"]["traces"]
        assert exported["layout"]["xaxis"] == result["plots"]["a"]["layout"]["xaxis"]
        assert exported["layout"]["yaxis"] == result["plots"]["a"]["layout"]["yaxis"]
        assert exported["layout"]["height"] == result["plots"]["a"]["layout"]["height"] + 50
    assert all("Grouped across Prompt" in guide["dataset"] for guide in result["guides"])
    assert all(guide["note"] == "Exact invented caption." for guide in result["guides"])


def test_empty_resource_view_needs_no_controls_or_exports() -> None:
    result = node(
        r"""
      const api=require(process.argv[1]),chart={},exports={};
      const panel={querySelector:s=>s==='.resource-chart'?chart:exports};
      api.initializeResources(panel,{none:{fields:[],rows:[]}},null,null,'empty');
      process.stdout.write(JSON.stringify({chart,exports}));
    """,
        "site/report.js",
    )
    assert result["exports"]["hidden"]
    assert "No completed pinned arms" in result["chart"]["textContent"]


def test_grouped_resource_inputs_deduplicate_storage_aliases() -> None:
    rows = [row(input_="title_and_desc"), row(input_="combined_text", value=80)]
    original = copy.deepcopy(rows)
    spec = node(
        "const api=require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(api.resourcePlotSpec("
        + json.dumps(views(rows)["input"])
        + ")));",
        "site/report.js",
    )
    assert spec["traces"][0]["customdata"][0][3] == "Input: title + desc<br>Prompt: Zero-shot"
    assert spec["traces"][0]["x"] == [70]
    assert spec["traces"][0]["error_x"]["array"] == pytest.approx([10])
    assert rows == original
