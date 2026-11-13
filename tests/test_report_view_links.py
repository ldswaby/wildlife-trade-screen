"""Verify narrative links select explicit views without changing evidence."""

import copy
import json
from typing import Any

import pytest
from site_builder.content import contextual_links, render_markdown
from site_builder.evidence import DatasetEvidence
from test_resource_links import node


def panel(task: str, index: int) -> DatasetEvidence:
    """Build an invented panel whose index is unrelated to its task."""
    return DatasetEvidence(index, task + "-invented", task, 10, 2, "a" * 64, [], "", "", "", "")


@pytest.mark.parametrize(
    "alias,target",
    [("figure-1", "roc-title"), ("figure-2", "resource-title"), ("table-1", "table-title")],
)
def test_explicit_view_links_resolve_task_not_active_panel(alias: str, target: str) -> None:
    source = render_markdown(f"[Keep *this*](#{alias}?dataset=animals&models=all)")
    for index in (0, 7):
        result = contextual_links(source, [panel("sharks", 2), panel("animals", index)])
        assert f'href="#{target}-{index}?dataset=animals&amp;models=all"' in result
        assert "Keep <em>this</em>" in result
        assert "data-active-dataset" not in result
    missing = contextual_links(source, [panel("sharks", 2)])
    assert f'href="#{alias}?dataset=animals&amp;models=all"' in missing
    ambiguous = contextual_links(source, [panel("animals", 0), panel("animals", 1)])
    assert ambiguous == missing


MODELS = ["gpt_6_luna_chat", "gpt_6_luna_decisions", "microsoft_decision_1", "d3_27b_q4_k_m"]


def rows(inputs: tuple[str, ...] = ("title", "title_and_desc")) -> list[dict[str, Any]]:
    """Supply complete invented configurations with a plottable binary point."""
    return [
        {"kind": "measured", "model": model, "prompt": prompt, "input": input_, "point": [0.1, 0.8]}
        for model in MODELS
        for prompt in ("zero-shot", "few-shot")
        for input_ in inputs
    ]


@pytest.mark.parametrize(
    "options,inputs,models,prompts,selected_inputs",
    [
        (
            {"models": "all", "prompts": "zero-shot", "inputs": "title"},
            ("title",),
            MODELS,
            ["zero-shot"],
            ["title"],
        ),
        (
            {"models": "all", "prompts": "zero-shot", "inputs": "title_and_desc"},
            ("combined_text",),
            MODELS,
            ["zero-shot"],
            ["combined_text"],
        ),
        (
            {"models": "decision", "prompts": "zero-shot", "inputs": "title,title_and_desc"},
            ("title", "title_and_desc"),
            MODELS[1:],
            ["zero-shot"],
            ["title", "title_and_desc"],
        ),
        (
            {
                "models": "gpt_6_luna_chat,gpt_6_luna_decisions",
                "prompts": "zero-shot,few-shot",
                "inputs": "title",
            },
            ("title",),
            MODELS[:2],
            ["zero-shot", "few-shot"],
            ["title"],
        ),
    ],
)
def test_requested_narrative_roc_views_preserve_baselines_and_source(
    options: dict[str, str],
    inputs: tuple[str, ...],
    models: list[str],
    prompts: list[str],
    selected_inputs: list[str],
) -> None:
    available = rows(inputs)
    selection = {
        "model": [MODELS[-1]],
        "prompt": ["few-shot"],
        "input": [inputs[0]],
        "baselines": ["chosen-reference"],
    }
    original = copy.deepcopy([available, selection])
    result = node(
        "const api=require(process.argv[1]);const rows="
        + json.dumps(available)
        + ",selection="
        + json.dumps(selection)
        + ";process.stdout.write(JSON.stringify({view:api.linkedRocSelection(rows,selection,"
        + json.dumps(options)
        + "),rows,selection}));",
        "site/report.js",
    )
    assert result["view"]["selection"] == {
        "model": models,
        "prompt": prompts,
        "input": selected_inputs,
        "baselines": ["chosen-reference"],
    }
    assert [result["rows"], result["selection"]] == original


@pytest.mark.parametrize("missing", ["model", "prompt", "input", "combination"])
def test_unavailable_roc_request_is_explicit_and_atomic(missing: str) -> None:
    available = rows()
    options = {"models": "all", "prompts": "zero-shot", "inputs": "title,title_and_desc"}
    if missing == "combination":
        available.pop(0)
    else:
        field = {"model": "models", "prompt": "prompts", "input": "inputs"}[missing]
        options[field] = "unavailable"
    selection = {"model": [MODELS[-1]], "prompt": ["few-shot"], "input": ["title"], "baselines": []}
    result = node(
        "const api=require(process.argv[1]),selection="
        + json.dumps(selection)
        + ";process.stdout.write(JSON.stringify({view:api.linkedRocSelection("
        + json.dumps(available)
        + ",selection,"
        + json.dumps(options)
        + "),selection}));",
        "site/report.js",
    )
    assert "unavailable" in result["view"]["error"].lower()
    assert result["selection"] == selection


def test_linked_table_grouping_replaces_both_dimensions_and_rejects_unavailable() -> None:
    result = node(
        r"""
      const api=require(process.argv[1]);
      const controls=['input','prompt'].map(value=>({value,checked:true,addEventListener(){}}));
      const table={innerHTML:'',querySelector:()=>({outerHTML:'caption'})}, status={};
      const figure={dataset:{},querySelectorAll:()=>controls,
        querySelector:s=>s==='.table-grouping-data'?{textContent:JSON.stringify({none:'original',input:'inputs',prompt:'prompts','input,prompt':'both'})}:s==='table'?table:status};
      api.initializeTableGrouping({querySelectorAll:()=>[figure]});
      figure.applyTableView('input');
      const first={html:table.innerHTML,checked:controls.map(c=>c.checked)};
      figure.applyTableView('prompt');
      const second={html:table.innerHTML,checked:controls.map(c=>c.checked)};
      const error=figure.applyTableView('unavailable');
      process.stdout.write(JSON.stringify({first,second,error,final:controls.map(c=>c.checked)}));
    """,
        "site/report.js",
    )
    assert result["first"] == {"html": "captioninputs", "checked": [True, False]}
    assert result["second"] == {"html": "captionprompts", "checked": [False, True]}
    assert "unavailable" in result["error"]
    assert result["final"] == [False, True]


@pytest.mark.parametrize("fragment", ["figure-1", "roc-title-99", "figure-2", "resource-title-99"])
def test_view_navigation_direct_repeat_history_and_explicit_errors(fragment: str) -> None:
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm'),events={},clicks={},calls=[];
      let status=null,scrolls=0,activated=0;
      const location={hash:'#FRAGMENT?dataset=animals&VIEW_SETTINGS',
        origin:'https://example.invalid',pathname:'/report.html',search:''};
      const originalHash=location.hash;
      const panel={id:'panel-7',dataset:{datasetKey:'animals-invented'},
        hidden:true,parentElement:null,
        tagName:'SECTION',getAttribute:name=>name==='role'?'tabpanel':'tab-7',
        applyResourceView:async group=>{calls.push(group);return null},
        applyRocView:async options=>{calls.push(options);return null}};
      const figure={applyTableView:group=>{calls.push(group);return null}};
      const target={id:'roc-title-7',parentElement:panel,tagName:'H3',
        scrollIntoView:()=>scrolls++,closest:()=>figure};
      panel.querySelector=()=>target;
      const main={querySelector:()=>null,querySelectorAll:s=>s==='.dataset-panel'?[panel]:[],
        prepend:el=>{status=el}};
      const document={baseURI:'https://example.invalid/report.html',
        querySelector:s=>s==='main'?main:s==='.page-index nav'?{replaceChildren(){}}:
          s==='.page-index details'?{addEventListener(){}}:
          {getBoundingClientRect:()=>({bottom:80})},
        querySelectorAll:()=>[],getElementById:id=>id==='tab-7'?{click:()=>{panel.hidden=false;activated++}}:
          id==='report-view-message'?status:null,
        createElement:()=>({setAttribute(){},scrollIntoView(){},remove(){status=null}}),
        addEventListener:(name,fn)=>clicks[name]=fn};
      const context=vm.createContext({document,location,URL,URLSearchParams,
        window:{addEventListener:(name,fn)=>events[name]=fn},
        matchMedia:()=>({matches:false,addEventListener(){}}),
        MutationObserver:class {observe(){}},requestAnimationFrame:fn=>fn()});
      vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
      const tick=()=>new Promise(setImmediate);
      (async()=>{
        await tick();
        let prevented=false;
        const link={href:document.baseURI+location.hash,target:'',hasAttribute:()=>false};
        clicks.click({button:0,target:{closest:()=>link},preventDefault(){prevented=true}});
        await tick();
        for(const group of ['input','prompt']) {
          location.hash='#table-1?dataset=animals&group='+group;events.hashchange();await tick();
        }
        location.hash=originalHash;events.hashchange();await tick();
        const count=calls.length;
        location.hash='#figure-1?dataset=missing&models=all';events.hashchange();await tick();
        const missing=status.textContent;
        location.hash='#figure-1?dataset=animals&bogus=bad';events.hashchange();await tick();
        const unsupported=status.textContent;
        location.hash='#figure-1?dataset=animals&models=all&models=decision';
        events.hashchange();await tick();
        process.stdout.write(JSON.stringify({calls,count,prevented,scrolls,activated,missing,
          unsupported,repeated:status.textContent}));
      })();
    """.replace("FRAGMENT", fragment).replace(
            "VIEW_SETTINGS",
            "group=prompt"
            if fragment in {"figure-2", "resource-title-99"}
            else "models=all&prompts=zero-shot&inputs=title",
        ),
        "site/site.js",
    )
    view: str | dict[str, str] = (
        "prompt"
        if fragment in {"figure-2", "resource-title-99"}
        else {"models": "all", "prompts": "zero-shot", "inputs": "title"}
    )
    assert result["calls"] == [view, view, "input", "prompt", view]
    assert result["count"] == 5 and result["prevented"]
    assert result["activated"] == 1 and result["scrolls"] == 5
    assert "missing" in result["missing"]
    assert "unsupported" in result["unsupported"]
    assert "repeated" in result["repeated"]


@pytest.mark.parametrize("fields", [["input", "prompt"], ["prompt"], []])
def test_resource_presets_apply_available_grouping_atomically(fields: list[str]) -> None:
    result = node(
        r"""
      const api=require(process.argv[1]);
      const controls=FIELDS.map(value=>({value,checked:false,addEventListener(){}}));
      const views=Object.fromEntries(['none','input','prompt','input,prompt'].map(key=>
        [key,{fields:key==='none'?[]:key.split(','),rows:[{series:'invented',model:'m',
          input:'title',prompt:'zero-shot',cost_per_advert:0.1,latency_per_advert:1,
          cost_sd:0,latency_sd:0,inputs:['title'],prompts:['zero-shot'],count:1}]}]));
      const elements={},renders=[];
      const panel={querySelectorAll:()=>controls,querySelector:s=>elements[s]??={style:{}},
        applyRocView:'unchanged',tableState:'unchanged'};
      const Plotly={react:async(el,traces)=>renders.push(traces)};
      api.initializeResources(panel,views,Plotly,()=>{},'invented');
      (async()=>{
        const group=FIELDS.join(',')||'none';
        const success=await panel.applyResourceView(group);
        const selected=controls.map(c=>c.checked);
        const error=await panel.applyResourceView('unavailable');
        const afterError=controls.map(c=>c.checked);
        await panel.applyResourceView('none');
        process.stdout.write(JSON.stringify({success,selected,error,afterError,
          cleared:controls.map(c=>c.checked),renders:renders.length,
          roc:panel.applyRocView,table:panel.tableState}));
      })();
    """.replace("FIELDS", json.dumps(fields)),
        "site/report.js",
    )
    assert result["success"] is None
    assert result["selected"] == [True] * len(fields)
    assert "unavailable" in result["error"]
    assert result["afterError"] == result["selected"]
    assert result["cleared"] == [False] * len(fields)
    assert result["renders"] == 3
    assert result["roc"] == result["table"] == "unchanged"
