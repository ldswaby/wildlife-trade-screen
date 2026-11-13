"""Check resource presentation and fragment navigation with invented evidence offline."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from site_builder.content import (
    DATASET_LINK_TARGETS,
    contextual_links,
    dataset_text,
    render_markdown,
)
from site_builder.evidence import DatasetEvidence

ROOT = Path(__file__).resolve().parents[1]


def node(script: str, path: str) -> Any:
    """Run a small offline browser-contract harness against the actual JavaScript."""
    binary = shutil.which("node")
    if not binary:
        pytest.skip("Node.js unavailable")
    result = subprocess.run(
        [binary, "-e", script, str(ROOT / path)], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("alias,target", DATASET_LINK_TARGETS.items())
def test_markdown_aliases_are_dataset_local_or_follow_active_panel(alias: str, target: str) -> None:
    source = render_markdown(f"[Go *here*](#{alias}) and [Explicit](#table-title-7).")
    for index in (0, 1, 2):
        local = dataset_text(source, "Invented", index)
        assert f'href="#{target}-{index}"' in local
        assert 'href="#table-title-7"' in local and "<em>here</em>" in local
        evidence = DatasetEvidence(index, "invented", "sharks", 2, 1, "a" * 64, [], "", "", "", "")
        shared = contextual_links(source, [evidence])
        assert f'href="#{target}-{index}" data-active-dataset-target="{target}"' in shared
    assert "No dataset results are selected." in contextual_links(source, [])
    assert dataset_text('<a href="#evaluation">Policy</a>', "Invented", 2) == (
        '<a href="#evaluation">Policy</a>'
    )


@pytest.mark.parametrize("dataset_name", ["Shark Products", "Animal Products", "Live Birds"])
def test_resource_bars_annotations_and_both_exports_share_per_thousand_values(
    dataset_name: str,
) -> None:
    result = node(
        r"""
      const api=require(process.argv[1]);
      const resources=[
        {series:'known',cost_per_advert:0.00001234,latency_per_advert:0.25},
        {series:'zero',cost_per_advert:0,latency_per_advert:0},
        {series:'unknown',cost_per_advert:null,latency_per_advert:null},
        {series:'local',cost_per_advert:0,latency_per_advert:2,local_api_free:true}];
      const views={none:{fields:[],rows:resources.map(row=>({...row,model:row.series,
        input:'title',prompt:'zero-shot',inputs:['title'],prompts:['zero-shot'],count:1,
        cost_sd:row.cost_per_advert===null?null:0,latency_sd:row.latency_per_advert===null?null:0}))}};
      const buttons={},exports=[],guides=[];let shown;
      const document={getElementById:()=>({textContent:'DATASET_NAME'}),
        createElement:()=>({style:{},remove(){}}),body:{append(){}}};
      const panel={querySelector:selector=>buttons[selector]??=(selector==='.resource-chart'
        ?{style:{},clientWidth:1000}:{textContent:selector.includes('title')
          ?'Resources per 1,000 adverts':
          'Summed request time per 1,000 adverts'}),
        querySelectorAll:()=>[],getAttribute:()=> 'tab'};
      const Plotly={react:async(el,traces,layout)=>{shown={traces,layout}},
        newPlot:async()=>{},downloadImage:async(el,options)=>exports.push({options,shown}),purge(){}};
      const guide=(layout,title,note,dataset)=>{
        guides.push({title,note:note.textContent,dataset});return 0};
      api.initializeResources(panel,views,Plotly,guide,'invented',document);
      (async()=>{
        await buttons['.resource-png'].onclick(); await buttons['.resource-svg'].onclick();
        process.stdout.write(JSON.stringify({shown,exports,guides,resources}));
      })();
    """.replace("DATASET_NAME", dataset_name),
        "site/report.js",
    )
    traces = result["shown"]["traces"]
    assert traces[0]["x"] == pytest.approx([0.01234, 0, None, 0])
    assert traces[1]["x"] == [250, 0, None, 2000]
    labels = [a["text"].replace("<br>", " ") for a in result["shown"]["layout"]["annotations"]]
    assert labels[:2] == [
        "API charge / 1,000 adverts (USD)",
        "Summed request time / 1,000 adverts (s)",
    ]
    assert labels[2:4] == ["0.01234", "250"]
    assert labels.count("N/A") == 2 and "0 external API" in labels
    assert all("1,000 adverts" in t["hovertemplate"] for t in traces)
    assert [e["options"]["format"] for e in result["exports"]] == ["png", "svg"]
    assert all(e["shown"]["traces"] == traces for e in result["exports"])
    assert all(
        "1,000 adverts" in g["title"] and "1,000 adverts" in g["note"] for g in result["guides"]
    )
    assert all(g["dataset"] == dataset_name for g in result["guides"])
    assert result["resources"][0]["cost_per_advert"] == 0.00001234


@pytest.mark.parametrize("panel_id", ["panel-2", "models-panel-3"])
def test_fragment_navigation_reveals_hidden_panel_and_all_disclosures_on_repeat(
    panel_id: str,
) -> None:
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm');
      const events={},clicks={}; let scrolls=0,activations=0;
      const panel={id:'PANEL_ID',hidden:true,parentElement:null,tagName:'SECTION',
        getAttribute:name=>name==='role'?'tabpanel':'tab-button'};
      const outer={open:false,parentElement:panel,tagName:'DETAILS'};
      const inner={open:false,parentElement:outer,tagName:'DETAILS'};
      const target={parentElement:inner,closest:()=>panel.hidden?panel:null,
        scrollIntoView:()=>scrolls++};
      const button={dataset:{panel:'panel-2'},click:()=>{panel.hidden=false;activations++}};
      const main={querySelector:()=>null,querySelectorAll:s=>s==='.tab'?[button]:[]};
      const index={replaceChildren(){}};
      const disclosure={open:false,addEventListener(){}};
      const document={baseURI:'https://example.invalid/report.html',
        querySelector:s=>s==='main'?main:s==='.page-index nav'?index:
          s==='.page-index details'?disclosure:{getBoundingClientRect:()=>({bottom:80})},
        querySelectorAll:()=>[],getElementById:id=>id==='nested'?target:id==='tab-button'?button:null,
        addEventListener:(name,fn)=>clicks[name]=fn};
      const location={hash:'#nested',origin:'https://example.invalid',pathname:'/report.html',search:''};
      const context=vm.createContext({document,location,URL,
        window:{addEventListener:(name,fn)=>events[name]=fn},
        matchMedia:()=>({matches:false,addEventListener(){}}),
        MutationObserver:class {observe(){}},requestAnimationFrame:fn=>fn()});
      vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
      const initial={scrolls,activations,inner:inner.open,outer:outer.open};
      inner.open=false;outer.open=false;panel.hidden=true;
      let prevented=false;
      const link={href:'https://example.invalid/report.html#nested',target:'',hasAttribute:()=>false};
      clicks.click({button:0,target:{closest:()=>link},preventDefault(){prevented=true}});
      const repeated={scrolls,activations,inner:inner.open,outer:outer.open,prevented};
      location.hash='#%E0%A4%A';events.hashchange();
      location.hash='#missing';events.hashchange();
      process.stdout.write(JSON.stringify({initial,repeated,scrolls}));
    """.replace("PANEL_ID", panel_id),
        "site/site.js",
    )
    assert result["initial"] == {"scrolls": 1, "activations": 1, "inner": True, "outer": True}
    assert result["repeated"] == {
        "scrolls": 2,
        "activations": 2,
        "inner": True,
        "outer": True,
        "prevented": True,
    }
    assert result["scrolls"] == 2
