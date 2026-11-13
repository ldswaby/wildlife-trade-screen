"""Check display-only dataset names and unprefixed navigation with invented content."""

import shutil
from pathlib import Path

from site_builder.build import initialize
from site_builder.content import report_content
from site_builder.evidence import DatasetEvidence
from test_resource_links import node

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_names_and_version_suffixes_preserve_evidence_keys(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    panels = [
        DatasetEvidence(
            i,
            key,
            task,
            4,
            1,
            "a" * 64,
            [],
            "<p>Invented caption for the {{ dataset_name }} dataset.</p>",
            "",
            "",
            "",
        )
        for i, (key, task) in enumerate(
            [("birds-one", "birds"), ("sharks-one", "sharks"), ("animals-one", "animals")]
        )
    ]
    page = report_content(directory, panels, (ROOT / "site/report.html").read_text())
    for name in ("Shark Products", "Animal Products", "Live Birds"):
        assert f">{name}</button>" in page
        assert f'aria-label="Results for {name}"' in page
        assert f"for the {name} dataset" in page
    assert page.index(">Shark Products</button>") < page.index(">Animal Products</button>")
    assert page.index(">Animal Products</button>") < page.index(">Live Birds</button>")
    for panel in panels:
        assert f'data-dataset-key="{panel.key}"' in page
        assert f'id="commentary-{panel.index}"' not in page
    panels.append(DatasetEvidence(3, "birds-two", "birds", 4, 1, "b" * 64, [], "", "", "", ""))
    page = report_content(directory, panels, (ROOT / "site/report.html").read_text())
    assert ">Live Birds · one</button>" in page and ">Live Birds · two</button>" in page


def test_task_heading_and_index_follow_dataset_switches_and_preserve_fallback() -> None:
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm');
      const names=['Shark Products','Animal Products','Live Birds'];
      const panels=names.map((name,i)=>({id:'panel-'+i,hidden:i!==0,
        getAttribute:()=> 'tab-'+i,querySelector:()=>null}));
      const tabs=names.map(textContent=>({textContent}));
      function heading(id,text,tag,panel=null) {
        return {id,textContent:text,tagName:tag,closest:()=>panel?.hidden?panel:null,
          matches:selector=>id.startsWith(selector.split('"')[1]),
          getBoundingClientRect:()=>({top:200})};
      }
      const results=heading('results','Results','H2');
      const labels=['Dataset Details','Prompt Viewer','ROC title','Table title','Resource title'];
      const prefixes=['dataset-data','dataset-prompts','roc-title','table-title','resource-title'];
      const bibliography=heading('bibliography','Bibliography','H2');
      const headings=[results,...panels.flatMap((p,i)=>labels.map((name,j)=>
        heading(prefixes[j]+'-'+i,name,j<6?'H3':'H4',p))),bibliography];
      let entries=[],changed;
      function element(tag) {
        return {tagName:tag.toUpperCase(),children:[],attributes:{},listeners:{},
          append(...nodes){this.children.push(...nodes)},
          prepend(node){this.children.unshift(node)},
          setAttribute(name,value){this.attributes[name]=value},
          removeAttribute(name){delete this.attributes[name]},
          addEventListener(name,fn){this.listeners[name]=fn}};
      }
      function links(nodes) {
        return nodes.flatMap(node=>node.tagName==='A'?[node]:links(node.children));
      }
      const index={replaceChildren:(...nodes)=>{entries=links(nodes)}};
      const main={querySelector:()=>panels.find(p=>!p.hidden),
        querySelectorAll:selector=>selector==='h2, h3, h4'?headings:[]};
      const document={querySelector:selector=>selector==='main'?main:
        selector==='.page-index nav'?index:selector==='.page-index details'?
          {addEventListener(){}}:{getBoundingClientRect:()=>({bottom:80})},
        querySelectorAll:selector=>selector==='[id]'?headings:[],
        getElementById:id=>id==='results'?results:id.startsWith('tab-')?tabs[Number(id.slice(4))]:null,
        createElement:element,
        addEventListener(){}};
      const context=vm.createContext({document,location:{hash:''},
        window:{addEventListener(){}},matchMedia:()=>({matches:false,addEventListener(){}}),
        MutationObserver:class {constructor(fn){changed=fn}observe(){}},
        requestAnimationFrame:fn=>fn()});
      vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
      const initial=results.textContent;
      const snapshots=[];
      for(let i=0;i<3;i++) {
        panels.forEach((p,j)=>p.hidden=i!==j);changed();
        snapshots.push(entries.map(e=>({text:e.textContent,href:e.href})));
      }
      tabs[2].textContent='Live Birds · version-two';changed();
      const version=results.textContent;
      panels.forEach(p=>p.hidden=true);changed();
      process.stdout.write(JSON.stringify({initial,snapshots,version,
        fallback:results.textContent,emptyIndex:entries.map(e=>e.textContent),
        tabs:tabs.map(t=>t.textContent)}));
    """,
        "site/site.js",
    )
    expected = [
        "Dataset Details",
        "Prompt Viewer",
        "Figure 1 · ROC",
        "Table 1 · Results",
        "Figure 2 · Resources",
    ]
    for i, name in enumerate(("Shark Products", "Animal Products", "Live Birds")):
        assert [entry["text"] for entry in result["snapshots"][i]] == [
            f"Task: {name}",
            *expected,
            "Bibliography",
        ]
        assert result["snapshots"][i][0]["href"] == "#results"
        assert result["snapshots"][i][-2]["href"] == f"#resource-title-{i}"
        assert result["snapshots"][i][-1]["href"] == "#bibliography"
    assert result["initial"] == "Task: Shark Products"
    assert result["version"] == "Task: Live Birds · version-two"
    assert result["fallback"] == "Results"
    assert result["emptyIndex"] == ["Results", "Bibliography"]
    assert result["tabs"] == ["Shark Products", "Animal Products", "Live Birds · version-two"]
