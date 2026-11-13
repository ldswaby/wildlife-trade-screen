"""Verify nested index state and navigation with an invented heading hierarchy."""

from test_resource_links import node


def test_nested_index_collapses_preserves_state_and_reveals_link_targets() -> None:
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm');
      const events={};let rebuild,roots=[],scroll=0;
      function element(tag) {
        return {tagName:tag.toUpperCase(),children:[],attributes:{},events:{},
          append(...nodes){this.children.push(...nodes)},
          prepend(child){this.children.unshift(child)},
          setAttribute(key,value){this.attributes[key]=value},
          removeAttribute(key){delete this.attributes[key]},
          addEventListener(key,fn){this.events[key]=fn}};
      }
      const panels=[0,1].map(i=>({id:'panel-'+i,hidden:i!==0,
        getAttribute:()=> 'tab-'+i,querySelector:()=>null}));
      const tabs=[{textContent:'Sharks'},{textContent:'Birds'}];
      function heading(id,level,top,panel=null) {
        return {id,tagName:'H'+level,textContent:id,parentElement:null,
          closest:()=>panel?.hidden?panel:null,contains(target){return target===this},
          matches:()=>false,getBoundingClientRect:()=>({top:top-scroll}),
          scrollIntoView(){}};
      }
      const headings=[heading('introduction',2,0),heading('setup',2,400),
        heading('evaluation',3,600),heading('detail',4,800),heading('results',2,1200),
        ...panels.flatMap((panel,i)=>[heading('dataset-'+i,3,1300,panel),
          heading('dataset-detail-'+i,4,1400,panel)]),
        heading('conclusion',2,1600)];
      const main={querySelector:()=>panels.find(p=>!p.hidden),
        querySelectorAll:s=>s==='h2, h3, h4'?headings:[]};
      const index={replaceChildren(...nodes){roots=nodes}};
      const document={querySelector:s=>s==='main'?main:s==='.page-index nav'?index:
          s==='.page-index details'?{addEventListener(){}}:
          {getBoundingClientRect:()=>({bottom:80})},
        querySelectorAll:s=>s==='[id]'?headings:[],createElement:element,
        getElementById:id=>id.startsWith('tab-')?tabs[Number(id.slice(4))]:headings.find(h=>h.id===id),
        addEventListener(){}};
      const location={hash:''};
      const context=vm.createContext({document,location,
        window:{addEventListener:(name,fn)=>events[name]=fn},
        matchMedia:()=>({matches:false,addEventListener(){}}),
        MutationObserver:class {constructor(fn){rebuild=fn}observe(){}},
        requestAnimationFrame:fn=>fn()});
      vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
      const all=nodes=>nodes.flatMap(n=>[n,...all(n.children)]);
      function group(id) {
        return all(roots).find(n=>n.className==='index-group'&&
          n.children[0].children.some(c=>c.href==='#'+id));
      }
      const toggle=id=>group(id).children[0].children.find(c=>c.tagName==='BUTTON');
      const children=id=>group(id).children.find(c=>c.className==='index-children');
      const initial={setup:children('setup').hidden,evaluation:children('evaluation').hidden,
        results:children('results').hidden,leafToggle:!!toggle('conclusion')};
      toggle('setup').events.click();
      const nestedStillOpen=!children('evaluation').hidden;
      toggle('evaluation').events.click();
      scroll=750;events.scroll();
      const current=all(roots).find(n=>n.attributes['aria-current']==='location').href;
      location.hash='#detail';events.hashchange();
      const revealed={setup:!children('setup').hidden,evaluation:!children('evaluation').hidden};
      toggle('results').events.click();toggle('dataset-0').events.click();
      panels[0].hidden=true;panels[1].hidden=false;rebuild();
      const switched={setup:!children('setup').hidden,evaluation:!children('evaluation').hidden,
        results:!children('results').hidden,newDataset:!!group('dataset-1'),oldDataset:!!group('dataset-0')};
      const newGroupExpanded=!children('dataset-1').hidden;
      panels[1].hidden=true;panels[0].hidden=false;rebuild();
      const restoredGroupCollapsed=children('dataset-0').hidden;
      const attributes=toggle('setup').attributes;
      process.stdout.write(JSON.stringify({initial,nestedStillOpen,current,revealed,switched,
        newGroupExpanded,restoredGroupCollapsed,attributes}));
    """,
        "site/site.js",
    )
    assert result["initial"] == {
        "setup": False,
        "evaluation": False,
        "results": False,
        "leafToggle": False,
    }
    assert result["nestedStillOpen"]
    assert result["newGroupExpanded"]
    assert result["restoredGroupCollapsed"]
    assert result["current"] == "#setup"
    assert result["revealed"] == {"setup": True, "evaluation": True}
    assert result["switched"] == {
        "setup": True,
        "evaluation": True,
        "results": False,
        "newDataset": True,
        "oldDataset": False,
    }
    assert result["attributes"]["aria-expanded"] == "true"
    assert result["attributes"]["aria-label"] == "Collapse subheadings under setup"
