"""Check image caption layout with invented rich text and fixed font measurements."""

from test_resource_links import node


def test_export_caption_preserves_words_emphasis_paragraphs_and_plot_area() -> None:
    result = node(
        r"""
      const fs=require('fs'),vm=require('vm');
      const source=fs.readFileSync(process.argv[1],'utf8').split('function draw(')[0];
      const text=value=>({nodeType:3,textContent:value});
      const el=(tag,...children)=>({tagName:tag.toUpperCase(),childNodes:children});
      const caption=el('figcaption',
        el('p',text('First '),el('strong',text('bold & clear')),text(' with '),
          el('em',text('italic μ')),text(' and '),el('a',text('readable link')),text('.')),
        el('p',text('Second paragraph. '),el('code',text('an_unusually_long_unbroken_token'))));
      const document={getElementById:()=>({textContent:'{}'}),
        createElement:()=>({getContext:()=>({measureText:text=>({width:[...text].length*6})})})};
      const context=vm.createContext({document,WildlifeReport:require('./site/report.js'),caption});
      vm.runInContext(source,context);
      const lines=vm.runInContext('exportCaptionLines(caption,120,13)',context);
      const layout={height:600,margin:{l:70,r:30,t:90,b:100},
        annotations:[{text:'Existing plot label'}],legend:{y:-.2}};
      context.layout=layout;
      const extra=vm.runInContext(
        "addExportGuide(layout,'Exact figure title',caption,'Dataset identity',500)",context);
      process.stdout.write(JSON.stringify({lines,layout,extra}));
    """,
        "site/charts.js",
    )
    import html
    import re

    lines = result["lines"]
    plain = [html.unescape(re.sub(r"<[^>]+>", "", line)) for line in lines]
    assert "" in lines
    assert all(len(line) <= 20 for line in plain)
    assert any("<b>" in line for line in lines)
    assert any("<i>" in line for line in lines)
    assert "First bold & clear with italic μ and readable link." in " ".join(plain)
    assert "an_unusually_long_unbroken_token" in "".join(plain)
    layout = result["layout"]
    assert layout["title"]["text"] == (
        'Exact figure title<br><span style="font-size:12px">Dataset identity</span>'
    )
    assert layout["annotations"][0] == {"text": "Existing plot label"}
    caption = layout["annotations"][-1]
    assert caption["y"] < -100 / 410  # Below the original plot and legend margin.
    assert caption["yanchor"] == "top" and caption["align"] == "left"
    assert layout["height"] == 600 + result["extra"]
    assert layout["height"] - layout["margin"]["t"] - layout["margin"]["b"] == 410
    assert layout["legend"] == {"y": -0.2}
    assert layout["margin"]["autoexpand"] is False
