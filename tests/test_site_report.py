"""Verify report-only structure and selectors with invented experiments."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from site_builder.build import initialize
from site_builder.content import report_content
from site_builder.evidence import DatasetEvidence

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "site" / "report.html").read_text()


def test_report_uses_private_prose_and_independent_tabs(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    (directory / "introduction.md").write_text("Keep *this & that* exactly.")
    (directory / "results/shark-products.md").write_text("Invented dataset commentary.")
    panel = DatasetEvidence(
        index=2,
        key="sharks-invented",
        task="sharks",
        n=4,
        positives=1,
        sha256="a" * 64,
        prompts=[],
        roc='<h3 id="roc-title-2">ROC</h3>',
        table="<table><tr><td>0.625</td></tr></table>",
        resources="",
        provenance="invented-hash",
    )
    page = report_content(directory, [panel], TEMPLATE)
    assert "Keep <em>this &amp; that</em> exactly." in page
    assert "<table><tr><td>0.625</td></tr></table>" in page
    assert page.count('role="tablist"') == 2
    assert 'aria-label="Models"' in page
    assert 'aria-label="Datasets"' in page
    assert 'id="dataset-data-2"' in page and 'id="models-panel-0"' in page
    assert '<h2 id="introduction">Introduction</h2>' in page
    assert '<h2 id="results">Results</h2>' in page
    assert '<section id="dataset-results-2" aria-label="Results for Shark Products">' in page
    assert "Invented dataset commentary." in page
    assert "Relevant means matching the monitoring policy" not in page
    assert '<div class="model-card" id="model-d3">' in page
    assert "Model summary placeholder: d3." in page
    assert 'id="discussion"' in page
    assert "<strong>Source:</strong> Source details to be added." in page
    assert (
        page.index("<strong>Labels/policy:</strong>")
        < page.index('class="table-wrap class-distribution"')
        < page.index("<strong>Caveats:</strong>")
    )
    assert "{{ class_distribution }}" not in page
    assert "dataset-statistics" not in page
    assert "dataset-heading" not in page
    assert 'id="dataset-sharks-invented" class="dataset-anchor"' in page
    assert 'aria-labelledby="results-tab-2"' in page
    with pytest.raises(ValueError, match="Expected one"):
        report_content(directory, [], TEMPLATE.replace("{{ dataset_workspace }}", ""))


def test_template_without_selected_results_still_has_methods_tabs(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    page = report_content(directory, [], TEMPLATE)
    assert "No saved dataset results" in page
    assert page.count('role="tablist"') == 1


@pytest.mark.parametrize(
    "task,slug", [("sharks", "shark-products"), ("animals", "animal-products"), ("birds", "birds")]
)
def test_dataset_evidence_renders_without_archived_commentary(
    tmp_path: Path, task: str, slug: str
) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    (directory / f"results/{slug}/commentary.md").write_text(
        "#### Screening performance\n\n**Invented ROC point:** *Keep this emphasis.*\n\n"
        "**Invented table point:** Keep this wording.\n\n"
        "#### Resource usage\n\nInvented resource commentary.\n\n"
        "#### Preparation\n\nPreparation commentary to be added."
    )
    panel = DatasetEvidence(
        index=0,
        key=f"{task}-invented",
        task=task,
        n=4,
        positives=1,
        sha256="a" * 64,
        prompts=[],
        roc="<figure>Invented ROC display<figcaption>ROC caption</figcaption></figure>",
        table="<table><tr><td>Invented table display</td></tr></table><p>Table caption</p>",
        resources="<figure>Invented resource display</figure>",
        provenance="invented-provenance",
    )
    archive = directory / f"results/{slug}/commentary.md"
    original = archive.read_bytes()
    (directory / "overall-results.md").write_text("**Invented overall discussion.**")
    page = report_content(directory, [panel], TEMPLATE)
    assert archive.read_bytes() == original
    archive.unlink()
    assert report_content(directory, [panel], TEMPLATE) == page
    assert "<strong>Invented overall discussion.</strong>" in page
    assert "Invented ROC point" not in page and "Invented resource commentary" not in page
    assert 'id="commentary-' not in page
    fragments = [
        panel.roc,
        panel.table,
        panel.resources,
        panel.provenance,
    ]
    positions = [page.index(fragment) for fragment in fragments]
    assert positions == sorted(positions)
    assert 'class="figure-commentary"' not in page


def _javascript(expression: str) -> Any:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for offline website-controller verification")
    script = (
        f"const api = require({json.dumps(str(ROOT / 'site' / 'report.js'))});\n"
        + "console.log(JSON.stringify((() => {"
        + expression
        + "})()));"
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_selected_filters_show_only_existing_combinations_and_independent_baselines() -> None:
    result = _javascript("""
      const rows=[];
      for (const model of ['a','b']) for (const prompt of ['zero-shot','few-shot'])
        for (const input of ['title','title_and_desc']) rows.push({
          series:model+'-'+prompt+'-'+input,model,prompt,input,kind:'measured'});
      rows.push({series:'baseline',kind:'published'});
      const selection={model:['a','b'],prompt:['zero-shot','few-shot'],
        input:['title','title_and_desc'],baselines:['baseline']};
      const all=api.selectRows(rows,selection).map(row=>row.series);
      selection.prompt=['zero-shot'];
      const zero=api.selectRows(rows,selection).map(row=>row.series);
      const sparse=api.selectRows(rows.filter(row=>row.series!=='b-zero-shot-title'),selection)
        .map(row=>row.series);
      selection.model=[];
      const empty=api.selectRows(rows,selection).map(row=>row.series);
      selection.baselines=[];
      return {all,zero,sparse,empty,none:api.selectRows(rows,selection)};
    """)
    assert len(result["all"]) == 9
    assert result["zero"] == [
        "a-zero-shot-title",
        "a-zero-shot-title_and_desc",
        "b-zero-shot-title",
        "b-zero-shot-title_and_desc",
        "baseline",
    ]
    assert result["sparse"] == [
        "a-zero-shot-title",
        "a-zero-shot-title_and_desc",
        "b-zero-shot-title_and_desc",
        "baseline",
    ]
    assert result["empty"] == ["baseline"]
    assert result["none"] == []


def test_dataset_defaults_available_options_and_friendly_labels() -> None:
    result = _javascript("""
      const rows=['title','title_and_desc','combined_text'].map(input=>({
        series:input,model:'d3_27b_q4_k_m',kind:'measured',prompt:'zero-shot',input,
        status:'complete',metrics:{positive_recall:0.5,false_positive_rate:0.2}}));
      rows.push({series:'published',kind:'published',model:'LTS-text',point:[0.1,0.5]});
      rows.push({series:'f1-only',kind:'published',model:'GPT-4 zero-shot'});
      const available=api.plottableRows(rows);
      return {
        sharks:api.defaultSelection('sharks-fixture',rows),
        animals:api.defaultSelection('animals-fixture',rows),
        birds:api.defaultSelection('birds-fixture',rows),
        missing:api.defaultSelection('sharks-fixture',rows.filter(row=>row.input!=='title_and_desc')),
        inputs:api.availableValues(available,'input'),
        label:api.seriesLabel(rows[1]), combinedLabel:api.seriesLabel(rows[2]),
        selected:api.selectRows(rows,{model:['d3_27b_q4_k_m'],prompt:['zero-shot'],
          input:['title_and_desc'],baselines:[]}).map(row=>row.input),
        selectedCombined:api.selectRows(rows,{model:['d3_27b_q4_k_m'],prompt:['zero-shot'],
          input:['combined_text'],baselines:[]}).map(row=>row.input),
        published:api.seriesLabel(rows[3]),
        plottable:available.map(row=>row.series)
      };
    """)
    for task, value in (
        ("sharks", "title_and_desc"),
        ("animals", "title"),
        ("birds", "combined_text"),
    ):
        assert result[task] == {
            "model": ["d3_27b_q4_k_m"],
            "prompt": ["zero-shot"],
            "input": [value],
            "baselines": ["published"],
        }
    assert result["missing"] == {"model": [], "prompt": [], "input": [], "baselines": ["published"]}
    assert result["inputs"] == ["title", "title_and_desc"]
    assert result["label"] == "d3 · Zero-shot · title + desc"
    assert result["combinedLabel"] == "d3 · Zero-shot · title + desc"
    assert result["selected"] == result["selectedCombined"] == ["title_and_desc", "combined_text"]
    assert result["published"] == "LTS · Text"
    assert "f1-only" not in result["plottable"]


def test_chart_export_notes_wrap_at_whitespace() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for offline chart-controller verification")
    script = r"""
      const fs=require('fs'), vm=require('vm');
      const source=fs.readFileSync(process.argv[1],'utf8').split('function draw(')[0];
      const document={getElementById:()=>({textContent:'{}'}),
        createElement:()=>({getContext:()=>({measureText:text=>({width:text.length*6})})})};
      const context=vm.createContext({document,WildlifeReport:require("./site/report.js")});
      vm.runInContext(source,context);
      const text='An invented figure guide with several words and a second sentence.';
      const result=vm.runInContext('wrapExportText('+JSON.stringify(text)+',120,12)',context);
      const titles=['Shark Products','Animal Products','Live Birds'].map(name=>{
        const layout={height:400,margin:{t:20,b:40,l:50,r:20}};
        context.layout=layout;context.name=name;
        context.caption={childNodes:[{nodeType:3,textContent:'Exact caption'}]};
        vm.runInContext(
          "addExportGuide(layout,'Invented figure',caption,name,1200)",context);
        return layout.title.text;
      });
      console.log(JSON.stringify({result,text,titles}));
    """
    result = subprocess.run(
        [node, "-e", script, str(ROOT / "site/charts.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert len(value["result"]) > 1
    assert " ".join(value["result"]) == value["text"]
    assert all(len(line) <= 20 for line in value["result"])
    for name, title in zip(
        ("Shark Products", "Animal Products", "Live Birds"), value["titles"], strict=True
    ):
        assert "Invented figure" in title and name in title


def test_workspace_orders_datasets_without_reassigning_evidence(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    panels = [
        DatasetEvidence(
            index=index,
            key=f"{task}-invented",
            task=task,
            n=10,
            positives=3,
            sha256=str(index) * 64,
            prompts=[],
            roc=f'<h3 id="roc-title-{index}">ROC for {task}</h3>',
            table=f"<table><caption>Table for {task}</caption></table>",
            resources=f'<h3 id="resource-title-{index}">Resources for {task}</h3>',
            provenance=f"Provenance for {task}",
        )
        for index, task in enumerate(["animals", "birds", "sharks"])
    ]
    page = report_content(directory, panels, TEMPLATE)
    assert page.index('data-panel="panel-2"') < page.index('data-panel="panel-0"')
    assert page.index('data-panel="panel-0"') < page.index('data-panel="panel-1"')
    from html.parser import HTMLParser

    class Panels(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.datasets: dict[str, str] = {}

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            values = dict(attrs)
            if values.get("class") == "dataset-panel":
                self.datasets[str(values["id"])] = str(values["data-dataset-key"])

    parser = Panels()
    parser.feed(page)
    assert parser.datasets == {
        "panel-2": "sharks-invented",
        "panel-0": "animals-invented",
        "panel-1": "birds-invented",
    }
    assert page.count("<td>7</td><td>3</td><td>10</td>") == 3
    assert "dataset-heading" not in page
    for task in ("sharks", "animals", "birds"):
        assert f'id="dataset-{task}-invented" class="dataset-anchor"' in page
    assert (
        page.index("ROC for sharks")
        < page.index("Table for sharks")
        < page.index("Resources for sharks")
    )
    assert (
        page.index('id="introduction"')
        < page.index('id="experimental-setup"')
        < page.index('class="dataset-workspace"')
    )
    assert (
        page.index('class="dataset-workspace"')
        < page.index('id="overall-results"')
        < page.index('id="discussion"')
    )
    assert page.count('aria-label="Datasets"') == 1
    assert page.count('<h2 id="results">Results</h2>') == 1
    assert page.index('<h2 id="results">') < page.index('class="dataset-workspace"')
    assert '<h2 id="dataset-results-' not in page
    assert '<h4 id="resources-reproducibility">' in page
    assert '<h3 id="limitations">' in page
    assert 'id="published-comparison-caveats" aria-hidden="true"' in page
    for index, name in ((2, "Shark Products"), (0, "Animal Products"), (1, "Live Birds")):
        intro = f"The following sections outline the details of the {name} dataset"
        assert page.index(intro) < page.index(f'id="dataset-data-{index}"')
        assert f'<h3 id="dataset-data-{index}">Dataset details</h3>' in page
        assert f'<h3 id="dataset-prompts-{index}">Prompt viewer</h3>' in page
        assert f'<section id="dataset-results-{index}" aria-label="Results for {name}">' in page
    assert "{{ dataset_name }}" not in page


def test_dataset_introduction_is_editable_and_init_preserves_it(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    target = directory / "results/dataset-introduction.md"
    source = "Invented **{{ dataset_name }}** summary with `notes` and *commentary*."
    target.write_text(source)
    initialize(tmp_path)
    assert target.read_text() == source
    from site_builder.content import dataset_content

    evidence = DatasetEvidence(0, "sharks-invented", "sharks", 10, 3, "a" * 64, [], "", "", "", "")
    page = dataset_content(directory, evidence, {}, active=True, dataset_name="Sharks & <invented>")
    assert "<strong>Sharks &amp; &lt;invented&gt;</strong>" in page
    assert "<code>notes</code> and <em>commentary</em>" in page
    assert page.index("Invented <strong>") < page.index('<h3 id="dataset-data-0">')


def test_prompt_choices_handle_sparse_conditions_and_versions() -> None:
    result = _javascript("""
      const records=[
        {family:'decision',condition:'zero-shot',sha256:'a',version:'v0'},
        {family:'decision',condition:'zero-shot',sha256:'b',version:'v1'},
        {family:'llm',condition:'zero-shot',sha256:'c',version:'v0'},
        {family:'llm',condition:'few-shot',sha256:'d',version:'v0'}
      ];
      return {initial:api.promptSelection(records),
        version:api.promptSelection(records,'decision','zero-shot','b'),
        missing:api.promptSelection(records,'decision','few-shot'),
        chat:api.promptSelection(records,'llm','few-shot'),empty:api.promptSelection([])};
    """)
    assert result["initial"]["selected"]["sha256"] == "a"
    assert len(result["initial"]["candidates"]) == 2
    assert result["version"]["selected"]["sha256"] == "b"
    assert result["missing"]["condition"] == "zero-shot"
    assert result["missing"]["conditions"] == ["zero-shot"]
    assert result["chat"]["selected"]["sha256"] == "d"
    assert result["empty"]["selected"] is None


def test_dataset_placeholders_resolve_only_authored_sections(tmp_path: Path) -> None:
    from site_builder.content import dataset_content, dataset_text

    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    (directory / "results/shark-products.md").write_text("**{{ dataset_name }}** _kept_.")
    (directory / "methods/data/shark-products.md").write_text(
        "- **Source:** {{ dataset_name }}\n\n{{ class_distribution }}\n\n- **Caveats:** unchanged."
    )
    panel = DatasetEvidence(
        index=2,
        key="invented",
        task="sharks",
        n=10,
        positives=3,
        sha256="a" * 64,
        prompts=[],
        roc="<h3>{{ dataset_name }}</h3><p hidden>{{ dataset_name }} export</p>"
        '<a href="#table-1">Table 1</a>',
        table="{{ dataset_name }} table",
        resources="{{ dataset_name }} resources",
        provenance="Technical {{ dataset_name }} remains literal",
    )
    for name in ("Sharks", "Animals", "Birds"):
        rendered = dataset_content(directory, panel, {}, active=True, dataset_name=name)
        assert f"<strong>{name}</strong> <em>kept</em>." in rendered
        assert f"<strong>Source:</strong> {name}" in rendered
        assert f"<h3>{name}</h3><p hidden>{name} export</p>" in rendered
        assert f"{name} table" in rendered and f"{name} resources" in rendered
        assert 'href="#table-title-2"' in rendered
        assert "Technical {{ dataset_name }} remains literal" in rendered
    assert dataset_text("{{ dataset_name }}", 'A & <B> "C"', 0) == "A &amp; &lt;B&gt; &quot;C&quot;"


def test_overall_results_heading_contains_h3_subsections(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    (directory / "overall-results.md").write_text(
        "### Screening Performance\n\nKeep *performance*.\n\n"
        "### Resource Usage\n\nKeep [resources](#figure-2)."
    )
    page = report_content(directory, [], TEMPLATE)
    assert '<h2 id="overall-results">Results</h2>' in page
    assert "Results Summary</h2>" not in page
    assert "<h3>Screening Performance</h3>" in page
    assert "<h3>Resource Usage</h3>" in page
    assert "Keep <em>performance</em>." in page
    assert page.index('id="overall-results"') < page.index("<h3>Screening Performance</h3>")
    assert page.index("<h3>Resource Usage</h3>") < page.index('id="discussion"')


def test_problem_formulation_initializes_renders_math_and_preserves_edits(tmp_path: Path) -> None:
    """Keep the editable formulation between prompts and evaluation with native math."""
    import re
    from xml.etree import ElementTree

    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    draft = directory / "methods/problem-formulation.md"
    source = draft.read_text()
    page = report_content(directory, [], TEMPLATE)
    start = page.index('<h3 id="problem-formulation">Problem formulation</h3>')
    end = page.index('<h3 id="evaluation">')
    assert page.index('<h3 id="prompts">') < start < end
    section = page[start:end]
    equations = re.findall(r"<math\b.*?</math>", section, re.DOTALL)
    assert len(equations) == 10
    assert section.count('display="block"') == 2
    for equation in equations:
        root = ElementTree.fromstring(equation)
        assert root.tag == "{http://www.w3.org/1998/Math/MathML}math"
    assert "<mo>≥</mo><mn>0.5</mn>" in section
    assert "Recorded inference failures are sent for review rather than discarded." in section
    assert "parameters, which remain unchanged" in section
    assert "{{ problem_formulation }}" not in page
    draft.write_text(source + "\nKeep this authored edit.\n")
    initialize(tmp_path)
    assert draft.read_text() == source + "\nKeep this authored edit.\n"
    assert "Keep this authored edit." in report_content(directory, [], TEMPLATE)
