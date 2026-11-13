"""Check configuration summaries independently of prediction pooling and ROC state."""

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from site_builder.descriptions import load_descriptions
from site_builder.evidence import _table, _table_view
from site_builder.table_grouping import format_stats, group_rows
from test_resource_links import node

from wildlife_screening.metrics import FINISHED_WITH_FAILURES

ROOT = Path(__file__).resolve().parents[1]


def measured(
    model: str = "a", input_: str = "title", prompt: str = "zero-shot", value: float = 0.6
) -> dict[str, Any]:
    return {
        "kind": "measured",
        "arm": model,
        "model": model,
        "input": input_,
        "prompt": prompt,
        "status": "complete",
        "coverage": 1.0,
        "failures": 0,
        "metrics": {
            "positive_precision": value,
            "positive_recall": value,
            "positive_f1": value,
            "false_positive_rate": 1 - value,
            "fpr_at_95_tpr": 0.9,
            "fn": 2,
        },
    }


@pytest.mark.parametrize("fields,n", [(("input",), 4), (("prompt",), 4), (("input", "prompt"), 2)])
def test_all_combinations_keep_models_and_unchecked_conditions_separate(
    fields: tuple[str, ...], n: int
) -> None:
    rows = [
        measured(model, input_, prompt)
        for model in ("a", "b")
        for input_ in ("title", "title_and_desc")
        for prompt in ("zero-shot", "few-shot")
    ]
    original = copy.deepcopy(rows)
    grouped = group_rows(rows, fields)
    assert len(grouped) == n and rows == original
    assert {row["model"] for row in grouped} == {"a", "b"}
    for row in grouped:
        assert row["_group_size"] == 8 // n
        assert len(row["_group_inputs"]) == (2 if "input" in fields else 1)
        assert len(row["_group_prompts"]) == (2 if "prompt" in fields else 1)


def test_original_unrounded_means_population_sd_counts_and_singletons() -> None:
    a, b = measured(value=0.6), measured(input_="title_and_desc", value=0.8)
    a["metrics"]["fn"], b["metrics"]["fn"] = 1, 4
    stats = group_rows([a, b], ("input",))[0]["_group_stats"]
    assert stats["balanced_accuracy"].mean == pytest.approx(0.7)
    assert stats["balanced_accuracy"].sd == pytest.approx(0.1)
    assert format_stats(stats["balanced_accuracy"], percent=True) == "70.0% (±10.0)"
    assert format_stats(stats["missed"]) == "2.5 (±1.5)"
    singleton = group_rows([a], ("input",))[0]["_group_stats"]
    assert format_stats(singleton["recall"], percent=True) == "60.0% (±0)"
    # Rounded display ties must not replace the source values used for arithmetic.
    a["metrics"]["positive_f1"], b["metrics"]["positive_f1"] = 0.60049, 0.60051
    assert group_rows([a, b], ("input",))[0]["_group_stats"]["f1"].mean == pytest.approx(0.6005)


def test_mean_f1_is_not_pooled_f1_and_does_not_weight_support() -> None:
    a, b = measured(), measured(input_="title_and_desc")
    # TP/FP/FN: 1/0/1 versus 1/2/0. Mean F1=7/12; pooled F1=4/7.
    a["metrics"].update(positive_f1=2 / 3, tp=1, fp=0, fn=1, n=2)
    b["metrics"].update(positive_f1=1 / 2, tp=1, fp=2, fn=0, n=3)
    mean = group_rows([a, b], ("input",))[0]["_group_stats"]["f1"].mean
    assert mean == pytest.approx(7 / 12) and mean != pytest.approx(4 / 7)


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_strict_missingness_is_per_metric(missing: float | None) -> None:
    a, b = measured(), measured(input_="title_and_desc")
    b["metrics"]["fpr_at_95_tpr"] = missing
    stats = group_rows([a, b], ("input",))[0]["_group_stats"]
    assert stats["fpr_at_95_tpr"].mean is None
    assert "1/2" in stats["fpr_at_95_tpr"].reason
    assert stats["f1"].mean == 0.6


def test_failure_handling_and_incomplete_groups_withhold_performance_only() -> None:
    a, b = measured(), measured(input_="title_and_desc")
    b.update(status=FINISHED_WITH_FAILURES, coverage=0.8, failures=3)
    group = group_rows([a, b], ("input",))[0]
    assert group["status"] == FINISHED_WITH_FAILURES
    assert group["_group_stats"]["f1"].mean == 0.6
    assert group["_group_stats"]["coverage"].mean == 0.9
    assert group["_group_stats"]["failures"].mean == 1.5
    b["status"] = "incomplete"
    group = group_rows([a, b], ("input",))[0]
    for key, value in group["_group_stats"].items():
        if key not in {"coverage", "failures"}:
            assert value.mean is None and "1/2 configurations incomplete" in value.reason
    assert group["_group_stats"]["coverage"].mean == 0.9
    b["failures"] = None
    assert group_rows([a, b], ("input",))[0]["_group_stats"]["failures"].mean is None


def test_precomputed_views_keep_published_rows_and_original_table_and_recompute_leaders() -> None:
    a, b, c = (
        measured(value=0.6),
        measured(input_="title_and_desc", value=1.0),
        measured(model="b", value=0.9),
    )
    a["metrics"].update(recall_ci_low=0.2, recall_ci_high=0.8)
    published = {
        "kind": "published",
        "arm": "Reference",
        "input_modality": "text",
        "recall": 0.85,
        "precision": 0.85,
        "f1": 0.85,
        "false_positive_rate": 0.15,
    }
    rows = [a, b, c, published]
    source = _table(rows, descriptions=load_descriptions(ROOT / "report"))
    match = re.search(
        r'<script class="table-grouping-data" type="application/json">(.*?)</script>', source
    )
    assert match
    views = json.loads(match[1])
    assert set(views) == {"none", "input", "prompt", "input,prompt"}
    assert views["none"] == _table_view(rows)
    assert "TPR [95% CI]" in views["none"]
    assert "60.0% [20.0%, 80.0%]" in views["none"]
    assert "<th>TPR</th>" in views["input"]
    original_headers = re.findall(r"<th>(.*?)</th>", views["none"])
    for key in ("input", "prompt", "input,prompt"):
        assert re.findall(r"<th>(.*?)</th>", views[key]) == [
            "TPR" if title == "TPR [95% CI]" else title for title in original_headers
        ]
        assert not re.search(r"\(±[^)]*pp\)", views[key])
    assert "1 config)" in views["input"]
    assert "100.0% (±0) / 0 (±0)" in views["input"]
    assert not re.search(r"<input[^>]*\schecked", source)

    def published_cells(view: str) -> str:
        match = re.search(r'<tr class="published-row">(.*?)</tr>', view)
        assert match
        return match[1]

    # This reference is not a column leader in either view; its formatting stays identical.
    assert all(published_cells(view) == published_cells(views["none"]) for view in views.values())
    reference = published_cells(views["none"])
    assert reference.count(">†</sup>") == 3
    assert reference.count(">‡</sup>") == 2
    assert "(est.)" not in reference
    assert "N/A<sup" not in reference
    for view in views.values():
        measured_rows = re.findall(r"<tr>(.*?)</tr>", view)
        assert all("published-origin" not in row for row in measured_rows)

    assert "<strong>100.0%</strong>" in views["none"]
    assert "<strong>90.0% (±0)</strong>" in views["input"]
    assert "2 configs" in views["input"] and "title; title + desc" in views["input"]
    assert "60.0% (±0)" in views["prompt"]
    assert "Metric available for 2/2" in views["input"]


@pytest.mark.parametrize(
    "inputs,prompts,expected",
    [
        (["title"], ["zero-shot"], []),
        (["title", "title_and_desc"], ["zero-shot"], ["input"]),
        (["title"], ["zero-shot", "few-shot"], ["prompt"]),
        (["title", "title_and_desc"], ["zero-shot", "few-shot"], ["input", "prompt"]),
        ([], [], []),
    ],
)
def test_controls_only_offer_dimensions_that_vary_in_measured_rows(
    inputs: list[str], prompts: list[str], expected: list[str]
) -> None:
    rows = [measured(input_=input_, prompt=prompt) for input_ in inputs for prompt in prompts]
    # Published modality/prompt variety cannot create grouping controls.
    rows.extend(
        {
            "kind": "published",
            "arm": f"Reference {i}",
            "input_modality": modality,
            "input": f"published-{i}",
            "prompt": f"published-{i}",
        }
        for i, modality in enumerate(("text", "text-image"))
    )
    source = _table(rows, descriptions=load_descriptions(ROOT / "report"))
    assert re.findall(r'<input type="checkbox" value="([^"]+)"', source) == expected
    assert ('<fieldset class="table-grouping">' in source) == bool(expected)
    assert ("Grouped values: mean ± population SD." in source) == bool(expected)
    assert ('class="grouping-status visually-hidden" role="status"' in source) == bool(expected)
    assert not re.search(r"<input[^>]*\schecked", source)


def test_grouping_rejects_already_grouped_values_and_uses_unrounded_leaders() -> None:
    a, b = measured(value=0.9001), measured(model="b", value=0.9002)
    grouped = group_rows([a, b], ("input",))
    with pytest.raises(ValueError, match="original configurations"):
        group_rows(grouped, ("prompt",))
    source = _table_view(grouped, grouped=True)
    # Same visible percentages, but only model b leads balanced accuracy.
    assert source.count("<strong>90.0% (±0)</strong>") == 6
    first = source.split("<tbody>")[1].split("</tr>")[0]
    assert "<strong>" not in re.findall(r"<td[^>]*>(.*?)</td>", first)[3]


def test_browser_controller_switches_all_views_restores_and_retains_independent_state() -> None:
    result = node(
        r"""
      const api=require(process.argv[1]);
      function figure(values=['input','prompt']){
        const controls=values.map(value=>({value,checked:false,
          addEventListener:(name,fn)=>{controls.find(c=>c.value===value).change=fn}}));
        const views={none:'original',input:'input',prompt:'prompt','input,prompt':'both'};
        const table={innerHTML:'captionoriginal',querySelector:()=>({outerHTML:'caption'})};
        const status={textContent:''};
        return {dataset:{},controls,table,status,
          querySelectorAll:()=>controls,querySelector:s=>s==='.table-grouping-data'
            ?{textContent:JSON.stringify(views)}:s==='table'?table:status};
      }
      const a=figure(),b=figure(['prompt']),c=figure([]),document={querySelectorAll:()=>[a,b,c]};
      api.initializeTableGrouping(document);api.initializeTableGrouping(document);
      const states=[];
      for(const checked of [[true,false],[true,true],[false,true],[false,false]]){
        a.controls.forEach((c,i)=>c.checked=checked[i]);a.controls[0].change();
        states.push(a.table.innerHTML);
      }
      a.controls[0].checked=true;a.controls[0].change();
      b.controls[0].checked=true;b.controls[0].change();
      process.stdout.write(JSON.stringify({states,a:a.table.innerHTML,b:b.table.innerHTML,
        aChecked:a.controls.map(c=>c.checked),bChecked:b.controls.map(c=>c.checked),status:a.status.textContent}));
    """,
        "site/report.js",
    )
    assert result["states"] == ["captioninput", "captionboth", "captionprompt", "captionoriginal"]
    assert result["a"] == "captioninput" and result["b"] == "captionprompt"
    assert result["aChecked"] == [True, False] and result["bChecked"] == [True]
    assert result["status"] == "Grouped by Input."


def test_storage_aliases_share_one_grouped_input_label() -> None:
    rows = [
        measured(input_="title_and_desc", value=0.6),
        measured(input_="combined_text", value=0.8),
    ]
    original = copy.deepcopy(rows)
    view = _table_view(group_rows(rows, ("input",)))
    assert "<td>title + desc</td>" in view
    assert "title + desc; title + desc" not in view
    assert "combined column" not in view and "combined_text" not in view
    assert rows == original
