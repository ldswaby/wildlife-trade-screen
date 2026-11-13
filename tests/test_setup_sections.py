"""Verify shared methods prose, stable anchors and evidence-aware links offline."""

import shutil
from pathlib import Path

import pytest
from site_builder.build import initialize, prepare_site
from site_builder.content import (
    MODEL_CARDS_MARKER,
    SHARED_METHODS,
    contextual_links,
    model_cards_source,
    model_section,
    prose,
    report_content,
)
from site_builder.evidence import DatasetEvidence

ROOT = Path(__file__).resolve().parents[1]


def panel(index: int, task: str) -> DatasetEvidence:
    return DatasetEvidence(index, f"{task}-invented", task, 10, 3, "a" * 64, [], "", "", "", "")


def drafts(root: Path) -> Path:
    shutil.copytree(ROOT / "site", root / "site", ignore=shutil.ignore_patterns("__pycache__"))
    return initialize(root)


def test_shared_sections_have_requested_order_and_preserve_authored_wording(tmp_path: Path) -> None:
    directory = drafts(tmp_path)
    authored = "Keep this *invented* wording verbatim, including a deliberate typo: relavant."
    (directory / "methods/prompts.md").write_text(
        authored + "\n\nThe precise prompts can be viewed [here](#selected-prompts)."
    )
    page = report_content(
        directory,
        [panel(0, "animals"), panel(1, "birds"), panel(2, "sharks")],
        (tmp_path / "site/report.html").read_text(),
    )
    headings = [
        "experimental-setup",
        "models",
        "data-summary",
        "inputs",
        "prompts",
        "evaluation",
        "resources-reproducibility",
        "limitations",
    ]
    offsets = [page.index(f'id="{heading}"') for heading in headings]
    assert offsets == sorted(offsets)
    assert offsets[-1] < page.index('class="dataset-workspace"')
    assert '<h2 id="experimental-setup">' in page
    for heading in headings[1:6]:
        assert f'<h3 id="{heading}">' in page
    assert '<h4 id="resources-reproducibility">' in page
    assert '<h3 id="limitations">' in page
    assert (
        "Keep this <em>invented</em> wording verbatim, including a deliberate typo: relavant."
        in page
    )
    assert '<a href="#dataset-prompts-2" data-active-dataset-section="prompts">here</a>' in page
    assert (
        '<a href="#dataset-data-2" data-active-dataset-section="data">Dataset details</a>' in page
    )
    assert '<a href="#dataset-data-1">Live Birds Dataset details</a>' in page
    assert page.count('class="prompt-viewer"') == 3
    assert '<h3 id="dataset-data-2">Dataset details</h3>' in page
    assert '<h3 id="dataset-prompts-2">Prompt viewer</h3>' in page
    assert '<details class="comparison-caveats">' in page
    assert '<span class="caveat-icon" aria-hidden="true">!</span>' in page
    assert (directory / "methods/prompts.md").read_text().startswith(authored)


@pytest.mark.parametrize("selected", [[], [panel(0, "sharks")]])
def test_context_links_explain_unavailable_targets(selected: list[DatasetEvidence]) -> None:
    markup = (
        '<a href="#selected-dataset">Dataset</a> '
        '<a href="#selected-prompts">here</a> '
        '<a href="#birds-dataset">Birds</a>'
    )
    page = contextual_links(markup, selected)
    assert "#birds-dataset" not in page and "Live Birds results are not selected." in page
    if selected:
        assert 'href="#dataset-data-0"' in page and 'href="#dataset-prompts-0"' in page
    else:
        assert "No dataset results are selected." in page
        assert "href=" not in page
    assert "#selected-" not in page


def test_new_drafts_initialise_without_overwriting_and_missing_sections_fail_atomically(
    tmp_path: Path,
) -> None:
    directory = drafts(tmp_path)
    for name in SHARED_METHODS.values():
        assert (directory / "methods" / name).is_file()
    target = directory / "methods/inputs.md"
    target.write_text("Invented researcher input-condition prose.")
    initialize(tmp_path)
    assert target.read_text() == "Invented researcher input-condition prose."
    output = prepare_site(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    target.unlink()
    with pytest.raises(ValueError, match="inputs.md"):
        prepare_site(tmp_path)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_models_draft_surrounds_unchanged_widget_and_preserves_rich_text(tmp_path: Path) -> None:
    directory = drafts(tmp_path)
    target = directory / "methods/models.md"
    source = (
        "Before **invented emphasis**, `identifier` and [source](https://example.org/model).\n\n"
        + MODEL_CARDS_MARKER
        + "\n\nAfter *invented commentary*."
    )
    original = model_section(directory)
    widget = original[original.index('<div class="report-tab-group" data-tab-group="models">') :]
    widget = widget[: widget.rindex("</div>") + len("</div>")]
    target.write_text(source)
    initialize(tmp_path)
    assert target.read_text() == source
    rendered = model_section(directory)
    assert widget in rendered
    assert rendered.index("Before <strong>") < rendered.index('data-tab-group="models"')
    assert rendered.index('data-tab-group="models"') < rendered.index("After <em>")
    assert "<code>identifier</code>" in rendered
    assert '<a href="https://example.org/model">source</a>' in rendered
    assert MODEL_CARDS_MARKER not in rendered
    assert "<p><div" not in rendered


@pytest.mark.parametrize(
    "source",
    ["No viewer", MODEL_CARDS_MARKER * 2, "Before " + MODEL_CARDS_MARKER, ""],
)
def test_models_marker_errors_precede_evidence_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    directory = drafts(tmp_path)
    output = prepare_site(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    (directory / "methods/models.md").write_text(source)

    def unexpected_aggregation(*args: object) -> None:
        pytest.fail("Invalid model insertion marker must fail before aggregation")

    monkeypatch.setattr("site_builder.build.build_evidence", unexpected_aggregation)
    with pytest.raises(ValueError, match=r"models.md.*exactly one.*model_cards.*own line"):
        prepare_site(tmp_path)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_models_marker_missing_file_and_symlink_are_rejected(tmp_path: Path) -> None:
    directory = drafts(tmp_path)
    target = directory / "methods/models.md"
    target.unlink()
    with pytest.raises(ValueError, match=r"models.md.*site/build.py init"):
        model_cards_source(directory)
    source = tmp_path / "invented.md"
    source.write_text(MODEL_CARDS_MARKER)
    target.symlink_to(source)
    with pytest.raises(ValueError, match="Missing regular draft"):
        model_cards_source(directory)


def test_caveat_bullets_links_and_authored_mathml_survive_markdown(tmp_path: Path) -> None:
    directory = drafts(tmp_path)
    equation = (
        '<span class="display-equation" role="region" aria-label="Invented equation" tabindex="0">'
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" '
        'aria-label="Seven divided by three">'
        "<mfrac><mn>7</mn><mn>3</mn></mfrac></math></span>"
    )
    source = "\n\n".join(
        [
            "- **Invented first:** [source](https://example.org/paper).",
            "- **Invented second:** `input` and *emphasis*.",
            "- **Invented third:**\n\n    " + equation + "\n\n    Preserved assumption.",
            "- **Invented fourth:** Independent caveat.",
            "- **Invented fifth:** [Birds Dataset](#birds-dataset).",
        ]
    )
    (directory / "methods/published-comparison-caveats.md").write_text(source)
    rendered = contextual_links(
        prose(directory, "methods/published-comparison-caveats.md"), [panel(1, "birds")]
    )
    assert rendered.count("<li>") == 5
    assert rendered.count("<ul>") == 1
    assert equation in rendered
    assert rendered.index("Preserved assumption.") < rendered.index("Invented fourth")
    assert '<a href="#dataset-data-1">Birds Dataset</a>' in rendered
    assert "<code>input</code>" in rendered and "<em>emphasis</em>" in rendered
    assert "<strong>Invented first:</strong>" in rendered
    assert '<a href="https://example.org/paper">source</a>' in rendered


def test_evaluation_figure_and_table_references_follow_selected_evidence() -> None:
    source = '<a href="#figure-1">Figure 1: ROC</a> and <a href="#table-1">Table 1</a>'
    rendered = contextual_links(source, [panel(2, "sharks")])
    assert (
        '<a href="#roc-title-2" data-active-dataset-target="roc-title">Figure 1: ROC</a>'
        in rendered
    )
    assert (
        '<a href="#table-title-2" data-active-dataset-target="table-title">Table 1</a>' in rendered
    )
    unavailable = contextual_links(source, [])
    assert "href=" not in unavailable
    assert unavailable.count("No dataset results are selected.") == 2


@pytest.mark.parametrize("selected", [[], [panel(0, "sharks")], [panel(1, "birds")]])
def test_bibliography_is_shared_final_markdown_and_initialization_preserves_edits(
    tmp_path: Path, selected: list[DatasetEvidence]
) -> None:
    directory = drafts(tmp_path)
    target = directory / "bibliography.md"
    assert target.read_text() == (ROOT / "site/content_templates/bibliography.md").read_text()
    source = "Invented, A. (2026). *Example paper*. [Publisher](https://example.org/paper).\n"
    target.write_text(source)
    initialize(tmp_path)
    assert target.read_text() == source
    page = report_content(directory, selected, (tmp_path / "site/report.html").read_text())
    assert page.count('<h2 id="bibliography">Bibliography</h2>') == 1
    assert page.index('id="discussion"') < page.index('id="bibliography"')
    section = page[page.index('<section aria-labelledby="bibliography">') :]
    assert "<em>Example paper</em>" in section
    assert '<a href="https://example.org/paper">Publisher</a>' in section
    assert "hidden" not in section and "tabpanel" not in section
    assert "{{ bibliography }}" not in page
    assert page.rindex("<h2") == page.index('<h2 id="bibliography"')


def test_conclusion_is_editable_and_follows_discussion(tmp_path: Path) -> None:
    directory = drafts(tmp_path)
    target = directory / "conclusion.md"
    assert target.read_text() == ""
    target.write_text("Invented conclusion with **editable wording**.")
    initialize(tmp_path)
    assert target.read_text() == "Invented conclusion with **editable wording**."
    page = report_content(
        directory, [panel(0, "sharks")], (tmp_path / "site/report.html").read_text()
    )
    assert page.count('<h2 id="conclusion">Conclusion</h2>') == 1
    assert (
        page.index('id="discussion"')
        < page.index('id="conclusion"')
        < page.index('id="bibliography"')
    )
    assert "Invented conclusion with <strong>editable wording</strong>." in page
    assert "{{ conclusion }}" not in page


@pytest.mark.parametrize("selected", [[panel(7, "birds")], [panel(2, "sharks")], []])
def test_summary_caption_resolves_bird_details_for_selected_evidence(
    tmp_path: Path, selected: list[DatasetEvidence]
) -> None:
    directory = drafts(tmp_path)
    summary = '<figure><figcaption><a href="#birds-dataset">Bird details</a></figcaption></figure>'
    page = report_content(directory, selected, (tmp_path / "site/report.html").read_text(), summary)
    assert 'href="#birds-dataset"' not in page
    if selected and selected[0].task == "birds":
        assert '<a href="#dataset-data-7">Bird details</a>' in page
        assert '<h3 id="dataset-data-7">Dataset details</h3>' in page
    else:
        assert "Live Birds results are not selected." in page
