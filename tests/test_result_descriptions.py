"""Verify editable Results descriptions without private drafts or inference."""

import shutil
from pathlib import Path

import pytest
from site_builder.build import initialize, prepare_site
from site_builder.descriptions import load_descriptions
from site_builder.evidence import _table

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "roc",
    "table",
    "resources",
    "table-gpt4-note",
    "summary",
)


def invented(root: Path) -> Path:
    shutil.copytree(ROOT / "tests/fixtures/report", root / "report")
    return root / "report"


def test_titles_markdown_and_definitions_are_authored(tmp_path: Path) -> None:
    report = invented(tmp_path)
    shared = report / "results/shared"
    (shared / "roc.md").write_text(
        "# Figure 1. **Invented** & [title](https://example.org)\n\n"
        "Description with *emphasis* and `<invented>`.\n\n"
        "Metric\n: Definition with **bold**."
    )
    descriptions = load_descriptions(report)
    assert descriptions.roc.title_text == "Figure 1. Invented & title"
    assert "<strong>Invented</strong>" in descriptions.roc.title_html
    assert '<a href="https://example.org">title</a>' in descriptions.roc.title_html
    assert "<em>emphasis</em>" in descriptions.roc.body_html
    assert "<code>&lt;invented&gt;</code>" in descriptions.roc.body_html
    assert "<dt>Metric</dt>" in descriptions.roc.body_html
    assert "<dd>Definition with <strong>bold</strong>.</dd>" in descriptions.roc.body_html


def test_editing_table_caption_and_conditional_note_changes_rendered_text(tmp_path: Path) -> None:
    report = invented(tmp_path)
    shared = report / "results/shared"
    (shared / "table.md").write_text("# Invented revised title\n\nRevised table **caption**.")
    (shared / "table-gpt4-note.md").write_text("A revised *conditional* note.")
    descriptions = load_descriptions(report)
    no_note = _table([], descriptions=descriptions)
    assert '<h3 id="table-title-0">Invented revised title</h3>' in no_note
    assert '<caption class="visually-hidden">Invented revised title</caption>' in no_note
    assert "Revised table <strong>caption</strong>." in no_note
    assert "conditional" not in no_note
    row = {"kind": "published", "arm": "GPT-4 zero-shot", "input_modality": "text"}
    with_note = _table([row], descriptions=descriptions)
    assert with_note.count("A revised <em>conditional</em> note.") == 1
    assert row == {"kind": "published", "arm": "GPT-4 zero-shot", "input_modality": "text"}


def test_title_contents_are_rendered_as_inline_markdown(tmp_path: Path) -> None:
    report = invented(tmp_path)
    (report / "results/shared/roc.md").write_text("# - **Invented** title\n\nCaption.")
    title = load_descriptions(report).roc
    assert title.title_html == "- <strong>Invented</strong> title"
    assert title.title_text == "- Invented title"


def test_literal_heading_inside_fenced_code_is_not_a_second_title(tmp_path: Path) -> None:
    report = invented(tmp_path)
    (report / "results/shared/roc.md").write_text(
        "# Invented title\n\nDescription.\n\n```text\n# Literal example\n```"
    )
    assert "# Literal example" in load_descriptions(report).roc.body_html


@pytest.mark.parametrize("name", NAMES)
def test_every_shared_description_is_required(tmp_path: Path, name: str) -> None:
    report = invented(tmp_path)
    (report / f"results/shared/{name}.md").unlink()
    with pytest.raises(ValueError, match=rf"{name}.md.*site/build.py init"):
        load_descriptions(report)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "Caption without title",
        "## Wrong heading\n\nCaption",
        "# Title",
        "# Title\n\n# Second",
        "# Title\n\n<!-- Empty caption -->",
    ],
)
def test_invalid_title_or_empty_caption_has_actionable_error(tmp_path: Path, source: str) -> None:
    report = invented(tmp_path)
    (report / "results/shared/roc.md").write_text(source)
    with pytest.raises(ValueError, match=r"roc.md.*(?:nonempty|Title)"):
        load_descriptions(report)


@pytest.mark.parametrize("name", ["table-gpt4-note"])
def test_empty_guide_or_note_is_rejected(tmp_path: Path, name: str) -> None:
    report = invented(tmp_path)
    (report / f"results/shared/{name}.md").write_text(" \n")
    with pytest.raises(ValueError, match=rf"{name}.md.*nonempty"):
        load_descriptions(report)


def test_invalid_description_precedes_aggregation_and_preserves_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(ROOT / "site", tmp_path / "site", ignore=shutil.ignore_patterns("__pycache__"))
    report = initialize(tmp_path)
    draft = report / "results/shared/roc.md"
    draft.write_text("# Invented edited title\n\nKeep my custom **caption**.")
    initialize(tmp_path)
    assert "Keep my custom **caption**." in draft.read_text()
    output = prepare_site(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    draft.write_text("Malformed draft")

    def unexpected(*args: object) -> None:
        pytest.fail("Invalid descriptions must fail before snapshot aggregation")

    monkeypatch.setattr("site_builder.build.build_evidence", unexpected)
    with pytest.raises(ValueError, match=r"roc.md.*# Title"):
        prepare_site(tmp_path)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_shared_description_symlink_is_rejected(tmp_path: Path) -> None:
    report = invented(tmp_path)
    path = report / "results/shared/roc.md"
    path.unlink()
    target = tmp_path / "invented.md"
    target.write_text("# Title\n\nInvented caption")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Missing regular draft"):
        load_descriptions(report)


def test_shared_caption_context_preserves_markdown(tmp_path: Path) -> None:
    from site_builder.content import dataset_text

    report = invented(tmp_path)
    shared = report / "results/shared"
    (shared / "table.md").write_text(
        "# Table 1. {{ dataset_name }} results\n\n"
        "Locally edited **{{ dataset_name }}** caption with `code` and _emphasis_."
    )
    descriptions = load_descriptions(report)
    for name in ("Sharks", "Animals", "Birds"):
        table = dataset_text(_table([], descriptions=descriptions), name, 2)
        assert f'<h3 id="table-title-0">Table 1. {name} results</h3>' in table
        assert f"Locally edited <strong>{name}</strong> caption" in table
        assert "<code>code</code> and <em>emphasis</em>" in table
    assert "{{ dataset_name }}" in (shared / "table.md").read_text()
