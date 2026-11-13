"""Check editable model cards without private prose or provider calls."""

import shutil
from pathlib import Path

from site_builder.build import initialize
from site_builder.content import MODELS, report_content

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("Version", "Type", "Size", "Inputs", "Outputs", "Source", "Notes")


def test_model_cards_preserve_editable_markdown_and_notes(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    notes = "Keep *these notes* and this [existing citation](https://example.invalid/notes).\n"
    draft = directory / "methods/models/d3.md"
    source = "\n\n".join(
        f"**{field}:** {value}"
        for field, value in zip(
            FIELDS,
            (
                "invented-checkpoint-" + "x" * 200,
                "Decision",
                "~1B parameters",
                "Invented text input",
                "Invented choice probabilities",
                "[Invented model card](https://example.invalid/model)",
                "\n\n" + notes,
            ),
            strict=True,
        )
    )
    draft.write_text(source)
    initialize(tmp_path)
    assert draft.read_text() == source
    page = report_content(directory, [], (ROOT / "site/report.html").read_text())
    assert page.count('class="model-card"') == 4
    assert page.count('role="tablist"') == 1
    assert 'aria-label="Models"' in page
    assert "Keep <em>these notes</em>" in page
    assert 'href="https://example.invalid/notes"' in page
    assert 'href="https://example.invalid/model"' in page
    assert "these experiments evaluate text inputs only" not in page
    assert "model-capabilities-note" not in page
    card = page.split('id="models-panel-3"', 1)[1].split("</div></div>", 1)[0]
    offsets = [card.index(f"<strong>{field}:</strong>") for field in FIELDS]
    assert offsets == sorted(offsets)
    assert draft.read_text() == source


def test_starter_model_cards_have_the_same_fields_without_overwriting(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "site/content_templates", tmp_path / "site/content_templates")
    directory = initialize(tmp_path)
    for slug in MODELS:
        draft = directory / f"methods/models/{slug}.md"
        text = draft.read_text()
        offsets = [text.index(f"**{field}:**") for field in FIELDS]
        assert offsets == sorted(offsets)
        assert text.count("**Notes:**") == 1
        assert "to be added" in text
        draft.write_text(text + "\nInvented researcher edit.\n")
    initialize(tmp_path)
    for slug in MODELS:
        assert (
            (directory / f"methods/models/{slug}.md")
            .read_text()
            .endswith("Invented researcher edit.\n")
        )
