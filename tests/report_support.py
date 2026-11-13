"""Shared offline evidence-rendering helper for invented snapshots."""

import json
from pathlib import Path
from typing import Any

from site_builder.descriptions import ResultDescriptions, load_descriptions
from site_builder.evidence import _html_document, build_evidence
from site_builder.evidence import _table as evidence_table

from wildlife_screening.snapshots import _read_selection


def fixture_descriptions() -> ResultDescriptions:
    """Load invented human-readable caption fixtures, never private report drafts."""
    return load_descriptions(Path(__file__).parent / "fixtures/report")


def _table(rows: list[dict[str, Any]], index: int = 0) -> str:
    """Exercise the table renderer using explicit invented descriptions."""
    return evidence_table(rows, index, descriptions=fixture_descriptions())


def render_evidence(selection: Path, output: Path) -> None:
    """Render only evidence; prose authoring is tested through the site builder."""
    panels, data = build_evidence(_read_selection(selection), fixture_descriptions())
    content = "".join(
        f'<section class="dataset-panel" data-dataset-key="{panel.key}">'
        f'<h2 class="dataset-heading">{panel.task}</h2>'
        + panel.roc
        + panel.table
        + panel.resources
        + panel.provenance
        + "</section>"
        for panel in panels
    )
    output.write_text(_html_document(content, json.dumps(data)), encoding="utf-8")
