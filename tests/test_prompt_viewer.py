"""Verify exact historical prompt sources and escaped presentation offline."""

import hashlib
import html
import re
from pathlib import Path

import pytest
import yaml
from site_builder.prompt_viewer import prompt_sources, prompt_viewer

from wildlife_screening.prompts import Prompt


def test_source_hashes_select_exact_prompt_bytes_and_escape_markup(tmp_path: Path) -> None:
    directory = tmp_path / "prompts" / "invented"
    directory.mkdir(parents=True)
    original = "Invented source <script>untrusted()</script> & text\n"
    source = directory / "llm_zeroshot_v0.txt"
    source.write_text(original)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = [{"family": "llm", "condition": "zero-shot", "version": "v0", "sha256": digest}]
    page = prompt_viewer(0, records, prompt_sources(tmp_path))
    assert "&lt;script&gt;untrusted()&lt;/script&gt; &amp; text" in page
    assert "<script>untrusted" not in page
    assert original == source.read_text()
    assert 'class="language-text"' in page
    assert str(tmp_path) not in page
    # Editing a same-named file must never present it as the historical source.
    source.write_text("Invented changed source")
    page = prompt_viewer(0, records, prompt_sources(tmp_path))
    assert "saved prompt source is unavailable" in page
    assert "Invented changed source" not in page


def test_yaml_multiple_versions_missing_sources_and_symlinks(tmp_path: Path) -> None:
    directory = tmp_path / "prompts"
    directory.mkdir()
    records = []
    for index in range(2):
        source = directory / f"decision-v{index}.yaml"
        source.write_text(
            yaml.safe_dump(
                {
                    "version": f"v{index}",
                    "family": "decision",
                    "condition": "zero-shot",
                    "instructions": f"Invented policy {index}",
                    "criteria": {
                        "relevant": "Matches invented policy",
                        "not relevant": "Does not match",
                    },
                    "examples": [],
                }
            )
        )
        records.append(
            {
                "family": "decision",
                "condition": "zero-shot",
                "version": f"v{index}",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    outside = tmp_path / "private.txt"
    outside.write_text("Invented excluded text")
    (directory / "linked.txt").symlink_to(outside)
    sources = prompt_sources(tmp_path)
    assert len(sources) == 2
    page = prompt_viewer(4, records, sources)
    assert page.count('class="prompt-source"') == 2
    assert page.count('class="language-yaml"') == 2
    assert 'name="prompt-family-4"' in page
    assert 'name="prompt-condition-4"' in page
    assert "Invented excluded text" not in page
    assert "Prompt version" in page
    assert "saved prompt source is unavailable" in prompt_viewer(4, records, {})


@pytest.mark.parametrize("few_shot", [False, True])
def test_decision_preview_matches_assembled_question(tmp_path: Path, few_shot: bool) -> None:
    directory = tmp_path / "prompts"
    directory.mkdir()
    source = directory / "invented.yaml"
    definition = {
        "version": "invented-version",
        "family": "decision",
        "condition": "few-shot" if few_shot else "zero-shot",
        "source": "Invented citation",
        "rendering": "native-choice-v0",
        "instructions": "Invented <policy> & instructions.",
        "criteria": {"relevant": "An invented positive", "not relevant": "An invented negative"},
        "examples": [
            {"text": "Invented alpha", "label": "relevant", "explanation": "Alpha rationale."},
            {"text": "Invented beta", "label": "not relevant", "explanation": "Beta rationale."},
        ]
        if few_shot
        else [],
    }
    source.write_text(yaml.safe_dump(definition, sort_keys=False))
    original = source.read_bytes()
    prompt = Prompt.load(source)
    record = {
        "family": "decision",
        "condition": prompt.condition,
        "version": "invented-version",
        "sha256": prompt.sha256,
    }
    page = prompt_viewer(0, [record], prompt_sources(tmp_path))
    code = re.search(r'<code class="language-yaml">(.*?)</code>', page, re.S)
    assert code is not None
    preview = yaml.safe_load(html.unescape(code.group(1)))
    question = prompt.question()
    assert preview == {"instructions": question["instructions"], "criteria": question["criteria"]}
    assert set(preview) == {"instructions", "criteria"}
    assert "&lt;policy&gt; &amp;" in page
    assert "Invented citation" not in code.group(1)
    assert "native-choice-v0" not in code.group(1)
    if few_shot:
        assert preview["instructions"].endswith(
            "Examples:\n1. Advertisement: Invented alpha\nLabel: relevant\nAlpha rationale.\n"
            "2. Advertisement: Invented beta\nLabel: not relevant\nBeta rationale."
        )
    else:
        assert "Examples:" not in preview["instructions"]
    assert "Prompt provenance" in page and prompt.sha256 in page
    assert '<p class="subtle">Advert text is supplied separately and is not shown here.</p>' in page
    assert "Decision previews show" not in page
    assert 'role="status" hidden' in page
    assert source.read_bytes() == original


@pytest.mark.parametrize("definition", ["[invalid", "[]", "instructions: Incomplete"])
def test_invalid_matched_definition_has_explicit_message(tmp_path: Path, definition: str) -> None:
    directory = tmp_path / "prompts"
    directory.mkdir()
    source = directory / "invented.yaml"
    source.write_text(definition)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {"family": "decision", "condition": "zero-shot", "version": "v0", "sha256": digest}
    page = prompt_viewer(0, [record], prompt_sources(tmp_path))
    assert "matched saved prompt definition cannot be rendered" in page
    assert 'class="language-yaml"' not in page
    assert "Prompt provenance" in page and digest in page
