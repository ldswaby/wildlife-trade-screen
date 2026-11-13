"""Verify report authoring, result imports and allowlisted site packaging offline."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from site_builder.build import ASSETS, initialize, prepare_site
from site_builder.selection import import_selection
from test_interactive_report import _invented_csv, _run

ROOT = Path(__file__).resolve().parents[1]


def assets(root: Path) -> Path:
    shutil.copytree(ROOT / "site", root / "site", ignore=shutil.ignore_patterns("__pycache__"))
    return initialize(root)


def test_initialization_preserves_drafts_and_has_complete_structure(tmp_path: Path) -> None:
    directory = assets(tmp_path)
    assert (directory / "overall-results.md").exists()
    assert (directory / "results/shark-products/commentary.md").exists()
    assert (directory / "results/dataset-introduction.md").exists()
    draft = directory / "abstract.md"
    draft.write_text("# Invented abstract\n\nKeep my text.")
    initialize(tmp_path)
    assert "Keep my text." in draft.read_text()
    assert (directory / "report-abstract.md").read_text() == ""
    assert (directory / "conclusion.md").read_text() == ""
    for name in ("report-abstract.md", "background.md", "conclusion.md"):
        (directory / name).write_text("Invented authored draft.")
    initialize(tmp_path)
    for name in ("report-abstract.md", "background.md", "conclusion.md"):
        assert (directory / name).read_text() == "Invented authored draft."
    assert yaml.safe_load((directory / "selection.yaml").read_text())["pins"] == []


def test_bundle_only_packages_allowed_files_and_refreshes_prose(tmp_path: Path) -> None:
    directory = assets(tmp_path)
    (tmp_path / "site/private.txt").write_text("Invented excluded text")
    (directory / "abstract.md").write_text("# Invented abstract\n\nLocal draft.")
    output = prepare_site(tmp_path)
    assert {path.name for path in output.iterdir()} == {*ASSETS, "report.html"}
    index = (output / "index.html").read_text()
    report = (output / "report.html").read_text()
    assert 'aria-current="page">Overview' in index
    assert "<title>Wildlife-trade screening</title>" in index
    assert "Research question" in index and "Reproduce the study" in index
    assert "Invented abstract" not in index + report
    assert '<h2 id="abstract">Abstract</h2>' in report
    abstract = report.split('<section aria-labelledby="abstract">')[1].split("</section>")[0]
    assert abstract.strip() == '<h2 id="abstract">Abstract</h2>'
    assert report.index('id="abstract"') < report.index('id="introduction"')
    assert 'aria-current="page">Report' in (output / "report.html").read_text()
    assert "No saved dataset results" in (output / "report.html").read_text()
    (directory / "introduction.md").write_text("Invented revised introduction.")
    (directory / "report-abstract.md").write_text("Invented **report abstract**.")
    (directory / "background.md").write_text("# Personal Note\n\nInvented personal note.")
    (directory / "abstract.md").unlink()
    prepare_site(tmp_path)
    assert "Invented revised introduction." in (output / "report.html").read_text()
    assert "Invented <strong>report abstract</strong>." in (output / "report.html").read_text()
    assert "Invented personal note." in (output / "index.html").read_text()
    assert "report abstract" not in (output / "index.html").read_text()
    assert "Invented personal note" not in (output / "report.html").read_text()
    assert "Invented excluded text" not in (output / "index.html").read_text()
    before = (output / "report.html").read_bytes()
    (output / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected files"):
        prepare_site(tmp_path)
    assert (output / "report.html").read_bytes() == before
    assert (output / "unexpected.json").exists()


def test_bundle_rejects_symlinks_and_missing_drafts_before_replacing(tmp_path: Path) -> None:
    directory = assets(tmp_path)
    output = prepare_site(tmp_path)
    before = (output / "report.html").read_bytes()
    (directory / "discussion.md").unlink()
    with pytest.raises(ValueError, match="run site/build.py init"):
        prepare_site(tmp_path)
    assert (output / "report.html").read_bytes() == before
    shutil.rmtree(tmp_path / "build")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "build").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        prepare_site(tmp_path)
    assert not list(outside.iterdir())


def test_import_run_is_explicit_idempotent_and_preserves_other_datasets(tmp_path: Path) -> None:
    directory = assets(tmp_path)
    first = _invented_csv(tmp_path / "first.csv")
    second = _invented_csv(tmp_path / "second.csv", revised=True)
    pin = _run(tmp_path, first, True)
    selected = directory / "selection.yaml"
    original = pin.read_bytes()
    assert import_selection(pin, selected, tmp_path, data=first, positive_label="1") == 1
    assert import_selection(pin, selected, tmp_path, data=first) == 0
    other = _run(tmp_path, second, True)
    assert import_selection(other, selected, tmp_path, data=second, positive_label="1") == 1
    content = yaml.safe_load(selected.read_text())
    assert len(content["pins"]) == 2 and len(content["datasets"]) == 2
    assert not any(Path(value).is_absolute() for value in content["pins"])
    assert pin.read_bytes() == original
    output = prepare_site(tmp_path)
    page = (output / "report.html").read_text()
    assert page.count('class="dataset-panel"') == 2
    for prefix in ("roc-title-", "table-title-", "resource-title-"):
        assert page.count(f'<h3 id="{prefix}') == 2
        assert f'<h4 id="{prefix}' not in page
    assert page.count('<h2 id="results">Results</h2>') == 1
    assert "Invented shark item" not in page and "raw_output" not in page
    assert str(tmp_path) not in page


def test_candidate_batch_merge_duplicates_and_conflicts_are_atomic(tmp_path: Path) -> None:
    directory = assets(tmp_path)
    data = _invented_csv(tmp_path / "first.csv")
    pin = _run(tmp_path, data)
    selected = directory / "selection.yaml"
    assert import_selection(pin, selected, tmp_path, data=data, positive_label="1") == 1
    # A second recorded run of the same configuration is an explicit replacement,
    # even when its scores happen to be identical.
    replacement = pin.with_name("manifest-other.json")
    shutil.copyfile(pin, replacement)
    shutil.copyfile(
        pin.with_name(pin.name.replace("manifest-", "scores-")),
        replacement.with_name("scores-other.json"),
    )
    candidate = tmp_path / "candidate.yaml"
    content: dict[str, Any] = yaml.safe_load(selected.read_text())
    content["pins"] = [str(pin), str(pin)]
    candidate.write_text(yaml.safe_dump(content))
    assert import_selection(candidate, selected, tmp_path, candidate=True) == 0
    before = selected.read_bytes()
    content["pins"] = [str(replacement)]
    candidate.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="Conflicting snapshots"):
        import_selection(candidate, selected, tmp_path, candidate=True)
    assert selected.read_bytes() == before
    content["pins"] = [str(pin)]
    next(iter(content["datasets"].values()))["positive_label"] = "0"
    candidate.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="Conflicting dataset"):
        import_selection(candidate, selected, tmp_path, candidate=True)
    assert selected.read_bytes() == before
    # A corrupt incoming record also leaves the active list untouched.
    broken = json.loads(replacement.read_text())
    broken["status"] = "aborted"
    replacement.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="completed scoring snapshot"):
        import_selection(replacement, selected, tmp_path)
    assert selected.read_bytes() == before


@pytest.mark.parametrize("markers", [0, 2])
def test_invalid_data_marker_leaves_existing_bundle_unchanged(tmp_path: Path, markers: int) -> None:
    directory = assets(tmp_path)
    output = prepare_site(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    draft = directory / "methods/data/birds.md"
    draft.write_text("- **Labels/policy:** Invented.\n\n" + "{{ class_distribution }}\n" * markers)
    # Even an unselected dataset must have valid authoring structure.
    with pytest.raises(ValueError, match="birds.md: expected exactly one"):
        prepare_site(tmp_path)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_public_pages_share_a_fixed_publication_date_and_indexable_navigation(
    tmp_path: Path,
) -> None:
    assets(tmp_path)
    output = prepare_site(tmp_path)
    pages = {name: (output / name).read_text() for name in ("index.html", "report.html")}
    for page in pages.values():
        assert '<em><time datetime="2026-11-13">Published: 13/11/2026</time></em>' in page
        assert "noindex" not in page
        assert "Private preview" not in page and "Draft research" not in page
        assert 'href="./"' in page and 'href="report.html"' in page
        assert 'href="https://github.com/ldswaby/wildlife-trade-screen"' in page
        assert 'rel="canonical"' in page
    prepare_site(tmp_path)
    assert pages == {name: (output / name).read_text() for name in pages}
