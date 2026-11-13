"""Verify safe consolidation of invented private commentary drafts."""

import shutil
from pathlib import Path

import pytest
from site_builder.build import initialize, prepare_site

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ("roc", "table", "resources")


def setup_templates(root: Path) -> Path:
    """Copy public starters without introducing private research text."""
    shutil.copytree(ROOT / "site", root / "site", ignore=shutil.ignore_patterns("__pycache__"))
    folder = root / "report/results/shark-products"
    folder.mkdir(parents=True)
    return folder


def test_init_combines_legacy_in_order_and_preserves_authored_files(tmp_path: Path) -> None:
    folder = setup_templates(tmp_path)
    originals = {
        name: f"**Invented {name}:** *Keep wording* and [links](https://example.invalid).\n"
        for name in LEGACY
    }
    for name, text in originals.items():
        (folder / f"{name}.md").write_text(text)
    initialize(tmp_path)
    combined = folder / "commentary.md"
    assert combined.read_text() == (
        "#### Screening performance\n\n"
        + originals["roc"].rstrip("\n")
        + "\n\n"
        + originals["table"].rstrip("\n")
        + "\n\n#### Resource usage\n\n"
        + originals["resources"].rstrip("\n")
        + "\n\n#### Preparation\n\nPreparation commentary to be added.\n"
    )
    combined.write_text("An edited combined draft.\n")
    initialize(tmp_path)
    assert combined.read_text() == "An edited combined draft.\n"
    assert {name: (folder / f"{name}.md").read_text() for name in LEGACY} == originals


@pytest.mark.parametrize(
    "present",
    [
        ("roc",),
        ("table",),
        ("resources",),
        ("roc", "table"),
        ("roc", "resources"),
        ("table", "resources"),
    ],
)
def test_incomplete_legacy_set_creates_no_partial_drafts(
    tmp_path: Path, present: tuple[str, ...]
) -> None:
    folder = setup_templates(tmp_path)
    for name in present:
        (folder / f"{name}.md").write_text("Invented original.\n")
    with pytest.raises(ValueError, match="Cannot combine incomplete legacy commentary") as error:
        initialize(tmp_path)
    for name in set(LEGACY) - set(present):
        assert f"{name}.md" in str(error.value)
    assert not (folder / "commentary.md").exists()
    assert not (tmp_path / "report/abstract.md").exists()


def test_existing_combined_draft_ignores_incomplete_inactive_backups(tmp_path: Path) -> None:
    folder = setup_templates(tmp_path)
    (folder / "commentary.md").write_text("Keep my combined edits.\n")
    (folder / "roc.md").write_text("Inactive backup.\n")
    initialize(tmp_path)
    assert (folder / "commentary.md").read_text() == "Keep my combined edits.\n"


def test_fresh_init_preserves_archive_structure_but_build_does_not_require_it(
    tmp_path: Path,
) -> None:
    setup_templates(tmp_path)
    report = initialize(tmp_path)
    for slug in ("shark-products", "animal-products", "birds"):
        folder = report / "results" / slug
        source = (folder / "commentary.md").read_text()
        assert [line for line in source.splitlines() if line.startswith("#### ")] == [
            "#### Screening performance",
            "#### Resource usage",
            "#### Preparation",
        ]
        assert source.endswith("Preparation commentary to be added.\n")
        assert all(not (folder / f"{name}.md").exists() for name in LEGACY)
    for slug in ("shark-products", "animal-products", "birds"):
        (report / "results" / slug / "commentary.md").unlink()
    output = prepare_site(tmp_path)
    assert (output / "report.html").is_file()
    assert 'id="commentary-' not in (output / "report.html").read_text()
