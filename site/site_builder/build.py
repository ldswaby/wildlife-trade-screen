"""Build the project website from authored drafts and explicitly selected results."""

import argparse
import hashlib
import html
import json
import shutil
from dataclasses import replace
from pathlib import Path

from wildlife_screening.snapshots import _read_selection, discover_runs

from .content import (
    DATASETS,
    FILES,
    MODELS,
    SHARED_METHODS,
    data_summary_source,
    model_cards_source,
    prose,
    report_content,
)
from .descriptions import load_descriptions
from .evidence import _html_document, build_evidence
from .layout import website_frame
from .selection import import_selection, resolved
from .summary import summary_data, summary_figure

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ("index.html", "styles.css", "site.js", "report.js")


def initialize(root: Path = ROOT) -> Path:
    """Create missing drafts, combining complete legacy commentary without overwrites."""
    directory = root / "report"
    templates = root / "site" / "content_templates"
    migrations: dict[Path, str] = {}
    # Validate every legacy set before creating any drafts: never substitute a
    # placeholder for a partially present discussion or lose an authored edit.
    for slug in FILES.values():
        target = directory / "results" / slug / "commentary.md"
        legacy = [target.with_name(f"{name}.md") for name in ("roc", "table", "resources")]
        if target.exists() or not any(path.exists() for path in legacy):
            continue
        missing = [str(path) for path in legacy if not path.is_file()]
        if missing:
            raise ValueError(
                "Cannot combine incomplete legacy commentary; missing: " + ", ".join(missing)
            )
        roc, table, resources = [path.read_text(encoding="utf-8").strip("\n") for path in legacy]
        migrations[target] = (
            f"#### Screening performance\n\n{roc}\n\n{table}\n\n"
            f"#### Resource usage\n\n{resources}\n\n"
            "#### Preparation\n\nPreparation commentary to be added.\n"
        )
    for source in sorted(templates.rglob("*")):
        if source.is_file():
            destination = directory / source.relative_to(templates)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with destination.open("x", encoding="utf-8") as handle:
                    handle.write(migrations.get(destination, source.read_text(encoding="utf-8")))
    return directory


def prepare_site(root: Path = ROOT) -> Path:
    """Package only website assets and aggregate results, never the outputs tree."""
    destination = root / "build" / "site"
    if any(path.is_symlink() for path in (root / "build", destination)):
        raise ValueError("Website output directories must not be symbolic links")
    allowed = {*ASSETS, "report.html"}
    if destination.exists() and any(
        path.name not in allowed or not path.is_file() or path.is_symlink()
        for path in destination.iterdir()
    ):
        raise ValueError("Unexpected files in build/site; inspect and move them before building")
    sources = [root / "site" / name for name in (*ASSETS, "report.html", "charts.js", "summary.js")]
    if not all(path.is_file() and not path.is_symlink() for path in sources):
        raise ValueError("Expected regular website assets in site/")
    directory = root / "report"
    # Validate active authored sections; archived dataset commentary is not a build input.
    descriptions = load_descriptions(directory)
    model_cards_source(directory)
    prose(directory, "results/dataset-introduction.md")
    for task in DATASETS:
        data_summary_source(directory, f"methods/data/{FILES[task]}.md")
        prose(directory, f"results/{FILES[task]}.md")
    for model in MODELS:
        prose(directory, f"methods/models/{model}.md")
    for filename in SHARED_METHODS.values():
        prose(directory, f"methods/{filename}")
    selection = _read_selection(directory / "selection.yaml")
    panels, data = build_evidence(resolved(selection, root), descriptions)
    panels = [
        replace(
            panel, provenance=panel.provenance.replace(html.escape(str(root.resolve()) + "/"), "")
        )
        for panel in panels
    ]
    arranged = report_content(
        directory,
        panels,
        (root / "site" / "report.html").read_text(encoding="utf-8"),
        summary_figure(summary_data(panels, data), descriptions),
    )
    rendered = _html_document(
        arranged, json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    )
    template = (root / "site" / "index.html").read_text(encoding="utf-8")
    if template.count("{{ site_content }}") != 1:
        raise ValueError("Expected one site_content marker in site/index.html")
    index = template.replace(
        "{{ site_content }}", website_frame(prose(directory, "background.md"), "Overview")
    )
    for name in ("styles.css", "site.js", "report.js"):
        version = hashlib.sha256((root / "site" / name).read_bytes()).hexdigest()[:12]
        index = index.replace(f'"{name}"', f'"{name}?v={version}"')
        rendered = rendered.replace(f'"{name}"', f'"{name}?v={version}"')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ASSETS[1:]:
        shutil.copyfile(root / "site" / name, destination / name)
    (destination / "index.html").write_text(index, encoding="utf-8")
    (destination / "report.html").write_text(rendered, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    """Expose offline initialisation, explicit result imports, discovery and building."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create missing report source files")
    commands.add_parser("build", help="Rebuild the project website without inference")
    discover = commands.add_parser("discover", help="List saved snapshots without selecting them")
    discover.add_argument("--runs-root", type=Path, default=Path("outputs/comparison"))
    run = commands.add_parser("import-run", help="Select one immutable run manifest")
    run.add_argument("manifest", type=Path)
    run.add_argument("--data", type=Path, help="Local evaluation CSV for a custom dataset")
    run.add_argument("--positive-label", choices=("0", "1"))
    run.add_argument(
        "--label-review", type=Path, help="Required only for historical review-backed runs"
    )
    candidate = commands.add_parser(
        "import-candidate", help="Merge a campaign's exact snapshot references"
    )
    candidate.add_argument("selection", type=Path)
    args = parser.parse_args(argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise ValueError("Run site/build.py from the project root")
        if args.command == "init":
            print(f"Report source: {initialize()}")
        elif args.command == "build":
            print(f"Built project website: {prepare_site()}")
        elif args.command == "discover":
            print(json.dumps(discover_runs(args.runs_root), indent=2))
        else:
            count = import_selection(
                args.selection if args.command == "import-candidate" else args.manifest,
                ROOT / "report" / "selection.yaml",
                ROOT,
                candidate=args.command == "import-candidate",
                data=getattr(args, "data", None),
                positive_label=getattr(args, "positive_label", None),
                label_review=getattr(args, "label_review", None),
            )
            print(f"Added {count} snapshot reference(s); run site/build.py build to refresh")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot prepare website: {exc}\n")
