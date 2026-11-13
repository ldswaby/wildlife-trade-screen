"""Compose authored Markdown and generated evidence in the shared report layout."""

import html
import re
from pathlib import Path
from urllib.parse import parse_qs

import markdown

from .evidence import DatasetEvidence
from .prompt_viewer import prompt_sources, prompt_viewer

DATASETS = {"animals": "Animal Products", "birds": "Live Birds", "sharks": "Shark Products"}
FILES = {"animals": "animal-products", "birds": "birds", "sharks": "shark-products"}
MODELS = {
    "gpt-6-luna-chat": "GPT-6 Luna Chat",
    "gpt-6-luna-decisions": "GPT-6 Luna Decisions",
    "microsoft-decision-1": "Microsoft-Decision-1",
    "d3": "d3",
}


SHARED_METHODS = {
    "experimental_setup": "experimental-setup.md",
    "data_summary": "data-summary.md",
    "inputs": "inputs.md",
    "prompt_overview": "prompts.md",
    "problem_formulation": "problem-formulation.md",
    "evaluation": "evaluation.md",
    "resources_reproducibility": "resources-reproducibility.md",
    "comparison_caveats": "published-comparison-caveats.md",
}

DATASET_LINK_TARGETS = {
    "figure-1": "roc-title",
    "table-1": "table-title",
    "figure-2": "resource-title",
}


def contextual_links(markup: str, panels: list[DatasetEvidence]) -> str:
    """Resolve authorable section links only to evidence selected in this build."""
    active = panels[0] if panels else None
    birds = next((panel for panel in panels if panel.task == "birds"), None)

    def resolve(match: re.Match[str]) -> str:
        marker, label = match.groups()
        panel = birds if marker == "birds-dataset" else active
        section = "prompts" if marker == "selected-prompts" else "data"
        if panel is None:
            message = (
                f"{DATASETS['birds']} results are not selected."
                if marker == "birds-dataset"
                else "No dataset results are selected."
            )
            return (
                f'<span class="context-link-unavailable">{label}</span> '
                f'<span class="subtle">({message})</span>'
            )
        if marker in DATASET_LINK_TARGETS:
            target = DATASET_LINK_TARGETS[marker]
            return (
                f'<a href="#{target}-{panel.index}" '
                f'data-active-dataset-target="{target}">{label}</a>'
            )
        context = f' data-active-dataset-section="{section}"' if marker != "birds-dataset" else ""
        return f'<a href="#dataset-{section}-{panel.index}"{context}>{label}</a>'

    def resolve_view(match: re.Match[str]) -> str:
        alias, query, label = match.groups()
        params = parse_qs(html.unescape(query))
        candidates = [panel for panel in panels if panel.task == params.get("dataset", [""])[0]]
        # Keep unresolved semantic links: the browser explains absent/ambiguous datasets
        # without substituting the active tab. Explicit links never follow tab changes.
        target = (
            f"{DATASET_LINK_TARGETS[alias]}-{candidates[0].index}"
            if len(candidates) == 1
            else alias
        )
        return f'<a href="#{target}?{query}">{label}</a>'

    markup = re.sub(
        r'<a href="#(figure-1|figure-2|table-1)\?([^"]*)">(.*?)</a>', resolve_view, markup
    )
    return re.sub(
        r'<a href="#(selected-dataset|selected-prompts|birds-dataset|'
        + "|".join(DATASET_LINK_TARGETS)
        + r')">(.*?)</a>',
        resolve,
        markup,
    )


def prose(directory: Path, relative: str) -> str:
    """Read trusted local authoring Markdown, including optional HTML formatting."""
    path = directory / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular draft {path}; run site/build.py init")
    return render_markdown(path.read_text(encoding="utf-8"))


def render_markdown(source: str) -> str:
    """Render trusted local prose, retaining authored HTML such as native MathML."""
    return str(markdown.markdown(source, extensions=["tables", "fenced_code"]))


MODEL_CARDS_MARKER = "{{ model_cards }}"


def model_cards_source(directory: Path) -> str:
    """Require one explicit, standalone widget insertion point before aggregation."""
    path = directory / "methods/models.md"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular draft {path}; run site/build.py init")
    source = path.read_text(encoding="utf-8")
    if source.count(MODEL_CARDS_MARKER) != 1 or MODEL_CARDS_MARKER not in (
        line.strip() for line in source.splitlines()
    ):
        raise ValueError(
            f"{path}: expected exactly one {MODEL_CARDS_MARKER} marker on its own line; "
            "place prose before and after it to surround the model viewer."
        )
    return source


def model_section(directory: Path) -> str:
    """Insert the existing viewer between independently rendered Markdown blocks."""
    before, after = model_cards_source(directory).split(MODEL_CARDS_MARKER)
    widget = _tabs(
        "models",
        [
            (
                label,
                f'<div class="model-card" id="model-{slug}">'
                + prose(directory, f"methods/models/{slug}.md")
                + "</div>",
            )
            for slug, label in MODELS.items()
        ],
    )
    return render_markdown(before) + widget + render_markdown(after)


DISTRIBUTION_MARKER = "{{ class_distribution }}"


def data_summary_source(directory: Path, relative: str) -> str:
    """Validate the explicit insertion point before building expensive evidence."""
    path = directory / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular draft {path}; run site/build.py init")
    source = path.read_text(encoding="utf-8")
    if source.count(DISTRIBUTION_MARKER) != 1:
        raise ValueError(
            f"{path}: expected exactly one {DISTRIBUTION_MARKER} marker; "
            "place it on its own line between Labels/policy and Caveats."
        )
    return source


def class_distribution(n: int, positives: int) -> str:
    """Describe normalised evaluation targets without changing recorded metrics."""
    if not 0 <= positives <= n:
        raise ValueError("Evaluation counts require 0 <= positives <= total")
    negatives = n - positives
    negative_percent = f"{100 * negatives / n:.1f}%" if n else "N/A"
    positive_percent = f"{100 * positives / n:.1f}%" if n else "N/A"
    rows = [
        ("Absolute count", f"{negatives:,}", f"{positives:,}", f"{n:,}"),
        ("Percentage (%)", negative_percent, positive_percent, "100%" if n else "N/A"),
    ]
    return (
        '<div class="table-wrap class-distribution" role="region" '
        'aria-label="Evaluation class distribution" tabindex="0"><table>'
        "<caption>Evaluation class distribution</caption><thead><tr>"
        '<th scope="col">Metric</th><th scope="col">Negative / not relevant (0)</th>'
        '<th scope="col">Positive / relevant (1)</th><th scope="col">Total</th>'
        "</tr></thead><tbody>"
        + "".join(
            f'<tr><th scope="row">{label}</th>'
            + "".join(f"<td>{value}</td>" for value in values)
            + "</tr>"
            for label, *values in rows
        )
        + '</tbody></table></div><p class="subtle">0 and 1 denote the normalised '
        "evaluation target under the selected label polarity: 0 = not relevant, "
        "1 = relevant. They need not match the original source label codes.</p>"
    )


def data_summary(directory: Path, relative: str, evidence: DatasetEvidence) -> str:
    """Insert validated class counts into the trusted authored dataset summary."""
    source = data_summary_source(directory, relative)
    # Render each side separately so the generated table cannot join a Markdown list.
    before, after = source.split(DISTRIBUTION_MARKER)
    return (
        str(markdown.markdown(before, extensions=["tables", "fenced_code"]))
        + class_distribution(evidence.n, evidence.positives)
        + str(markdown.markdown(after, extensions=["tables", "fenced_code"]))
    )


def _tabs(group: str, entries: list[tuple[str, str]], *, results: bool = False) -> str:
    buttons = []
    panels = []
    for index, (label, content) in enumerate(entries):
        panel_id = f"panel-{index}" if results else f"{group}-panel-{index}"
        selected = index == 0
        buttons.append(
            f'<button type="button" class="segment-tab{" tab" if results else ""}" '
            f'id="{group}-tab-{index}" role="tab" aria-controls="{panel_id}" '
            f'data-panel="{panel_id}" aria-selected="{str(selected).lower()}" '
            f'tabindex="{0 if selected else -1}">{html.escape(label)}</button>'
        )
        if not results:
            panels.append(
                f'<div role="tabpanel" id="{panel_id}" aria-labelledby="{group}-tab-{index}" '
                f'tabindex="0"{"" if selected else " hidden"}>{content}</div>'
            )
    return (
        f'<div class="report-tab-group" data-tab-group="{group}">'
        f'<div class="segment-scroll"><div class="segmented-tabs" role="tablist" '
        f'aria-label="{"Results datasets" if results else group.title()}">'
        + "".join(buttons)
        + "</div></div>"
        + "".join(panels)
        + "</div>"
    )


def dataset_text(markup: str, dataset_name: str, index: int) -> str:
    """Resolve authored panel text without touching prompt sources or provenance.

    Names are escaped for HTML, including hidden plain-text export guides. The
    short figure aliases resolve within this dataset, never another tab.
    """
    markup = markup.replace("{{ dataset_name }}", html.escape(dataset_name))
    for alias, target in DATASET_LINK_TARGETS.items():
        markup = markup.replace(f'href="#{alias}"', f'href="#{target}-{index}"')
    return markup


def dataset_content(
    directory: Path,
    evidence: DatasetEvidence,
    sources: dict[str, tuple[str, str | None]],
    *,
    active: bool,
    dataset_name: str,
) -> str:
    """Compose dataset introduction, details, prompt viewer and complete evidence."""
    index, task = evidence.index, evidence.task
    slug = FILES.get(task, task)

    def authored(markup: str) -> str:
        return dataset_text(markup, dataset_name, index)

    general = authored(prose(directory, f"results/{slug}.md"))
    introduction = authored(prose(directory, "results/dataset-introduction.md"))
    return (
        f'<section class="dataset-panel" id="panel-{index}" role="tabpanel" '
        f'aria-labelledby="results-tab-{index}" data-dataset-key="{html.escape(evidence.key)}" '
        f"{'' if active else 'hidden'}>"
        # Retain old dataset fragments without adding a redundant heading/index entry.
        f'<span id="dataset-{html.escape(evidence.key)}" class="dataset-anchor" '
        'aria-hidden="true"></span>'
        + introduction
        + f'<section aria-labelledby="dataset-data-{index}">'
        f'<h3 id="dataset-data-{index}">Dataset details</h3>'
        + authored(data_summary(directory, f"methods/data/{slug}.md", evidence))
        + "</section>"
        f'<section aria-labelledby="dataset-prompts-{index}">'
        f'<h3 id="dataset-prompts-{index}">Prompt viewer</h3>'
        + prompt_viewer(index, evidence.prompts, sources)
        + f'</section><section id="dataset-results-{index}" '
        f'aria-label="Results for {html.escape(dataset_name)}">'
        + general
        + authored(evidence.roc)
        + authored(evidence.table)
        + authored(evidence.resources)
        + evidence.provenance
        + f'<p class="subtle">Dataset: {html.escape(evidence.key)} · SHA-256: '
        + html.escape(evidence.sha256)
        + "</p></section></section>"
    )


def report_content(
    directory: Path, panels: list[DatasetEvidence], template: str, summary: str = ""
) -> str:
    """Keep shared prose outside a single, explicitly keyed dataset workspace."""
    order = {"sharks": 0, "animals": 1, "birds": 2}
    ordered = sorted(panels, key=lambda panel: (order.get(panel.task, 3), panel.key))
    sources = prompt_sources(directory.parent)
    buttons, content = [], []
    for position, panel in enumerate(ordered):
        label = DATASETS.get(panel.task, panel.task.title())
        if sum(other.task == panel.task for other in ordered) > 1:
            label += f" · {panel.key.split('-', 1)[-1]}"
        active = position == 0
        buttons.append(
            f'<button type="button" class="segment-tab tab" id="results-tab-{panel.index}" '
            f'role="tab" aria-controls="panel-{panel.index}" data-panel="panel-{panel.index}" '
            f'aria-selected="{str(active).lower()}" tabindex="{0 if active else -1}">'
            f"{html.escape(label)}</button>"
        )
        content.append(
            dataset_content(directory, panel, sources, active=active, dataset_name=label)
        )
    workspace = (
        '<div class="report-tab-group" data-tab-group="datasets">'
        '<div class="segment-scroll"><div class="segmented-tabs" role="tablist" '
        'aria-label="Datasets">' + "".join(buttons) + "</div></div>" + "".join(content) + "</div>"
        if ordered
        else "<p>No saved dataset results are selected yet.</p>"
    )
    replacements = {
        "report_title": "<h1>Wildlife-trade screening report</h1>",
        "abstract": contextual_links(prose(directory, "report-abstract.md"), ordered),
        "introduction": contextual_links(prose(directory, "introduction.md"), ordered),
        "model_section": contextual_links(model_section(directory), ordered),
        **{
            name: contextual_links(prose(directory, f"methods/{filename}"), ordered)
            for name, filename in SHARED_METHODS.items()
        },
        "overall_results": contextual_links(prose(directory, "overall-results.md"), ordered),
        "discussion": contextual_links(prose(directory, "discussion.md"), ordered),
        "conclusion": contextual_links(prose(directory, "conclusion.md"), ordered),
        "bibliography": contextual_links(prose(directory, "bibliography.md"), ordered),
        "dataset_workspace": workspace,
        "summary_figure": contextual_links(summary, ordered),
    }
    for name, value in replacements.items():
        marker = "{{ " + name + " }}"
        if template.count(marker) != 1:
            raise ValueError(f"Expected one {marker} marker in site/report.html")
        template = template.replace(marker, value)
    return template
