"""Preview model-facing prompts without rendering evaluation adverts."""

import hashlib
import html
from pathlib import Path

import yaml

from wildlife_screening.prompts import Prompt


class _PreviewDumper(yaml.SafeDumper):
    """Keep assembled multiline instructions readable without changing their value."""


def _text(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    )


_PreviewDumper.add_representer(str, _text)


def prompt_sources(root: Path) -> dict[str, tuple[str, str | None]]:
    """Index model-facing previews of regular sources by their complete byte hash.

    File names and version labels cannot establish historical identity. Only an
    exact hash match can be presented as the source used by a selected snapshot.
    """
    directory = root / "prompts"
    sources: dict[str, tuple[str, str | None]] = {}
    for path in sorted(directory.rglob("*")):
        if path.suffix not in {".txt", ".yaml"} or not path.is_file() or path.is_symlink():
            continue
        if not path.resolve().is_relative_to(directory.resolve()):
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        preview: str | None = content.decode("utf-8")
        if path.suffix == ".yaml":
            try:
                prompt = Prompt.load(path)
                # Never associate a changed file with an earlier catalogued hash.
                if prompt.sha256 != digest:
                    continue
                question = prompt.question()
                preview = yaml.dump(
                    {"instructions": question["instructions"], "criteria": question["criteria"]},
                    Dumper=_PreviewDumper,
                    allow_unicode=True,
                    sort_keys=False,
                    width=88,
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
                preview = None
        sources.setdefault(digest, (str(path.relative_to(root)), preview))
    return sources


def prompt_viewer(
    index: int, records: list[dict[str, str]], sources: dict[str, tuple[str, str | None]]
) -> str:
    """Render escaped text/YAML sources with independent, mutually exclusive choices."""
    controls = []
    for field, title, values in (
        ("family", "Model type", (("decision", "Decision"), ("llm", "Chat"))),
        ("condition", "Prompt type", (("zero-shot", "Zero-shot"), ("few-shot", "Few-shot"))),
    ):
        controls.append(
            f'<fieldset class="prompt-{field}"><legend>{title}</legend>'
            + "".join(
                '<label class="selection-box">'
                f'<input type="radio" name="prompt-{field}-{index}" value="{value}">'
                f"<span>{label}</span></label>"
                for value, label in values
            )
            + "</fieldset>"
        )
    cards = []
    for number, record in enumerate(
        sorted(records, key=lambda item: (item["version"], item["sha256"]))
    ):
        digest = record["sha256"]
        source = sources.get(digest)
        expected = ".txt" if record["family"] == "llm" else ".yaml"
        if source and not source[0].endswith(expected):
            source = None
        content = (
            f'<pre tabindex="0"><code class="language-{"text" if expected == ".txt" else "yaml"}">'
            + html.escape(source[1])
            + "</code></pre>"
            if source and source[1] is not None
            else '<p class="prompt-unavailable">The saved prompt source is unavailable. '
            "No current file matches this experiment's prompt hash.</p>"
        )
        if source and source[1] is None:
            content = (
                '<p class="prompt-unavailable">The matched saved prompt definition cannot '
                "be rendered as a model-facing preview. Check its source definition; "
                "no substitute prompt is shown.</p>"
            )
        attrs = " ".join(
            f'data-{key}="{html.escape(record[key], quote=True)}"'
            for key in ("family", "condition", "version", "sha256")
        )
        cards.append(
            f'<div class="prompt-source" id="prompt-source-{index}-{number}" {attrs} hidden>'
            + content
            + "<details><summary>Prompt provenance</summary>"
            + (f"<p>Source: {html.escape(source[0])}</p>" if source else "")
            + f"<p>Version: {html.escape(record['version'])}<br>SHA-256: "
            + html.escape(digest)
            + "</p></details></div>"
        )
    return (
        '<div class="prompt-viewer"><div class="prompt-options">'
        + "".join(controls)
        + '</div><p class="subtle">Advert text is supplied separately and is not shown here.</p>'
        '<label class="prompt-version" hidden>Prompt version '
        '<select aria-label="Prompt version"></select></label>'
        '<p class="prompt-message subtle" role="status" hidden></p>' + "".join(cards) + "</div>"
    )
