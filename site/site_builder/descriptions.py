"""Load editable shared Results wording independently of numerical evidence."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import markdown


class _Text(HTMLParser):
    """Collect readable text for accessible titles and caption validation."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "li", "dt", "dd", "div"}:
            self.parts.append(" ")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append(" ")


def _plain_text(markup: str) -> str:
    parser = _Text()
    parser.feed(markup)
    return " ".join("".join(parser.parts).split())


def _markdown(source: str) -> str:
    return str(markdown.markdown(source, extensions=["tables", "fenced_code", "def_list"]))


@dataclass(frozen=True)
class Caption:
    """Keep the authored title and body separate from generated figure structure."""

    title_html: str
    title_text: str
    body_html: str


@dataclass(frozen=True)
class ResultDescriptions:
    """Require all shared captions explicitly, without fallbacks."""

    roc: Caption
    table: Caption
    resources: Caption
    table_gpt4_note: str
    summary: Caption


def load_descriptions(directory: Path) -> ResultDescriptions:
    """Validate shared Markdown before snapshot aggregation or destination writes.

    Args:
        directory: Private report directory containing results/shared.

    Returns:
        Rendered titles, captions and the conditional note.
    """
    shared = directory / "results/shared"

    def read(name: str) -> str:
        path = shared / f"{name}.md"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular draft {path}; run site/build.py init")
        source = path.read_text(encoding="utf-8").strip()
        if not source:
            raise ValueError(f"{path}: add nonempty Markdown text, then rebuild")
        return source

    def caption(name: str) -> Caption:
        source = read(name)
        heading, _, body = source.partition("\n")
        if not heading.startswith("# ") or not heading[2:].strip() or not body.strip():
            raise ValueError(
                f"{shared / (name + '.md')}: start with # Title followed by a nonempty caption"
            )
        body_html = _markdown(body.strip())
        if re.search(r"<h1(?:\s[^>]*)?>", body_html):
            raise ValueError(f"{shared / (name + '.md')}: use only one # Title heading")
        title_html = _markdown(heading).removeprefix("<h1>").removesuffix("</h1>")
        title_text = _plain_text(title_html)
        if not title_text:
            raise ValueError(f"{shared / (name + '.md')}: add a readable # Title")
        if not _plain_text(body_html):
            raise ValueError(f"{shared / (name + '.md')}: add a nonempty readable caption")
        return Caption(title_html, title_text, body_html)

    return ResultDescriptions(
        summary=caption("summary"),
        roc=caption("roc"),
        table=caption("table"),
        resources=caption("resources"),
        table_gpt4_note=_markdown(read("table-gpt4-note")),
    )
