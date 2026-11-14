"""Share the website frame between the project overview and report."""

import html
import re
from datetime import date

# First public release date; deliberately independent of build or update time.
PUBLICATION_DATE = date(2026, 11, 13)
PROJECT_TITLE = "Decision models for zero-shot wildlife advert screening"


def website_frame(content: str, active: str) -> str:
    """Wrap trusted, locally generated content in the shared website navigation."""
    hero = ""
    title = re.search(r"<h1(?:\s[^>]*)?>.*?</h1>", content, flags=re.DOTALL)
    if title:
        heading = title.group()
        remainder = content[title.end() :]
        if active == "Overview":
            lead = re.match(r'\s*<p class="lead">.*?</p>', remainder, flags=re.DOTALL)
            if lead:
                heading += lead.group()
                remainder = remainder[lead.end() :]
        # Move authored heading markup intact: one accessible title, no duplicate copy.
        content = content[: title.start()] + remainder
        hero = '<div class="page-hero"><div class="hero-inner">' + heading + "</div></div>"
    links = (("Overview", "./"), ("Report", "report.html"))
    navigation = "".join(
        f'<a href="{url}"'
        + (' aria-current="page"' if name == active else "")
        + f">{html.escape(name)}</a>"
        for name, url in links
    )
    return (
        '<a class="skip" href="#main">Skip to content</a>'
        '<header class="site-header"><div class="header-inner">'
        f'<a class="brand" href="./">{html.escape(PROJECT_TITLE)}</a>'
        '<nav aria-label="Main navigation">'
        + navigation
        + '<a href="https://github.com/ldswaby/wildlife-trade-screen">Code '
        '<span aria-hidden="true">↗</span></a></nav></div></header>'
        '<main id="main" tabindex="-1">'
        '<aside class="page-index" aria-label="Page index">'
        "<details open><summary>On this page</summary>"
        '<nav aria-label="On this page"></nav></details></aside>'
        + hero
        + '<div class="publication"><em>'
        f'<time datetime="{PUBLICATION_DATE.isoformat()}">'
        f"Published: {PUBLICATION_DATE:%d/%m/%Y}</time></em></div>"
        '<div class="page-layout"><div class="page-content">' + content + "</div></div></main>"
        f'<footer class="site-footer">{html.escape(PROJECT_TITLE)} · '
        "Code and reproducible results</footer>"
    )
