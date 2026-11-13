"""Share the website frame between the project overview and report."""

import html
from datetime import date

# First public release date; deliberately independent of build or update time.
PUBLICATION_DATE = date(2026, 11, 13)


def website_frame(content: str, active: str) -> str:
    """Wrap trusted, locally generated content in the shared website navigation."""
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
        '<a class="brand" href="./">Wildlife-trade screening</a>'
        '<nav aria-label="Main navigation">'
        + navigation
        + '<a href="https://github.com/ldswaby/wildlife-trade-screen">Code '
        '<span aria-hidden="true">↗</span></a></nav></div></header>'
        '<div class="publication"><em>'
        f'<time datetime="{PUBLICATION_DATE.isoformat()}">'
        f"Published: {PUBLICATION_DATE:%d/%m/%Y}</time></em></div>"
        '<div class="page-layout"><main id="main" tabindex="-1">'
        + content
        + '</main><aside class="page-index" aria-label="Page index">'
        "<details open><summary>On this page</summary>"
        '<nav aria-label="On this page"></nav></details></aside></div>'
        '<footer class="site-footer">Wildlife-trade screening · '
        "Code and reproducible results</footer>"
    )
