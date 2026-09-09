from bs4 import BeautifulSoup

STRIP_TAGS = {"nav", "header", "footer", "script", "style", "noscript"}
STRIP_CLASSES = {"navbar", "nav", "footer", "header", "sidebar", "menu", "breadcrumb"}

# Elements whose end reads as a paragraph break rather than a line break.
BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "pre", "li", "tr", "table",
    "h1", "h2", "h3", "h4", "h5", "h6",
}

# Table cells need separating from each other, but not onto their own lines.
CELL_TAGS = {"td", "th"}


def _collapse_blank_lines(text: str) -> str:
    """Strip each line and reduce runs of blank lines to one paragraph break."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    return "\n".join(out).strip()


def extract_text_from_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": lambda c: c and any(s in str(c).lower() for s in STRIP_CLASSES)}):
        tag.decompose()
    chapters = []
    for heading in soup.find_all(["h1", "h2", "h3"]):
        text = heading.get_text(strip=True)
        if text:
            chapters.append(text)
    body = soup.find("body")
    if body is None:
        return {"text": "", "chapters": []}
    # Mark boundaries explicitly, then flatten without a separator. Passing
    # separator="\n" breaks on every string, which splits a sentence wherever it
    # contains inline markup ("<i>as rendered</i>", a linked name) and leaves a
    # paragraph break indistinguishable from a wrapped line.
    for tag in body.find_all("br"):
        tag.replace_with("\n")
    for tag in body.find_all(CELL_TAGS):
        tag.insert_after("\t")
    for tag in body.find_all(BLOCK_TAGS):
        tag.insert_after("\n\n")
    text = body.get_text()
    return {"text": _collapse_blank_lines(text), "chapters": chapters}
