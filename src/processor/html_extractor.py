from bs4 import BeautifulSoup

STRIP_TAGS = {"nav", "header", "footer", "script", "style", "noscript"}
STRIP_CLASSES = {"navbar", "nav", "footer", "header", "sidebar", "menu", "breadcrumb"}

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
    text = body.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return {"text": text, "chapters": chapters}
