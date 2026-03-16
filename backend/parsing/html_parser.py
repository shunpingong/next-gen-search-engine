from __future__ import annotations

import re
from urllib.parse import urljoin

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


from utils.text_utils import normalize_document_text


def parse_html_document(html: str, url: str) -> tuple[str, dict[str, object]]:
    text = ""
    title = ""
    headings: list[str] = []
    doi = ""
    authors: list[str] = []
    links: list[dict[str, str]] = []

    if trafilatura is not None:
        try:
            text = (
                trafilatura.extract(
                    html,
                    output_format="txt",
                    include_links=False,
                    include_images=False,
                    favor_precision=True,
                )
                or ""
            )
        except Exception:
            text = ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if title_tag is not None:
            title = title_tag.get_text(" ", strip=True)
        for meta in soup.find_all("meta"):
            key = (meta.get("name") or meta.get("property") or meta.get("itemprop") or "").strip().lower()
            value = (meta.get("content") or "").strip()
            if not key or not value:
                continue
            if key in {"citation_author", "dc.creator", "author"}:
                if value not in authors:
                    authors.append(value)
                continue
            if key in {"citation_doi", "dc.identifier", "dc.identifier.doi", "prism.doi", "doi"}:
                doi = value
                continue
        headings = [
            heading.get_text(" ", strip=True)
            for level in range(1, 5)
            for heading in soup.find_all(f"h{level}")[:6]
        ]
        seen_links: set[tuple[str, str]] = set()
        for anchor in soup.find_all("a", href=True)[:160]:
            href = (anchor.get("href") or "").strip()
            text_value = anchor.get_text(" ", strip=True)
            if not href or not text_value:
                continue
            if href.startswith(("#", "javascript:", "mailto:")):
                continue
            absolute_url = urljoin(url, href)
            dedupe_key = (absolute_url.lower(), text_value.lower())
            if dedupe_key in seen_links:
                continue
            seen_links.add(dedupe_key)
            links.append({"text": text_value, "url": absolute_url})
        if not text:
            paragraphs = [
                paragraph.get_text(" ", strip=True)
                for paragraph in soup.find_all(["p", "li"])[:120]
                if paragraph.get_text(" ", strip=True)
            ]
            text = "\n".join(paragraphs)

    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    cleaned = normalize_document_text(cleaned)
    return cleaned, {"title": title, "headings": headings, "url": url, "doi": doi, "authors": authors, "links": links[:120]}
