from __future__ import annotations

import re
from io import BytesIO

from utils.text_utils import normalize_document_text

try:
    import fitz
except ImportError:
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None

MAX_PDF_TEXT_CHARS = 200_000


def parse_pdf_document(pdf_bytes: bytes) -> tuple[str, dict[str, object]]:
    chunks: list[str] = []
    title = ""
    page_count = 0

    if fitz is not None:
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                page_count = document.page_count
                title = document.metadata.get("title", "") or ""
                total_chars = 0
                for page in document:
                    page_text = page.get_text("text") or ""
                    if not page_text.strip():
                        continue
                    chunks.append(page_text)
                    total_chars += len(page_text)
                    if total_chars >= MAX_PDF_TEXT_CHARS:
                        break
        except Exception:
            chunks = []

    combined = "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
    if len(combined) < 200 and pdfminer_extract_text is not None:
        try:
            combined = pdfminer_extract_text(BytesIO(pdf_bytes)) or combined
        except Exception:
            pass

    combined = re.sub(r"\n{3,}", "\n\n", combined.strip())
    combined = normalize_document_text(combined)
    return combined, {"title": title, "page_count": page_count}
