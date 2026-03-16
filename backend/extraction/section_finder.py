from __future__ import annotations

import re

from agent.models import SectionMatch
from utils.text_utils import (
    ACKNOWLEDGEMENT_SECTION_TITLES,
    REFERENCE_SECTION_TITLES,
    is_heading_candidate,
    normalize_heading,
)

SECTION_END_TITLES = {
    "abstract",
    "introduction",
    "table of contents",
    "contents",
    "references",
    "bibliography",
    "appendix",
    "chapter 1",
    "chapter one",
    "dedication",
}


def find_named_section(text: str, titles: tuple[str, ...]) -> SectionMatch | None:
    if not text:
        return None

    normalized_titles = {normalize_heading(title) for title in titles}
    lines = [line.rstrip() for line in re.split(r"\r?\n", text) if line.strip()]
    if not lines:
        return None

    for index, line in enumerate(lines):
        heading_key = normalize_heading(line.strip().strip(":"))
        if heading_key not in normalized_titles:
            continue

        inline_text = ""
        if ":" in line:
            prefix, suffix = line.split(":", 1)
            if normalize_heading(prefix) in normalized_titles:
                inline_text = suffix.strip()

        end_index = len(lines)
        for next_index in range(index + 1, len(lines)):
            next_heading = normalize_heading(lines[next_index].strip().strip(":"))
            if next_heading in SECTION_END_TITLES:
                end_index = next_index
                break
            if next_index > index + 1 and is_heading_candidate(lines[next_index]):
                end_index = next_index
                break

        content_lines = [inline_text] if inline_text else []
        content_lines.extend(lines[index + 1 : end_index])
        section_text = "\n".join(line for line in content_lines if line).strip()
        if section_text:
            return SectionMatch(
                heading=line.strip(),
                text=section_text,
                start_index=index,
                end_index=end_index,
            )

    inline_pattern = re.compile(
        r"(?is)\b(acknowledgements|acknowledgments|acknowledgement)\b\s*[:\-]\s*(.+)"
    )
    match = inline_pattern.search(text)
    if not match:
        return None

    tail = match.group(2)
    next_heading = re.search(
        r"(?im)^\s*(abstract|introduction|references|bibliography|appendix|chapter\s+\d+)\s*$",
        tail,
    )
    section_text = tail[: next_heading.start()].strip() if next_heading else tail.strip()
    if not section_text:
        return None
    return SectionMatch(
        heading=match.group(1),
        text=section_text,
        start_index=match.start(),
        end_index=match.end(),
    )


def find_acknowledgements_section(text: str) -> SectionMatch | None:
    return find_named_section(text, ACKNOWLEDGEMENT_SECTION_TITLES)


def find_references_section(text: str) -> SectionMatch | None:
    return find_named_section(text, REFERENCE_SECTION_TITLES)
