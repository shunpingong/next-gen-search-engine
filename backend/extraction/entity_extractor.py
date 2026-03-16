from __future__ import annotations

import re

from utils.text_utils import PERSON_NAME_PATTERN, filter_candidate_names, split_sentences

THANK_PATTERNS = (
    re.compile(
        r"\b(?:thank|thanks|grateful to|gratitude to|appreciate|indebted to)\s+"
        r"(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?"
        r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})"
    ),
    re.compile(
        r"\b(?:to|for)\s+"
        r"(?:Dr\.?\s+|Prof\.?\s+|Professor\s+)?"
        r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})"
        r"\b(?:,|\s+who|\s+for|\s+whose)"
    ),
)


def extract_person_entities(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in THANK_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(match.group(1))

    for sentence in split_sentences(text):
        if not re.search(r"\b(?:thank|grateful|gratitude|appreciate)\b", sentence, re.IGNORECASE):
            continue
        for match in PERSON_NAME_PATTERN.finditer(sentence):
            candidates.append(match.group(1))

    return filter_candidate_names(candidates)


def extract_name_candidates_with_evidence(text: str) -> list[tuple[str, str]]:
    names = extract_person_entities(text)
    sentences = split_sentences(text)
    supported: list[tuple[str, str]] = []
    for name in names:
        evidence = next((sentence for sentence in sentences if name in sentence), "")
        supported.append((name, evidence))
    return supported
