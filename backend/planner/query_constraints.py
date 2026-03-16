from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.text_utils import lexical_relevance_score, normalize_whitespace

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

SAMPLE_ENTITY_HINTS = (
    "couple",
    "couples",
    "participant",
    "participants",
    "subject",
    "subjects",
    "adult",
    "adults",
    "people",
    "respondent",
    "respondents",
)

AUTHOR_LINE_NAME_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3}\b"
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_YEAR_PATTERN = re.compile(
    r"\b("
    + "|".join(MONTH_NAMES)
    + r")\s+(19|20)\d{2}\b",
    re.IGNORECASE,
)
AUTHOR_COUNT_PATTERN = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+authors?\b",
    re.IGNORECASE,
)
SAMPLE_RANGE_PATTERN = re.compile(
    r"\b(?:sample(?: size)?|participants?|subjects?|couples?|respondents?|adults?|people)\b"
    r"[^.]{0,80}?\b(?:between|ranging from)\s+(\d{2,5})\s+(?:and|to|-)\s+(\d{2,5})\b",
    re.IGNORECASE,
)
REFERENCE_RANGE_PATTERN = re.compile(
    r"\breferences?(?: cited)?[^.]{0,120}?\bbetween\s+(\d{4})\s+(?:and|to|-)\s+(\d{4})\b",
    re.IGNORECASE,
)
INSTITUTION_RANGE_PATTERN = re.compile(
    r"\b(?:university|institution|college|school)[^.]{0,80}?\bfounded\b[^.]{0,60}?\bbetween\s+(\d{4})\s+(?:and|to|-)\s+(\d{4})\b",
    re.IGNORECASE,
)
PUBLICATION_YEAR_PATTERN = re.compile(
    r"\b(?:as of|published|dated|from|in)\s+(?:"
    + "|".join(MONTH_NAMES)
    + r"\s+)?((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
SAMPLE_SIZE_PATTERNS = (
    re.compile(r"\bN\s*=\s*(\d{2,5})\b", re.IGNORECASE),
    re.compile(
        r"\b(?:sample of|included|examined|surveyed|studied|followed)\s+(\d{2,5})\s+"
        r"(?:married\s+|cohabiting\s+|older\s+|young\s+)?"
        r"(?:couples?|participants?|subjects?|adults?|people|respondents?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{2,5})\s+(?:married\s+|cohabiting\s+|older\s+|young\s+)?"
        r"(?:couples?|participants?|subjects?|adults?|people|respondents?)\b",
        re.IGNORECASE,
    ),
)
FOUNDED_YEAR_PATTERN = re.compile(
    r"\b(?:founded|established|chartered)\s+(?:in\s+)?((?:19|20)\d{2})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryConstraints:
    author_count: int | None = None
    sample_min: int | None = None
    sample_max: int | None = None
    sample_terms: tuple[str, ...] = field(default_factory=tuple)
    publication_year: int | None = None
    publication_month: str = ""
    reference_year_min: int | None = None
    reference_year_max: int | None = None
    institution_year_min: int | None = None
    institution_year_max: int | None = None


@dataclass(frozen=True)
class ConstraintAssessment:
    score: float
    matched: tuple[str, ...] = field(default_factory=tuple)
    missing: tuple[str, ...] = field(default_factory=tuple)
    contradicted: tuple[str, ...] = field(default_factory=tuple)


def parse_query_constraints(query: str) -> QueryConstraints:
    query_lower = normalize_whitespace(query).lower()

    author_count = None
    author_match = AUTHOR_COUNT_PATTERN.search(query_lower)
    if author_match:
        author_count = _parse_int(author_match.group("count"))

    sample_min = None
    sample_max = None
    sample_match = SAMPLE_RANGE_PATTERN.search(query_lower)
    if sample_match:
        first = int(sample_match.group(1))
        second = int(sample_match.group(2))
        sample_min, sample_max = sorted((first, second))

    publication_month = ""
    publication_year = None
    month_year_match = MONTH_YEAR_PATTERN.search(query_lower)
    if month_year_match:
        publication_month = month_year_match.group(1).lower()
        publication_year = int(month_year_match.group(2))
    else:
        publication_match = PUBLICATION_YEAR_PATTERN.search(query_lower)
        if publication_match:
            publication_year = int(publication_match.group(1))

    reference_year_min = None
    reference_year_max = None
    reference_match = REFERENCE_RANGE_PATTERN.search(query_lower)
    if reference_match:
        first = int(reference_match.group(1))
        second = int(reference_match.group(2))
        reference_year_min, reference_year_max = sorted((first, second))

    institution_year_min = None
    institution_year_max = None
    institution_match = INSTITUTION_RANGE_PATTERN.search(query_lower)
    if institution_match:
        first = int(institution_match.group(1))
        second = int(institution_match.group(2))
        institution_year_min, institution_year_max = sorted((first, second))

    sample_terms = tuple(
        term
        for term in SAMPLE_ENTITY_HINTS
        if term in query_lower
    )

    return QueryConstraints(
        author_count=author_count,
        sample_min=sample_min,
        sample_max=sample_max,
        sample_terms=sample_terms,
        publication_year=publication_year,
        publication_month=publication_month,
        reference_year_min=reference_year_min,
        reference_year_max=reference_year_max,
        institution_year_min=institution_year_min,
        institution_year_max=institution_year_max,
    )


def assess_document_constraints(
    query: str,
    title: str,
    content: str,
    *,
    reference_text: str = "",
    metadata: dict[str, object] | None = None,
) -> ConstraintAssessment:
    constraints = parse_query_constraints(query)
    metadata = metadata or {}
    authors = metadata.get("authors") or ()
    author_text = " ".join(author for author in authors if isinstance(author, str))
    published_date = normalize_whitespace(str(metadata.get("published_date", "")))
    front_matter = normalize_whitespace(f"{title} {author_text} {published_date} {(content or '')[:4000]}")
    reference_text = normalize_whitespace(reference_text)

    matched: list[str] = []
    missing: list[str] = []
    contradicted: list[str] = []

    score = 0.25 * lexical_relevance_score(query, front_matter)

    if constraints.publication_year is not None:
        publication_assessment = _assess_publication_date(
            constraints.publication_year,
            constraints.publication_month,
            front_matter,
            published_date,
        )
        score += publication_assessment[0]
        _append_status(publication_assessment[1], "publication_date", matched, missing, contradicted)

    if constraints.author_count is not None:
        author_assessment = _assess_author_count(constraints.author_count, content, metadata)
        score += author_assessment[0]
        _append_status(author_assessment[1], "author_count", matched, missing, contradicted)

    if constraints.sample_min is not None and constraints.sample_max is not None:
        sample_assessment = _assess_sample_range(
            constraints.sample_min,
            constraints.sample_max,
            constraints.sample_terms,
            content,
        )
        score += sample_assessment[0]
        _append_status(sample_assessment[1], "sample_size", matched, missing, contradicted)

    if constraints.reference_year_min is not None and constraints.reference_year_max is not None:
        reference_assessment = _assess_reference_year_range(
            constraints.reference_year_min,
            constraints.reference_year_max,
            reference_text,
        )
        score += reference_assessment[0]
        _append_status(reference_assessment[1], "reference_year", matched, missing, contradicted)

    if constraints.institution_year_min is not None and constraints.institution_year_max is not None:
        institution_assessment = _assess_institution_range(
            constraints.institution_year_min,
            constraints.institution_year_max,
            content,
        )
        score += institution_assessment[0]
        _append_status(institution_assessment[1], "institution_year", matched, missing, contradicted)

    contradiction_penalty = min(0.45, 0.12 * len(contradicted))
    score = max(0.0, min(1.0, score - contradiction_penalty))
    return ConstraintAssessment(
        score=round(score, 4),
        matched=tuple(matched),
        missing=tuple(missing),
        contradicted=tuple(contradicted),
    )


def _append_status(
    status: str,
    label: str,
    matched: list[str],
    missing: list[str],
    contradicted: list[str],
) -> None:
    if status == "matched":
        matched.append(label)
    elif status == "contradicted":
        contradicted.append(label)
    elif status == "missing":
        missing.append(label)


def _assess_publication_date(
    target_year: int,
    target_month: str,
    front_matter: str,
    published_date: str,
) -> tuple[float, str]:
    search_text = normalize_whitespace(f"{published_date} {front_matter}")
    month_year_pairs = {
        (match.group(1).lower(), int(match.group(2)))
        for match in MONTH_YEAR_PATTERN.finditer(search_text)
    }
    years = {int(match.group(0)) for match in YEAR_PATTERN.finditer(search_text[:1800])}

    if target_month and (target_month, target_year) in month_year_pairs:
        return 0.24, "matched"
    if target_year in years:
        return 0.16, "matched"
    if years:
        return -0.18, "contradicted"
    return 0.0, "missing"


def _assess_author_count(
    expected_count: int,
    content: str,
    metadata: dict[str, object],
) -> tuple[float, str]:
    inferred_count = infer_author_count(content, metadata)
    if inferred_count is None:
        return 0.0, "missing"
    if inferred_count == expected_count:
        return 0.14, "matched"
    return -0.14, "contradicted"


def _assess_sample_range(
    minimum: int,
    maximum: int,
    sample_terms: tuple[str, ...],
    content: str,
) -> tuple[float, str]:
    sample_sizes = extract_sample_sizes(content, sample_terms=sample_terms)
    if not sample_sizes:
        return 0.0, "missing"
    if any(minimum <= size <= maximum for size in sample_sizes):
        return 0.2, "matched"
    return -0.22, "contradicted"


def _assess_reference_year_range(
    minimum: int,
    maximum: int,
    reference_text: str,
) -> tuple[float, str]:
    if not reference_text:
        return 0.0, "missing"
    years = [int(match.group(0)) for match in YEAR_PATTERN.finditer(reference_text)]
    if not years:
        return 0.0, "missing"
    if any(minimum <= year <= maximum for year in years):
        return 0.1, "matched"
    return -0.06, "contradicted"


def _assess_institution_range(
    minimum: int,
    maximum: int,
    content: str,
) -> tuple[float, str]:
    years = [int(match.group(1)) for match in FOUNDED_YEAR_PATTERN.finditer(content[:4000])]
    if not years:
        return 0.0, "missing"
    if any(minimum <= year <= maximum for year in years):
        return 0.08, "matched"
    return -0.08, "contradicted"


def infer_author_count(content: str, metadata: dict[str, object] | None = None) -> int | None:
    metadata = metadata or {}
    authors = metadata.get("authors")
    if isinstance(authors, (list, tuple)):
        normalized_authors = [normalize_whitespace(str(author)) for author in authors if normalize_whitespace(str(author))]
        if normalized_authors:
            return len(normalized_authors)

    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    for line in lines[:12]:
        if len(line) > 180:
            continue
        names = AUTHOR_LINE_NAME_PATTERN.findall(line)
        unique_names = _unique_preserve_order(name for name in names if "abstract" not in name.lower())
        if 2 <= len(unique_names) <= 6 and ("," in line or " and " in line.lower()):
            return len(unique_names)
    return None


def extract_sample_sizes(text: str, *, sample_terms: tuple[str, ...] = ()) -> list[int]:
    text = text or ""
    sizes: list[int] = []
    lowered_text = text.lower()
    for pattern in SAMPLE_SIZE_PATTERNS:
        for match in pattern.finditer(text[:8000]):
            value = int(match.group(1))
            if value < 20:
                continue
            window = lowered_text[max(0, match.start() - 40) : min(len(lowered_text), match.end() + 60)]
            if sample_terms and not any(term in window for term in sample_terms):
                # Keep general N= counts, but otherwise prefer context words that match the query.
                if not match.group(0).lower().startswith("n"):
                    continue
            sizes.append(value)
    return _unique_ints(sizes)


def _parse_int(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique_values: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        key = normalize_whitespace(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_values.append(normalize_whitespace(value))
    return unique_values
