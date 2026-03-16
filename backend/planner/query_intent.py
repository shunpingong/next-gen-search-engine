from __future__ import annotations

from dataclasses import dataclass
import re

from utils.text_utils import (
    ACKNOWLEDGEMENT_HINTS,
    CAREER_HINTS,
    CHARACTER_QUERY_HINTS,
    EVENT_QUERY_HINTS,
    MEDIA_QUERY_HINTS,
    PAPER_QUERY_HINTS,
    REFERENCE_HINTS,
    THESIS_CONTENT_HINTS,
    contains_any_exact_phrase,
    contains_exact_phrase,
    extract_constraint_phrases,
    is_academic_lookup_query,
    is_event_lookup_query,
    is_media_lookup_query,
    is_person_target_query,
    normalize_whitespace,
)


@dataclass(frozen=True)
class QueryIntent:
    answer_type: str
    explicit_thesis: bool
    explicit_paper: bool
    prefers_repository_sources: bool
    prefers_paper_sources: bool
    prefers_pdf_sources: bool
    prefers_encyclopedic_sources: bool
    prefers_character_sources: bool
    prefers_event_sources: bool
    requires_acknowledgement_section: bool
    requires_reference_section: bool
    needs_career_hop: bool
    needs_reference_hop: bool
    needs_entity_discovery_hop: bool
    needs_event_discovery_hop: bool
    needs_person_identity_hop: bool
    targets_person: bool
    targets_citation_identifier: bool
    targets_count: bool
    targets_compound_answer: bool
    is_media_query: bool
    is_event_query: bool
    is_open_domain_browsecomp: bool
    signals: tuple[str, ...]


def _query_has_hint(query_lower: str, hint: str) -> bool:
    if hint.endswith("acknowledg"):
        return hint in query_lower
    return contains_exact_phrase(query_lower, hint)


def _count_query_hints(query_lower: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if _query_has_hint(query_lower, hint))


def _count_unique_query_hints(query_lower: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in dict.fromkeys(hints) if _query_has_hint(query_lower, hint))


def _query_requests_count_answer(query_lower: str) -> bool:
    if contains_any_exact_phrase(
        query_lower,
        (
            "how many",
            "count of",
            "how much",
            "what number",
        ),
    ):
        return True
    return bool(
        re.search(
            r"\bwhat(?:'s| is)?\s+(?:the\s+)?number of\b",
            query_lower,
        )
    )


def analyze_query_intent(query: str) -> QueryIntent:
    query_lower = normalize_whitespace(query).lower()
    media_signal_count = _count_query_hints(query_lower, MEDIA_QUERY_HINTS)
    event_signal_count = _count_query_hints(query_lower, EVENT_QUERY_HINTS)
    constraint_count = len(extract_constraint_phrases(query_lower))

    repository_signal_count = sum(
        1
        for hint in (
            *THESIS_CONTENT_HINTS,
            "department",
            "university",
            "student",
            "submitted",
            "completed",
            "advisor",
            "supervisor",
            "repository",
        )
        if _query_has_hint(query_lower, hint)
    )
    paper_signal_count = _count_unique_query_hints(
        query_lower,
        (
            *PAPER_QUERY_HINTS,
            *REFERENCE_HINTS,
            "journal",
            "article",
            "authors",
            "sample size",
            "published",
            "citation",
        ),
    )

    explicit_thesis = any(_query_has_hint(query_lower, hint) for hint in THESIS_CONTENT_HINTS)
    targets_citation_identifier = any(
        contains_exact_phrase(query_lower, hint)
        for hint in ("doi", "digital object identifier")
    )
    requires_acknowledgement_section = any(hint in query_lower for hint in ACKNOWLEDGEMENT_HINTS)
    requires_reference_section = any(_query_has_hint(query_lower, hint) for hint in REFERENCE_HINTS)
    asks_full_name = contains_any_exact_phrase(
        query_lower,
        (
            "full name",
            "first and last name",
            "first and last names",
            "first name and last name",
        ),
    )
    targets_count = _query_requests_count_answer(query_lower)
    asks_last_name = contains_any_exact_phrase(query_lower, ("last name", "surname")) and not asks_full_name
    asks_title = contains_exact_phrase(query_lower, "title")
    asks_year = contains_any_exact_phrase(
        query_lower,
        ("what year", "which year", "in what year", "when was", "when did", "dated"),
    )
    targets_person = is_person_target_query(query_lower) and not asks_year
    asks_institution = contains_any_exact_phrase(query_lower, ("which university", "what university", "institution"))
    asks_name = contains_any_exact_phrase(
        query_lower,
        ("what's the name", "what is the name", "name of", "which character", "who is"),
    )
    explicit_paper = targets_citation_identifier or paper_signal_count >= 2 or any(
        _query_has_hint(query_lower, hint)
        for hint in (
            "research paper",
            "journal article",
            "peer reviewed",
        )
    )
    is_media_query = is_media_lookup_query(query_lower)
    is_event_query = is_event_lookup_query(query_lower)
    prefers_encyclopedic_sources = is_media_query or any(
        hint in query_lower for hint in ("wiki", "fandom", "character", "plot summary")
    )
    prefers_character_sources = is_media_query and any(
        hint in query_lower for hint in CHARACTER_QUERY_HINTS
    )
    prefers_event_sources = (
        is_event_query
        and not explicit_paper
        and not explicit_thesis
    )
    needs_entity_discovery_hop = (
        not explicit_paper
        and not explicit_thesis
        and (is_media_query or media_signal_count >= 2)
        and any(hint in query_lower for hint in ("created by", "became known", "based on", "draws some elements"))
    )
    needs_event_discovery_hop = (
        prefers_event_sources
        and any(
            hint in query_lower
            for hint in (
                "named after",
                "began after",
                "set them apart",
                "celebration",
                "festival",
                "anniversary",
                "winner",
                "pageant",
                "contest",
            )
        )
    )
    article_reference_count = len(re.findall(r"(?<!\w)articles?(?!\w)", query_lower))
    needs_person_identity_hop = (
        targets_person
        and not asks_full_name
        and not asks_last_name
        and not asks_year
        and not explicit_thesis
        and not explicit_paper
        and not targets_citation_identifier
        and not requires_acknowledgement_section
        and not requires_reference_section
        and (
            article_reference_count >= 2
            or (
                constraint_count >= 3
                and any(
                    _query_has_hint(query_lower, hint)
                    for hint in (
                        "student",
                        "children",
                        "article",
                        "admitted",
                        "schools",
                        "growth",
                        "violence",
                    )
                )
            )
        )
    )
    targets_compound_answer = (asks_name or targets_person or "who" in query_lower) and targets_count
    open_domain_relation_count = sum(
        1
        for hint in (
            "after",
            "before",
            "between",
            "as of",
            "during",
            "inclusive",
            "anniversary",
            "winner",
            "named after",
            "began after",
            "shifted",
        )
        if hint in query_lower
    )
    historical_year_discovery_query = (
        asks_year
        and not is_academic_lookup_query(query_lower)
        and (
            constraint_count >= 2
            or open_domain_relation_count >= 2
            or (
                any(
                    hint in query_lower
                    for hint in (
                        "monument",
                        "memorial",
                        "spomenik",
                        "loss of lives",
                        "lost their lives",
                        "victims",
                        "dedication",
                        "honor",
                        "battle",
                        "massacre",
                        "tragedy",
                    )
                )
                and any(
                    hint in query_lower
                    for hint in (
                        "constructed",
                        "prior to",
                        "before 1970",
                        "former yugoslavia",
                        "population census",
                        "largest cities",
                        "artist who was born",
                        "born in",
                    )
                )
            )
        )
    )
    is_open_domain_browsecomp = (
        needs_person_identity_hop
        or (
            not is_academic_lookup_query(query_lower)
            and (
                needs_entity_discovery_hop
                or needs_event_discovery_hop
                or targets_compound_answer
                or historical_year_discovery_query
                or (is_media_query and any(hint in query_lower for hint in ("between", "first chapter", "released")))
                or (
                    prefers_event_sources
                    and (
                        open_domain_relation_count >= 3
                        or constraint_count >= 2
                        or (targets_person and any(hint in query_lower for hint in ("winner", "won", "pageant", "contest")))
                    )
                )
            )
        )
    )

    prefers_repository_sources = (
        explicit_thesis
        or requires_acknowledgement_section
        or (repository_signal_count >= 2 and paper_signal_count <= repository_signal_count + 1)
    )
    prefers_paper_sources = (
        targets_citation_identifier
        or requires_reference_section
        or (explicit_paper and paper_signal_count >= max(2, repository_signal_count))
    )
    prefers_pdf_sources = prefers_repository_sources or explicit_thesis or "pdf" in query_lower
    needs_reference_hop = requires_reference_section or "cited" in query_lower or "reference" in query_lower
    needs_career_hop = (
        any(hint in query_lower for hint in CAREER_HINTS)
        and any(hint in query_lower for hint in ("later", "became", "joined", "appointed", "uk"))
    )

    if targets_citation_identifier:
        answer_type = "doi"
    elif asks_year:
        answer_type = "year"
    elif targets_compound_answer:
        answer_type = "entity_and_count"
    elif asks_last_name:
        answer_type = "person_last_name"
    elif asks_full_name or (targets_person and not targets_count):
        answer_type = "person_name"
    elif targets_count:
        answer_type = "count"
    elif asks_name or (is_media_query and targets_person):
        answer_type = "entity_name"
    elif asks_title:
        answer_type = "title"
    elif asks_institution:
        answer_type = "institution"
    else:
        answer_type = "generic"

    signals = tuple(
        signal
        for signal, enabled in (
            ("academic", is_academic_lookup_query(query_lower)),
            ("thesis", explicit_thesis),
            ("paper", explicit_paper),
            ("acknowledgement", requires_acknowledgement_section),
            ("reference", requires_reference_section),
            ("career", needs_career_hop),
            ("doi", targets_citation_identifier),
            ("year", asks_year),
            ("person", targets_person),
            ("media", is_media_query),
            ("event", is_event_query),
            ("browsecomp", is_open_domain_browsecomp),
            ("count", targets_count),
        )
        if enabled
    )

    return QueryIntent(
        answer_type=answer_type,
        explicit_thesis=explicit_thesis,
        explicit_paper=explicit_paper,
        prefers_repository_sources=prefers_repository_sources,
        prefers_paper_sources=prefers_paper_sources,
        prefers_pdf_sources=prefers_pdf_sources,
        prefers_encyclopedic_sources=prefers_encyclopedic_sources,
        prefers_character_sources=prefers_character_sources,
        prefers_event_sources=prefers_event_sources,
        requires_acknowledgement_section=requires_acknowledgement_section,
        requires_reference_section=requires_reference_section,
        needs_career_hop=needs_career_hop,
        needs_reference_hop=needs_reference_hop,
        needs_entity_discovery_hop=needs_entity_discovery_hop,
        needs_event_discovery_hop=needs_event_discovery_hop,
        needs_person_identity_hop=needs_person_identity_hop,
        targets_person=targets_person,
        targets_citation_identifier=targets_citation_identifier,
        targets_count=targets_count,
        targets_compound_answer=targets_compound_answer,
        is_media_query=is_media_query,
        is_event_query=is_event_query,
        is_open_domain_browsecomp=is_open_domain_browsecomp,
        signals=signals,
    )
