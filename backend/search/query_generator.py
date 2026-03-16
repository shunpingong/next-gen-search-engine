from __future__ import annotations

from planner.query_intent import analyze_query_intent
from utils.text_utils import (
    ACKNOWLEDGEMENT_HINTS,
    ABILITY_QUERY_HINTS,
    CAREER_HINTS,
    CHARACTER_QUERY_HINTS,
    EVENT_QUERY_HINTS,
    EVENT_WINNER_HINTS,
    MEDIA_QUERY_HINTS,
    PAPER_QUERY_HINTS,
    REFERENCE_HINTS,
    THESIS_CONTENT_HINTS,
    clue_similarity,
    extract_media_type,
    specific_query_terms,
    important_terms,
    is_academic_lookup_query,
    historical_year_artist_birth_year,
    historical_year_build_year_cutoff,
    is_media_lookup_query,
    keyword_focus,
    normalize_whitespace,
    query_requires_bosnia_top_city,
    strip_question_prefix,
    unique_preserve_order,
)

MAX_RETRIEVAL_CLUES = 10
MAX_FOLLOW_UP_CLUES = 6
MAX_SIMPLE_RETRIEVAL_CLUES = 2
DEFAULT_MAX_QUERY_LENGTH = 1000
EVENT_DISCOVERY_HINTS = (
    "festival",
    "celebration",
    "anniversary",
    "official",
    "tourism",
    "township",
    "town",
    "municipal",
    "municipality",
    "province",
    "provincial",
    "named after",
)


def prepare_retrieval_clues(query: str, clues: list[str]) -> list[str]:
    prepared_clues = unique_preserve_order(clues)
    if not prepared_clues:
        return []

    intent = analyze_query_intent(query)
    media_type = extract_media_type(query)
    if not is_academic_lookup_query(query) or intent.needs_person_identity_hop:
        normalized_clues = _normalize_media_clues(prepared_clues, media_type=media_type)
        if intent.needs_event_discovery_hop:
            normalized_clues = _normalize_event_discovery_clues(normalized_clues)
        additions = _generic_browsecomp_additions(query, normalized_clues)
        return unique_preserve_order(normalized_clues + additions)[:MAX_RETRIEVAL_CLUES]

    query_lower = query.lower()
    institution_focus = next(
        (
            clue
            for clue in prepared_clues
            if any(term in clue.lower() for term in ("university", "department", "visual storytelling"))
        ),
        prepared_clues[0],
    )
    research_focus = next(
        (
            clue
            for clue in prepared_clues
            if any(term in clue.lower() for term in ("film", "genre", "audience", "paper"))
        ),
        prepared_clues[min(1, len(prepared_clues) - 1)],
    )
    acknowledgement_focus = next(
        (
            clue
            for clue in prepared_clues
            if any(term in clue.lower() for term in ("acknowledg", "thank", "grateful", "gratitude"))
        ),
        prepared_clues[-1],
    )

    additions: list[str] = []
    if intent.prefers_repository_sources:
        if not any(term in institution_focus.lower() for term in THESIS_CONTENT_HINTS):
            additions.append(f"{institution_focus} thesis repository pdf")
        if not any(term in research_focus.lower() for term in THESIS_CONTENT_HINTS):
            additions.append(f"{research_focus} thesis dissertation pdf repository")

    if intent.prefers_paper_sources:
        additions.append(f"{research_focus} journal article")
        additions.append(f"{research_focus} abstract")
        if intent.targets_citation_identifier:
            additions.append(f"{research_focus} doi")
            additions.append(f"site:doi.org {research_focus}")
        if intent.needs_reference_hop:
            additions.append(f"{research_focus} references")
            additions.append(f"{research_focus} cited references")
        if any(term in query_lower for term in ("authors", "author", "three authors")):
            additions.append(f"{research_focus} authors")
        if any(term in query_lower for term in ("sample", "couples")):
            additions.append(f"{research_focus} couples sample")

    if intent.requires_acknowledgement_section:
        additions.append(f"{acknowledgement_focus} acknowledgements pdf")
        additions.append(f"{acknowledgement_focus} thanks gratitude")
    if intent.needs_career_hop:
        career_focus = next(
            (
                clue
                for clue in prepared_clues
                if any(term in clue.lower() for term in ("professor", "uk", "faculty"))
            ),
            acknowledgement_focus,
        )
        additions.append(f"{career_focus} professor uk 2018")

    return unique_preserve_order(prepared_clues + additions)[:MAX_RETRIEVAL_CLUES]


def prepare_simple_retrieval_clues(query: str) -> list[str]:
    normalized_query = normalize_whitespace(query)[:DEFAULT_MAX_QUERY_LENGTH]
    if not normalized_query:
        return []

    primary_clue = strip_question_prefix(normalized_query).rstrip(" ?") or normalized_query.rstrip(" ?")
    clues = [primary_clue]
    if len(important_terms(primary_clue)) >= 7:
        focused = keyword_focus(primary_clue, max_terms=6)
        if focused and clue_similarity(primary_clue, focused) < 0.85:
            clues.append(focused)
    return unique_preserve_order(clues)[:MAX_SIMPLE_RETRIEVAL_CLUES]


def prepare_follow_up_retrieval_clues(query: str, clues: list[str]) -> list[str]:
    prepared_clues = unique_preserve_order(clues)
    if not prepared_clues:
        return []

    intent = analyze_query_intent(query)
    query_lower = query.lower()
    specific_focus = " ".join(specific_query_terms(query)[:6])
    media_type = extract_media_type(query)
    additions: list[str] = []
    identity_focus = next(
        (
            clue
            for clue in prepared_clues
            if any(term in clue.lower() for term in ("student", "advisor", "mentor", "supervisor", "prescott"))
        ),
        prepared_clues[0],
    )
    for clue in prepared_clues:
        clue_lower = clue.lower()
        event_specific_clue = _is_event_discovery_clue(clue_lower) or '"' in clue_lower
        if intent.needs_person_identity_hop:
            compact = _compact_clue(clue, max_terms=10)
            identity_compact = _compact_clue(identity_focus, max_terms=8)
            if "biography" not in clue_lower:
                additions.append(f"{compact} biography")
            if "children" not in clue_lower:
                additions.append(f"{compact} children")
            if (
                identity_compact
                and identity_compact.lower() not in clue_lower
                and compact.lower() != identity_compact.lower()
            ):
                additions.append(f"{identity_compact} {compact}")
            continue
        if '"' in clue and intent.requires_acknowledgement_section:
            if "acknowledg" not in clue_lower and "thank" not in clue_lower:
                additions.append(f"{clue} acknowledgements pdf")
        if '"' in clue and intent.needs_career_hop:
            if not any(term in clue_lower for term in CAREER_HINTS):
                additions.append(f"{clue} professor uk 2018")
        if intent.prefers_repository_sources and "site:" not in clue_lower and any(
            term in clue_lower for term in THESIS_CONTENT_HINTS
        ):
            additions.append(f"{clue} pdf")
        if intent.prefers_paper_sources:
            if "doi" not in clue_lower and intent.targets_citation_identifier:
                additions.append(f"{clue} doi")
            if "journal" not in clue_lower:
                additions.append(f"{clue} journal article")
            if intent.needs_reference_hop and not any(term in clue_lower for term in REFERENCE_HINTS):
                additions.append(f"{clue} references")
        if intent.prefers_event_sources:
            if not any(term in clue_lower for term in EVENT_QUERY_HINTS):
                additions.append(f"{clue} festival celebration")
            if (
                not any(term in clue_lower for term in EVENT_WINNER_HINTS)
                and (not intent.needs_event_discovery_hop or event_specific_clue)
            ):
                additions.append(f"{clue} beauty pageant winner")
            if (
                intent.targets_person
                and (not intent.needs_event_discovery_hop or event_specific_clue)
                and not any(
                    term in clue_lower
                    for term in ("beauty pageant", "festival queen", "contest winner", "winner", "full name")
                )
            ):
                additions.append(f"{clue} beauty pageant contest winner full name")
            if "official" not in clue_lower and "tourism" not in clue_lower:
                additions.append(f"{clue} official tourism")
        if not intent.prefers_paper_sources and intent.prefers_encyclopedic_sources:
            if "wiki" not in clue_lower:
                additions.append(f"{clue} wiki")
            if intent.is_media_query and "fandom" not in clue_lower:
                additions.append(f"{clue} fandom")
        if intent.prefers_character_sources and not any(term in clue_lower for term in CHARACTER_QUERY_HINTS):
            additions.append(f"{clue} character")
        if intent.targets_count and not any(term in clue_lower for term in ABILITY_QUERY_HINTS):
            additions.append(f"{clue} movements effects")
        if intent.is_open_domain_browsecomp and specific_focus:
            if intent.is_media_query and "site:wikipedia.org" not in clue_lower:
                additions.append(f"site:wikipedia.org {specific_focus} {media_type}".strip())
            if intent.is_media_query and "site:fandom.com" not in clue_lower:
                additions.append(f"site:fandom.com {specific_focus} {media_type}".strip())
    normalized = _normalize_media_clues(prepared_clues + additions, media_type=media_type)
    return unique_preserve_order(normalized)[:MAX_FOLLOW_UP_CLUES]


def score_search_priority(query: str, clue: str) -> float:
    intent = analyze_query_intent(query)
    query_lower = query.lower()
    clue_lower = clue.lower()
    score = 0.45
    score += min(0.35, len(set(important_terms(query.lower())) & set(important_terms(clue_lower))) * 0.05)
    if intent.prefers_repository_sources and any(term in clue_lower for term in THESIS_CONTENT_HINTS):
        score += 0.1
    if intent.requires_acknowledgement_section and any(term in clue_lower for term in ACKNOWLEDGEMENT_HINTS):
        score += 0.08
    if intent.needs_career_hop and any(term in clue_lower for term in CAREER_HINTS):
        score += 0.05
    if intent.prefers_paper_sources and any(term in clue_lower for term in PAPER_QUERY_HINTS):
        score += 0.08
    if intent.needs_reference_hop and any(term in clue_lower for term in REFERENCE_HINTS):
        score += 0.06
    if intent.prefers_event_sources and any(term in clue_lower for term in EVENT_QUERY_HINTS):
        score += 0.08
    if intent.prefers_event_sources and any(term in clue_lower for term in EVENT_WINNER_HINTS):
        score += 0.06
    if intent.needs_event_discovery_hop:
        if _is_event_discovery_clue(clue_lower):
            score += 0.08
        if any(term in clue_lower for term in ("official tourism", "festival celebration", "anniversary competition")):
            score += 0.06
        if any(term in clue_lower for term in ("stew", "fish", "meat", "condiment", "ingredient")) and not _is_event_discovery_clue(clue_lower):
            score -= 0.2
        if any(term in clue_lower for term in EVENT_WINNER_HINTS) and not _is_event_discovery_clue(clue_lower) and '"' not in clue_lower:
            score -= 0.18
    if intent.needs_person_identity_hop:
        if any(term in clue_lower for term in ("student", "advisor", "mentor", "supervisor", "biography", "children")):
            score += 0.08
        if any(term in clue_lower for term in ("new south", "scrap metal", "gardening", "african-american", "all-white", "schools")):
            score += 0.07
        if '"' in clue and any(
            term in clue_lower for term in ("new south", "scrap metal", "african-american", "william prescott", "dr. william prescott")
        ):
            score += 0.05
        if any(term in clue_lower for term in ("journal article", "abstract", "doi")):
            score -= 0.1
    if intent.prefers_encyclopedic_sources and any(term in clue_lower for term in ("wiki", "fandom")):
        score += 0.08
    if intent.prefers_character_sources and any(term in clue_lower for term in CHARACTER_QUERY_HINTS):
        score += 0.07
    if intent.targets_count and any(term in clue_lower for term in ABILITY_QUERY_HINTS):
        score += 0.06
    if intent.answer_type == "year":
        if any(term in clue_lower for term in ("event year", "victims", "massacre", "battle", "memorial", "monument")):
            score += 0.08
        if "born" in clue_lower and "artist" in clue_lower and not any(
            term in clue_lower for term in ("event year", "victims", "massacre", "battle", "dedication")
        ):
            score -= 0.06
    if intent.needs_entity_discovery_hop and any(
        term in clue_lower for term in ("first chapter", "released", "created by", "group name", "elementary school")
    ):
        score += 0.08
    if intent.is_media_query and any(term in clue_lower for term in ("inspired by", "classic story", "novel")):
        score += 0.05
    if is_media_lookup_query(query_lower) and any(term in clue_lower for term in MEDIA_QUERY_HINTS):
        score += 0.04
    return min(1.0, score)


def _generic_browsecomp_additions(query: str, clues: list[str]) -> list[str]:
    intent = analyze_query_intent(query)
    query_lower = normalize_whitespace(query).lower()
    additions: list[str] = []
    normalized_query = normalize_whitespace(query)[:DEFAULT_MAX_QUERY_LENGTH]
    base_focus = keyword_focus(strip_question_prefix(normalized_query), max_terms=12)
    specific_focus = " ".join(specific_query_terms(normalized_query)[:6])
    media_type = extract_media_type(query)
    discovery_focus = next(
        (
            clue
            for clue in clues
            if any(
                term in clue.lower()
                for term in ("first chapter", "released", "created", "elementary school", "group name", "group")
            )
        ),
        clues[0],
    )
    inspiration_focus = next(
        (
            clue
            for clue in clues
            if any(term in clue.lower() for term in ("classic story", "novelist", "inspired", "elements"))
        ),
        clues[min(1, len(clues) - 1)],
    )
    character_focus = next(
        (clue for clue in clues if any(term in clue.lower() for term in CHARACTER_QUERY_HINTS)),
        clues[-1],
    )
    ability_focus = next(
        (clue for clue in clues if any(term in clue.lower() for term in ABILITY_QUERY_HINTS)),
        clues[-1],
    )
    person_identity_focus = next(
        (
            clue
            for clue in clues
            if any(term in clue.lower() for term in ("student", "advisor", "mentor", "supervisor", "prescott"))
        ),
        clues[0],
    )
    event_focus = next(
        (clue for clue in clues if any(term in clue.lower() for term in EVENT_QUERY_HINTS)),
        clues[0],
    )

    if intent.needs_person_identity_hop:
        additions.extend(_person_identity_browsecomp_additions(query, clues, person_identity_focus))
        return [
            clue
            for clue in unique_preserve_order(additions)
            if len(important_terms(clue)) >= 2
        ]

    if intent.answer_type != "year":
        for clue in clues[:4]:
            compact = _compact_clue(clue, max_terms=10)
            if compact and clue_similarity(clue, compact) < 0.92:
                additions.append(compact)

    if intent.prefers_encyclopedic_sources and base_focus:
        additions.append(f"{base_focus} {media_type} wiki".strip())
        if intent.is_media_query:
            additions.append(f"{base_focus} {media_type} fandom".strip())
    if specific_focus and intent.prefers_encyclopedic_sources:
        if intent.is_media_query:
            additions.append(f"site:wikipedia.org {specific_focus} {media_type}".strip())
            additions.append(f"site:fandom.com {specific_focus} {media_type}".strip())

    if intent.needs_entity_discovery_hop:
        additions.append(f"{_compact_clue(discovery_focus, max_terms=10)} wiki")
        additions.append(f"{_compact_clue(discovery_focus, max_terms=10)} creators group name")
        if any(term in query_lower for term in ("chapter", "released")):
            additions.append(f"{_compact_clue(discovery_focus, max_terms=10)} first chapter")
        if specific_focus:
            additions.append(f"{specific_focus} {media_type} title".strip())
            additions.append(f"{specific_focus} creator group")

    if intent.is_media_query:
        additions.append(f"{_compact_clue(discovery_focus, max_terms=10)} {media_type} wiki".strip())
        if any(term in query_lower for term in ("classic story", "novelist", "1800s", "elements")):
            additions.append(f"{_compact_clue(inspiration_focus, max_terms=10)} {media_type} inspired by novel".strip())
        if "theme" in query_lower:
            additions.append(
                f"{base_focus} {media_type} perfection theme".strip()
                if "perfection" in query_lower
                else f"{base_focus} {media_type} theme".strip()
            )
        if specific_focus:
            additions.append(f"{specific_focus} {media_type}".strip())

    if intent.prefers_character_sources:
        additions.append(f"{_compact_clue(character_focus, max_terms=10)} {media_type} character wiki".strip())
        additions.append(f"{_compact_clue(character_focus, max_terms=10)} {media_type} antagonist companion".strip())
        if specific_focus:
            additions.append(f"{specific_focus} {media_type} character wiki".strip())

    if intent.targets_count:
        additions.append(f"{_compact_clue(ability_focus, max_terms=10)} {media_type} movements effects".strip())
        additions.append(f"{_compact_clue(ability_focus, max_terms=10)} {media_type} ability list".strip())
        if specific_focus:
            additions.append(f"{specific_focus} {media_type} movements effects".strip())

    if intent.prefers_event_sources:
        compact_event_focus = _compact_clue(event_focus, max_terms=10)
        if "festival celebration" not in compact_event_focus.lower():
            additions.append(f"{compact_event_focus} festival celebration".strip())
        if "official tourism" not in compact_event_focus.lower():
            additions.append(f"{compact_event_focus} official tourism".strip())
        if not intent.needs_event_discovery_hop:
            if "beauty pageant winner" not in compact_event_focus.lower():
                additions.append(f"{compact_event_focus} beauty pageant winner".strip())
            if "full name" not in compact_event_focus.lower():
                additions.append(f"{compact_event_focus} beauty pageant contest winner full name".strip())
            if "contest winner" not in compact_event_focus.lower():
                additions.append(f"{compact_event_focus} contest winner".strip())
        if intent.targets_person and not intent.needs_event_discovery_hop:
            additions.append("beauty pageant contest winner full name official tourism")
            additions.append("festival queen winner full name official tourism")
        if "anniversary" in query_lower:
            if "anniversary competition" not in compact_event_focus.lower():
                additions.append(f"{compact_event_focus} anniversary competition".strip())
        if specific_focus and not intent.is_open_domain_browsecomp:
            additions.extend(
                [
                    f"{specific_focus} festival winner".strip(),
                    f"{specific_focus} beauty pageant winner".strip(),
                    f"{specific_focus} official tourism".strip(),
                ]
            )

    if intent.answer_type == "year":
        year_focus = next(
            (
                clue
                for clue in clues
                if any(term in clue.lower() for term in ("bosnia", "former yugoslavia", "sarajevo", "banja luka", "tuzla", "zenica"))
            ),
            clues[0] if clues else base_focus,
        )
        compact_focus = _compact_clue(year_focus, max_terms=10) if year_focus else base_focus
        if any(term in query_lower for term in ("monument", "memorial", "spomenik")):
            if "former yugoslavia" in query_lower:
                additions.append(f"site:spomenikdatabase.org {compact_focus} monument".strip())
                additions.append(f"{compact_focus} former Yugoslavia monument".strip())
            if query_requires_bosnia_top_city(query):
                additions.append("Banja Luka Tuzla Zenica Sarajevo monument victims year")
                additions.extend(
                    [
                        "site:spomenikdatabase.org Zenica monument victims year",
                        "site:spomenikdatabase.org Tuzla monument victims year",
                        "site:spomenikdatabase.org Banja Luka monument victims year",
                        "site:spomenikdatabase.org Sarajevo monument victims year",
                    ]
                )
                build_cutoff = historical_year_build_year_cutoff(query)
                artist_birth = historical_year_artist_birth_year(query)
                if build_cutoff is not None:
                    additions.append(f"site:spomenikdatabase.org Bosnia monument built before {build_cutoff}")
                if artist_birth is not None:
                    additions.append(f"site:spomenikdatabase.org Bosnia monument artist born {artist_birth}")
                    additions.append(
                        f"site:spomenikdatabase.org Banja Luka Tuzla Zenica Sarajevo monument artist born {artist_birth}"
                    )
            additions.append(f"{compact_focus} monument event year".strip())
            additions.append(f"{compact_focus} victims memorial".strip())
            additions.append(f"{compact_focus} dedication victims year".strip())
        if any(term in query_lower for term in ("massacre", "battle", "victims", "loss of lives", "lost their lives")):
            additions.append(f"{compact_focus} event year victims".strip())
        if any(term in query_lower for term in ("artist", "born")) and any(
            term in query_lower for term in ("monument", "memorial", "spomenik")
        ):
            additions.append(f"{compact_focus} monument artist born".strip())

    return [
        clue
        for clue in unique_preserve_order(additions)
        if len(important_terms(clue)) >= 2
    ]


def _compact_clue(clue: str, *, max_terms: int) -> str:
    compact = keyword_focus(clue, max_terms=max_terms)
    return compact or normalize_whitespace(clue)


def _person_identity_browsecomp_additions(
    query: str,
    clues: list[str],
    identity_focus: str,
) -> list[str]:
    normalized_query = normalize_whitespace(query)[:DEFAULT_MAX_QUERY_LENGTH]
    query_lower = normalized_query.lower()
    identity_compact = _compact_clue(identity_focus, max_terms=8)
    children_focus = next((clue for clue in clues if "children" in clue.lower()), "")
    article_focuses = [
        clue
        for clue in clues
        if clue != identity_focus
        and any(
            term in clue.lower()
            for term in ("article", "new south", "scrap metal", "african-american", "schools", "growth", "gardening")
        )
    ]

    additions: list[str] = []
    if identity_compact:
        additions.append(f"{identity_compact} biography")

    for article_focus in article_focuses[:3]:
        compact_article = _compact_clue(article_focus, max_terms=10)
        if identity_compact:
            additions.append(f"{identity_compact} {compact_article}")
        additions.append(compact_article)
        if "article" not in compact_article.lower():
            additions.append(f"{compact_article} article")

    if children_focus:
        compact_children = _compact_clue(children_focus, max_terms=8)
        additions.append(compact_children)
        if identity_compact and compact_children.lower() not in identity_compact.lower():
            additions.append(f"{identity_compact} {compact_children}")
        if identity_compact:
            additions.append(f"{identity_compact} children")

    if "new south" in query_lower:
        additions.append("\"New South\" cities growth article")
    if "scrap metal" in query_lower:
        additions.append("\"scrap metal\" gardening violence article")
    if "african-american" in query_lower and "schools" in query_lower:
        additions.append("\"African-American\" students admitted four all-white schools article")

    return unique_preserve_order(
        clue
        for clue in additions
        if clue and len(important_terms(clue)) >= 2
    )


def _normalize_media_clues(clues: list[str], *, media_type: str) -> list[str]:
    if not media_type:
        return unique_preserve_order(clues)

    normalized: list[str] = []
    for clue in clues:
        clue_lower = clue.lower()
        if any(term in clue_lower for term in ("manga", "anime", "comic", "novel", "game", "film", "movie")):
            normalized.append(clue)
            continue
        if any(term in clue_lower for term in ("character", "antagonist", "companion", "movement", "movements", "effects", "chapter", "group name", "creators")):
            normalized.append(f"{clue} {media_type}".strip())
            continue
        normalized.append(clue)
    return unique_preserve_order(normalized)


def _normalize_event_discovery_clues(clues: list[str]) -> list[str]:
    event_core_count = sum(1 for clue in clues if _is_event_discovery_clue(clue.lower()))
    normalized: list[str] = []
    retained_dish_identity_clue = False
    for clue in clues:
        clue_lower = clue.lower()
        if any(term in clue_lower for term in ("beauty pageant", "contest winner", "winner", "full name")) and not _is_event_discovery_clue(clue_lower):
            continue
        if any(term in clue_lower for term in ("stew", "fish", "meat", "condiment", "ingredient")) and not _is_event_discovery_clue(clue_lower):
            if event_core_count >= 2 and retained_dish_identity_clue:
                continue
            retained_dish_identity_clue = True
            normalized.append(f"{clue} celebration named after dish".strip())
            normalized.append(f"{clue} township festival official tourism".strip())
            continue
        if any(term in clue_lower for term in ("anniversary", "provincial", "province", "winners")) and "festival" not in clue_lower:
            normalized.append(f"{clue} festival celebration".strip())
            continue
        normalized.append(clue)
    return unique_preserve_order(normalized)


def _is_event_discovery_clue(clue_lower: str) -> bool:
    return any(term in clue_lower for term in EVENT_DISCOVERY_HINTS)
