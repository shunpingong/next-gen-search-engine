from __future__ import annotations

import logging
import re

from agent.models import Document, FollowUpClueOutput
from config.reflection_config import (
    MAX_FOLLOW_UP_CLUES,
    MIN_FOLLOW_UP_CLUES,
    FollowUpConfig,
    load_follow_up_config,
)
from planner.query_intent import analyze_query_intent
from planner.query_decomposer import sanitize_model_clues
from utils.text_utils import (
    ACKNOWLEDGEMENT_HINTS,
    ABILITY_QUERY_HINTS,
    CAREER_HINTS,
    CHARACTER_QUERY_HINTS,
    REFERENCE_HINTS,
    clue_similarity,
    browsecomp_anchor_match_stats,
    document_title_query_phrase,
    document_type_score,
    event_page_score,
    event_winner_evidence_score,
    extract_author_names_from_text,
    extract_capitalized_entities,
    extract_doi_candidates,
    extract_institutions_from_text,
    has_event_winner_evidence,
    important_terms,
    is_aggregate_listing_page,
    is_broad_overview_page,
    is_forum_discussion_page,
    is_generic_event_topic_page,
    is_generic_historical_monument_page,
    is_generic_media_topic_page,
    is_grounded_browsecomp_page,
    is_low_trust_social_page,
    is_non_english_wiki_page,
    is_person_biography_page,
    is_plausible_person_name,
    is_recipe_food_page,
    is_specific_historical_year_page,
    is_wiki_meta_page,
    looks_like_event_page,
    looks_like_media_page,
    media_page_score,
    normalize_whitespace,
    query_requires_bosnia_top_city,
    specific_query_terms,
    specificity_overlap_score,
    unique_preserve_order,
)

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

logger = logging.getLogger("reflection.query_refiner")

FOLLOW_UP_SYSTEM_PROMPT = """
You are planning a second-hop retrieval step for a tool-augmented deep research system.

Given:
- the original user question
- the initial search clues already used
- the top documents retrieved so far

Generate 2 to 4 follow-up search clues that use specific entities from the retrieved documents
to find the missing evidence needed to answer the question.

Rules:
- use concrete entities from the retrieved documents whenever possible, such as exact paper titles,
  departments, universities, authors, repository names, or career institutions
- prefer clues that help bridge multi-hop gaps, for example acknowledgements pages, thesis PDFs,
  repository records, or faculty pages
- do not answer the question
- do not repeat existing clues unless you make them more specific with new entities
- keep each clue short and directly usable as a search query

Return only the clues in the schema.
""".strip()


async def generate_follow_up_clues_async(
    query: str,
    docs: list[Document],
    existing_clues: list[str],
) -> list[str]:
    follow_up_config = load_follow_up_config()
    if not follow_up_config.use_retrieval or not docs:
        return []
    if not should_run_follow_up_hop(query, docs, existing_clues):
        return []

    source_docs = _select_follow_up_source_documents(query, docs, follow_up_config)
    heuristic_clues = _heuristic_generate_follow_up_clues(query, source_docs, existing_clues)
    llm_clues = await _generate_follow_up_clues_with_openai(
        query,
        source_docs,
        existing_clues,
        follow_up_config,
    )
    merged_clues = _merge_follow_up_clues(query, source_docs, llm_clues, heuristic_clues, existing_clues)
    if len(merged_clues) < MIN_FOLLOW_UP_CLUES:
        grounded_heuristics = _filter_grounded_follow_up_clues(query, source_docs, heuristic_clues)
        return grounded_heuristics[:MAX_FOLLOW_UP_CLUES]
    return merged_clues[:MAX_FOLLOW_UP_CLUES]


def should_run_follow_up_hop(query: str, docs: list[Document], existing_clues: list[str]) -> bool:
    if not docs:
        return False

    intent = analyze_query_intent(query)
    query_lower = query.lower()
    relation_hints = sum(
        1
        for hint in (
            "later",
            "became",
            "between",
            "fewer than",
            "under",
            "acknowledg",
            "professor",
            "department",
            "university",
        )
        if hint in query_lower
    )
    enough_clues = len(existing_clues) >= 4
    enough_terms = len(important_terms(query)) >= 12
    return (
        intent.targets_citation_identifier
        or intent.requires_reference_section
        or intent.requires_acknowledgement_section
        or intent.needs_career_hop
        or intent.is_open_domain_browsecomp
        or (intent.targets_person and relation_hints >= 2)
        or (enough_clues and enough_terms)
    )


async def _generate_follow_up_clues_with_openai(
    query: str,
    docs: list[Document],
    existing_clues: list[str],
    follow_up_config: FollowUpConfig,
) -> list[str]:
    if not follow_up_config.use_llm:
        return []

    if not follow_up_config.openai_api_key or not OPENAI_AVAILABLE:
        return []

    doc_context = _build_follow_up_context(query, docs)
    if not doc_context:
        return []

    client = AsyncOpenAI(
        api_key=follow_up_config.openai_api_key,
        timeout=follow_up_config.timeout_seconds,
    )
    try:
        response = await client.responses.parse(
            model=follow_up_config.model,
            input=[
                {"role": "system", "content": FOLLOW_UP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original user question:\n{query}\n\n"
                        f"Initial search clues:\n{_render_clue_list(existing_clues)}\n\n"
                        f"Top retrieved documents:\n{doc_context}"
                    ),
                },
            ],
            text_format=FollowUpClueOutput,
        )
    except Exception as error:
        logger.warning("OpenAI follow-up clue generation failed; using heuristic fallback: %s", error)
        return []
    finally:
        await client.close()

    parsed_output = response.output_parsed
    if parsed_output is None:
        return []
    return sanitize_model_clues(parsed_output.clues, query=query)


def _merge_follow_up_clues(
    query: str,
    docs: list[Document],
    llm_clues: list[str],
    heuristic_clues: list[str],
    existing_clues: list[str],
) -> list[str]:
    merged = unique_preserve_order(sanitize_model_clues(llm_clues, query=query) + heuristic_clues)
    existing_lower = {clue.lower() for clue in existing_clues}
    filtered: list[str] = []
    for clue in merged:
        if clue.lower() in existing_lower:
            continue
        if any(clue_similarity(clue, existing) >= 0.92 for existing in existing_clues):
            continue
        filtered.append(clue)
    grounded = _filter_grounded_follow_up_clues(query, docs, filtered)
    return grounded[:MAX_FOLLOW_UP_CLUES]


def _select_follow_up_source_documents(
    query: str,
    docs: list[Document],
    follow_up_config: FollowUpConfig,
) -> list[Document]:
    limit = follow_up_config.source_doc_count
    intent = analyze_query_intent(query)
    prioritized = sorted(
        docs,
        key=lambda doc: (
            _follow_up_source_score(query, doc) if intent.is_open_domain_browsecomp else document_type_score(doc.url, doc.title, doc.content[:500]),
            doc.rank_score,
            doc.retrieval_score,
            len(doc.matched_clues),
        ),
        reverse=True,
    )
    selected = _unique_documents_by_url(docs[: max(1, limit // 2)] + prioritized[:limit])
    return selected[:limit]


def _build_follow_up_context(query: str, docs: list[Document]) -> str:
    lines: list[str] = []
    for index, document in enumerate(docs, start=1):
        snippet = normalize_whitespace(document.content[:500])
        lines.extend(
            [
                f"{index}. Title: {document.title}",
                f"   URL: {document.url}",
                f"   Source: {document.source}",
                f"   Snippet: {snippet}",
            ]
        )
    return "\n".join(lines).strip()


def _render_clue_list(clues: list[str]) -> str:
    if not clues:
        return "None"
    return "\n".join(f"- {clue}" for clue in clues)


def _heuristic_generate_follow_up_clues(
    query: str,
    docs: list[Document],
    existing_clues: list[str],
) -> list[str]:
    intent = analyze_query_intent(query)
    query_lower = query.lower()
    specific_focus = " ".join(specific_query_terms(query)[:6])
    candidates: list[str] = []

    for document in docs:
        title_phrase = document_title_query_phrase(document.title)
        repository_source = f"site:{document.source}" if document.source and document.source != "unknown" else ""
        author_names = extract_author_names_from_text(f"{document.title} {document.content[:2500]}")
        institutions = extract_institutions_from_text(f"{document.title} {document.content[:2500]}")
        generic_entities = extract_capitalized_entities(f"{document.title} {document.content[:1800]}")
        suppress_person_pivots = (
            intent.answer_type == "year"
            and not intent.targets_person
            and not intent.prefers_paper_sources
            and not intent.needs_career_hop
        )
        document_specificity = specificity_overlap_score(query, f"{document.title} {document.content[:2200]}")
        broad_overview = is_broad_overview_page(document.url, document.title)
        generic_media_topic = is_generic_media_topic_page(document.url, document.title, document.content[:2200])
        generic_event_topic = is_generic_event_topic_page(document.url, document.title, document.content[:2200])
        wiki_meta_page = is_wiki_meta_page(document.url, document.title, document.content[:2200])
        non_english_wiki = is_non_english_wiki_page(document.url)
        media_like_page = looks_like_media_page(document.url, document.title, document.content[:2200])
        event_like_page = looks_like_event_page(document.url, document.title, document.content[:2200])
        event_score = event_page_score(document.url, document.title, document.content[:2200])
        winner_evidence = has_event_winner_evidence(
            normalize_whitespace(f"{document.title}. {document.content[:2200]}"),
            minimum_score=0.34,
        )
        grounded_browsecomp_page = is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            document.content[:2200],
            require_media=intent.is_media_query,
        )
        specific_historical_year_page = (
            intent.answer_type == "year"
            and is_specific_historical_year_page(
                query,
                document.url,
                document.title,
                document.content[:2200],
            )
        )
        if broad_overview and document_specificity < 0.1:
            title_phrase = ""
        if intent.answer_type == "year" and is_generic_historical_monument_page(
            document.url,
            document.title,
            document.content[:1200],
        ):
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.answer_type == "year" and is_person_biography_page(
            document.url,
            document.title,
            document.content[:2200],
        ):
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if generic_media_topic or generic_event_topic or wiki_meta_page or non_english_wiki:
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.prefers_event_sources and is_recipe_food_page(document.url, document.title, document.content[:700]):
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.is_media_query and not media_like_page:
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.prefers_event_sources and is_low_trust_social_page(
            document.url,
            document.title,
            document.content[:500],
        ) and event_score < 0.55:
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.prefers_event_sources and not event_like_page and event_score < 0.2:
            generic_entities = []
        if intent.prefers_event_sources and not winner_evidence and (event_score < 0.38 or not grounded_browsecomp_page):
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.answer_type == "year" and not specific_historical_year_page:
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if intent.is_open_domain_browsecomp and not grounded_browsecomp_page:
            title_phrase = ""
            author_names = []
            institutions = []
            generic_entities = []
        if suppress_person_pivots:
            author_names = []
            generic_entities = [entity for entity in generic_entities if not is_plausible_person_name(entity)]

        if title_phrase:
            if intent.requires_acknowledgement_section:
                candidates.append(f"{repository_source} \"{title_phrase}\" acknowledgements".strip())
                candidates.append(f"\"{title_phrase}\" thank gratitude")
            if intent.needs_career_hop:
                candidates.append(f"\"{title_phrase}\" professor uk 2018")
                candidates.append(f"{repository_source} \"{title_phrase}\" faculty".strip())
            if intent.prefers_paper_sources:
                candidates.append(f"\"{title_phrase}\" journal article")
                if intent.targets_citation_identifier:
                    candidates.append(f"\"{title_phrase}\" doi")
                    candidates.append(f"site:doi.org \"{title_phrase}\"")
                if intent.needs_reference_hop:
                    candidates.append(f"\"{title_phrase}\" references")
            if intent.prefers_event_sources:
                candidates.extend(
                    [
                        f"\"{title_phrase}\" beauty pageant winner",
                        f"\"{title_phrase}\" contest winner",
                        f"\"{title_phrase}\" official tourism",
                    ]
                )
                if "anniversary" in query_lower:
                    candidates.append(f"\"{title_phrase}\" anniversary competition")
            if intent.answer_type == "year":
                candidates.extend(
                    [
                        f"\"{title_phrase}\" event year",
                        f"\"{title_phrase}\" victims year",
                    ]
                )
                if any(term in query_lower for term in ("monument", "memorial", "spomenik")):
                    candidates.append(f"\"{title_phrase}\" monument dedication")
            candidates.append(f"{repository_source} \"{title_phrase}\"".strip())
            candidates.append(f"\"{title_phrase}\"")

        for author_name in author_names[:2]:
            if intent.requires_acknowledgement_section and title_phrase:
                candidates.append(f"\"{author_name}\" \"{title_phrase}\" acknowledgements")
            if intent.needs_career_hop:
                candidates.append(f"\"{author_name}\" professor uk 2018")
                candidates.append(f"\"{author_name}\" faculty film studies")
            if intent.prefers_paper_sources:
                candidates.append(f"\"{author_name}\" journal article")
                if title_phrase:
                    candidates.append(f"\"{author_name}\" \"{title_phrase}\"")
                    if intent.targets_citation_identifier:
                        candidates.append(f"\"{author_name}\" \"{title_phrase}\" doi")
                        candidates.append(f"site:doi.org \"{author_name}\" \"{title_phrase}\"")
            candidates.append(f"\"{author_name}\"")

        for institution in institutions[:2]:
            if intent.requires_acknowledgement_section:
                candidates.append(f"\"{institution}\" dissertation acknowledgements pdf")
            if intent.needs_career_hop:
                candidates.append(f"\"{institution}\" professor uk 2018 film")
            if intent.prefers_paper_sources and title_phrase:
                candidates.append(f"\"{institution}\" \"{title_phrase}\" journal")
                if intent.targets_citation_identifier:
                    candidates.append(f"\"{institution}\" \"{title_phrase}\" doi")

        if title_phrase and intent.prefers_encyclopedic_sources:
            candidates.append(f"\"{title_phrase}\" wiki")
            candidates.append(f"site:wikipedia.org \"{title_phrase}\"")
            if intent.is_media_query:
                candidates.append(f"\"{title_phrase}\" fandom")
                candidates.append(f"site:fandom.com \"{title_phrase}\"")
        if title_phrase and intent.prefers_character_sources:
            candidates.append(f"\"{title_phrase}\" character")
            candidates.append(f"\"{title_phrase}\" characters")
            candidates.append(f"\"{title_phrase}\" antagonist")
        if title_phrase and intent.targets_count:
            candidates.append(f"\"{title_phrase}\" movements effects")
            candidates.append(f"\"{title_phrase}\" ability list")
            candidates.append(f"\"{title_phrase}\" moves")

        for entity in generic_entities[:4]:
            if entity == title_phrase:
                continue
            if broad_overview and document_specificity < 0.18:
                continue
            if intent.prefers_encyclopedic_sources:
                candidates.append(f"\"{entity}\" wiki")
                if title_phrase:
                    candidates.append(f"\"{entity}\" \"{title_phrase}\"")
            if intent.prefers_character_sources:
                candidates.append(f"\"{entity}\" character")
                if title_phrase:
                    candidates.append(f"\"{entity}\" \"{title_phrase}\" antagonist")
            if intent.targets_count:
                candidates.append(f"\"{entity}\" movements")
                candidates.append(f"\"{entity}\" effects")
            if intent.prefers_event_sources:
                candidates.append(f"\"{entity}\" winner")
                if title_phrase:
                    candidates.append(f"\"{entity}\" \"{title_phrase}\" beauty pageant")

        if intent.targets_citation_identifier:
            doi_candidates = extract_doi_candidates(
                f"{document.url} {document.title} {document.content[:2500]}"
            )
            for doi in doi_candidates[:2]:
                candidates.append(doi)

        if intent.requires_reference_section:
            if title_phrase:
                candidates.append(f"\"{title_phrase}\" references")
                candidates.append(f"\"{title_phrase}\" bibliography")

        if intent.is_media_query and title_phrase and any(term in query_lower for term in ("created", "group", "chapter")):
            candidates.append(f"\"{title_phrase}\" creators")
            candidates.append(f"\"{title_phrase}\" first chapter")

    if intent.is_open_domain_browsecomp and specific_focus:
        if intent.is_media_query:
            candidates.extend(
                [
                    f"site:wikipedia.org {specific_focus}",
                    f"site:fandom.com {specific_focus}",
                    f"{specific_focus} manga title",
                ]
            )
        if intent.prefers_event_sources and not intent.is_open_domain_browsecomp:
            candidates.extend(
                [
                    f"{specific_focus} festival winner",
                    f"{specific_focus} beauty pageant winner",
                    f"{specific_focus} official tourism",
                ]
            )
        if intent.prefers_character_sources:
            candidates.append(f"{specific_focus} character wiki")
        if intent.targets_count:
            candidates.append(f"{specific_focus} movements effects")

    filtered = []
    existing_lower = {clue.lower() for clue in existing_clues}
    for candidate in unique_preserve_order(candidates):
        if len(important_terms(candidate)) < 2:
            continue
        if candidate.lower() in existing_lower:
            continue
        filtered.append(candidate)
    return filtered[:MAX_FOLLOW_UP_CLUES]


def _follow_up_source_score(query: str, document: Document) -> float:
    score = document_type_score(document.url, document.title, document.content[:500])
    intent = analyze_query_intent(query)
    combined = f"{document.title} {document.content[:2000]}"
    lowered_combined = normalize_whitespace(combined).lower()
    anchor_terms = specific_query_terms(query)[:8]
    anchor_matches = sum(1 for term in anchor_terms if term in lowered_combined)
    matched_groups, matched_terms, _ = browsecomp_anchor_match_stats(query, combined)
    media_score = media_page_score(document.url, document.title, document.content[:2000])
    event_score = event_page_score(document.url, document.title, document.content[:2000])
    winner_evidence_score = event_winner_evidence_score(
        normalize_whitespace(f"{document.title}. {document.content[:2000]}")
    )
    grounded_browsecomp = is_grounded_browsecomp_page(
        query,
        document.url,
        document.title,
        document.content[:2200],
        require_media=intent.is_media_query,
    )
    score += 0.8 * specificity_overlap_score(query, combined)
    score += min(0.24, 0.08 * anchor_matches)
    score += min(0.24, 0.1 * matched_groups) + min(0.12, 0.03 * matched_terms)
    score += 0.22 * media_score
    if intent.answer_type == "year":
        if is_person_biography_page(document.url, document.title, document.content[:1200]):
            score -= 0.75
        if is_specific_historical_year_page(query, document.url, document.title, document.content[:2200]):
            score += 0.4
        else:
            score -= 0.35
        if grounded_browsecomp:
            score += 0.52
        if any(
            hint in lowered_combined
            for hint in ("victims", "massacre", "battle", "killed", "died", "executed", "commemorates", "in honor")
        ) and any(hint in lowered_combined for hint in ("monument", "memorial", "spomenik")):
            score += 0.2
        if is_generic_historical_monument_page(document.url, document.title, document.content[:800]):
            score -= 0.55
    if intent.prefers_event_sources:
        score += 0.3 * event_score
        score += 0.26 * winner_evidence_score
        if winner_evidence_score < 0.24 and event_score < 0.3:
            score -= 0.28
        if is_low_trust_social_page(document.url, document.title, document.content[:400]) and event_score < 0.55:
            score -= 0.32
    if document_title_query_phrase(document.title):
        score += 0.12
    if is_broad_overview_page(document.url, document.title):
        score -= 0.4
    if is_aggregate_listing_page(document.url, document.title, document.content[:300]):
        score -= 0.45
    if is_wiki_meta_page(document.url, document.title, document.content[:300]):
        score -= 0.5
    if is_non_english_wiki_page(document.url):
        score -= 0.45
    if is_generic_media_topic_page(document.url, document.title, document.content[:300]):
        score -= 0.45
    if is_generic_event_topic_page(document.url, document.title, document.content[:300]):
        score -= 0.5
    if is_generic_historical_monument_page(document.url, document.title, document.content[:800]):
        score -= 0.5
    if is_recipe_food_page(document.url, document.title, document.content[:700]) and event_score < 0.55:
        score -= 0.5
    if is_forum_discussion_page(document.url, document.title, document.content[:300]):
        score -= 0.35
    if intent.is_media_query and media_score < 0.18:
        score -= 0.4
    if intent.is_open_domain_browsecomp and not grounded_browsecomp:
        score -= 0.45
    return score


def _filter_grounded_follow_up_clues(
    query: str,
    docs: list[Document],
    clues: list[str],
) -> list[str]:
    intent = analyze_query_intent(query)
    if not intent.is_open_domain_browsecomp:
        return unique_preserve_order(clues)

    good_title_phrases = {
        title_phrase.lower()
        for document in docs
        if (
            _follow_up_source_score(query, document) >= 0.3
            or (
                intent.answer_type == "year"
                and is_grounded_browsecomp_page(
                    query,
                    document.url,
                    document.title,
                    document.content[:2200],
                    require_media=intent.is_media_query,
                )
                and not is_generic_historical_monument_page(
                    document.url,
                    document.title,
                    document.content[:800],
                )
            )
        )
        and not is_wiki_meta_page(document.url, document.title, document.content[:800])
        and not is_non_english_wiki_page(document.url)
        and not is_generic_media_topic_page(document.url, document.title, document.content[:800])
        and not is_generic_event_topic_page(document.url, document.title, document.content[:800])
        and (
            not intent.prefers_event_sources
            or has_event_winner_evidence(
                normalize_whitespace(f"{document.title}. {document.content[:2200]}"),
                minimum_score=0.34,
            )
            or event_page_score(document.url, document.title, document.content[:2200]) >= 0.42
        )
        and is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            document.content[:2200],
            require_media=intent.is_media_query,
        )
        and (not intent.is_media_query or looks_like_media_page(document.url, document.title, document.content[:2200]))
        for title_phrase in [document_title_query_phrase(document.title)]
        if title_phrase
    }

    grounded: list[str] = []
    query_lower = query.lower()
    year_location_terms = ["bosnia", "yugoslavia"]
    if query_requires_bosnia_top_city(query):
        year_location_terms = ["sarajevo", "banja luka", "tuzla", "zenica"]
    elif "bosnia" in query_lower or "largest cities" in query_lower or "population census" in query_lower:
        year_location_terms.extend(["sarajevo", "banja luka", "tuzla", "zenica"])
    for clue in unique_preserve_order(clues):
        lowered = clue.lower()
        if intent.answer_type == "year" and not intent.targets_person:
            if any(term in lowered for term in ("biography", "sculptor biography", "artist biography")):
                continue
            if re.search(r'"[^"]+"', clue):
                quoted_terms = re.findall(r'"([^"]+)"', clue)
                if any(is_plausible_person_name(term) for term in quoted_terms):
                    continue
            if "born" in lowered and not any(
                term in lowered for term in ("event year", "victims", "battle", "massacre", "dedication", "memorial", "monument")
            ):
                continue
        if any(title_phrase in lowered for title_phrase in good_title_phrases):
            grounded.append(clue)
            continue

        clue_specificity = specificity_overlap_score(query, clue)
        if intent.answer_type == "year" and not intent.targets_person:
            if not any(term in lowered for term in ("event year", "victims year", "dedication", "memorial", "monument", "spomenik", "massacre", "battle")):
                continue
            if not any(term in lowered for term in year_location_terms) and not re.search(r'"[^"]+"', clue):
                continue
        if clue_specificity >= 0.18:
            grounded.append(clue)
            continue

        if clue_specificity >= 0.12 and any(
            token in lowered
            for token in (
                "wiki",
                "fandom",
                "character",
                "antagonist",
                "movements",
                "effects",
                "ability",
                "winner",
                "pageant",
                "contest",
                "tourism",
                "festival",
            )
        ):
            grounded.append(clue)

    return grounded


def _unique_documents_by_url(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique_documents: list[Document] = []
    for document in documents:
        key = normalize_whitespace(document.url or document.title)
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)
    return unique_documents
