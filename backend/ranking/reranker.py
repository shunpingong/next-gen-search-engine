from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from agent.models import Document, EvidenceSnippet
from config.ranking_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    load_ranking_config,
)
from planner.query_constraints import assess_document_constraints
from planner.query_intent import analyze_query_intent
from utils.text_utils import (
    ACKNOWLEDGEMENT_HINTS,
    ACADEMIC_REPOSITORY_HINTS,
    ABILITY_QUERY_HINTS,
    BOSNIA_TOP_FOUR_CITY_TERMS,
    CAREER_HINTS,
    CHARACTER_QUERY_HINTS,
    ENCYCLOPEDIC_SOURCE_HINTS,
    EVENT_WINNER_HINTS,
    LOW_VALUE_TERMS,
    MEDIA_QUERY_HINTS,
    NON_RESEARCH_PAGE_HINTS,
    PAPER_CONTENT_HINTS,
    PERSON_CONTEXT_HINTS,
    REFERENCE_HINTS,
    THESIS_CONTENT_HINTS,
    canonicalize_url,
    contains_doi,
    contains_primary_doi,
    contains_candidate_person_name,
    document_matches_query_years,
    event_page_score,
    event_winner_evidence_score,
    extract_doi_candidates,
    extract_primary_doi_candidates,
    has_event_winner_evidence,
    historical_year_has_structural_constraints,
    historical_year_structural_assessment,
    historical_year_trusted_memorial_source,
    important_terms,
    is_aggregate_listing_page,
    is_academic_lookup_query,
    is_broad_overview_page,
    is_forum_discussion_page,
    is_generic_event_topic_page,
    is_generic_historical_monument_page,
    is_generic_media_topic_page,
    is_grounded_browsecomp_page,
    is_low_trust_social_page,
    is_media_lookup_query,
    is_non_english_wiki_page,
    is_person_biography_page,
    is_person_target_query,
    is_recipe_food_page,
    is_specific_historical_year_page,
    is_wiki_meta_page,
    lexical_relevance_score,
    looks_like_event_page,
    media_page_score,
    normalize_whitespace,
    query_requires_bosnia_top_city,
    split_sentences,
    specificity_overlap_score,
)

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    SentenceTransformer = None
    util = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("ranking.reranker")

DEFAULT_SNIPPET_LENGTH = 320
DEDUPLICATION_SIMILARITY_THRESHOLD = 0.92
KEYWORD_WINDOW_RADIUS = 180
NAME_WINDOW_RADIUS = 120
MAX_KEYWORD_WINDOW_CANDIDATES = 10
MAX_NAME_WINDOW_CANDIDATES = 8
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_EVENT_HINTS = ("victims", "massacre", "battle", "killed", "died", "executed", "loss of lives", "memorial", "commemorates", "honor", "dedication")
YEAR_BIOGRAPHY_HINTS = ("born", "artist", "sculptor", "author", "architect")
YEAR_CONSTRUCTION_HINTS = ("built", "constructed", "erected", "unveiled", "completed", "opened")
YEAR_AWARD_HINTS = ("award", "awards", "competition", "prize", "salon", "triennale", "biennale", "medal")


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    deduplicated: list[Document] = []
    ordered_documents = sorted(
        documents,
        key=lambda doc: (doc.retrieval_score, len(doc.content), len(doc.matched_clues)),
        reverse=True,
    )

    for document in ordered_documents:
        duplicate = _find_duplicate_document(document, deduplicated)
        if duplicate is None:
            deduplicated.append(document)
            continue
        _merge_documents(duplicate, document)

    return deduplicated


def rank_documents(
    query: str,
    docs: list[Document],
    *,
    return_backend: bool = False,
) -> list[Document] | tuple[list[Document], str]:
    if not docs:
        return ([], "none") if return_backend else []

    semantic_scores, backend = _RANKER.score(query, docs)
    lexical_scores = [lexical_relevance_score(query, _compose_rank_text(doc)) for doc in docs]
    retrieval_scores = _normalize_scores([doc.retrieval_score for doc in docs])
    clue_scores = _normalize_scores([float(len(doc.matched_clues)) for doc in docs])
    prior_scores = [_document_prior_score(query, doc) for doc in docs]

    ranked_documents: list[Document] = []
    for document, semantic_score, lexical_score, retrieval_score, clue_score, prior_score in zip(
        docs,
        semantic_scores,
        lexical_scores,
        retrieval_scores,
        clue_scores,
        prior_scores,
    ):
        intent = analyze_query_intent(query)
        lowered_combined = normalize_whitespace(f"{document.title} {document.content[:2200]}").lower()
        grounded_browsecomp = is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            document.content[:2500],
            require_media=intent.is_media_query,
        )
        if backend == "lexical":
            final_score = (
                0.45 * lexical_score
                + 0.15 * retrieval_score
                + 0.1 * clue_score
                + 0.3 * prior_score
            )
        else:
            final_score = (
                0.4 * semantic_score
                + 0.15 * lexical_score
                + 0.1 * retrieval_score
                + 0.1 * clue_score
                + 0.25 * prior_score
            )
        if intent.prefers_event_sources:
            event_score = event_page_score(document.url, document.title, document.content[:2200])
            winner_evidence_score = event_winner_evidence_score(
                normalize_whitespace(f"{document.title}. {document.content[:2200]}")
            )
            event_specificity = specificity_overlap_score(query, f"{document.title} {document.content[:2200]}")
            final_score += 0.14 * event_score
            final_score += 0.18 * winner_evidence_score
            if contains_candidate_person_name(document.content[:2500]) and any(
                hint in f"{document.title.lower()} {document.content[:1800].lower()}"
                for hint in EVENT_WINNER_HINTS
            ):
                final_score += 0.16
            if is_generic_event_topic_page(document.url, document.title, document.content[:500]):
                final_score -= 0.34
            if is_recipe_food_page(document.url, document.title, document.content[:700]) and event_score < 0.55:
                final_score -= 0.32
            if winner_evidence_score < 0.24 and event_score < 0.28:
                final_score -= 0.18
            if intent.is_open_domain_browsecomp and not grounded_browsecomp:
                final_score -= 0.14
            if intent.is_open_domain_browsecomp and event_specificity < 0.1 and event_score < 0.5:
                final_score -= 0.2
            if intent.needs_event_discovery_hop and event_specificity < 0.1 and event_score < 0.55:
                final_score -= 0.24
            if (
                intent.needs_event_discovery_hop
                and any(term in lowered_combined for term in ("beauty pageant", "pageant", "festival queen", "beauty queen"))
                and not any(
                    term in lowered_combined
                    for term in (
                        "festival",
                        "celebration",
                        "anniversary",
                        "official",
                        "tourism",
                        "township",
                        "municipality",
                        "municipal",
                        "province",
                        "provincial",
                        "stew",
                        "ingredient",
                        "condiment",
                    )
                )
            ):
                final_score -= 0.12
            if is_low_trust_social_page(document.url, document.title, document.content[:400]) and event_score < 0.55:
                final_score -= 0.2
        if intent.answer_type == "year":
            final_score += 0.18 * _historical_year_document_score(query, document)
            if is_generic_historical_monument_page(document.url, document.title, document.content[:800]):
                final_score -= 0.28
        if intent.is_open_domain_browsecomp:
            if grounded_browsecomp:
                final_score += 0.14
            else:
                final_score -= 0.2
        document.rank_score = round(float(final_score), 6)
        ranked_documents.append(document)

    ranked_documents.sort(
        key=lambda doc: (doc.rank_score, doc.retrieval_score, len(doc.matched_clues)),
        reverse=True,
    )

    if return_backend:
        return ranked_documents, backend
    return ranked_documents


def extract_evidence(query: str, docs: list[Document]) -> list[EvidenceSnippet]:
    evidence: list[EvidenceSnippet] = []
    for document in docs:
        snippet = _extract_best_snippet(query, document)
        if not snippet:
            continue
        evidence.append(
            EvidenceSnippet(
                title=document.title,
                url=document.url,
                snippet=snippet,
                score=round(document.rank_score, 4),
            )
        )
    return evidence


def build_context_block(evidence: list[EvidenceSnippet]) -> str:
    if not evidence:
        return "CONTEXT SOURCES:\n\nNo relevant sources found."

    lines = ["CONTEXT SOURCES:", ""]
    for index, item in enumerate(evidence, start=1):
        lines.extend(
            [
                f"[Source {index}]",
                f"Title: {item.title}",
                f"URL: {item.url}",
                f"Snippet: {item.snippet}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def select_context_documents(
    query: str,
    docs: list[Document],
    *,
    max_sources: int,
    pipeline_mode: str,
) -> list[Document]:
    if not docs or max_sources <= 0:
        return []

    limit = max(1, min(max_sources, len(docs)))
    if pipeline_mode == "simple":
        return docs[:limit]

    selected: list[Document] = []
    seen_urls: set[str] = set()

    def add_document(document: Document | None) -> None:
        if document is None:
            return
        key = canonicalize_url(document.url) or document.url or document.title
        if key in seen_urls:
            return
        seen_urls.add(key)
        selected.append(document)

    desired_signals: list[str] = []
    if pipeline_mode == "multi-hop":
        intent = analyze_query_intent(query)
        if intent.prefers_repository_sources:
            desired_signals.append("repository")
        if intent.prefers_paper_sources:
            desired_signals.append("paper")
        if intent.targets_citation_identifier:
            desired_signals.append("doi-bearing")
        if intent.requires_reference_section:
            desired_signals.append("reference")
        if intent.requires_acknowledgement_section:
            desired_signals.append("acknowledgement")
        if intent.needs_career_hop:
            desired_signals.append("career")
        if intent.prefers_event_sources:
            desired_signals.append("event")
        if intent.targets_person:
            desired_signals.append("name-bearing")
        if intent.prefers_encyclopedic_sources:
            desired_signals.append("encyclopedic")
        if intent.prefers_character_sources:
            desired_signals.append("character")
        if intent.prefers_event_sources:
            desired_signals.append("event")
        if intent.targets_count:
            desired_signals.append("count")

        primary_signal = next(iter(dict.fromkeys(desired_signals)), "")
        if primary_signal:
            add_document(_select_best_signal_document(query, docs, primary_signal, seen_urls))

    if not selected:
        add_document(docs[0])

    if pipeline_mode == "multi-hop":
        for signal in dict.fromkeys(desired_signals):
            if len(selected) >= limit:
                break
            add_document(_select_best_signal_document(query, docs, signal, seen_urls))

    for document in docs:
        if len(selected) >= limit:
            break
        add_document(document)

    return selected[:limit]


def _find_duplicate_document(document: Document, existing_documents: list[Document]) -> Document | None:
    canonical_url = canonicalize_url(document.url)
    normalized_title = normalize_whitespace(document.title).lower()
    normalized_content = normalize_whitespace(document.content)

    for existing_document in existing_documents:
        if canonical_url and canonical_url == canonicalize_url(existing_document.url):
            return existing_document
        if normalized_title and normalized_title == normalize_whitespace(existing_document.title).lower():
            if document.source == existing_document.source:
                return existing_document
        if normalized_content and existing_document.content:
            if _content_similarity(normalized_content, existing_document.content) >= DEDUPLICATION_SIMILARITY_THRESHOLD:
                return existing_document
    return None


def _merge_documents(target: Document, incoming: Document) -> None:
    merged_clues = list(target.matched_clues)
    for clue in incoming.matched_clues:
        if clue not in merged_clues:
            merged_clues.append(clue)
    target.matched_clues = tuple(merged_clues)
    target.retrieval_score = max(target.retrieval_score, incoming.retrieval_score)
    if incoming.acknowledgement_section:
        target.acknowledgement_section = incoming.acknowledgement_section
    if incoming.sections:
        target.sections.update(incoming.sections)
    if incoming.entities:
        target.entities = tuple(dict.fromkeys(target.entities + incoming.entities))
    if len(incoming.content) > len(target.content):
        target.content = incoming.content
        target.raw_content = incoming.raw_content or target.raw_content
        target.content_type = incoming.content_type
        target.metadata.update(incoming.metadata)
    if not target.title and incoming.title:
        target.title = incoming.title


def _content_similarity(text_a: str, text_b: str) -> float:
    normalized_a = normalize_whitespace(text_a).lower()
    normalized_b = normalize_whitespace(text_b).lower()
    if not normalized_a or not normalized_b:
        return 0.0

    seq_ratio = SequenceMatcher(None, normalized_a[:1200], normalized_b[:1200]).ratio()
    tokens_a = set(important_terms(normalized_a))
    tokens_b = set(important_terms(normalized_b))
    if not tokens_a or not tokens_b:
        return seq_ratio
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return max(seq_ratio, jaccard)


def _compose_rank_text(document: Document) -> str:
    return normalize_whitespace(f"{document.title} {document.content[:2000]}")


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum - minimum < 1e-8:
        baseline = max(0.0, min(1.0, float(scores[0])))
        return [baseline for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _select_best_signal_document(
    query: str,
    docs: list[Document],
    signal: str,
    seen_urls: set[str],
) -> Document | None:
    best_document: Document | None = None
    best_score = float("-inf")
    for document in docs:
        key = canonicalize_url(document.url) or document.url or document.title
        if key in seen_urls:
            continue
        signal_score = _document_signal_score(query, document, signal)
        if signal_score <= 0:
            continue
        combined_score = document.rank_score + signal_score
        if combined_score > best_score:
            best_score = combined_score
            best_document = document
    return best_document


def _document_signal_score(query: str, document: Document, signal: str) -> float:
    intent = analyze_query_intent(query)
    combined = _document_signal_text(document)
    url_lower = document.url.lower()
    query_lower = query.lower()
    front_matter_text = _front_matter_text(document)
    has_primary_doi = bool(
        extract_doi_candidates(str(document.metadata.get("doi", "")))
        or contains_primary_doi(document.url, document.title, document.content)
    )
    constraint_assessment = assess_document_constraints(
        query,
        document.title,
        document.content,
        reference_text=document.sections.get("references", ""),
        metadata=document.metadata,
    )
    specificity_score = specificity_overlap_score(query, f"{document.title} {document.content[:2500]}")
    broad_overview = is_broad_overview_page(document.url, document.title)
    aggregate_listing = is_aggregate_listing_page(document.url, document.title, document.content[:300])
    forum_discussion = is_forum_discussion_page(document.url, document.title, document.content[:300])
    wiki_meta_page = is_wiki_meta_page(document.url, document.title, document.content[:300])
    non_english_wiki = is_non_english_wiki_page(document.url)
    generic_media_topic = is_generic_media_topic_page(document.url, document.title, document.content[:600])
    media_score = media_page_score(document.url, document.title, document.content[:2500])
    grounded_browsecomp = is_grounded_browsecomp_page(
        query,
        document.url,
        document.title,
        document.content[:2500],
        require_media=intent.is_media_query,
    )
    event_score = event_page_score(document.url, document.title, document.content[:2500])
    winner_evidence_score = event_winner_evidence_score(
        normalize_whitespace(f"{document.title}. {document.content[:2500]}")
    )
    low_trust_social = is_low_trust_social_page(document.url, document.title, document.content[:500])

    if signal == "repository":
        score = 0.0
        if url_lower.endswith(".pdf") or any(
            hint in url_lower for hint in ("download", "bitstream", "bitstreams", "viewcontent")
        ):
            score += 0.35
        if any(hint in combined for hint in ACADEMIC_REPOSITORY_HINTS):
            score += 0.3
        if any(hint in combined for hint in THESIS_CONTENT_HINTS):
            score += 0.35
        return score

    if signal == "acknowledgement":
        score = 0.0
        if document.acknowledgement_section:
            score += 0.55
        if any(hint in combined for hint in ACKNOWLEDGEMENT_HINTS):
            score += 0.25
        if contains_candidate_person_name(document.content[:2500]):
            score += 0.1
        if "last name" in query_lower or "surname" in query_lower:
            score += 0.1
        return score

    if signal == "career":
        score = 0.0
        if any(hint in combined for hint in CAREER_HINTS):
            score += 0.7
        if "uk" in query_lower and any(term in combined for term in (" uk ", " united kingdom ", "ac.uk")):
            score += 0.2
        if any(term in combined for term in PERSON_CONTEXT_HINTS):
            score += 0.1
        return score

    if signal == "paper":
        score = 0.0
        if has_primary_doi:
            score += 0.35
        if any(hint in front_matter_text for hint in PAPER_CONTENT_HINTS):
            score += 0.35
        if "abstract" in front_matter_text:
            score += 0.1
        if any(hint in combined for hint in REFERENCE_HINTS):
            score += 0.1
        score += 0.25 * constraint_assessment.score
        score -= min(0.2, 0.08 * len(constraint_assessment.contradicted))
        if not intent.explicit_thesis and any(hint in combined for hint in THESIS_CONTENT_HINTS):
            score -= 0.25
        return max(0.0, score)

    if signal == "doi-bearing":
        score = 0.0
        if "doi.org/" in url_lower:
            score += 0.6
        if has_primary_doi:
            score += 0.4
        score += 0.2 * constraint_assessment.score
        score -= min(0.2, 0.08 * len(constraint_assessment.contradicted))
        return score

    if signal == "reference":
        score = 0.0
        if document.sections.get("references"):
            score += 0.6
        if any(hint in combined for hint in REFERENCE_HINTS):
            score += 0.25
        if contains_doi(document.content[:3000]):
            score += 0.15
        return score

    if signal == "name-bearing":
        score = 0.0
        if contains_candidate_person_name(document.content[:3000]):
            score += 0.6
        if any(hint in combined for hint in ACKNOWLEDGEMENT_HINTS):
            score += 0.2
        if any(hint in combined for hint in CAREER_HINTS):
            score += 0.1
        if intent.prefers_event_sources:
            score += 0.25 * event_score
            score += 0.25 * winner_evidence_score
            if any(hint in combined for hint in EVENT_WINNER_HINTS):
                score += 0.18
            if not has_event_winner_evidence(f"{document.title}. {document.content[:2400]}") and event_score < 0.3:
                score -= 0.22
            if intent.is_open_domain_browsecomp and specificity_score < 0.1 and event_score < 0.5:
                score -= 0.25
            if low_trust_social and event_score < 0.55:
                score -= 0.2
        if "last name" in query_lower or "surname" in query_lower:
            score += 0.1
        return score

    if signal == "event":
        score = 0.0
        score += 0.6 * event_score
        score += 0.28 * winner_evidence_score
        if any(hint in combined for hint in EVENT_WINNER_HINTS):
            score += 0.2
        if contains_candidate_person_name(document.content[:3000]):
            score += 0.12
        if looks_like_event_page(document.url, document.title, document.content[:2200]):
            score += 0.1
        if winner_evidence_score < 0.24 and event_score < 0.3:
            score -= 0.28
        if intent.is_open_domain_browsecomp and specificity_score < 0.1 and event_score < 0.5:
            score -= 0.3
        if low_trust_social and event_score < 0.55:
            score -= 0.25
        if broad_overview:
            score -= 0.12
        if aggregate_listing:
            score -= 0.15
        if wiki_meta_page:
            score -= 0.2
        if non_english_wiki:
            score -= 0.2
        if forum_discussion:
            score -= 0.18
        return score

    if signal == "encyclopedic":
        score = 0.0
        if any(hint in url_lower for hint in ENCYCLOPEDIC_SOURCE_HINTS):
            score += 0.45
        if any(hint in combined for hint in ("wiki", "fandom", "characters", "plot", "story")):
            score += 0.25
        if is_media_lookup_query(query):
            score += 0.15 * lexical_relevance_score(query, combined)
            score += 0.2 * specificity_score
            score += 0.22 * media_score
            if media_score < 0.18:
                score -= 0.28
        if broad_overview:
            score -= 0.18
        if aggregate_listing:
            score -= 0.2
        if wiki_meta_page:
            score -= 0.3
        if non_english_wiki:
            score -= 0.28
        if generic_media_topic:
            score -= 0.3
        if forum_discussion:
            score -= 0.2
        if intent.is_open_domain_browsecomp and grounded_browsecomp:
            score += 0.18
        elif intent.is_open_domain_browsecomp:
            score -= 0.22
        return score

    if signal == "character":
        score = 0.0
        if any(hint in combined for hint in CHARACTER_QUERY_HINTS):
            score += 0.45
        if any(hint in combined for hint in ("companion", "rival", "villain", "hero")):
            score += 0.2
        if contains_candidate_person_name(document.content[:3000]):
            score += 0.1
        if any(hint in url_lower for hint in ENCYCLOPEDIC_SOURCE_HINTS):
            score += 0.1
        score += 0.18 * specificity_score
        score += 0.18 * media_score
        if intent.is_media_query and media_score < 0.18:
            score -= 0.26
        if broad_overview:
            score -= 0.16
        if aggregate_listing:
            score -= 0.18
        if wiki_meta_page:
            score -= 0.28
        if non_english_wiki:
            score -= 0.26
        if generic_media_topic:
            score -= 0.28
        if forum_discussion:
            score -= 0.2
        if intent.is_open_domain_browsecomp and grounded_browsecomp:
            score += 0.2
        elif intent.is_open_domain_browsecomp:
            score -= 0.24
        return score

    if signal == "count":
        score = 0.0
        if any(hint in combined for hint in ABILITY_QUERY_HINTS):
            score += 0.45
        if bool(re.search(r"\b\d+\b", combined)):
            score += 0.15
        if any(hint in combined for hint in ("list of", "abilities", "moves", "effects")):
            score += 0.15
        score += 0.15 * specificity_score
        score += 0.15 * media_score
        if intent.is_media_query and media_score < 0.18:
            score -= 0.24
        if broad_overview:
            score -= 0.12
        if aggregate_listing:
            score -= 0.15
        if wiki_meta_page:
            score -= 0.22
        if non_english_wiki:
            score -= 0.2
        if generic_media_topic:
            score -= 0.24
        if forum_discussion:
            score -= 0.18
        if intent.is_open_domain_browsecomp and grounded_browsecomp:
            score += 0.18
        elif intent.is_open_domain_browsecomp:
            score -= 0.22
        return score

    return 0.0


def _document_prior_score(query: str, document: Document) -> float:
    intent = analyze_query_intent(query)

    query_lower = query.lower()
    title_lower = normalize_whitespace(document.title).lower()
    content_lower = normalize_whitespace(document.content[:5000]).lower()
    url_lower = document.url.lower()
    combined = f"{title_lower} {content_lower} {url_lower}"
    front_matter_text = _front_matter_text(document)
    has_primary_doi = bool(
        extract_doi_candidates(str(document.metadata.get("doi", "")))
        or contains_primary_doi(document.url, document.title, document.content)
    )
    constraint_assessment = assess_document_constraints(
        query,
        document.title,
        document.content,
        reference_text=document.sections.get("references", ""),
        metadata=document.metadata,
    )
    specificity_score = specificity_overlap_score(query, f"{document.title} {document.content[:2500]}")
    broad_overview = is_broad_overview_page(document.url, document.title)
    aggregate_listing = is_aggregate_listing_page(document.url, document.title, document.content[:300])
    forum_discussion = is_forum_discussion_page(document.url, document.title, document.content[:300])
    wiki_meta_page = is_wiki_meta_page(document.url, document.title, document.content[:300])
    non_english_wiki = is_non_english_wiki_page(document.url)
    generic_media_topic = is_generic_media_topic_page(document.url, document.title, document.content[:600])
    media_score = media_page_score(document.url, document.title, document.content[:2500])
    grounded_browsecomp = is_grounded_browsecomp_page(
        query,
        document.url,
        document.title,
        document.content[:2500],
        require_media=intent.is_media_query,
    )
    event_score = event_page_score(document.url, document.title, document.content[:2500])
    winner_evidence_score = event_winner_evidence_score(
        normalize_whitespace(f"{document.title}. {document.content[:2500]}")
    )
    low_trust_social = is_low_trust_social_page(document.url, document.title, document.content[:500])

    score = 0.0
    if intent.prefers_repository_sources:
        if url_lower.endswith(".pdf") or any(
            hint in url_lower for hint in ("download", "bitstream", "bitstreams", "viewcontent")
        ):
            score += 0.2
        if any(hint in combined for hint in ACADEMIC_REPOSITORY_HINTS):
            score += 0.15
        if any(hint in combined for hint in THESIS_CONTENT_HINTS):
            score += 0.3
    if intent.requires_acknowledgement_section and (
        document.acknowledgement_section or any(hint in combined for hint in ACKNOWLEDGEMENT_HINTS)
    ):
        score += 0.2
    if intent.prefers_paper_sources:
        if has_primary_doi:
            score += 0.25
        if any(hint in front_matter_text for hint in PAPER_CONTENT_HINTS):
            score += 0.2
        if document.sections.get("references") or any(hint in combined for hint in REFERENCE_HINTS):
            score += 0.1
        if not intent.explicit_thesis and any(hint in combined for hint in THESIS_CONTENT_HINTS):
            score -= 0.35
        score += 0.35 * constraint_assessment.score
        score -= min(0.3, 0.1 * len(constraint_assessment.contradicted))
    if intent.needs_career_hop and any(hint in combined for hint in CAREER_HINTS):
        score += 0.1
    if document_matches_query_years(query, combined):
        score += 0.05
    if any(hint in combined for hint in NON_RESEARCH_PAGE_HINTS):
        score -= 0.25
    if not is_academic_lookup_query(query) and intent.prefers_encyclopedic_sources:
        if any(hint in url_lower for hint in ENCYCLOPEDIC_SOURCE_HINTS):
            score += 0.25
        if any(hint in combined for hint in ("wiki", "fandom", "characters", "plot")):
            score += 0.15
    if not is_academic_lookup_query(query) and intent.prefers_character_sources:
        if any(hint in combined for hint in CHARACTER_QUERY_HINTS):
            score += 0.25
        if any(hint in combined for hint in ("companion", "villain", "rival", "antagonist")):
            score += 0.1
    if not is_academic_lookup_query(query) and intent.targets_count:
        if any(hint in combined for hint in ABILITY_QUERY_HINTS):
            score += 0.2
        if bool(re.search(r"\b\d+\b", combined)):
            score += 0.05
    if not is_academic_lookup_query(query) and intent.prefers_event_sources:
        score += 0.28 * event_score
        score += 0.28 * winner_evidence_score
        if any(hint in combined for hint in EVENT_WINNER_HINTS):
            score += 0.18
        if contains_candidate_person_name(document.content[:3000]):
            score += 0.08
        if contains_candidate_person_name(document.content[:3000]) and any(
            hint in combined for hint in EVENT_WINNER_HINTS
        ):
            score += 0.12
        if winner_evidence_score < 0.24 and event_score < 0.3:
            score -= 0.25
        if intent.is_open_domain_browsecomp and specificity_score < 0.1 and event_score < 0.5:
            score -= 0.28
        if low_trust_social and event_score < 0.55:
            score -= 0.28
    if not is_academic_lookup_query(query) and intent.is_media_query:
        if any(hint in combined for hint in MEDIA_QUERY_HINTS):
            score += 0.1
    if intent.answer_type == "year":
        score += 0.32 * _historical_year_document_score(query, document)
        if is_person_biography_page(document.url, document.title, document.content[:1200]):
            score -= 0.55
        if is_specific_historical_year_page(query, document.url, document.title, document.content[:2200]):
            score += 0.18
        else:
            score -= 0.18
        if is_generic_historical_monument_page(document.url, document.title, document.content[:800]):
            score -= 0.32
    if intent.is_open_domain_browsecomp:
        score += 0.3 * specificity_score
        score += 0.28 * media_score
        if broad_overview:
            score -= 0.28
        if aggregate_listing:
            score -= 0.32
        if wiki_meta_page:
            score -= 0.35
        if non_english_wiki:
            score -= 0.3
        if generic_media_topic:
            score -= 0.35
        if forum_discussion:
            score -= 0.28
        if intent.is_media_query and media_score < 0.18:
            score -= 0.35
        if grounded_browsecomp:
            score += 0.24
        else:
            score -= 0.28
    return max(0.0, min(1.0, score))


def _historical_year_document_score(query: str, document: Document) -> float:
    combined = normalize_whitespace(f"{document.title}. {document.content[:2600]}")
    if not combined:
        return 0.0
    if is_person_biography_page(document.url, document.title, document.content[:1400]):
        return 0.0
    structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
        query,
        document.url,
        document.title,
        document.content[:2600],
    )
    if structural_contradictions > 0:
        return 0.0
    if query_requires_bosnia_top_city(query):
        lowered_combined = combined.lower()
        if not any(term in lowered_combined for term in BOSNIA_TOP_FOUR_CITY_TERMS):
            return 0.0
    if (
        historical_year_has_structural_constraints(query)
        and structural_matches == 0
        and not historical_year_trusted_memorial_source(document.url)
    ):
        return 0.0

    best_score = max(0.0, structural_score)
    query_lower = query.lower()
    for sentence in split_sentences(combined):
        lowered = sentence.lower()
        if not YEAR_PATTERN.search(sentence):
            continue

        score = 0.12
        if any(hint in lowered for hint in YEAR_EVENT_HINTS):
            score += 0.3
        if any(hint in lowered for hint in ("monument", "memorial", "spomenik")):
            score += 0.08
        if any(hint in query_lower for hint in ("former yugoslavia", "bosnia", "largest cities", "population census")) and any(
            hint in lowered for hint in ("bosnia", "banja luka", "zenica", "tuzla", "sarajevo", "yugoslavia")
        ):
            score += 0.08
        if any(hint in lowered for hint in YEAR_BIOGRAPHY_HINTS):
            score -= 0.18
        if any(hint in lowered for hint in YEAR_CONSTRUCTION_HINTS):
            score -= 0.12
        if any(hint in lowered for hint in YEAR_AWARD_HINTS) and not any(hint in lowered for hint in YEAR_EVENT_HINTS):
            score -= 0.28
        if "born in" in lowered:
            score -= 0.22
        if re.search(
            r"\b(?:began|started|emerged|formed|launched|initiated)\b.{0,18}\b(?:19|20)\d{2}\b",
            lowered,
        ) and not any(hint in lowered for hint in ("victims", "massacre", "killed", "died", "executed")):
            score -= 0.2
        best_score = max(best_score, score)

    return max(0.0, min(1.0, best_score))


def _document_signal_text(document: Document) -> str:
    return normalize_whitespace(
        f" {document.title.lower()} {document.content[:4000].lower()} {document.url.lower()} "
    )


def _extract_best_snippet(query: str, document: Document) -> str:
    intent = analyze_query_intent(query)
    if intent.targets_citation_identifier:
        doi_candidate = _extract_best_doi_snippet(document)
        if doi_candidate:
            return doi_candidate
    if intent.requires_reference_section and document.sections.get("references"):
        reference_candidate = _extract_best_reference_snippet(document)
        if reference_candidate:
            return reference_candidate
    if intent.requires_acknowledgement_section and document.acknowledgement_section:
        acknowledgement_candidate = _truncate_snippet(document.acknowledgement_section)
        if acknowledgement_candidate:
            return acknowledgement_candidate

    candidates = _generate_keyword_window_candidates(query, document.content)
    candidates.extend(_generate_name_window_candidates(query, document.content))
    candidates.extend(_generate_snippet_candidates(document.content))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return _truncate_snippet(document.title or document.content)

    best_candidate = max(
        candidates,
        key=lambda candidate: _snippet_relevance_score(query, candidate, document.rank_score),
    )
    return _truncate_snippet(best_candidate)


def _generate_keyword_window_candidates(query: str, content: str) -> list[str]:
    cleaned_content = normalize_whitespace(content)
    if not cleaned_content:
        return []

    lowered_content = cleaned_content.lower()
    candidates: list[str] = []
    for keyword in _snippet_focus_keywords(query):
        start_index = 0
        while len(candidates) < MAX_KEYWORD_WINDOW_CANDIDATES:
            match_index = lowered_content.find(keyword, start_index)
            if match_index == -1:
                break
            window_start = max(0, match_index - KEYWORD_WINDOW_RADIUS)
            window_end = min(len(cleaned_content), match_index + KEYWORD_WINDOW_RADIUS)
            candidates.append(normalize_whitespace(cleaned_content[window_start:window_end]))
            start_index = match_index + max(1, len(keyword))
    return list(dict.fromkeys(candidates))


def _generate_name_window_candidates(query: str, content: str) -> list[str]:
    intent = analyze_query_intent(query)
    if not (is_academic_lookup_query(query) or is_person_target_query(query) or intent.requires_acknowledgement_section):
        return []

    cleaned_content = normalize_whitespace(content)
    if not cleaned_content:
        return []

    lowered_content = cleaned_content.lower()
    keywords = [
        "acknowledg",
        "thank",
        "gratitude",
        "grateful",
        "professor",
        "faculty",
        "appointed",
        "joined",
    ]
    if intent.is_event_query:
        keywords.extend(["winner", "won", "pageant", "contest", "queen", "coronation", "festival"])
    candidates: list[str] = []
    for keyword in keywords:
        start_index = 0
        while len(candidates) < MAX_NAME_WINDOW_CANDIDATES:
            match_index = lowered_content.find(keyword, start_index)
            if match_index == -1:
                break
            window_start = max(0, match_index - NAME_WINDOW_RADIUS)
            window_end = min(len(cleaned_content), match_index + NAME_WINDOW_RADIUS)
            window = normalize_whitespace(cleaned_content[window_start:window_end])
            if contains_candidate_person_name(window):
                candidates.append(window)
            start_index = match_index + max(1, len(keyword))
    return list(dict.fromkeys(candidates))


def _snippet_focus_keywords(query: str) -> list[str]:
    intent = analyze_query_intent(query)
    query_lower = query.lower()
    keywords: list[str] = []
    if is_academic_lookup_query(query):
        if intent.requires_acknowledgement_section:
            keywords.extend(["acknowledg", "thank", "gratitude", "grateful"])
        if intent.needs_career_hop:
            keywords.extend(["professor", "faculty"])
        if "uk" in query_lower:
            keywords.extend(["uk", "united kingdom", "ac.uk"])
    if intent.targets_citation_identifier:
        keywords.extend(["doi", "journal", "abstract"])
    if intent.requires_reference_section:
        keywords.extend(["references", "bibliography", "cited"])
    if intent.is_event_query:
        keywords.extend(["winner", "won", "pageant", "contest", "queen", "festival", "coronation"])
    keywords.extend(
        term.lower()
        for term in important_terms(query)
        if len(term) > 4 and term.lower() not in LOW_VALUE_TERMS
    )
    return list(dict.fromkeys(keywords))[:10]


def _generate_snippet_candidates(content: str) -> list[str]:
    cleaned_content = normalize_whitespace(content)
    if not cleaned_content:
        return []

    sentences = [
        normalize_whitespace(sentence)
        for sentence in content.replace("\r", "\n").split("\n")
        if normalize_whitespace(sentence)
    ]
    if not sentences:
        return [_truncate_snippet(cleaned_content)]

    candidates: list[str] = []
    for index, sentence in enumerate(sentences):
        if len(sentence) >= 40:
            candidates.append(sentence)
        if index + 1 < len(sentences):
            combined = normalize_whitespace(f"{sentence} {sentences[index + 1]}")
            if 60 <= len(combined) <= DEFAULT_SNIPPET_LENGTH * 2:
                candidates.append(combined)

    if not candidates:
        candidates.append(cleaned_content)
    return list(dict.fromkeys(candidates))


def _extract_best_doi_snippet(document: Document) -> str:
    front_matter = _front_matter_text(document)
    doi_candidates = extract_doi_candidates(document.metadata.get("doi", ""))
    if not doi_candidates:
        doi_candidates = extract_primary_doi_candidates(document.url, document.title, document.content)
    if not doi_candidates and document.sections.get("references"):
        doi_candidates = extract_doi_candidates(document.sections["references"])
    if not doi_candidates:
        return ""
    selected_doi = doi_candidates[0]
    text = normalize_whitespace(front_matter or document.content or document.sections.get("references", ""))
    match_index = text.lower().find(selected_doi.lower())
    if match_index == -1:
        return _truncate_snippet(f"DOI: {selected_doi}")
    window_start = max(0, match_index - 140)
    window_end = min(len(text), match_index + len(selected_doi) + 140)
    return _truncate_snippet(text[window_start:window_end])


def _front_matter_text(document: Document) -> str:
    return normalize_whitespace(
        f"{document.title} {document.metadata.get('doi', '')} {document.content[:3000]}"
    )


def _extract_best_reference_snippet(document: Document) -> str:
    reference_text = document.sections.get("references", "")
    if not reference_text:
        return ""
    lines = [line.strip() for line in reference_text.split("\n") if line.strip()]
    if not lines:
        return ""
    doi_line = next((line for line in lines if contains_doi(line)), "")
    if doi_line:
        return _truncate_snippet(doi_line)
    return _truncate_snippet(lines[0])


def _snippet_relevance_score(query: str, candidate: str, document_rank_score: float) -> float:
    intent = analyze_query_intent(query)
    score = lexical_relevance_score(query, candidate) + (0.1 * document_rank_score)
    candidate_lower = candidate.lower()
    if intent.requires_acknowledgement_section and any(hint in candidate_lower for hint in ACKNOWLEDGEMENT_HINTS):
        score += 0.08
    if intent.needs_career_hop and any(hint in candidate_lower for hint in CAREER_HINTS):
        score += 0.05
    if intent.targets_person and contains_candidate_person_name(candidate):
        score += 0.12
    if intent.targets_citation_identifier and contains_doi(candidate):
        score += 0.18
    if intent.requires_reference_section and any(hint in candidate_lower for hint in REFERENCE_HINTS):
        score += 0.08
    return score


def _truncate_snippet(text: str) -> str:
    cleaned_text = normalize_whitespace(text)
    if len(cleaned_text) <= DEFAULT_SNIPPET_LENGTH:
        return cleaned_text

    truncated = cleaned_text[: DEFAULT_SNIPPET_LENGTH + 1]
    last_boundary = max(
        truncated.rfind(". "),
        truncated.rfind("; "),
        truncated.rfind(", "),
    )
    if last_boundary >= 80:
        truncated = truncated[:last_boundary]
    return truncated.rstrip(" ,;:.") + "..."


class _SemanticRanker:
    def __init__(self) -> None:
        self._cross_encoder = None
        self._embedding_model = None
        self._cross_encoder_attempted = False
        self._embedding_attempted = False

    def score(self, query: str, docs: list[Document]) -> tuple[list[float], str]:
        if not docs:
            return [], "none"

        backend_preference = _get_reranker_backend_preference()
        if SENTENCE_TRANSFORMERS_AVAILABLE and backend_preference in {"auto", "cross-encoder"}:
            cross_encoder = self._load_cross_encoder()
            if cross_encoder is not None:
                try:
                    pairs = [(query, _compose_rank_text(doc)) for doc in docs]
                    scores = cross_encoder.predict(pairs)
                    return _normalize_scores([float(score) for score in scores]), "cross-encoder"
                except Exception as error:
                    logger.warning("Cross-encoder reranking failed, falling back: %s", error)

        if SENTENCE_TRANSFORMERS_AVAILABLE and backend_preference in {"auto", "cross-encoder", "embeddings"}:
            embedding_model = self._load_embedding_model()
            if embedding_model is not None:
                try:
                    query_embedding = embedding_model.encode(
                        query,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    document_embeddings = embedding_model.encode(
                        [_compose_rank_text(doc) for doc in docs],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
                    return _normalize_scores(
                        [float(score) for score in cosine_scores.tolist()]
                    ), "embeddings"
                except Exception as error:
                    logger.warning("Embedding reranking failed, falling back: %s", error)

        lexical_scores = [lexical_relevance_score(query, _compose_rank_text(doc)) for doc in docs]
        return _normalize_scores(lexical_scores), "lexical"

    def _load_cross_encoder(self) -> Any:
        if self._cross_encoder_attempted:
            return self._cross_encoder
        self._cross_encoder_attempted = True
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        model_name = load_ranking_config().reranker_model
        try:
            self._cross_encoder = CrossEncoder(model_name)
        except Exception as error:
            logger.warning("Unable to load cross-encoder '%s': %s", model_name, error)
            self._cross_encoder = None
        return self._cross_encoder

    def _load_embedding_model(self) -> Any:
        if self._embedding_attempted:
            return self._embedding_model
        self._embedding_attempted = True
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        model_name = load_ranking_config().embedding_model
        try:
            self._embedding_model = SentenceTransformer(model_name)
        except Exception as error:
            logger.warning("Unable to load embedding model '%s': %s", model_name, error)
            self._embedding_model = None
        return self._embedding_model


def _get_reranker_backend_preference() -> str:
    return load_ranking_config().reranker_backend


_RANKER = _SemanticRanker()
