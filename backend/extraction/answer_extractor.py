from __future__ import annotations

import logging
import re

from agent.models import AnswerCandidate, AnswerExtractionOutput, Document, EvidenceSnippet
from config.extraction_config import (
    DEFAULT_ANSWER_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    load_answer_extraction_config,
)
from extraction.entity_extractor import extract_name_candidates_with_evidence
from planner.query_constraints import assess_document_constraints
from planner.query_intent import QueryIntent, analyze_query_intent
from utils.text_utils import (
    BOSNIA_TOP_FOUR_CITY_TERMS,
    ABILITY_QUERY_HINTS,
    CHARACTER_QUERY_HINTS,
    EVENT_WINNER_HINTS,
    PAPER_CONTENT_HINTS,
    THESIS_CONTENT_HINTS,
    contains_doi,
    contains_primary_doi,
    contains_candidate_person_name,
    document_title_query_phrase,
    event_page_score,
    event_winner_evidence_score,
    extract_capitalized_entities,
    extract_doi_candidates,
    extract_institutions_from_text,
    extract_primary_doi_candidates,
    has_event_winner_evidence,
    historical_year_has_structural_constraints,
    historical_year_structural_assessment,
    historical_year_trusted_memorial_source,
    is_plausible_person_name,
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
    is_recipe_food_page,
    is_specific_historical_year_page,
    is_wiki_meta_page,
    lexical_relevance_score,
    looks_like_event_page,
    looks_like_media_page,
    looks_like_reference_citation,
    media_page_score,
    normalize_whitespace,
    query_requires_bosnia_top_city,
    specificity_overlap_score,
    split_sentences,
)

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

logger = logging.getLogger("extraction.answer_extractor")

DOI_FRONT_MATTER_CHARS = 5000
DOI_BODY_CHARS = 8000

REFERENCE_DOI_TARGET_HINTS = (
    "doi of the reference",
    "doi for the reference",
    "reference doi",
    "cited reference doi",
    "doi of the cited reference",
    "doi for the cited reference",
    "doi of the cited paper",
    "doi for the cited paper",
    "reference's doi",
)

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_EVENT_HINTS = (
    "event",
    "occur",
    "loss of lives",
    "lost their lives",
    "victims",
    "victim",
    "massacre",
    "battle",
    "attack",
    "tragedy",
    "killed",
    "died",
    "death",
    "executed",
    "honor",
    "in their honor",
    "commemorates",
    "commemorating",
    "dedicated",
    "dedication",
    "memorial",
    "monument",
    "spomenik",
)
YEAR_BIOGRAPHICAL_HINTS = (
    "born",
    "birth",
    "artist",
    "sculptor",
    "author",
    "architect",
)
YEAR_CONSTRUCTION_HINTS = (
    "built",
    "constructed",
    "erected",
    "unveiled",
    "opened",
    "completed",
    "installed",
)
YEAR_CENSUS_HINTS = ("census", "population")
YEAR_AWARD_HINTS = (
    "award",
    "awards",
    "competition",
    "prize",
    "salon",
    "triennale",
    "biennale",
    "medal",
)

ANSWER_EXTRACTION_SYSTEM_PROMPT = """
You extract grounded answers from research evidence.

Rules:
- Use only the supplied evidence.
- If the question asks for a last name or surname, return only the surname in `answer`.
- If the question asks for a DOI, return only the DOI string in `answer`.
- If the question asks for a year, return only the 4-digit year in `answer`.
- If the question asks for both a name and a count, return them compactly as `Name, count`.
- For DOI questions, return the DOI of the target work itself, not a DOI that appears only in cited references unless the user explicitly asks for a cited reference's DOI.
- Put the shortest direct supporting sentence or phrase in `evidence`.
- Put the full person name, if present, in `supporting_person`.
- Confidence must be between 0 and 1.
- If the evidence is insufficient, leave `answer` blank and set confidence below 0.4.
""".strip()


class AnswerExtractor:
    async def extract(
        self,
        query: str,
        documents: list[Document],
        evidence_snippets: list[EvidenceSnippet],
    ) -> AnswerCandidate:
        intent = analyze_query_intent(query)
        candidate_docs = self._select_candidate_documents(query, intent, documents)
        if not candidate_docs and evidence_snippets:
            return self._heuristic_from_snippets(query, intent, evidence_snippets)
        if not candidate_docs:
            return AnswerCandidate()

        if intent.targets_citation_identifier:
            heuristic_doi = self._heuristic_doi_from_documents(query, intent, candidate_docs)
            if heuristic_doi.answer and heuristic_doi.confidence >= 0.7:
                return heuristic_doi

            llm_answer = await self._extract_with_openai(query, intent, candidate_docs)
            if self._is_valid_doi_answer(query, llm_answer, candidate_docs):
                return llm_answer

            if heuristic_doi.answer:
                return heuristic_doi

            if evidence_snippets:
                return self._heuristic_from_snippets(query, intent, evidence_snippets)
            return AnswerCandidate()

        if intent.answer_type == "year":
            heuristic_year = self._heuristic_year_from_documents(query, candidate_docs)
            if heuristic_year.answer and heuristic_year.confidence >= 0.72:
                return heuristic_year

            llm_answer = await self._extract_with_openai(query, intent, candidate_docs)
            if self._is_valid_year_answer(query, llm_answer, candidate_docs):
                return llm_answer

            if heuristic_year.answer:
                return heuristic_year

            if evidence_snippets:
                snippet_candidate = self._heuristic_from_snippets(query, intent, evidence_snippets)
                if snippet_candidate.answer:
                    return snippet_candidate
            return AnswerCandidate()

        if intent.answer_type in {"person_last_name", "person_name"}:
            llm_answer = await self._extract_with_openai(query, intent, candidate_docs)
            if self._is_valid_person_answer(query, intent, llm_answer, candidate_docs):
                return llm_answer

            heuristic_person = self._heuristic_person_from_documents(query, candidate_docs)
            if heuristic_person.answer:
                return heuristic_person

            if evidence_snippets:
                snippet_candidate = self._heuristic_from_snippets(query, intent, evidence_snippets)
                if snippet_candidate.answer:
                    return snippet_candidate
            return AnswerCandidate()

        llm_answer = await self._extract_with_openai(query, intent, candidate_docs)
        if llm_answer.answer:
            return llm_answer
        return self._heuristic_from_documents(query, intent, candidate_docs, evidence_snippets)

    async def _extract_with_openai(
        self,
        query: str,
        intent: QueryIntent,
        documents: list[Document],
    ) -> AnswerCandidate:
        extraction_config = load_answer_extraction_config()
        if not extraction_config.openai_api_key or not OPENAI_AVAILABLE:
            return AnswerCandidate()

        evidence_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"[Document {index}]",
                        f"Title: {document.title}",
                        f"URL: {document.url}",
                        f"Relevant Evidence:\n{self._relevant_document_excerpt(query, intent, document)[:2500]}",
                    ]
                )
                for index, document in enumerate(documents[:4], start=1)
            ]
        )

        client = AsyncOpenAI(
            api_key=extraction_config.openai_api_key,
            timeout=extraction_config.timeout_seconds,
        )
        try:
            response = await client.responses.parse(
                model=extraction_config.model,
                input=[
                    {"role": "system", "content": ANSWER_EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{query}\n\n"
                            f"Task Type: {intent.answer_type}\n\n"
                            f"Evidence:\n{evidence_block}"
                        ),
                    },
                ],
                text_format=AnswerExtractionOutput,
            )
        except Exception as error:
            logger.warning("OpenAI answer extraction failed; using heuristic fallback: %s", error)
            return AnswerCandidate()
        finally:
            await client.close()

        parsed_output = response.output_parsed
        if parsed_output is None or not parsed_output.answer:
            return AnswerCandidate()

        matched_source = next(
            (
                document.url
                for document in documents
                if parsed_output.evidence
                and parsed_output.evidence
                in self._relevant_document_excerpt(query, intent, document)
            ),
            documents[0].url,
        )
        return AnswerCandidate(
            answer=parsed_output.answer.strip(),
            evidence=parsed_output.evidence.strip(),
            source=matched_source,
            confidence=float(parsed_output.confidence),
            supporting_person=parsed_output.supporting_person.strip(),
        )

    def _select_candidate_documents(
        self,
        query: str,
        intent: QueryIntent,
        documents: list[Document],
    ) -> list[Document]:
        if intent.targets_citation_identifier:
            selected = [
                document
                for document in documents
                if self._document_has_primary_doi(document) or self._document_looks_like_paper(document)
            ]
            if selected:
                return selected
            if intent.requires_reference_section:
                selected = [
                    document
                    for document in documents
                    if document.sections.get("references") or contains_doi(document.content)
                ]
                if selected:
                    return selected
            return documents

        if intent.answer_type == "year":
            selected = sorted(
                documents,
                key=lambda document: self._year_document_score(query, document),
                reverse=True,
            )
            filtered = [
                document
                for document in selected
                if self._year_document_score(query, document) > 0.18
                and (
                    not intent.is_open_domain_browsecomp
                    or is_specific_historical_year_page(
                        query,
                        document.url,
                        document.title,
                        document.content[:2200],
                    )
                )
            ]
            return filtered or selected[: max(1, min(4, len(selected)))]

        if intent.requires_acknowledgement_section:
            selected = [
                document
                for document in documents
                if document.acknowledgement_section or document.sections.get("acknowledgements")
            ]
            return selected or documents

        if intent.requires_reference_section:
            selected = [
                document
                for document in documents
                if document.sections.get("references") or contains_doi(document.content)
            ]
            return selected or documents

        if intent.answer_type in {"person_last_name", "person_name"}:
            selected = sorted(
                documents,
                key=lambda document: self._person_document_score(query, document, intent),
                reverse=True,
            )
            threshold = 0.22 if intent.prefers_event_sources else 0.08
            filtered = [
                document
                for document in selected
                if self._person_document_score(query, document, intent) > threshold
                and (
                    not intent.prefers_event_sources
                    or self._is_grounded_event_person_document(query, intent, document)
                )
            ]
            if intent.prefers_event_sources:
                return filtered[: max(1, min(4, len(filtered)))] if filtered else []
            return filtered or selected[: max(1, min(4, len(selected)))]

        if intent.answer_type in {"entity_and_count", "entity_name", "count"} or intent.is_open_domain_browsecomp:
            selected = sorted(
                documents,
                key=lambda document: self._generic_document_score(query, document, intent),
                reverse=True,
            )
            threshold = 0.18 if intent.is_open_domain_browsecomp else 0.0
            filtered = [
                document
                for document in selected
                if self._generic_document_score(query, document, intent) > threshold
                and (
                    not intent.is_media_query
                    or (
                        looks_like_media_page(document.url, document.title, document.content[:2500], minimum_score=0.18)
                        and not is_wiki_meta_page(document.url, document.title, document.content[:600])
                        and not is_non_english_wiki_page(document.url)
                        and not is_generic_media_topic_page(document.url, document.title, document.content[:1200])
                    )
                    or document_title_query_phrase(document.title) != ""
                )
            ]
            return filtered or selected[: max(1, min(4, len(selected)))]

        return documents

    def _relevant_document_excerpt(self, query: str, intent: QueryIntent, document: Document) -> str:
        if intent.requires_acknowledgement_section:
            return document.acknowledgement_section or document.sections.get("acknowledgements", "") or document.content
        if intent.targets_citation_identifier:
            primary_evidence = self._build_primary_doi_evidence(document)
            if primary_evidence:
                return primary_evidence
            if self._query_targets_reference_doi(query) and document.sections.get("references"):
                return document.sections["references"]
        if intent.requires_reference_section and document.sections.get("references"):
            return document.sections["references"]
        if intent.answer_type in {"entity_and_count", "entity_name", "count"} or intent.is_open_domain_browsecomp:
            generic_excerpt = self._build_generic_evidence(query, intent, document)
            if generic_excerpt:
                return generic_excerpt
        return document.content

    def _heuristic_from_documents(
        self,
        query: str,
        intent: QueryIntent,
        documents: list[Document],
        evidence_snippets: list[EvidenceSnippet],
    ) -> AnswerCandidate:
        if intent.targets_citation_identifier:
            doi_candidate = self._heuristic_doi_from_documents(query, intent, documents)
            if doi_candidate.answer:
                return doi_candidate
        if intent.answer_type in {"person_last_name", "person_name"}:
            person_candidate = self._heuristic_person_from_documents(query, documents)
            if person_candidate.answer:
                return person_candidate
        if intent.answer_type == "title":
            title_candidate = self._heuristic_title_from_documents(documents)
            if title_candidate.answer:
                return title_candidate
        if intent.answer_type == "institution":
            institution_candidate = self._heuristic_institution_from_documents(documents)
            if institution_candidate.answer:
                return institution_candidate
        if intent.answer_type == "year":
            year_candidate = self._heuristic_year_from_documents(query, documents)
            if year_candidate.answer:
                return year_candidate
        if intent.answer_type in {"entity_and_count", "entity_name", "count", "generic"} or intent.is_open_domain_browsecomp:
            generic_candidate = self._heuristic_generic_from_documents(query, intent, documents)
            if generic_candidate.answer:
                return generic_candidate
        if evidence_snippets:
            return self._heuristic_from_snippets(query, intent, evidence_snippets)
        return AnswerCandidate()

    def _heuristic_doi_from_documents(
        self,
        query: str,
        intent: QueryIntent,
        documents: list[Document],
    ) -> AnswerCandidate:
        target_reference_doi = self._query_targets_reference_doi(query)
        best_candidate = AnswerCandidate()
        best_score = float("-inf")

        for document in documents:
            candidate_locations = self._document_doi_locations(document)
            if not candidate_locations:
                continue

            constraint_assessment = assess_document_constraints(
                query,
                document.title,
                document.content,
                reference_text=document.sections.get("references", ""),
                metadata=document.metadata,
            )
            paper_context_score = lexical_relevance_score(
                query,
                f"{document.title} {self._front_matter_text(document)}",
            ) * 0.3
            thesis_penalty = (
                0.25
                if not target_reference_doi
                and any(hint in self._front_matter_text(document).lower() for hint in THESIS_CONTENT_HINTS)
                else 0.0
            )
            if not target_reference_doi and self._has_critical_constraint_conflict(constraint_assessment):
                continue

            for doi, locations in candidate_locations.items():
                if not target_reference_doi and locations <= {"reference"}:
                    continue

                evidence_text = self._best_doi_source_text(document, doi, target_reference_doi)
                evidence = self._best_doi_evidence(evidence_text, preferred_doi=doi) or f"DOI: {doi}"
                score = 0.05
                score += 0.2 * max(0.0, min(1.0, document.rank_score))
                score += 0.1 * max(0.0, min(1.0, document.retrieval_score))
                score += paper_context_score + (0.5 * constraint_assessment.score)
                score += (0.08 if self._document_looks_like_paper(document) else 0.0)
                score += (0.05 if self._document_has_primary_doi(document) else 0.0)
                score -= thesis_penalty
                score -= min(0.5, 0.22 * len(constraint_assessment.contradicted))

                if "metadata" in locations:
                    score += 0.22
                if "url" in locations:
                    score += 0.2 if "doi.org/" in document.url.lower() else 0.08
                if "title" in locations:
                    score += 0.05
                if "front" in locations:
                    score += 0.15
                if "body" in locations:
                    score += 0.04
                if "reference" in locations:
                    score += 0.06 if target_reference_doi else -0.25
                if "doi" in evidence.lower():
                    score += 0.05
                if any(hint in evidence.lower() for hint in PAPER_CONTENT_HINTS):
                    score += 0.05

                confidence = round(
                    min(
                        0.95,
                        max(
                            0.2,
                            (0.35 * constraint_assessment.score)
                            + score
                            - min(0.35, 0.1 * len(constraint_assessment.contradicted)),
                        ),
                    ),
                    2,
                )
                if score > best_score:
                    best_score = score
                    best_candidate = AnswerCandidate(
                        answer=doi,
                        evidence=evidence,
                        source=document.url,
                        confidence=confidence,
                    )

        if best_candidate.answer:
            return best_candidate
        return AnswerCandidate()

    def _heuristic_person_from_documents(self, query: str, documents: list[Document]) -> AnswerCandidate:
        intent = analyze_query_intent(query)
        best_candidate = AnswerCandidate()
        best_score = float("-inf")

        for document in documents:
            if intent.prefers_event_sources and not self._is_grounded_event_person_document(query, intent, document):
                continue
            evidence_text = self._build_person_evidence(query, intent, document)
            full_name, evidence, score = self._extract_person_candidate(query, intent, evidence_text)
            if not full_name:
                continue

            combined_score = score + (0.1 * max(0.0, min(1.0, document.rank_score)))
            if intent.prefers_event_sources:
                combined_score += 0.12 * event_page_score(document.url, document.title, document.content[:2000])

            if combined_score <= best_score:
                continue

            best_score = combined_score
            best_candidate = AnswerCandidate(
                answer=self._extract_last_name(query, full_name),
                evidence=evidence or evidence_text[:320],
                source=document.url,
                confidence=round(min(0.82, max(0.4, combined_score)), 2),
                supporting_person=full_name,
            )

        return best_candidate

    def _is_grounded_event_person_document(
        self,
        query: str,
        intent: QueryIntent,
        document: Document,
    ) -> bool:
        if not intent.prefers_event_sources:
            return True

        content_excerpt = normalize_whitespace(document.content[:3000])
        combined = normalize_whitespace(f"{document.title}. {content_excerpt}")
        event_score = event_page_score(document.url, document.title, content_excerpt)
        winner_score = event_winner_evidence_score(combined)
        specificity = specificity_overlap_score(
            query,
            normalize_whitespace(f"{document.title} {document.url} {content_excerpt}"),
        )
        grounded_browsecomp = is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            content_excerpt,
            require_media=intent.is_media_query,
        )

        if grounded_browsecomp and (winner_score >= 0.24 or event_score >= 0.42):
            return True
        if winner_score >= 0.48 and specificity >= 0.08:
            return True
        if not intent.is_open_domain_browsecomp and event_score >= 0.58 and winner_score >= 0.34:
            return True
        return False

    def _heuristic_year_from_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerCandidate:
        best_candidate = AnswerCandidate()
        best_score = float("-inf")
        for document in documents:
            candidate = self._best_year_candidate_from_document(query, document)
            if not candidate.answer or candidate.confidence <= best_score:
                continue
            best_score = candidate.confidence
            best_candidate = candidate
        return best_candidate

    def _heuristic_title_from_documents(self, documents: list[Document]) -> AnswerCandidate:
        for document in documents:
            title_phrase = document_title_query_phrase(document.title)
            if not title_phrase:
                continue
            return AnswerCandidate(
                answer=title_phrase,
                evidence=document.title,
                source=document.url,
                confidence=0.58,
            )
        return AnswerCandidate()

    def _heuristic_institution_from_documents(self, documents: list[Document]) -> AnswerCandidate:
        for document in documents:
            institutions = extract_institutions_from_text(
                f"{document.title} {document.content[:2500]} {document.sections.get('references', '')}"
            )
            if not institutions:
                continue
            return AnswerCandidate(
                answer=institutions[0],
                evidence=institutions[0],
                source=document.url,
                confidence=0.56,
            )
        return AnswerCandidate()

    def _heuristic_generic_from_documents(
        self,
        query: str,
        intent: QueryIntent,
        documents: list[Document],
    ) -> AnswerCandidate:
        if intent.answer_type == "entity_and_count" or (intent.targets_count and intent.targets_person):
            candidate = self._heuristic_entity_and_count_from_documents(query, documents)
            if candidate.answer:
                return candidate

        if intent.answer_type == "entity_name":
            candidate = self._heuristic_entity_name_from_documents(query, documents)
            if candidate.answer:
                return candidate

        if intent.answer_type == "count" or intent.targets_count:
            candidate = self._heuristic_count_from_documents(query, documents)
            if candidate.answer:
                return candidate

        if intent.answer_type == "generic":
            named_candidate = self._heuristic_entity_name_from_documents(query, documents)
            if named_candidate.answer:
                return named_candidate
            count_candidate = self._heuristic_count_from_documents(query, documents)
            if count_candidate.answer:
                return count_candidate

        return AnswerCandidate()

    def _heuristic_entity_and_count_from_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerCandidate:
        best_candidate = AnswerCandidate()
        best_score = float("-inf")

        for document in documents:
            excerpt = self._build_generic_evidence(query, analyze_query_intent(query), document)
            entity_name, entity_evidence, entity_score = self._extract_entity_candidate(query, excerpt)
            count_value, count_evidence, count_score = self._extract_count_candidate(query, excerpt)
            if not entity_name or not count_value:
                continue
            combined_score = (0.45 * entity_score) + (0.45 * count_score) + (0.1 * document.rank_score)
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = AnswerCandidate(
                    answer=f"{entity_name}, {count_value}",
                    evidence=f"{entity_evidence} {count_evidence}".strip(),
                    source=document.url,
                    confidence=round(min(0.78, max(0.45, combined_score)), 2),
                    supporting_person=entity_name,
                )

        return best_candidate

    def _heuristic_entity_name_from_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerCandidate:
        best_candidate = AnswerCandidate()
        best_score = float("-inf")
        for document in documents:
            excerpt = self._build_generic_evidence(query, analyze_query_intent(query), document)
            entity_name, evidence, score = self._extract_entity_candidate(query, excerpt)
            if not entity_name or score <= best_score:
                continue
            best_score = score
            best_candidate = AnswerCandidate(
                answer=entity_name,
                evidence=evidence,
                source=document.url,
                confidence=round(min(0.72, max(0.35, score)), 2),
                supporting_person=entity_name,
            )
        return best_candidate

    def _heuristic_count_from_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerCandidate:
        best_candidate = AnswerCandidate()
        best_score = float("-inf")
        for document in documents:
            excerpt = self._build_generic_evidence(query, analyze_query_intent(query), document)
            count_value, evidence, score = self._extract_count_candidate(query, excerpt)
            if not count_value or score <= best_score:
                continue
            best_score = score
            best_candidate = AnswerCandidate(
                answer=count_value,
                evidence=evidence,
                source=document.url,
                confidence=round(min(0.7, max(0.35, score)), 2),
            )
        return best_candidate

    def _build_person_evidence(self, query: str, intent: QueryIntent, document: Document) -> str:
        if intent.requires_acknowledgement_section:
            return (
                document.acknowledgement_section
                or document.sections.get("acknowledgements", "")
                or document.content
            )

        content = normalize_whitespace(f"{document.title}. {document.content[:8000]}")
        if not content:
            return normalize_whitespace(f"{document.title} {document.url}")

        sentences = split_sentences(content)
        if not sentences:
            return content[:2500]

        scored_sentences: list[tuple[float, str]] = []
        strong_event_sentences: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            score = lexical_relevance_score(query, sentence)
            score += 0.12 * specificity_overlap_score(query, sentence)
            if contains_candidate_person_name(sentence):
                score += 0.12
            if any(hint in lowered for hint in EVENT_WINNER_HINTS):
                score += 0.18
            if any(hint in lowered for hint in ("named", "called", "crowned", "proclaimed")):
                score += 0.08
            if intent.prefers_event_sources:
                sentence_event_score = event_winner_evidence_score(sentence)
                score += 0.2 * sentence_event_score
                if sentence_event_score >= 0.38:
                    strong_event_sentences.append(sentence)
            scored_sentences.append((score, sentence))

        if intent.prefers_event_sources and strong_event_sentences:
            return "\n".join(strong_event_sentences[:6])

        ranked = [
            sentence
            for score, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True)
            if score >= 0.16
        ]
        if ranked:
            return "\n".join(ranked[:6])
        return content[:2500]

    def _extract_person_candidate(
        self,
        query: str,
        intent: QueryIntent,
        text: str,
    ) -> tuple[str, str, float]:
        if not text:
            return "", "", 0.0

        if intent.requires_acknowledgement_section:
            name_candidates = extract_name_candidates_with_evidence(text)
            if name_candidates:
                full_name, evidence = name_candidates[0]
                return full_name, evidence, 0.72

        best_name = ""
        best_evidence = ""
        best_score = 0.0
        query_lower = query.lower()
        sentences = split_sentences(text)
        if not sentences:
            sentences = [normalize_whitespace(text)]

        explicit_patterns = (
            re.compile(
                r"\b(?:winner|won by|won|crowned|proclaimed|named|declared)\b.{0,60}?\b"
                r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
            ),
            re.compile(
                r"\b([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b.{0,90}?\b"
                r"(?:won|was crowned|was proclaimed|was declared|emerged as)\b"
            ),
            re.compile(
                r"\b(?:beauty\s+pageant|pageant|contest|festival queen|queen|candidate)\b.{0,40}?\b"
                r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
            ),
            re.compile(
                r"\b([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
                r"\s+(?:was\s+)?(?:crowned|proclaimed|declared|named)\s+"
                r"(?:the\s+)?(?:winner|queen|festival queen)\b"
            ),
            re.compile(
                r"\bwinner\b.{0,30}\b(?:was|is|goes to|went to)\s+"
                r"([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b"
            ),
        )
        blocked_terms = {
            "festival",
            "celebration",
            "township",
            "province",
            "instagram",
            "facebook",
            "wikipedia",
            "stew",
            "dish",
            "contest",
            "winner",
        }

        for sentence in sentences:
            lowered = sentence.lower()
            sentence_event_score = event_winner_evidence_score(sentence) if intent.prefers_event_sources else 0.0
            if intent.prefers_event_sources and sentence_event_score < 0.32:
                continue
            local_score = lexical_relevance_score(query, sentence)
            sentence_specificity = specificity_overlap_score(query, sentence)
            local_score += 0.12 * sentence_specificity
            if intent.prefers_event_sources and intent.is_open_domain_browsecomp:
                if local_score < 0.12 and sentence_specificity < 0.08 and sentence_event_score < 0.38:
                    continue
            if any(hint in lowered for hint in EVENT_WINNER_HINTS):
                local_score += 0.2
            if contains_candidate_person_name(sentence):
                local_score += 0.1
            if intent.prefers_event_sources:
                local_score += 0.2 * sentence_event_score
            if any(term in query_lower for term in ("winner", "won", "pageant", "contest")) and any(
                term in lowered for term in ("winner", "won", "pageant", "contest", "queen")
            ):
                local_score += 0.12

            for pattern in explicit_patterns:
                match = pattern.search(sentence)
                if not match:
                    continue
                candidate = normalize_whitespace(match.group(1))
                if self._is_suspicious_person_candidate_span(sentence, match, candidate):
                    continue
                tokens = candidate.split()
                if any(token.lower() in blocked_terms for token in tokens):
                    continue
                if not is_plausible_person_name(candidate):
                    continue
                candidate_score = local_score + 0.18
                if intent.prefers_event_sources:
                    proximity = self._event_person_proximity(sentence, candidate)
                    if proximity < 0.0 and sentence_event_score < 0.45:
                        continue
                    candidate_score += proximity
                if candidate_score > best_score:
                    best_name = candidate
                    best_evidence = sentence
                    best_score = candidate_score

            if best_score >= 0.6:
                continue

            for match in re.finditer(
                r"\b([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){1,3})\b",
                sentence,
            ):
                candidate = normalize_whitespace(match.group(1))
                if self._is_suspicious_person_candidate_span(sentence, match, candidate):
                    continue
                tokens = candidate.split()
                if any(token.lower() in blocked_terms for token in tokens):
                    continue
                if not is_plausible_person_name(candidate):
                    continue
                if intent.prefers_event_sources and sentence_event_score < 0.4:
                    continue
                candidate_score = local_score
                if any(term in lowered for term in ("winner", "won", "pageant", "contest", "queen")):
                    candidate_score += 0.12
                if intent.prefers_event_sources:
                    proximity = self._event_person_proximity(sentence, candidate)
                    if proximity < 0.0 and sentence_event_score < 0.45:
                        continue
                    candidate_score += proximity
                if candidate_score > best_score:
                    best_name = candidate
                    best_evidence = sentence
                    best_score = candidate_score

        return best_name, best_evidence, best_score

    def _is_suspicious_person_candidate_span(
        self,
        sentence: str,
        match: re.Match[str],
        candidate: str,
    ) -> bool:
        if not candidate:
            return True

        trailing_text = sentence[match.end() :]
        if re.match(r"\s+[A-Z][A-Za-z'`.-]{2,}\b", trailing_text):
            return True

        consecutive_titlecase_run = re.search(
            r"\b(?:[A-Z][A-Za-z'`.-]+\s+){4,}[A-Z][A-Za-z'`.-]+\b",
            sentence,
        )
        if consecutive_titlecase_run and match.start() >= consecutive_titlecase_run.start() and match.end() <= consecutive_titlecase_run.end():
            return True

        return False

    def _event_person_proximity(self, sentence: str, candidate: str) -> float:
        lowered_sentence = sentence.lower()
        lowered_candidate = candidate.lower()
        candidate_index = lowered_sentence.find(lowered_candidate)
        if candidate_index == -1:
            return -1.0

        keyword_indices = [
            match.start()
            for pattern in (
                r"\bwinner\b",
                r"\bwon\b",
                r"\bcrowned\b",
                r"\bproclaimed\b",
                r"\bdeclared\b",
                r"\bpageant\b",
                r"\bcontest\b",
                r"\bqueen\b",
                r"\bcoronation\b",
            )
            for match in re.finditer(pattern, lowered_sentence)
        ]
        if not keyword_indices:
            return -1.0

        min_distance = min(abs(candidate_index - index) for index in keyword_indices)
        if min_distance <= 24:
            return 0.12
        if min_distance <= 48:
            return 0.05
        return -0.05

    def _person_document_score(
        self,
        query: str,
        document: Document,
        intent: QueryIntent,
    ) -> float:
        combined = normalize_whitespace(f"{document.title} {document.url} {document.content[:4000]}").lower()
        event_score = event_page_score(document.url, document.title, document.content[:2500])
        winner_evidence_score = event_winner_evidence_score(
            normalize_whitespace(f"{document.title}. {document.content[:2500]}")
        )
        specificity_score = specificity_overlap_score(query, combined)
        grounded_browsecomp = is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            document.content[:2500],
            require_media=intent.is_media_query,
        )
        score = 0.2 * max(0.0, min(1.0, document.rank_score))
        score += 0.15 * max(0.0, min(1.0, document.retrieval_score))
        score += 0.25 * lexical_relevance_score(query, combined)
        score += 0.12 * specificity_score
        if contains_candidate_person_name(document.content[:3000]):
            score += 0.2
        if intent.prefers_event_sources:
            score += 0.3 * event_score
            score += 0.28 * winner_evidence_score
            if any(hint in combined for hint in EVENT_WINNER_HINTS):
                score += 0.18
            if looks_like_event_page(document.url, document.title, document.content[:2200]):
                score += 0.08
            if is_generic_event_topic_page(document.url, document.title, document.content[:600]):
                score -= 0.34
            if is_recipe_food_page(document.url, document.title, document.content[:700]) and event_score < 0.55:
                score -= 0.32
            if winner_evidence_score < 0.3 and event_score < 0.25:
                score -= 0.22
            if intent.is_open_domain_browsecomp and not grounded_browsecomp:
                score -= 0.18
            if intent.is_open_domain_browsecomp and specificity_score < 0.1 and event_score < 0.5:
                score -= 0.22
            if intent.needs_event_discovery_hop and specificity_score < 0.1 and event_score < 0.55:
                score -= 0.24
            if (
                intent.needs_event_discovery_hop
                and any(term in combined for term in ("beauty pageant", "pageant", "festival queen", "beauty queen"))
                and not any(
                    term in combined
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
                score -= 0.12
        if is_broad_overview_page(document.url, document.title):
            score -= 0.2
        if is_aggregate_listing_page(document.url, document.title, document.content[:300]):
            score -= 0.2
        if is_wiki_meta_page(document.url, document.title, document.content[:300]):
            score -= 0.25
        if is_non_english_wiki_page(document.url):
            score -= 0.22
        if is_generic_media_topic_page(document.url, document.title, document.content[:600]):
            score -= 0.25
        if is_generic_event_topic_page(document.url, document.title, document.content[:600]):
            score -= 0.28
        if is_forum_discussion_page(document.url, document.title, document.content[:300]):
            score -= 0.18
        if is_low_trust_social_page(document.url, document.title, document.content[:300]) and event_score < 0.55:
            score -= 0.28
        if intent.is_open_domain_browsecomp:
            if grounded_browsecomp:
                score += 0.15
            else:
                score -= 0.12
        return score

    def _is_valid_person_answer(
        self,
        query: str,
        intent: QueryIntent,
        answer_candidate: AnswerCandidate,
        documents: list[Document],
    ) -> bool:
        answer = normalize_whitespace(answer_candidate.answer)
        if not answer:
            return False

        blocked_terms = {
            "festival",
            "celebration",
            "township",
            "province",
            "instagram",
            "facebook",
            "wikipedia",
            "stew",
            "dish",
            "winner",
        }
        if intent.answer_type == "person_last_name":
            if not re.fullmatch(r"[A-Za-z][A-Za-z'`.-]{1,40}", answer):
                return False
        else:
            tokens = answer.split()
            if len(tokens) < 2 or len(tokens) > 4:
                return False
            if any(token.lower() in blocked_terms for token in tokens):
                return False
            if not is_plausible_person_name(answer):
                return False

        supported_text = answer_candidate.supporting_person or answer
        for document in documents:
            if intent.prefers_event_sources and not self._is_grounded_event_person_document(query, intent, document):
                continue
            evidence_text = self._build_person_evidence(query, intent, document)
            lowered_evidence = evidence_text.lower()
            if supported_text.lower() in lowered_evidence or answer.lower() in lowered_evidence:
                if intent.requires_acknowledgement_section:
                    return True
                if intent.prefers_event_sources:
                    if not has_event_winner_evidence(evidence_text):
                        continue
                    heuristic_name, _, _ = self._extract_person_candidate(query, intent, evidence_text)
                    if heuristic_name:
                        if intent.answer_type == "person_last_name":
                            if heuristic_name.split()[-1].lower() == answer.lower():
                                return True
                        elif heuristic_name.lower() == supported_text.lower() or heuristic_name.lower() == answer.lower():
                            return True
                    continue
                if contains_candidate_person_name(evidence_text):
                    return True
        return False

    def _heuristic_from_snippets(
        self,
        query: str,
        intent: QueryIntent,
        evidence_snippets: list[EvidenceSnippet],
    ) -> AnswerCandidate:
        if intent.targets_citation_identifier:
            target_reference_doi = self._query_targets_reference_doi(query)
            best_candidate = AnswerCandidate()
            best_score = float("-inf")
            for snippet in evidence_snippets:
                doi_candidates = extract_doi_candidates(f"{snippet.url} {snippet.title} {snippet.snippet}")
                if not doi_candidates:
                    continue
                if (
                    not target_reference_doi
                    and looks_like_reference_citation(snippet.snippet)
                    and "doi.org/" not in snippet.url.lower()
                ):
                    continue

                snippet_score = lexical_relevance_score(query, f"{snippet.title} {snippet.snippet}")
                for doi in doi_candidates:
                    score = 0.2 + (0.45 * snippet_score)
                    if "doi.org/" in snippet.url.lower():
                        score += 0.3
                    if "doi" in snippet.snippet.lower():
                        score += 0.08
                    if looks_like_reference_citation(snippet.snippet):
                        score += 0.08 if target_reference_doi else -0.2
                    if "abstract" in snippet.snippet.lower() or "journal" in snippet.snippet.lower():
                        score += 0.06
                    if score > best_score:
                        best_score = score
                        best_candidate = AnswerCandidate(
                            answer=doi,
                            evidence=self._best_doi_evidence(snippet.snippet, preferred_doi=doi) or snippet.snippet,
                            source=snippet.url,
                            confidence=round(min(0.82, max(0.4, score)), 2),
                        )
            return best_candidate

        if intent.answer_type in {"person_last_name", "person_name"}:
            for snippet in evidence_snippets:
                person_text = normalize_whitespace(f"{snippet.title}. {snippet.snippet}")
                if intent.prefers_event_sources:
                    snippet_event_score = event_winner_evidence_score(person_text)
                    snippet_specificity = specificity_overlap_score(
                        query,
                        normalize_whitespace(f"{snippet.title} {snippet.url} {snippet.snippet}"),
                    )
                    if (
                        snippet_event_score < 0.38
                        or (
                            snippet_specificity < 0.12
                            and not looks_like_event_page(
                                snippet.url,
                                snippet.title,
                                snippet.snippet,
                                minimum_score=0.42,
                            )
                        )
                        or is_recipe_food_page(snippet.url, snippet.title, snippet.snippet)
                        or is_generic_event_topic_page(snippet.url, snippet.title, snippet.snippet)
                    ):
                        continue
                full_name, evidence, score = self._extract_person_candidate(query, intent, person_text)
                if not full_name:
                    continue
                return AnswerCandidate(
                    answer=self._extract_last_name(query, full_name),
                    evidence=evidence or snippet.snippet,
                    source=snippet.url,
                    confidence=round(min(0.72, max(0.4, score)), 2),
                    supporting_person=full_name,
                )

        if intent.answer_type == "year":
            best_candidate = AnswerCandidate()
            best_score = float("-inf")
            for snippet in evidence_snippets:
                if intent.is_open_domain_browsecomp and not is_specific_historical_year_page(
                    query,
                    snippet.url,
                    snippet.title,
                    snippet.snippet,
                ):
                    continue
                pseudo_document = Document(
                    title=snippet.title,
                    url=snippet.url,
                    content=snippet.snippet,
                    source=snippet.url,
                    retrieval_score=snippet.score,
                    fetched=False,
                    content_type="search-result",
                )
                candidate = self._best_year_candidate_from_document(query, pseudo_document)
                if not candidate.answer or candidate.confidence <= best_score:
                    continue
                best_score = candidate.confidence
                best_candidate = candidate
            return best_candidate

        if intent.answer_type in {"entity_and_count", "entity_name", "count", "generic"}:
            for snippet in evidence_snippets:
                entity_name, entity_evidence, entity_score = self._extract_entity_candidate(query, snippet.snippet)
                count_value, count_evidence, count_score = self._extract_count_candidate(query, snippet.snippet)
                if intent.answer_type == "entity_and_count" and entity_name and count_value:
                    return AnswerCandidate(
                        answer=f"{entity_name}, {count_value}",
                        evidence=f"{entity_evidence} {count_evidence}".strip(),
                        source=snippet.url,
                        confidence=round(min(0.68, max(0.4, (entity_score + count_score) / 2)), 2),
                        supporting_person=entity_name,
                    )
                if intent.answer_type in {"entity_name", "generic"} and entity_name:
                    return AnswerCandidate(
                        answer=entity_name,
                        evidence=entity_evidence or snippet.snippet,
                        source=snippet.url,
                        confidence=round(min(0.62, max(0.35, entity_score)), 2),
                        supporting_person=entity_name,
                    )
                if intent.answer_type == "count" and count_value:
                    return AnswerCandidate(
                        answer=count_value,
                        evidence=count_evidence or snippet.snippet,
                        source=snippet.url,
                        confidence=round(min(0.6, max(0.35, count_score)), 2),
                    )

        return AnswerCandidate()

    def _best_doi_evidence(self, text: str, *, preferred_doi: str = "") -> str:
        cleaned = normalize_whitespace(text)
        doi_candidates = extract_doi_candidates(cleaned)
        if not doi_candidates:
            return ""
        selected_doi = preferred_doi or doi_candidates[0]
        sentences = split_sentences(cleaned)
        evidence = next((sentence for sentence in sentences if selected_doi.lower() in sentence.lower()), "")
        if evidence:
            return evidence
        return f"DOI: {selected_doi}"

    def _is_valid_doi_answer(
        self,
        query: str,
        answer_candidate: AnswerCandidate,
        documents: list[Document],
    ) -> bool:
        if not answer_candidate.answer:
            return False
        target_reference_doi = self._query_targets_reference_doi(query)
        normalized_answer = answer_candidate.answer.lower()
        for document in documents:
            candidate_locations = self._document_doi_locations(document)
            locations = candidate_locations.get(normalized_answer, set())
            if not locations:
                continue
            constraint_assessment = assess_document_constraints(
                query,
                document.title,
                document.content,
                reference_text=document.sections.get("references", ""),
                metadata=document.metadata,
            )
            if target_reference_doi or (locations - {"reference"} and constraint_assessment.score >= 0.35):
                if not target_reference_doi and self._has_critical_constraint_conflict(constraint_assessment):
                    continue
                return True
        return False

    def _is_valid_year_answer(
        self,
        query: str,
        answer_candidate: AnswerCandidate,
        documents: list[Document],
    ) -> bool:
        answer = normalize_whitespace(answer_candidate.answer)
        if not YEAR_PATTERN.fullmatch(answer):
            return False

        query_years = self._years_in_text(query)
        for document in documents:
            candidate = self._best_year_candidate_from_document(
                query,
                document,
                preferred_year=answer,
            )
            if not candidate.answer:
                continue
            if answer in query_years and candidate.confidence < 0.7:
                continue
            if candidate.confidence >= 0.48:
                return True
        return False

    def _best_year_candidate_from_document(
        self,
        query: str,
        document: Document,
        *,
        preferred_year: str = "",
    ) -> AnswerCandidate:
        structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
            query,
            document.url,
            document.title,
            document.content[:3500],
        )
        if structural_contradictions > 0:
            return AnswerCandidate()
        if (
            historical_year_has_structural_constraints(query)
            and structural_matches == 0
            and not historical_year_trusted_memorial_source(document.url)
        ):
            return AnswerCandidate()
        if query_requires_bosnia_top_city(query):
            document_scope = normalize_whitespace(
                f"{document.title} {document.url} {document.content[:3500]}"
            ).lower()
            if not any(term in document_scope for term in BOSNIA_TOP_FOUR_CITY_TERMS):
                return AnswerCandidate()

        best_candidate = AnswerCandidate()
        best_score = float("-inf")
        title = normalize_whitespace(document.title)
        sentences = split_sentences(normalize_whitespace(f"{title}. {document.content[:3500]}"))
        for sentence in sentences:
            for match in YEAR_PATTERN.finditer(sentence):
                year = match.group(0)
                if preferred_year and year != preferred_year:
                    continue
                score = self._score_year_mention(
                    query,
                    title,
                    sentence,
                    year,
                    match.start(),
                    match.end(),
                )
                score += 0.08 * max(0.0, min(1.0, document.rank_score))
                score += 0.05 * max(0.0, min(1.0, document.retrieval_score))
                score += 0.18 * structural_score
                if score <= best_score:
                    continue
                best_score = score
                best_candidate = AnswerCandidate(
                    answer=year,
                    evidence=sentence,
                    source=document.url,
                    confidence=round(min(0.84, max(0.0, score)), 2),
                )

        if best_candidate.answer and best_candidate.confidence >= 0.26:
            return best_candidate
        return AnswerCandidate()

    def _score_year_mention(
        self,
        query: str,
        title: str,
        sentence: str,
        year: str,
        start: int,
        end: int,
    ) -> float:
        query_lower = query.lower()
        title_lower = title.lower()
        sentence_lower = sentence.lower()
        window = sentence_lower[max(0, start - 48) : min(len(sentence_lower), end + 48)]
        query_years = self._years_in_text(query)

        score = 0.12
        score += 0.35 * lexical_relevance_score(query, sentence)
        score += 0.2 * specificity_overlap_score(query, f"{title} {sentence}")
        if year not in query_years:
            score += 0.08
        else:
            score -= 0.18
        if any(hint in sentence_lower for hint in YEAR_EVENT_HINTS):
            score += 0.08
        if any(hint in window for hint in YEAR_EVENT_HINTS):
            score += 0.14
        if year in title_lower and any(hint in title_lower for hint in YEAR_EVENT_HINTS):
            score += 0.16
        if re.search(
            rf"\b(?:victims?|dead|killed|died|executed|massacre|battle|attack|tragedy|uprising|revolt)\b"
            rf".{{0,24}}\b{year}\b",
            sentence_lower,
        ) or re.search(
            rf"\b{year}\b.{{0,24}}\b(?:victims?|dead|killed|died|executed|massacre|battle|attack|tragedy|uprising|revolt)\b",
            sentence_lower,
        ):
            score += 0.18
        if re.search(rf"\bborn in {year}\b", sentence_lower):
            score -= 0.65
        elif re.search(rf"\bborn\b.{{0,20}}\b{year}\b", sentence_lower):
            score -= 0.45
        if re.search(
            rf"\b(?:built|constructed|erected|unveiled|opened|completed|installed)\b.{{0,18}}\b{year}\b",
            sentence_lower,
        ):
            score -= 0.42
        if "census" in window or "population" in window:
            score -= 0.36
        if any(hint in window for hint in YEAR_BIOGRAPHICAL_HINTS):
            score -= 0.18
        if any(hint in window for hint in YEAR_CONSTRUCTION_HINTS):
            score -= 0.18
        if any(hint in window for hint in YEAR_CENSUS_HINTS):
            score -= 0.2
        if any(hint in window for hint in YEAR_AWARD_HINTS) and not any(hint in window for hint in YEAR_EVENT_HINTS):
            score -= 0.38
        if re.search(
            rf"\b(?:began|started|emerged|formed|launched|initiated)\b.{{0,18}}\b{year}\b",
            sentence_lower,
        ) and not any(hint in sentence_lower for hint in ("victims", "massacre", "killed", "died", "executed")):
            score -= 0.24
        if "broader" in sentence_lower and "resistance" in sentence_lower:
            score -= 0.08
        if "constructed prior to" in query_lower and any(hint in sentence_lower for hint in YEAR_CONSTRUCTION_HINTS):
            score -= 0.08
        return score

    def _year_document_score(
        self,
        query: str,
        document: Document,
    ) -> float:
        structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
            query,
            document.url,
            document.title,
            document.content[:2200],
        )
        intent = analyze_query_intent(query)
        if intent.is_open_domain_browsecomp and intent.answer_type == "year":
            if not is_specific_historical_year_page(
                query,
                document.url,
                document.title,
                document.content[:2200],
            ):
                return -0.35
        candidate = self._best_year_candidate_from_document(query, document)
        combined = normalize_whitespace(f"{document.title} {document.url} {document.content[:2500]}")
        score = candidate.confidence
        score += 0.12 * lexical_relevance_score(query, combined)
        score += 0.08 * specificity_overlap_score(query, combined)
        score += 0.22 * structural_score
        if query_requires_bosnia_top_city(query):
            lowered_combined = combined.lower()
            if any(term in lowered_combined for term in BOSNIA_TOP_FOUR_CITY_TERMS):
                score += 0.18
            else:
                score -= 0.55
        if structural_contradictions > 0:
            score -= 0.65
        if (
            historical_year_has_structural_constraints(query)
            and structural_matches == 0
            and not historical_year_trusted_memorial_source(document.url)
        ):
            score -= 0.28
        if is_person_biography_page(document.url, document.title, document.content[:1400]):
            score -= 0.6
        if is_specific_historical_year_page(query, document.url, document.title, document.content[:2200]):
            score += 0.22
        else:
            score -= 0.22
        if is_generic_historical_monument_page(document.url, document.title, document.content[:800]):
            score -= 0.35
        if is_low_trust_social_page(document.url, document.title, document.content[:400]):
            score -= 0.18
        if is_broad_overview_page(document.url, document.title):
            score -= 0.12
        if is_aggregate_listing_page(document.url, document.title, document.content[:300]):
            score -= 0.15
        if is_wiki_meta_page(document.url, document.title, document.content[:300]):
            score -= 0.18
        return score

    def _years_in_text(self, text: str) -> set[str]:
        return set(YEAR_PATTERN.findall(text))

    def _document_has_primary_doi(self, document: Document) -> bool:
        metadata_doi = normalize_whitespace(str(document.metadata.get("doi", "")))
        return bool(
            extract_doi_candidates(metadata_doi)
            or contains_primary_doi(document.url, document.title, document.content, max_chars=DOI_FRONT_MATTER_CHARS)
        )

    def _document_looks_like_paper(self, document: Document) -> bool:
        front_matter = self._front_matter_text(document).lower()
        if any(hint in front_matter for hint in PAPER_CONTENT_HINTS):
            return True
        if document.content_type in {"pdf", "html"} and not any(hint in front_matter for hint in THESIS_CONTENT_HINTS):
            return True
        return False

    def _front_matter_text(self, document: Document) -> str:
        return normalize_whitespace(
            f"{document.title} {document.metadata.get('doi', '')} {document.content[:DOI_FRONT_MATTER_CHARS]}"
        )

    def _document_doi_locations(self, document: Document) -> dict[str, set[str]]:
        location_map: dict[str, set[str]] = {}
        segments = [
            ("metadata", normalize_whitespace(str(document.metadata.get("doi", "")))),
            ("url", document.url),
            ("title", document.title),
            ("front", self._front_matter_text(document)),
            ("body", normalize_whitespace(document.content[:DOI_BODY_CHARS])),
            ("reference", document.sections.get("references", "")),
        ]
        for location, text in segments:
            for doi in extract_doi_candidates(text):
                location_map.setdefault(doi.lower(), set()).add(location)
        return location_map

    def _best_doi_source_text(
        self,
        document: Document,
        doi: str,
        target_reference_doi: bool,
    ) -> str:
        ordered_segments = [
            normalize_whitespace(str(document.metadata.get("doi", ""))),
            document.url,
            document.title,
            self._front_matter_text(document),
            normalize_whitespace(document.content[:DOI_BODY_CHARS]),
        ]
        if target_reference_doi:
            ordered_segments.insert(4, document.sections.get("references", ""))
        else:
            ordered_segments.append(document.sections.get("references", ""))

        for segment in ordered_segments:
            if doi.lower() in normalize_whitespace(segment).lower():
                return segment
        return self._front_matter_text(document) or document.sections.get("references", "") or document.content

    def _build_primary_doi_evidence(self, document: Document) -> str:
        metadata_doi = normalize_whitespace(str(document.metadata.get("doi", "")))
        if extract_doi_candidates(metadata_doi):
            return f"DOI metadata: {metadata_doi}\nTitle: {document.title}\nURL: {document.url}"

        primary_candidates = extract_primary_doi_candidates(
            document.url,
            document.title,
            document.content,
            max_chars=DOI_FRONT_MATTER_CHARS,
        )
        if not primary_candidates:
            return ""

        target_doi = primary_candidates[0]
        primary_text = self._front_matter_text(document)
        evidence = self._best_doi_evidence(primary_text, preferred_doi=target_doi)
        if evidence:
            return evidence
        return f"Title: {document.title}\nURL: {document.url}\nDOI: {target_doi}"

    def _query_targets_reference_doi(self, query: str) -> bool:
        query_lower = normalize_whitespace(query).lower()
        return any(hint in query_lower for hint in REFERENCE_DOI_TARGET_HINTS)

    def _has_critical_constraint_conflict(self, assessment) -> bool:
        critical_labels = {"publication_date", "sample_size", "author_count"}
        return len(critical_labels & set(assessment.contradicted)) >= 2

    def _extract_last_name(self, query: str, full_name: str) -> str:
        if not full_name:
            return ""
        query_lower = query.lower()
        asks_full_name = any(
            phrase in query_lower
            for phrase in (
                "full name",
                "first and last name",
                "first and last names",
                "first name and last name",
            )
        )
        if ("last name" in query_lower or "surname" in query_lower) and not asks_full_name:
            return full_name.split()[-1]
        return full_name

    def _generic_document_score(
        self,
        query: str,
        document: Document,
        intent: QueryIntent,
    ) -> float:
        combined = normalize_whitespace(f"{document.title} {document.url} {document.content[:4000]}").lower()
        media_score = media_page_score(document.url, document.title, document.content[:2500])
        event_score = event_page_score(document.url, document.title, document.content[:2500])
        grounded_browsecomp = is_grounded_browsecomp_page(
            query,
            document.url,
            document.title,
            document.content[:2500],
            require_media=intent.is_media_query,
        )
        score = 0.2 * max(0.0, min(1.0, document.rank_score))
        score += 0.15 * max(0.0, min(1.0, document.retrieval_score))
        if intent.prefers_encyclopedic_sources and any(term in combined for term in ("wiki", "fandom")):
            score += 0.2
        if intent.prefers_character_sources and any(term in combined for term in CHARACTER_QUERY_HINTS):
            score += 0.25
        if intent.targets_count and any(term in combined for term in ABILITY_QUERY_HINTS):
            score += 0.2
        if any(term in combined for term in ("named", "called", "known as")):
            score += 0.05
        if re.search(r"\b\d+\b", combined):
            score += 0.05
        score += 0.35 * specificity_overlap_score(query, combined)
        score += 0.22 * media_score if intent.is_media_query else 0.0
        if intent.prefers_event_sources:
            score += 0.24 * event_score
            if any(term in combined for term in EVENT_WINNER_HINTS):
                score += 0.16
        if is_broad_overview_page(document.url, document.title):
            score -= 0.3
        if is_aggregate_listing_page(document.url, document.title, document.content[:300]):
            score -= 0.25
        if is_wiki_meta_page(document.url, document.title, document.content[:300]):
            score -= 0.35
        if is_non_english_wiki_page(document.url):
            score -= 0.3
        if is_generic_media_topic_page(document.url, document.title, document.content[:600]):
            score -= 0.35
        if is_forum_discussion_page(document.url, document.title, document.content[:300]):
            score -= 0.25
        if is_low_trust_social_page(document.url, document.title, document.content[:300]) and event_score < 0.55:
            score -= 0.28
        if intent.is_media_query and media_score < 0.18:
            score -= 0.3
        if document_title_query_phrase(document.title):
            score += 0.06
        if intent.is_open_domain_browsecomp and grounded_browsecomp:
            score += 0.22
        elif intent.is_open_domain_browsecomp:
            score -= 0.28
        return score

    def _build_generic_evidence(self, query: str, intent: QueryIntent, document: Document) -> str:
        content = normalize_whitespace(document.content[:8000])
        if not content:
            return ""

        sentences = split_sentences(content)
        if not sentences:
            return content[:2500]

        scored_sentences: list[tuple[float, str]] = []
        for sentence in sentences:
            lowered = sentence.lower()
            score = lexical_relevance_score(query, sentence)
            score += 0.14 * specificity_overlap_score(query, sentence)
            if intent.prefers_character_sources and any(term in lowered for term in CHARACTER_QUERY_HINTS):
                score += 0.18
            if intent.targets_count and any(term in lowered for term in ABILITY_QUERY_HINTS):
                score += 0.18
            if intent.prefers_event_sources and any(term in lowered for term in EVENT_WINNER_HINTS):
                score += 0.18
            if intent.prefers_event_sources and contains_candidate_person_name(sentence):
                score += 0.1
            if any(term in lowered for term in ("named", "called", "known as", "companion")):
                score += 0.07
            if re.search(r"\b\d+\b", sentence):
                score += 0.05 if intent.targets_count else 0.02
            scored_sentences.append((score, sentence))

        ranked = [sentence for score, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True) if score >= 0.16]
        if ranked:
            return "\n".join(ranked[:6])
        return content[:2500]

    def _extract_entity_candidate(self, query: str, text: str) -> tuple[str, str, float]:
        best_name = ""
        best_evidence = ""
        best_score = 0.0
        query_lower = query.lower()
        sentences = split_sentences(text)
        if not sentences:
            sentences = [normalize_whitespace(text)]

        explicit_patterns = (
            re.compile(
                r"\b([A-Z][A-Za-z0-9'`.-]+(?:\s+[A-Z][A-Za-z0-9'`.-]+){0,3})\b"
                r".{0,60}\b(?:potential\s+)?(?:antagonist|villain|rival|companion)\b",
            ),
            re.compile(
                r"\b(?:potential\s+)?(?:antagonist|villain|rival|character)\b.{0,60}\b"
                r"([A-Z][A-Za-z0-9'`.-]+(?:\s+[A-Z][A-Za-z0-9'`.-]+){0,3})\b",
            ),
            re.compile(
                r"\b(?:named|called|known as)\s+([A-Z][A-Za-z0-9'`.-]+(?:\s+[A-Z][A-Za-z0-9'`.-]+){0,3})\b"
            ),
        )

        for sentence in sentences:
            lowered = sentence.lower()
            local_score = lexical_relevance_score(query, sentence)
            if any(term in lowered for term in CHARACTER_QUERY_HINTS):
                local_score += 0.18
            if any(term in query_lower for term in ("antagonist", "villain", "rival")) and any(
                term in lowered for term in ("antagonist", "villain", "rival")
            ):
                local_score += 0.18
            if any(term in lowered for term in ("named", "called", "known as", "companion")):
                local_score += 0.07

            for pattern in explicit_patterns:
                match = pattern.search(sentence)
                if not match:
                    continue
                candidate = normalize_whitespace(match.group(1))
                if not candidate:
                    continue
                if candidate.lower() in {"the", "this", "that", "her", "his", "their", "its"}:
                    continue
                candidate_score = local_score + 0.15
                if candidate_score > best_score:
                    best_name = candidate
                    best_evidence = sentence
                    best_score = candidate_score

            for candidate in extract_capitalized_entities(sentence):
                candidate_score = local_score
                if any(term in lowered for term in ("antagonist", "villain", "rival", "companion")):
                    candidate_score += 0.08
                if candidate_score > best_score:
                    best_name = candidate
                    best_evidence = sentence
                    best_score = candidate_score

        return best_name, best_evidence, best_score

    def _extract_count_candidate(self, query: str, text: str) -> tuple[str, str, float]:
        best_value = ""
        best_evidence = ""
        best_score = 0.0
        patterns = (
            re.compile(
                r"\b(\d+)\s+(?:different\s+)?(?:movements|moves|abilities|effects|techniques|attacks)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:used|has used|performed|wields)\s+(\d+)\s+(?:different\s+)?"
                r"(?:movements|moves|abilities|effects|techniques|attacks)\b",
                re.IGNORECASE,
            ),
        )
        for sentence in split_sentences(text):
            lowered = sentence.lower()
            score = lexical_relevance_score(query, sentence)
            if any(term in lowered for term in ABILITY_QUERY_HINTS):
                score += 0.18
            for pattern in patterns:
                match = pattern.search(sentence)
                if not match:
                    continue
                candidate_score = score + 0.15
                if candidate_score > best_score:
                    best_value = match.group(1)
                    best_evidence = sentence
                    best_score = candidate_score
        return best_value, best_evidence, best_score
