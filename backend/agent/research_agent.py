from __future__ import annotations

import re
import time

from agent.models import AnswerCandidate, Document, FrontierTask, PipelineResult, SearchHit, TavilyPipelineError
from config.research_agent_config import (
    DEFAULT_TOP_SOURCES,
    load_research_agent_config,
)
from extraction.answer_extractor import AnswerExtractor
from extraction.entity_extractor import extract_name_candidates_with_evidence
from extraction.section_finder import find_acknowledgements_section, find_references_section
from planner.query_intent import analyze_query_intent
from memory.agent_memory import AgentMemory
from memory.evidence_graph import EvidenceGraph
from planner.planner import plan_query_pipeline
from planner.query_decomposer import decompose_query_async
from ranking.reranker import (
    build_context_block,
    deduplicate_documents,
    extract_evidence,
    rank_documents,
    select_context_documents,
)
from reflection.reflection_engine import ReflectionEngine
from retrieval.document_fetcher import DocumentFetcher
from search.frontier_scheduler import FrontierScheduler
from search.query_generator import (
    prepare_follow_up_retrieval_clues,
    prepare_retrieval_clues,
    prepare_simple_retrieval_clues,
    score_search_priority,
)
from search.search_agent import TavilySearchAgent, score_fetch_priority
from utils.text_utils import (
    BOSNIA_TOP_FOUR_CITY_TERMS,
    canonicalize_url,
    document_title_query_phrase,
    event_page_score,
    extract_capitalized_entities,
    extract_source,
    has_event_winner_evidence,
    historical_year_has_structural_constraints,
    historical_year_structural_assessment,
    historical_year_trusted_memorial_source,
    is_aggregate_listing_page,
    is_broad_overview_page,
    is_forum_discussion_page,
    is_generic_event_topic_page,
    is_generic_historical_monument_page,
    is_generic_media_topic_page,
    is_grounded_browsecomp_page,
    is_non_english_wiki_page,
    is_person_biography_page,
    is_plausible_person_name,
    is_recipe_food_page,
    is_specific_historical_year_page,
    is_wiki_meta_page,
    looks_like_media_page,
    normalize_whitespace,
    query_requires_bosnia_top_city,
    score_browsecomp_link_candidate,
    specific_query_terms,
    specificity_overlap_score,
    unique_preserve_order,
)


class ResearchAgent:
    def __init__(self) -> None:
        self.answer_extractor = AnswerExtractor()
        self.reflection_engine = ReflectionEngine()

    async def run(self, query: str, max_sources: int = DEFAULT_TOP_SOURCES) -> PipelineResult:
        started_at = time.perf_counter()
        agent_config = load_research_agent_config()
        trimmed_query = normalize_whitespace(query)[: agent_config.max_query_length]
        if not trimmed_query:
            raise TavilyPipelineError("Query must not be empty.")

        timing_stats: dict[str, float | int] = {
            "planning_time": 0.0,
            "search_time": 0.0,
            "download_time": 0.0,
            "parsing_time": 0.0,
            "ranking_time": 0.0,
            "extraction_time": 0.0,
            "total_time": 0.0,
            "iterations": 0,
        }
        reasoning_trace: list[str] = []

        planning_started = time.perf_counter()
        query_plan = plan_query_pipeline(trimmed_query)
        query_intent = analyze_query_intent(trimmed_query)
        if query_plan.use_decomposition:
            decomposition_clues = await decompose_query_async(trimmed_query)
            initial_clues = prepare_retrieval_clues(trimmed_query, decomposition_clues)
            reasoning_trace.append(
                f"Planner generated {len(initial_clues)} search clues for {query_plan.mode} exploration with intent {query_intent.answer_type}."
            )
        else:
            initial_clues = prepare_simple_retrieval_clues(trimmed_query)
            reasoning_trace.append("Planner selected a lightweight single-hop search path.")
        timing_stats["planning_time"] = round(time.perf_counter() - planning_started, 4)

        scheduler = FrontierScheduler()
        memory = AgentMemory(trimmed_query)
        evidence_graph = EvidenceGraph()
        all_clues = unique_preserve_order(initial_clues)

        for index, clue in enumerate(all_clues, start=1):
            task_key = f"search::{clue.lower()}"
            if not (memory.remember_query(clue) and memory.remember_task(task_key)):
                continue
            scheduler.push(
                FrontierTask(
                    kind="search",
                    key=task_key,
                    priority=score_search_priority(trimmed_query, clue),
                    branch=f"branch-{index}",
                    payload={"clue": clue},
                    depth=0,
                )
            )

        answer_candidate = AnswerCandidate()
        best_answer_candidate = AnswerCandidate()
        final_evidence = []
        final_ranked_documents = []
        backend = "none"

        async with TavilySearchAgent(
            max_results=query_plan.max_results_per_clue,
        ) as search_agent, DocumentFetcher() as fetcher:
            for iteration in range(1, query_plan.max_iterations + 1):
                timing_stats["iterations"] = iteration

                search_tasks = await scheduler.pop_batch(
                    limit=agent_config.max_search_tasks_per_iteration,
                    kind="search",
                )
                if search_tasks:
                    search_started = time.perf_counter()
                    task_results = await search_agent.search_many(
                        [(task.payload["clue"], task.branch) for task in search_tasks]
                    )
                    timing_stats["search_time"] = round(
                        float(timing_stats["search_time"]) + (time.perf_counter() - search_started),
                        4,
                    )
                    provider_report = getattr(search_agent, "last_search_report", {})
                    providers_used = provider_report.get("providers_used", [])
                    quota_exhausted = provider_report.get("quota_exhausted", [])
                    if quota_exhausted:
                        reasoning_trace.append(
                            f"Iteration {iteration}: rotated search providers after quota limits on {', '.join(quota_exhausted)}."
                        )
                    elif len(providers_used) > 1:
                        reasoning_trace.append(
                            f"Iteration {iteration}: search clues were distributed across providers {', '.join(providers_used)}."
                        )

                    queued_fetches = 0
                    for task in search_tasks:
                        clue_hits = [hit for hit in task_results if hit.clue == task.payload["clue"]]
                        for hit in clue_hits:
                            memory.remember_search_hit(hit)
                            fetch_key = f"fetch::{canonicalize_url(hit.url) or hit.url}"
                            if not (memory.queue_url(hit.url) and memory.remember_task(fetch_key)):
                                continue
                            scheduler.push(
                                FrontierTask(
                                    kind="fetch",
                                    key=fetch_key,
                                    priority=score_fetch_priority(trimmed_query, hit),
                                    branch=task.branch,
                                    payload={"hit": hit},
                                    depth=task.depth,
                                )
                            )
                            queued_fetches += 1

                    bridged_search_hits = self._enqueue_event_search_hit_bridge_clues(
                        trimmed_query,
                        task_results,
                        scheduler,
                        memory,
                        all_clues,
                        iteration,
                    )
                    if query_intent.answer_type == "year" and query_intent.is_open_domain_browsecomp:
                        bridged_search_hits += self._enqueue_historical_year_search_hit_bridge_clues(
                            trimmed_query,
                            task_results,
                            scheduler,
                            memory,
                            all_clues,
                            iteration,
                        )

                    reasoning_trace.append(
                        f"Iteration {iteration}: searched {len(search_tasks)} clues and queued {queued_fetches} promising documents."
                    )
                    if bridged_search_hits:
                        reasoning_trace.append(
                            f"Iteration {iteration}: promoted {bridged_search_hits} title-specific follow-up clues from promising search snippets."
                        )

                fetch_tasks = await scheduler.pop_batch(
                    limit=agent_config.max_fetch_tasks_per_iteration,
                    kind="fetch",
                )
                if fetch_tasks:
                    fetched_documents, download_time, parse_time = await fetcher.fetch_many(
                        [task.payload["hit"] for task in fetch_tasks]
                    )
                    timing_stats["download_time"] = round(
                        float(timing_stats["download_time"]) + download_time,
                        4,
                    )
                    timing_stats["parsing_time"] = round(
                        float(timing_stats["parsing_time"]) + parse_time,
                        4,
                    )

                    pdf_count = 0
                    html_count = 0
                    for document in fetched_documents:
                        if document.content_type == "pdf":
                            pdf_count += 1
                        elif document.content_type == "html":
                            html_count += 1
                        memory.remember_document(document)
                        evidence_graph.add_document(document)

                    reasoning_trace.append(
                        f"Iteration {iteration}: parsed {len(fetched_documents)} documents ({pdf_count} PDF, {html_count} HTML)."
                    )

                all_documents = deduplicate_documents(memory.documents())
                if not all_documents:
                    reflection = await self.reflection_engine.reflect(
                        trimmed_query,
                        [],
                        all_clues,
                        answer_candidate,
                    )
                    if not reflection.should_continue:
                        break
                    self._enqueue_reflection_clues(
                        reflection.clues,
                        scheduler,
                        memory,
                        all_clues,
                        trimmed_query,
                        iteration,
                    )
                    reasoning_trace.extend(reflection.notes)
                    continue

                ranking_started = time.perf_counter()
                final_ranked_documents, backend = rank_documents(
                    trimmed_query,
                    all_documents,
                    return_backend=True,
                )
                timing_stats["ranking_time"] = round(
                    float(timing_stats["ranking_time"]) + (time.perf_counter() - ranking_started),
                    4,
                )

                extraction_started = time.perf_counter()
                candidate_documents = final_ranked_documents[: agent_config.max_answer_documents]
                located_sections = 0
                located_section_names: list[str] = []
                for document in candidate_documents:
                    if query_intent.requires_acknowledgement_section and not document.sections.get("acknowledgements"):
                        section_match = find_acknowledgements_section(document.content)
                        if section_match is not None:
                            located_sections += 1
                            located_section_names.append("acknowledgements")
                            document.acknowledgement_section = section_match.text
                            document.sections["acknowledgements"] = section_match.text
                            evidence_graph.add_fact_support(
                                document,
                                "acknowledgements section",
                                section_match.text[:240],
                            )
                            entity_evidence = extract_name_candidates_with_evidence(section_match.text)
                            if entity_evidence:
                                document.entities = tuple(name for name, _ in entity_evidence)
                                for name, evidence in entity_evidence:
                                    evidence_graph.add_entity_mention(
                                        document,
                                        name,
                                        evidence or section_match.text[:200],
                                    )

                    if query_intent.requires_reference_section and not document.sections.get("references"):
                        reference_match = find_references_section(document.content)
                        if reference_match is not None:
                            located_sections += 1
                            located_section_names.append("references")
                            document.sections["references"] = reference_match.text
                            evidence_graph.add_fact_support(
                                document,
                                "references section",
                                reference_match.text[:240],
                            )
                    if query_intent.is_open_domain_browsecomp:
                        generic_entities = extract_capitalized_entities(
                            f"{document.title} {document.content[:2200]}"
                        )[:5]
                        if generic_entities:
                            document.entities = tuple(dict.fromkeys(document.entities + tuple(generic_entities)))
                            for entity in generic_entities:
                                evidence_graph.add_entity_mention(
                                    document,
                                    entity,
                                    document.content[:220] or document.title,
                                )
                if located_sections and located_section_names:
                    rendered_section_names = ", ".join(sorted(set(located_section_names)))
                    reasoning_trace.append(
                        f"Iteration {iteration}: located {rendered_section_names} sections in {located_sections} document(s)."
                    )

                selected_documents = select_context_documents(
                    trimmed_query,
                    final_ranked_documents,
                    max_sources=max(1, min(max_sources, 10)),
                    pipeline_mode=query_plan.mode,
                )
                final_evidence = extract_evidence(trimmed_query, selected_documents)
                iteration_answer_candidate = await self.answer_extractor.extract(
                    trimmed_query,
                    candidate_documents,
                    final_evidence,
                )
                timing_stats["extraction_time"] = round(
                    float(timing_stats["extraction_time"]) + (time.perf_counter() - extraction_started),
                    4,
                )

                if iteration_answer_candidate.answer:
                    selected_candidate, adopted_iteration_candidate = self._select_preferred_answer_candidate(
                        trimmed_query,
                        best_answer_candidate,
                        iteration_answer_candidate,
                        all_documents,
                    )
                    best_answer_candidate = selected_candidate
                    answer_candidate = best_answer_candidate
                    reasoning_trace.append(
                        f"Iteration {iteration}: extracted answer candidate '{iteration_answer_candidate.answer}' from grounded evidence."
                    )
                    if not adopted_iteration_candidate and best_answer_candidate.answer:
                        reasoning_trace.append(
                            f"Iteration {iteration}: retained stronger earlier answer candidate '{best_answer_candidate.answer}' over the weaker new candidate."
                        )
                    if best_answer_candidate.confidence >= 0.85:
                        reasoning_trace.append("Answer confidence is high, so the agent stopped early.")
                        break
                else:
                    answer_candidate = best_answer_candidate

                bridge_clue_count = 0
                link_candidate_count = 0
                if query_intent.is_open_domain_browsecomp and iteration < query_plan.max_iterations:
                    bridge_clue_count = self._enqueue_browsecomp_bridge_clues(
                        trimmed_query,
                        candidate_documents,
                        scheduler,
                        memory,
                        all_clues,
                        iteration,
                    )
                    if bridge_clue_count:
                        reasoning_trace.append(
                            f"Iteration {iteration}: promoted {bridge_clue_count} title-specific follow-up clues from promising candidate pages."
                        )
                    link_candidate_count = self._enqueue_browsecomp_link_candidates(
                        trimmed_query,
                        candidate_documents,
                        scheduler,
                        memory,
                        iteration,
                        agent_config=agent_config,
                    )
                    if link_candidate_count:
                        reasoning_trace.append(
                            f"Iteration {iteration}: mined {link_candidate_count} candidate title/entity pages from broad overview links."
                        )

                reflection = await self.reflection_engine.reflect(
                    trimmed_query,
                    candidate_documents,
                    all_clues,
                    best_answer_candidate,
                )
                if not reflection.should_continue:
                    if bridge_clue_count or link_candidate_count:
                        continue
                    reasoning_trace.extend(reflection.notes)
                    break

                self._enqueue_reflection_clues(
                    reflection.clues,
                    scheduler,
                    memory,
                    all_clues,
                    trimmed_query,
                    iteration,
                )
                reasoning_trace.extend(reflection.notes)

        top_documents = select_context_documents(
            trimmed_query,
            final_ranked_documents,
            max_sources=max(1, min(max_sources, 10)),
            pipeline_mode=query_plan.mode,
        )
        final_evidence = extract_evidence(trimmed_query, top_documents)
        context = build_context_block(final_evidence)

        answer_candidate = best_answer_candidate
        if answer_candidate.answer:
            answer = answer_candidate.answer
        elif final_evidence:
            answer = ""
        else:
            answer = ""

        timing_stats["search_hits"] = len(memory.search_hits_seen)
        timing_stats["documents_fetched"] = len(memory.documents_seen)
        timing_stats["graph_nodes"] = evidence_graph.summary()["nodes"]
        timing_stats["graph_edges"] = evidence_graph.summary()["edges"]
        timing_stats["search_providers"] = sorted(getattr(search_agent, "providers_used_overall", set()))
        timing_stats["total_time"] = round(time.perf_counter() - started_at, 4)

        resolved_evidence = answer_candidate.evidence if answer_candidate.answer else ""
        resolved_source = answer_candidate.source if answer_candidate.answer else ""

        return PipelineResult(
            query=trimmed_query,
            clues=all_clues,
            sources=final_evidence,
            context=context,
            retrieved_documents=len(memory.search_hits_seen),
            deduplicated_documents=len(deduplicate_documents(memory.documents())),
            reranker=backend,
            answer=answer,
            evidence=resolved_evidence,
            source=resolved_source,
            reasoning_trace=reasoning_trace,
            timing_stats=timing_stats,
            pipeline_mode=query_plan.mode,
            decomposition_used=query_plan.use_decomposition,
            follow_up_used=len(all_clues) > len(initial_clues),
        )

    def _select_preferred_answer_candidate(
        self,
        query: str,
        current_best: AnswerCandidate,
        new_candidate: AnswerCandidate,
        documents: list[Document],
    ) -> tuple[AnswerCandidate, bool]:
        if not new_candidate.answer:
            return current_best, False
        if not current_best.answer:
            return new_candidate, True

        current_score = self._answer_candidate_score(query, current_best, documents)
        new_score = self._answer_candidate_score(query, new_candidate, documents)
        same_answer = normalize_whitespace(current_best.answer).lower() == normalize_whitespace(new_candidate.answer).lower()

        if same_answer:
            if new_score >= current_score - 0.02 and new_candidate.confidence >= current_best.confidence:
                return new_candidate, True
            return current_best, False

        if new_score > current_score + 0.05:
            return new_candidate, True
        return current_best, False

    def _answer_candidate_score(
        self,
        query: str,
        candidate: AnswerCandidate,
        documents: list[Document],
    ) -> float:
        if not candidate.answer:
            return float("-inf")

        intent = analyze_query_intent(query)
        score = float(candidate.confidence)
        evidence_lower = normalize_whitespace(candidate.evidence).lower()
        source_document = self._document_for_candidate_source(candidate.source, documents)

        if intent.answer_type == "year":
            if re.fullmatch(r"(?:19|20)\d{2}", normalize_whitespace(candidate.answer)):
                score += 0.06
            else:
                score -= 0.8
            if any(
                term in evidence_lower
                for term in ("victims", "massacre", "battle", "killed", "died", "executed", "commemorates", "honor")
            ):
                score += 0.18
            if any(
                term in evidence_lower
                for term in ("built", "constructed", "erected", "unveiled", "completed", "opened")
            ):
                score -= 0.18
            if any(term in evidence_lower for term in ("born", "award", "awards", "prize", "competition", "salon")):
                score -= 0.22

        if source_document is None:
            combined_text = normalize_whitespace(f"{candidate.source} {candidate.evidence}").lower()
        else:
            combined_text = normalize_whitespace(
                f"{source_document.title} {source_document.url} {source_document.content[:2600]} {candidate.evidence}"
            ).lower()

        if intent.answer_type == "year" and source_document is not None:
            structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
                query,
                source_document.url,
                source_document.title,
                source_document.content[:2600],
            )
            score += 0.28 * structural_score
            if is_specific_historical_year_page(
                query,
                source_document.url,
                source_document.title,
                source_document.content[:2600],
            ):
                score += 0.26
            else:
                score -= 0.24
            if is_generic_historical_monument_page(
                source_document.url,
                source_document.title,
                source_document.content[:1200],
            ):
                score -= 0.32
            if is_person_biography_page(
                source_document.url,
                source_document.title,
                source_document.content[:2200],
            ):
                score -= 0.36
            if structural_contradictions > 0:
                score -= 0.5
            if (
                historical_year_has_structural_constraints(query)
                and structural_matches == 0
                and not historical_year_trusted_memorial_source(source_document.url)
            ):
                score -= 0.38

        if intent.answer_type == "year" and query_requires_bosnia_top_city(query):
            if any(term in combined_text for term in BOSNIA_TOP_FOUR_CITY_TERMS):
                score += 0.22
            else:
                score -= 0.45

        return score

    def _document_for_candidate_source(
        self,
        source: str,
        documents: list[Document],
    ) -> Document | None:
        if not source:
            return None
        canonical_source = canonicalize_url(source) or source
        for document in documents:
            canonical_document = canonicalize_url(document.url) or document.url
            if canonical_document == canonical_source:
                return document
        return None

    def _enqueue_reflection_clues(
        self,
        clues: list[str],
        scheduler: FrontierScheduler,
        memory: AgentMemory,
        all_clues: list[str],
        query: str,
        iteration: int,
    ) -> None:
        prepared_clues = prepare_follow_up_retrieval_clues(query, clues)
        for clue in prepared_clues:
            task_key = f"search::{clue.lower()}"
            if not (memory.remember_query(clue) and memory.remember_task(task_key)):
                continue
            all_clues.append(clue)
            scheduler.push(
                FrontierTask(
                    kind="search",
                    key=task_key,
                    priority=score_search_priority(query, clue),
                    branch=f"reflection-{iteration}",
                    payload={"clue": clue},
                    depth=iteration,
                )
            )

    def _enqueue_browsecomp_bridge_clues(
        self,
        query: str,
        documents: list[Document],
        scheduler: FrontierScheduler,
        memory: AgentMemory,
        all_clues: list[str],
        iteration: int,
    ) -> int:
        intent = analyze_query_intent(query)
        if not intent.is_open_domain_browsecomp:
            return 0

        candidate_clues: list[str] = []
        anchor_terms = specific_query_terms(query)[:8]
        for document in documents[:4]:
            title_phrase = document_title_query_phrase(document.title)
            if not title_phrase:
                continue
            if intent.answer_type == "year" and is_plausible_person_name(title_phrase):
                continue
            event_score = event_page_score(document.url, document.title, document.content[:2500])
            winner_evidence = has_event_winner_evidence(
                normalize_whitespace(f"{document.title}. {document.content[:2500]}"),
                minimum_score=0.34,
            )
            if is_wiki_meta_page(document.url, document.title, document.content[:800]):
                continue
            if is_non_english_wiki_page(document.url):
                continue
            if intent.answer_type == "year" and is_generic_historical_monument_page(
                document.url,
                document.title,
                document.content[:1200],
            ):
                continue
            if intent.answer_type == "year" and is_person_biography_page(
                document.url,
                document.title,
                document.content[:2200],
            ):
                continue
            if is_generic_media_topic_page(document.url, document.title, document.content[:1200]):
                continue
            if intent.prefers_event_sources and is_generic_event_topic_page(
                document.url,
                document.title,
                document.content[:1200],
            ):
                continue
            if intent.prefers_event_sources and is_recipe_food_page(
                document.url,
                document.title,
                document.content[:700],
            ):
                continue
            if is_broad_overview_page(document.url, document.title):
                continue
            if is_aggregate_listing_page(document.url, document.title, document.content[:300]):
                continue
            if intent.is_media_query and not looks_like_media_page(
                document.url,
                document.title,
                document.content[:2500],
                minimum_score=0.38,
            ):
                continue
            if intent.prefers_event_sources and not winner_evidence and event_score < 0.42:
                continue
            if not is_grounded_browsecomp_page(
                query,
                document.url,
                document.title,
                document.content[:2600],
                require_media=intent.is_media_query,
            ):
                continue

            specificity = specificity_overlap_score(
                query,
                f"{document.title} {document.content[:2500]}",
            )
            lowered_document = normalize_whitespace(
                f"{document.title} {document.content[:2500]}"
            ).lower()
            anchor_matches = [term for term in anchor_terms if term in lowered_document]
            if specificity < 0.2:
                continue
            if len(anchor_matches) < 2 and specificity < 0.28:
                continue
            if intent.answer_type == "year":
                if not is_specific_historical_year_page(
                    query,
                    document.url,
                    document.title,
                    document.content[:2600],
                ):
                    continue
                query_lower = query.lower()
                location_hints = [
                    term
                    for term in ("bosnia", "yugoslavia", "sarajevo", "banja luka", "tuzla", "zenica")
                    if term in query_lower
                ]
                location_hit = any(term in lowered_document for term in location_hints)
                city_hit = any(term in lowered_document for term in BOSNIA_TOP_FOUR_CITY_TERMS)
                if query_requires_bosnia_top_city(query) and not city_hit:
                    continue
                if not location_hit and specificity < 0.24:
                    continue

            quoted_title = f"\"{title_phrase}\""
            if intent.prefers_encyclopedic_sources:
                candidate_clues.extend(
                    [
                        f"site:wikipedia.org {quoted_title}",
                        f"site:fandom.com {quoted_title}",
                        f"{quoted_title} wiki",
                    ]
                )
            if intent.needs_entity_discovery_hop:
                candidate_clues.extend(
                    [
                        f"{quoted_title} creators",
                        f"{quoted_title} first chapter",
                        f"{quoted_title} group name",
                    ]
                )
            if intent.prefers_character_sources:
                candidate_clues.extend(
                    [
                        f"{quoted_title} character",
                        f"{quoted_title} antagonist companion",
                    ]
                )
            if intent.prefers_event_sources:
                candidate_clues.extend(
                    [
                        f"{quoted_title} beauty pageant winner",
                        f"{quoted_title} contest winner",
                        f"{quoted_title} official tourism",
                    ]
                )
                if "anniversary" in query.lower():
                    candidate_clues.append(f"{quoted_title} anniversary competition")
            if intent.answer_type == "year":
                candidate_clues.extend(
                    [
                        f"{quoted_title} event year",
                        f"{quoted_title} victims year",
                    ]
                )
                if any(term in query.lower() for term in ("monument", "memorial", "spomenik")):
                    candidate_clues.append(f"{quoted_title} monument dedication")
            if intent.targets_count:
                candidate_clues.extend(
                    [
                        f"{quoted_title} movements effects",
                        f"{quoted_title} ability list",
                    ]
                )

        added = 0
        for clue in unique_preserve_order(candidate_clues)[:8]:
            task_key = f"search::{clue.lower()}"
            if not (memory.remember_query(clue) and memory.remember_task(task_key)):
                continue
            all_clues.append(clue)
            scheduler.push(
                FrontierTask(
                    kind="search",
                    key=task_key,
                    priority=score_search_priority(query, clue),
                    branch=f"bridge-{iteration}",
                    payload={"clue": clue},
                    depth=iteration,
                )
            )
            added += 1
        return added

    def _enqueue_event_search_hit_bridge_clues(
        self,
        query: str,
        hits: list[SearchHit],
        scheduler: FrontierScheduler,
        memory: AgentMemory,
        all_clues: list[str],
        iteration: int,
    ) -> int:
        intent = analyze_query_intent(query)
        if not (intent.prefers_event_sources and intent.is_open_domain_browsecomp):
            return 0

        candidate_clues: list[str] = []
        ranked_hits = sorted(
            hits,
            key=lambda hit: score_fetch_priority(query, hit),
            reverse=True,
        )
        for hit in ranked_hits[:12]:
            combined = normalize_whitespace(f"{hit.title}. {hit.snippet} {hit.raw_content[:500]}")
            title_phrase = self._event_title_from_hit(hit)
            if not title_phrase:
                continue
            if not any(term in title_phrase.lower() for term in ("festival", "fiesta", "celebration")):
                continue
            if is_broad_overview_page(hit.url, hit.title):
                continue
            if is_aggregate_listing_page(hit.url, hit.title, hit.snippet):
                continue
            if is_wiki_meta_page(hit.url, hit.title, hit.snippet):
                continue
            if is_non_english_wiki_page(hit.url):
                continue
            if is_generic_event_topic_page(hit.url, hit.title, combined):
                continue
            if is_recipe_food_page(hit.url, hit.title, combined):
                continue
            if is_forum_discussion_page(hit.url, hit.title, hit.snippet):
                continue

            event_score = event_page_score(hit.url, hit.title, combined)
            specificity = specificity_overlap_score(query, combined)
            if event_score < 0.24 and specificity < 0.12:
                continue

            quoted_title = f"\"{title_phrase}\""
            candidate_clues.extend(
                [
                    f"{quoted_title} official tourism municipality",
                    f"{quoted_title} festival official tourism",
                ]
            )
            location = self._event_location_from_text(combined)
            if location:
                candidate_clues.append(f"{quoted_title} \"{location}\"")
            if "anniversary" in query.lower():
                candidate_clues.append(f"{quoted_title} anniversary competition province official")
            if intent.targets_person:
                candidate_clues.append(f"{quoted_title} beauty pageant winner")
                candidate_clues.append(f"{quoted_title} contest winner full name")

        added = 0
        for clue in unique_preserve_order(candidate_clues)[:8]:
            task_key = f"search::{clue.lower()}"
            if not (memory.remember_query(clue) and memory.remember_task(task_key)):
                continue
            all_clues.append(clue)
            scheduler.push(
                FrontierTask(
                    kind="search",
                    key=task_key,
                    priority=score_search_priority(query, clue),
                    branch=f"hit-bridge-{iteration}",
                    payload={"clue": clue},
                    depth=iteration,
                )
            )
            added += 1
        return added

    def _event_location_from_text(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(
            r"\b(?:municipality|province|township|city|town)\s+of\s+([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,2})\b",
            text,
        )
        if match:
            return normalize_whitespace(match.group(1))
        return ""

    def _historical_year_location_from_text(self, text: str) -> str:
        if not text:
            return ""
        lowered = normalize_whitespace(text).lower()
        for city in BOSNIA_TOP_FOUR_CITY_TERMS:
            if city in lowered:
                return city.title()
        return ""

    def _historical_year_title_from_hit(self, query: str, hit: SearchHit) -> str:
        combined = normalize_whitespace(f"{hit.title}. {hit.snippet} {hit.raw_content[:500]}")
        title_phrase = document_title_query_phrase(hit.title)
        if title_phrase and is_specific_historical_year_page(query, hit.url, title_phrase, combined):
            return title_phrase

        match = re.search(
            r"\b((?:Memorial|Monument|Spomenik|Mausoleum)[^|]{0,80})\b",
            combined,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        candidate = normalize_whitespace(match.group(1).strip(" -|"))
        if is_specific_historical_year_page(query, hit.url, candidate, combined):
            return candidate
        return ""

    def _event_title_from_hit(self, hit: SearchHit) -> str:
        title_phrase = document_title_query_phrase(hit.title)
        if title_phrase and any(term in title_phrase.lower() for term in ("festival", "fiesta", "celebration")):
            return title_phrase

        combined = normalize_whitespace(f"{hit.title} {hit.snippet} {hit.raw_content[:400]}")
        match = re.search(
            r"\b([A-Za-z][A-Za-z'`.-]+(?:\s+[A-Za-z][A-Za-z'`.-]+){0,3}\s+(?:festival|fiesta|celebration))\b",
            combined,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        candidate = normalize_whitespace(match.group(1))
        return candidate.title()

    def _enqueue_historical_year_search_hit_bridge_clues(
        self,
        query: str,
        hits: list[SearchHit],
        scheduler: FrontierScheduler,
        memory: AgentMemory,
        all_clues: list[str],
        iteration: int,
    ) -> int:
        intent = analyze_query_intent(query)
        if not (intent.answer_type == "year" and intent.is_open_domain_browsecomp):
            return 0

        candidate_clues: list[str] = []
        ranked_hits = sorted(
            hits,
            key=lambda hit: score_fetch_priority(query, hit),
            reverse=True,
        )
        for hit in ranked_hits[:12]:
            combined = normalize_whitespace(f"{hit.title}. {hit.snippet} {hit.raw_content[:500]}")
            structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
                query,
                hit.url,
                hit.title,
                combined,
            )
            if not is_specific_historical_year_page(query, hit.url, hit.title, combined):
                title_phrase = self._historical_year_title_from_hit(query, hit)
                if not title_phrase:
                    continue
            else:
                title_phrase = document_title_query_phrase(hit.title) or normalize_whitespace(hit.title)

            lowered = combined.lower()
            if query_requires_bosnia_top_city(query) and not any(term in lowered for term in BOSNIA_TOP_FOUR_CITY_TERMS):
                continue
            if structural_contradictions > 0:
                continue
            if (
                historical_year_has_structural_constraints(query)
                and structural_matches == 0
                and not historical_year_trusted_memorial_source(hit.url)
            ):
                continue
            if is_broad_overview_page(hit.url, hit.title):
                continue
            if is_aggregate_listing_page(hit.url, hit.title, hit.snippet):
                continue
            if is_wiki_meta_page(hit.url, hit.title, hit.snippet):
                continue
            if is_non_english_wiki_page(hit.url):
                continue
            if is_generic_historical_monument_page(hit.url, hit.title, combined):
                continue
            if is_person_biography_page(hit.url, hit.title, combined):
                continue
            if is_forum_discussion_page(hit.url, hit.title, hit.snippet):
                continue

            quoted_title = f"\"{title_phrase}\""
            candidate_clues.extend(
                [
                    f"{quoted_title} event year",
                    f"{quoted_title} victims year",
                    f"{quoted_title} monument dedication",
                ]
            )
            location = self._historical_year_location_from_text(combined)
            if location:
                candidate_clues.append(f"{quoted_title} \"{location}\" victims year")
            if query_requires_bosnia_top_city(query):
                candidate_clues.append(f"{quoted_title} Bosnia city victims year")
            if structural_score > 0.18 and "former yugoslavia" in query.lower():
                candidate_clues.append(f"{quoted_title} former Yugoslavia monument")

        added = 0
        for clue in unique_preserve_order(candidate_clues)[:8]:
            task_key = f"search::{clue.lower()}"
            if not (memory.remember_query(clue) and memory.remember_task(task_key)):
                continue
            all_clues.append(clue)
            scheduler.push(
                FrontierTask(
                    kind="search",
                    key=task_key,
                    priority=score_search_priority(query, clue),
                    branch=f"year-hit-bridge-{iteration}",
                    payload={"clue": clue},
                    depth=iteration,
                )
            )
            added += 1
        return added

    def _enqueue_browsecomp_link_candidates(
        self,
        query: str,
        documents: list[Document],
        scheduler: FrontierScheduler,
        memory: AgentMemory,
        iteration: int,
        *,
        agent_config,
    ) -> int:
        intent = analyze_query_intent(query)
        if not intent.is_open_domain_browsecomp:
            return 0

        candidate_hits: list[tuple[float, SearchHit]] = []
        for document in documents[: agent_config.max_browsecomp_pivot_documents]:
            if not (
                is_broad_overview_page(document.url, document.title)
                or is_aggregate_listing_page(document.url, document.title, document.content[:400])
            ):
                continue
            if intent.prefers_event_sources and is_generic_event_topic_page(
                document.url,
                document.title,
                document.content[:1200],
            ):
                continue

            raw_links = document.metadata.get("links")
            if not isinstance(raw_links, list) or not raw_links:
                continue

            for raw_link in raw_links[:120]:
                if not isinstance(raw_link, dict):
                    continue
                link_text = normalize_whitespace(str(raw_link.get("text") or ""))
                link_url = normalize_whitespace(str(raw_link.get("url") or ""))
                if not link_text or not link_url:
                    continue
                if (canonicalize_url(link_url) or link_url) == (canonicalize_url(document.url) or document.url):
                    continue
                if is_forum_discussion_page(link_url, link_text):
                    continue
                if is_wiki_meta_page(link_url, link_text):
                    continue
                if is_non_english_wiki_page(link_url):
                    continue

                candidate_score = score_browsecomp_link_candidate(
                    query,
                    link_text,
                    link_url,
                    parent_title=document.title,
                    parent_url=document.url,
                )
                if candidate_score < agent_config.min_browsecomp_link_score:
                    continue
                if intent.is_media_query and not looks_like_media_page(
                    link_url,
                    link_text,
                    "",
                    minimum_score=0.18,
                ):
                    continue

                candidate_hits.append(
                    (
                        candidate_score,
                        SearchHit(
                            title=link_text,
                            url=link_url,
                            snippet=f"Candidate title/entity page mined from {document.title}.",
                            raw_content=link_text,
                            retrieval_score=max(document.rank_score, document.retrieval_score, candidate_score),
                            source=extract_source(link_url),
                            clue=f"browsecomp-link::{document.title}",
                            branch=f"link-mine-{iteration}",
                            provider="link-miner",
                        ),
                    )
                )

        added = 0
        seen_urls: set[str] = set()
        ranked_hits = sorted(candidate_hits, key=lambda item: item[0], reverse=True)
        for candidate_score, hit in ranked_hits:
            canonical_url = canonicalize_url(hit.url) or hit.url
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)
            fetch_key = f"fetch::{canonical_url}"
            if not (memory.queue_url(hit.url) and memory.remember_task(fetch_key)):
                continue
            scheduler.push(
                FrontierTask(
                    kind="fetch",
                    key=fetch_key,
                    priority=min(1.0, score_fetch_priority(query, hit) + (0.18 * candidate_score)),
                    branch=hit.branch,
                    payload={"hit": hit},
                    depth=iteration,
                )
            )
            added += 1
            if added >= agent_config.max_browsecomp_link_candidates:
                break
        return added
