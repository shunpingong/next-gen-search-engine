from __future__ import annotations

import re

from agent.models import QueryPlan
from config.planner_config import (
    ACADEMIC_PER_CLUE_RESULTS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PER_CLUE_RESULTS,
    DEFAULT_PIPELINE_MODE,
    PIPELINE_MODE_VALUES,
    load_query_planner_config,
)
from planner.query_intent import analyze_query_intent
from utils.text_utils import (
    MULTI_HOP_RELATIONSHIP_HINTS,
    extract_constraint_phrases,
    important_terms,
    is_academic_lookup_query,
    is_media_lookup_query,
    is_person_target_query,
    normalize_whitespace,
    split_into_clauses,
    strip_question_prefix,
)

SIMPLE_QUERY_IMPORTANT_TERM_LIMIT = 8
COMPLEX_QUERY_IMPORTANT_TERM_THRESHOLD = 12
SIMPLE_QUERY_CLAUSE_LIMIT = 2
SIMPLE_QUERY_CONSTRAINT_LIMIT = 1
BROWSECOMP_PER_CLUE_RESULTS = 8


def plan_query_pipeline(query: str) -> QueryPlan:
    planner_config = load_query_planner_config()
    override = planner_config.pipeline_mode_override
    normalized_query = normalize_whitespace(query)
    browsecomp_iterations = _get_iteration_limit(normalized_query, planner_config)
    if override == "simple":
        return QueryPlan(
            mode="simple",
            use_decomposition=False,
            use_follow_up=False,
            max_results_per_clue=DEFAULT_PER_CLUE_RESULTS,
            max_iterations=1,
        )
    if override == "decomposition":
        return QueryPlan(
            mode="decomposition",
            use_decomposition=True,
            use_follow_up=False,
            max_results_per_clue=DEFAULT_PER_CLUE_RESULTS,
            max_iterations=browsecomp_iterations,
        )
    if override == "multi-hop":
        return QueryPlan(
            mode="multi-hop",
            use_decomposition=True,
            use_follow_up=True,
            max_results_per_clue=_get_per_clue_result_limit(query),
            max_iterations=browsecomp_iterations,
        )

    constraints = extract_constraint_phrases(normalized_query)
    clauses = split_into_clauses(strip_question_prefix(normalized_query), constraints)
    important_term_count = len(important_terms(normalized_query))
    relationship_hint_count = sum(
        1 for hint in MULTI_HOP_RELATIONSHIP_HINTS if hint in normalized_query.lower()
    )
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", normalized_query))
    person_target = is_person_target_query(normalized_query)
    academic_lookup = is_academic_lookup_query(normalized_query)

    if (
        not academic_lookup
        and important_term_count <= SIMPLE_QUERY_IMPORTANT_TERM_LIMIT
        and len(clauses) <= SIMPLE_QUERY_CLAUSE_LIMIT
        and len(constraints) <= SIMPLE_QUERY_CONSTRAINT_LIMIT
        and relationship_hint_count <= 1
        and not has_year
    ):
        return QueryPlan(
            mode="simple",
            use_decomposition=False,
            use_follow_up=False,
            max_results_per_clue=DEFAULT_PER_CLUE_RESULTS,
            max_iterations=1,
        )

    if (
        academic_lookup
        or len(constraints) >= 2
        or len(clauses) >= 3
        or relationship_hint_count >= 3
        or (person_target and has_year)
        or important_term_count >= COMPLEX_QUERY_IMPORTANT_TERM_THRESHOLD
    ):
        return QueryPlan(
            mode="multi-hop",
            use_decomposition=True,
            use_follow_up=True,
            max_results_per_clue=_get_per_clue_result_limit(normalized_query),
            max_iterations=browsecomp_iterations,
        )

    return QueryPlan(
        mode="decomposition",
        use_decomposition=True,
        use_follow_up=False,
        max_results_per_clue=DEFAULT_PER_CLUE_RESULTS,
        max_iterations=browsecomp_iterations,
    )


def get_per_clue_result_limit(query: str) -> int:
    return _get_per_clue_result_limit(query)


def get_max_iterations() -> int:
    return _get_max_iterations()


def _get_per_clue_result_limit(query: str) -> int:
    if is_academic_lookup_query(query):
        return ACADEMIC_PER_CLUE_RESULTS
    if is_media_lookup_query(query) or len(important_terms(query)) >= COMPLEX_QUERY_IMPORTANT_TERM_THRESHOLD:
        return BROWSECOMP_PER_CLUE_RESULTS
    return DEFAULT_PER_CLUE_RESULTS


def _get_pipeline_mode_override() -> str:
    return load_query_planner_config().pipeline_mode_override


def _get_max_iterations() -> int:
    return load_query_planner_config().max_iterations


def _get_iteration_limit(query: str, planner_config) -> int:
    if analyze_query_intent(query).is_open_domain_browsecomp:
        return max(planner_config.max_iterations, planner_config.browsecomp_max_iterations)
    return planner_config.max_iterations
