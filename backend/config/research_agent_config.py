from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_float, get_env_int

DEFAULT_MAX_QUERY_LENGTH = 1000
DEFAULT_TOP_SOURCES = 5
DEFAULT_MAX_SEARCH_TASKS_PER_ITERATION = 6
DEFAULT_MAX_FETCH_TASKS_PER_ITERATION = 8
DEFAULT_MAX_ANSWER_DOCUMENTS = 6
DEFAULT_MAX_BROWSECOMP_LINK_CANDIDATES = 6
DEFAULT_MAX_BROWSECOMP_PIVOT_DOCUMENTS = 4
DEFAULT_MIN_BROWSECOMP_LINK_SCORE = 0.42


@dataclass(frozen=True)
class ResearchAgentConfig:
    max_query_length: int
    default_top_sources: int
    max_search_tasks_per_iteration: int
    max_fetch_tasks_per_iteration: int
    max_answer_documents: int
    max_browsecomp_link_candidates: int
    max_browsecomp_pivot_documents: int
    min_browsecomp_link_score: float


def load_research_agent_config() -> ResearchAgentConfig:
    return ResearchAgentConfig(
        max_query_length=DEFAULT_MAX_QUERY_LENGTH,
        default_top_sources=DEFAULT_TOP_SOURCES,
        max_search_tasks_per_iteration=get_env_int(
            "TAVILY_MAX_SEARCH_TASKS_PER_ITERATION",
            DEFAULT_MAX_SEARCH_TASKS_PER_ITERATION,
            minimum=1,
            maximum=12,
        ),
        max_fetch_tasks_per_iteration=get_env_int(
            "TAVILY_MAX_FETCH_TASKS_PER_ITERATION",
            DEFAULT_MAX_FETCH_TASKS_PER_ITERATION,
            minimum=1,
            maximum=20,
        ),
        max_answer_documents=get_env_int(
            "TAVILY_MAX_ANSWER_DOCUMENTS",
            DEFAULT_MAX_ANSWER_DOCUMENTS,
            minimum=1,
            maximum=12,
        ),
        max_browsecomp_link_candidates=get_env_int(
            "TAVILY_MAX_BROWSECOMP_LINK_CANDIDATES",
            DEFAULT_MAX_BROWSECOMP_LINK_CANDIDATES,
            minimum=1,
            maximum=12,
        ),
        max_browsecomp_pivot_documents=get_env_int(
            "TAVILY_MAX_BROWSECOMP_PIVOT_DOCUMENTS",
            DEFAULT_MAX_BROWSECOMP_PIVOT_DOCUMENTS,
            minimum=1,
            maximum=8,
        ),
        min_browsecomp_link_score=get_env_float(
            "TAVILY_MIN_BROWSECOMP_LINK_SCORE",
            DEFAULT_MIN_BROWSECOMP_LINK_SCORE,
            minimum=0.1,
            maximum=0.95,
        ),
    )
