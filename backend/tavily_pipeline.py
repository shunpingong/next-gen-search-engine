from __future__ import annotations

from agent.models import (
    AnswerCandidate,
    Document,
    EvidenceSnippet,
    FollowUpClueOutput,
    PipelineResult,
    QueryDecompositionOutput,
    QueryPlan,
    ReflectionOutput,
    SearchProviderError,
    SearchProviderQuotaError,
    SearchHit,
    TavilyPipelineError,
    TavilySearchError,
)
from agent.research_agent import ResearchAgent
from config.research_agent_config import DEFAULT_TOP_SOURCES
from planner.planner import DEFAULT_PER_CLUE_RESULTS, plan_query_pipeline
from planner.query_decomposer import (
    DEFAULT_DECOMPOSITION_TIMEOUT_SECONDS,
    DECOMPOSITION_TIMEOUT_BACKOFF_MULTIPLIER,
    decompose_query,
    decompose_query_async,
    merge_decomposition_clues,
    sanitize_model_clues,
)
from ranking.reranker import (
    build_context_block,
    deduplicate_documents,
    extract_evidence,
    rank_documents,
    select_context_documents,
)
from reflection.query_refiner import generate_follow_up_clues_async
from search.query_generator import (
    prepare_follow_up_retrieval_clues,
    prepare_retrieval_clues,
    prepare_simple_retrieval_clues,
)
from search.search_agent import (
    DEFAULT_TAVILY_ENDPOINT,
    retrieve_documents,
    retrieve_documents_async,
)

_AGENT = ResearchAgent()


async def run_tavily_pipeline(query: str, max_sources: int = DEFAULT_TOP_SOURCES) -> PipelineResult:
    return await _AGENT.run(query, max_sources=max_sources)


__all__ = [
    "AnswerCandidate",
    "Document",
    "EvidenceSnippet",
    "FollowUpClueOutput",
    "PipelineResult",
    "QueryDecompositionOutput",
    "QueryPlan",
    "ReflectionOutput",
    "SearchProviderError",
    "SearchProviderQuotaError",
    "SearchHit",
    "TavilyPipelineError",
    "TavilySearchError",
    "DEFAULT_DECOMPOSITION_TIMEOUT_SECONDS",
    "DEFAULT_PER_CLUE_RESULTS",
    "DEFAULT_TAVILY_ENDPOINT",
    "DECOMPOSITION_TIMEOUT_BACKOFF_MULTIPLIER",
    "build_context_block",
    "deduplicate_documents",
    "decompose_query",
    "decompose_query_async",
    "extract_evidence",
    "generate_follow_up_clues_async",
    "merge_decomposition_clues",
    "plan_query_pipeline",
    "prepare_follow_up_retrieval_clues",
    "prepare_retrieval_clues",
    "prepare_simple_retrieval_clues",
    "rank_documents",
    "retrieve_documents",
    "retrieve_documents_async",
    "run_tavily_pipeline",
    "sanitize_model_clues",
    "select_context_documents",
]
