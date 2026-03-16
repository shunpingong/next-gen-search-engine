from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    raw_content: str
    retrieval_score: float
    source: str
    clue: str
    branch: str = ""
    published_date: str = ""
    provider: str = ""


@dataclass
class Document:
    title: str
    url: str
    content: str
    source: str
    matched_clues: tuple[str, ...] = field(default_factory=tuple)
    retrieval_score: float = 0.0
    rank_score: float = 0.0
    raw_content: str = ""
    content_type: str = "search-result"
    branch: str = ""
    fetched: bool = False
    fetch_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledgement_section: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    entities: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class EvidenceSnippet:
    title: str
    url: str
    snippet: str
    score: float


@dataclass
class AnswerCandidate:
    answer: str = ""
    evidence: str = ""
    source: str = ""
    confidence: float = 0.0
    supporting_person: str = ""


@dataclass
class PipelineResult:
    query: str
    clues: list[str]
    sources: list[EvidenceSnippet]
    context: str
    retrieved_documents: int
    deduplicated_documents: int
    reranker: str
    answer: str = ""
    evidence: str = ""
    source: str = ""
    reasoning_trace: list[str] = field(default_factory=list)
    timing_stats: dict[str, Any] = field(default_factory=dict)
    pipeline_mode: str = "decomposition"
    decomposition_used: bool = False
    follow_up_used: bool = False

    def to_response(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "source": self.source,
            "reasoning_trace": self.reasoning_trace,
            "timing_stats": self.timing_stats,
            "query": self.query,
            "clues": self.clues,
            "sources": [asdict(source) for source in self.sources],
            "context": self.context,
            "stats": {
                "retrieved_documents": self.retrieved_documents,
                "deduplicated_documents": self.deduplicated_documents,
                "reranker": self.reranker,
                "pipeline_mode": self.pipeline_mode,
                "decomposition_used": self.decomposition_used,
                "follow_up_used": self.follow_up_used,
                "timing_stats": self.timing_stats,
            },
        }


@dataclass(frozen=True)
class QueryPlan:
    mode: str
    use_decomposition: bool
    use_follow_up: bool
    max_results_per_clue: int
    max_iterations: int


@dataclass
class FrontierTask:
    kind: str
    key: str
    priority: float
    branch: str
    payload: dict[str, Any] = field(default_factory=dict)
    depth: int = 0


@dataclass
class SectionMatch:
    heading: str
    text: str
    start_index: int
    end_index: int


@dataclass
class ReflectionResult:
    should_continue: bool
    clues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class TavilyPipelineError(RuntimeError):
    """Base class for deep-research pipeline errors."""


class TavilySearchError(TavilyPipelineError):
    """Represents an upstream Tavily API failure."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SearchProviderError(TavilyPipelineError):
    """Represents an upstream search-provider failure after fallback is exhausted."""

    def __init__(self, provider: str, detail: str, status_code: int = 503):
        super().__init__(detail)
        self.provider = provider
        self.detail = detail
        self.status_code = status_code


class SearchProviderQuotaError(SearchProviderError):
    """Raised when a search provider reports that quota or credits are exhausted."""


class QueryDecompositionOutput(BaseModel):
    clues: list[str] = Field(
        default_factory=list,
        min_length=3,
        max_length=5,
        description="Three to five independent search clues for direct web search.",
    )


class FollowUpClueOutput(BaseModel):
    clues: list[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=4,
        description="Two to four document-grounded follow-up search clues.",
    )


class ReflectionOutput(BaseModel):
    should_continue: bool = Field(default=True)
    reason: str = Field(default="")
    clues: list[str] = Field(default_factory=list, max_length=4)


class AnswerExtractionOutput(BaseModel):
    answer: str = Field(default="")
    evidence: str = Field(default="")
    supporting_person: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
