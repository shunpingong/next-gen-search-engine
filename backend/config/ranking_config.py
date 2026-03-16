from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_str

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_BACKEND = "auto"
RERANKER_BACKEND_VALUES = {"auto", "cross-encoder", "embeddings", "lexical"}


@dataclass(frozen=True)
class RankingConfig:
    reranker_backend: str
    reranker_model: str
    embedding_model: str


def load_ranking_config() -> RankingConfig:
    backend = get_env_str("TAVILY_RERANKER_BACKEND", DEFAULT_RERANKER_BACKEND).strip().lower()
    if backend not in RERANKER_BACKEND_VALUES:
        backend = DEFAULT_RERANKER_BACKEND
    return RankingConfig(
        reranker_backend=backend,
        reranker_model=get_env_str("TAVILY_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        embedding_model=get_env_str("TAVILY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
