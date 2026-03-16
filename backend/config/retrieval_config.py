from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_float
from config.search_config import DEFAULT_USER_AGENT

DEFAULT_FETCH_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class DocumentFetchConfig:
    timeout_seconds: float
    user_agent: str


def load_document_fetch_config() -> DocumentFetchConfig:
    return DocumentFetchConfig(
        timeout_seconds=get_env_float(
            "TAVILY_FETCH_TIMEOUT_SECONDS",
            DEFAULT_FETCH_TIMEOUT_SECONDS,
            minimum=1.0,
        ),
        user_agent=DEFAULT_USER_AGENT,
    )
