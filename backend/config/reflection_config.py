from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_bool, get_env_float, get_env_int, get_env_str

MIN_FOLLOW_UP_CLUES = 2
MAX_FOLLOW_UP_CLUES = 4
DEFAULT_FOLLOW_UP_SOURCE_DOC_COUNT = 4
DEFAULT_FOLLOW_UP_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FollowUpConfig:
    openai_api_key: str
    use_retrieval: bool
    use_llm: bool
    source_doc_count: int
    model: str
    timeout_seconds: float


def load_follow_up_config() -> FollowUpConfig:
    return FollowUpConfig(
        openai_api_key=get_env_str("OPENAI_API_KEY"),
        use_retrieval=get_env_bool("TAVILY_USE_FOLLOW_UP_RETRIEVAL", True),
        use_llm=get_env_bool("TAVILY_USE_LLM_FOLLOW_UP", True),
        source_doc_count=get_env_int(
            "TAVILY_FOLLOW_UP_SOURCE_DOC_COUNT",
            DEFAULT_FOLLOW_UP_SOURCE_DOC_COUNT,
            minimum=1,
            maximum=8,
        ),
        model=get_env_str("TAVILY_FOLLOW_UP_MODEL", DEFAULT_FOLLOW_UP_MODEL),
        timeout_seconds=get_env_float("TAVILY_FOLLOW_UP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, minimum=1.0),
    )
