from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_float, get_env_str

DEFAULT_ANSWER_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class AnswerExtractionConfig:
    openai_api_key: str
    model: str
    timeout_seconds: float


def load_answer_extraction_config() -> AnswerExtractionConfig:
    return AnswerExtractionConfig(
        openai_api_key=get_env_str("OPENAI_API_KEY"),
        model=get_env_str("TAVILY_ANSWER_MODEL", DEFAULT_ANSWER_MODEL),
        timeout_seconds=get_env_float("TAVILY_ANSWER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, minimum=1.0),
    )
