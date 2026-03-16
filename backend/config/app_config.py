from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_str

DEFAULT_TEXTGRAD_MODEL = "gpt-4o"
DEFAULT_TEXTGRAD_FALLBACK_MODEL = "gpt-4o"


@dataclass(frozen=True)
class ApplicationConfig:
    openai_api_key: str
    textgrad_backward_model: str
    textgrad_advanced_model: str
    textgrad_fallback_model: str


def load_application_config() -> ApplicationConfig:
    return ApplicationConfig(
        openai_api_key=get_env_str("OPENAI_API_KEY"),
        textgrad_backward_model=get_env_str("TEXTGRAD_BACKWARD_MODEL", DEFAULT_TEXTGRAD_MODEL),
        textgrad_advanced_model=get_env_str("TEXTGRAD_ADVANCED_MODEL", DEFAULT_TEXTGRAD_MODEL),
        textgrad_fallback_model=DEFAULT_TEXTGRAD_FALLBACK_MODEL,
    )
