from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_csv, get_env_int, get_env_str

DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_PER_CLUE_RESULTS = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_PROVIDER_ORDER = (
    "tavily",
    "serper",
    "serpapi",
    "google_custom_search",
    "duckduckgo",
)
DEFAULT_PROVIDER_STRATEGY = "fallback"
PROVIDER_STRATEGY_VALUES = {"fallback", "round_robin"}
DEFAULT_QUOTA_COOLDOWN_SECONDS = 3600
DEFAULT_FAILURE_COOLDOWN_SECONDS = 300
DEFAULT_SELENIUM_ENGINE = "google"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class SearchConfig:
    tavily_api_key: str
    tavily_endpoint: str
    google_api_key: str
    google_search_engine_id: str
    serper_api_key: str
    serpapi_api_key: str
    max_results: int
    timeout_seconds: float
    provider_order: list[str]
    provider_strategy: str
    quota_cooldown_seconds: int
    failure_cooldown_seconds: int
    selenium_engine: str
    user_agent: str


def load_search_config(
    *,
    tavily_api_key: str | None = None,
    tavily_endpoint: str | None = None,
    max_results: int | None = None,
    timeout_seconds: float | None = None,
    provider_order: list[str] | None = None,
    provider_strategy: str | None = None,
) -> SearchConfig:
    resolved_provider_strategy = (
        provider_strategy.strip().lower()
        if provider_strategy is not None
        else get_env_str("SEARCH_PROVIDER_STRATEGY", DEFAULT_PROVIDER_STRATEGY).strip().lower()
    )
    if resolved_provider_strategy not in PROVIDER_STRATEGY_VALUES:
        resolved_provider_strategy = DEFAULT_PROVIDER_STRATEGY

    resolved_provider_order = (
        [item.strip().lower() for item in provider_order if item and item.strip()]
        if provider_order is not None
        else get_env_csv("SEARCH_PROVIDER_ORDER", DEFAULT_PROVIDER_ORDER)
    )

    return SearchConfig(
        tavily_api_key=tavily_api_key if tavily_api_key is not None else get_env_str("TAVILY_API_KEY"),
        tavily_endpoint=(
            tavily_endpoint
            if tavily_endpoint is not None
            else get_env_str("TAVILY_ENDPOINT", DEFAULT_TAVILY_ENDPOINT)
        ),
        google_api_key=get_env_str("GOOGLE_API_KEY"),
        google_search_engine_id=get_env_str("GOOGLE_SEARCH_ENGINE_ID"),
        serper_api_key=get_env_str("SERPER_API_KEY"),
        serpapi_api_key=get_env_str("SERPAPI_API_KEY"),
        max_results=max(1, min(10, max_results if max_results is not None else DEFAULT_PER_CLUE_RESULTS)),
        timeout_seconds=max(1.0, timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS),
        provider_order=resolved_provider_order or list(DEFAULT_PROVIDER_ORDER),
        provider_strategy=resolved_provider_strategy,
        quota_cooldown_seconds=get_env_int(
            "SEARCH_PROVIDER_QUOTA_COOLDOWN_SECONDS",
            DEFAULT_QUOTA_COOLDOWN_SECONDS,
            minimum=60,
            maximum=86_400,
        ),
        failure_cooldown_seconds=get_env_int(
            "SEARCH_PROVIDER_FAILURE_COOLDOWN_SECONDS",
            DEFAULT_FAILURE_COOLDOWN_SECONDS,
            minimum=30,
            maximum=3_600,
        ),
        selenium_engine=get_env_str("SEARCH_SELENIUM_ENGINE", DEFAULT_SELENIUM_ENGINE).strip().lower()
        or DEFAULT_SELENIUM_ENGINE,
        user_agent=DEFAULT_USER_AGENT,
    )
