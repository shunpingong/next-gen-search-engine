from __future__ import annotations

from dataclasses import dataclass

from config.env_utils import get_env_bool, get_env_float, get_env_int, get_env_str

DEFAULT_PER_CLUE_RESULTS = 5
ACADEMIC_PER_CLUE_RESULTS = 8
DEFAULT_PIPELINE_MODE = "auto"
PIPELINE_MODE_VALUES = {"auto", "simple", "decomposition", "multi-hop"}
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_DECOMPOSITION_MODEL = "gpt-5-mini"
DEFAULT_DECOMPOSITION_TIMEOUT_SECONDS = 30.0
DEFAULT_DECOMPOSITION_MAX_ATTEMPTS = 2
DEFAULT_DECOMPOSITION_MAX_QUERY_LENGTH = 1000


@dataclass(frozen=True)
class PipelinePlannerConfig:
    default_per_clue_results: int
    academic_per_clue_results: int
    pipeline_mode_override: str
    max_iterations: int
    browsecomp_max_iterations: int


@dataclass(frozen=True)
class DecompositionConfig:
    openai_api_key: str
    use_llm: bool
    model: str
    timeout_seconds: float
    max_attempts: int
    max_query_length: int


def load_query_planner_config() -> PipelinePlannerConfig:
    override = get_env_str("TAVILY_PIPELINE_MODE", DEFAULT_PIPELINE_MODE).strip().lower()
    if override not in PIPELINE_MODE_VALUES:
        override = DEFAULT_PIPELINE_MODE
    return PipelinePlannerConfig(
        default_per_clue_results=DEFAULT_PER_CLUE_RESULTS,
        academic_per_clue_results=ACADEMIC_PER_CLUE_RESULTS,
        pipeline_mode_override=override,
        max_iterations=get_env_int(
            "TAVILY_MAX_ITERATIONS",
            DEFAULT_MAX_ITERATIONS,
            minimum=1,
            maximum=4,
        ),
        browsecomp_max_iterations=get_env_int(
            "TAVILY_BROWSECOMP_MAX_ITERATIONS",
            3,
            minimum=2,
            maximum=5,
        ),
    )


def load_decomposition_config() -> DecompositionConfig:
    return DecompositionConfig(
        openai_api_key=get_env_str("OPENAI_API_KEY"),
        use_llm=get_env_bool("TAVILY_USE_LLM_DECOMPOSITION", True),
        model=get_env_str("TAVILY_DECOMPOSITION_MODEL", DEFAULT_DECOMPOSITION_MODEL),
        timeout_seconds=get_env_float(
            "TAVILY_DECOMPOSITION_TIMEOUT_SECONDS",
            DEFAULT_DECOMPOSITION_TIMEOUT_SECONDS,
            minimum=1.0,
        ),
        max_attempts=get_env_int(
            "TAVILY_DECOMPOSITION_MAX_ATTEMPTS",
            DEFAULT_DECOMPOSITION_MAX_ATTEMPTS,
            minimum=1,
        ),
        max_query_length=DEFAULT_DECOMPOSITION_MAX_QUERY_LENGTH,
    )
