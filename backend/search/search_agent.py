from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.models import (
    Document,
    SearchHit,
    SearchProviderError,
    SearchProviderQuotaError,
    TavilyPipelineError,
)
from planner.query_intent import analyze_query_intent
from config.search_config import (
    DEFAULT_FAILURE_COOLDOWN_SECONDS,
    DEFAULT_PER_CLUE_RESULTS,
    DEFAULT_PROVIDER_ORDER,
    DEFAULT_PROVIDER_STRATEGY,
    DEFAULT_QUOTA_COOLDOWN_SECONDS,
    DEFAULT_SELENIUM_ENGINE,
    DEFAULT_TAVILY_ENDPOINT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    SearchConfig,
    load_search_config,
)
from utils.text_utils import (
    ABILITY_QUERY_HINTS,
    BOSNIA_TOP_FOUR_CITY_TERMS,
    CHARACTER_QUERY_HINTS,
    ENCYCLOPEDIC_SOURCE_HINTS,
    document_type_score,
    domain_reliability_score,
    event_page_score,
    extract_source,
    historical_year_has_structural_constraints,
    historical_year_structural_assessment,
    historical_year_trusted_memorial_source,
    is_aggregate_listing_page,
    is_broad_overview_page,
    is_forum_discussion_page,
    is_generic_event_topic_page,
    is_generic_media_topic_page,
    is_grounded_browsecomp_page,
    is_low_trust_social_page,
    is_non_english_wiki_page,
    is_recipe_food_page,
    is_specific_historical_year_page,
    is_wiki_meta_page,
    lexical_relevance_score,
    media_page_score,
    query_requires_bosnia_top_city,
    specificity_overlap_score,
    unique_preserve_order,
)

logger = logging.getLogger("search.search_agent")

_PROVIDER_COOLDOWNS: dict[str, float] = {}
_PROVIDER_COOLDOWN_REASONS: dict[str, str] = {}
_PROVIDER_ROTATION_CURSOR = 0


@dataclass
class SearchProviderReport:
    attempted_providers: list[str] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    quota_exhausted: list[str] = field(default_factory=list)
    provider_failures: list[str] = field(default_factory=list)
    retried_clues: int = 0
    remaining_clues: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted_providers": list(dict.fromkeys(self.attempted_providers)),
            "providers_used": list(dict.fromkeys(self.providers_used)),
            "quota_exhausted": list(dict.fromkeys(self.quota_exhausted)),
            "provider_failures": list(dict.fromkeys(self.provider_failures)),
            "retried_clues": self.retried_clues,
            "remaining_clues": self.remaining_clues,
        }


class _BaseSearchProvider:
    def __init__(self, name: str, *, max_results: int, timeout_seconds: float) -> None:
        self.name = name
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        raise NotImplementedError


class _AsyncHTTPProvider(_BaseSearchProvider):
    def __init__(self, name: str, *, max_results: int, timeout_seconds: float) -> None:
        super().__init__(name, max_results=max_results, timeout_seconds=timeout_seconds)
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class _TavilyProvider(_AsyncHTTPProvider):
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        max_results: int,
        timeout_seconds: float,
    ) -> None:
        super().__init__("tavily", max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.endpoint = endpoint

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        if self._client is None:
            raise TavilyPipelineError("Search provider must be opened before use.")

        payload = {
            "query": clue,
            "search_depth": "advanced",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": True,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = await self._client.post(self.endpoint, json=payload, headers=headers)
        if response.status_code >= 400:
            raise _build_provider_http_error("tavily", response)

        data = response.json()
        results = data.get("results", [])
        hits: list[SearchHit] = []
        for item in results:
            url = item.get("url", "")
            hits.append(
                SearchHit(
                    title=item.get("title", "") or clue,
                    url=url,
                    snippet=item.get("content", "") or "",
                    raw_content=item.get("raw_content", "") or "",
                    retrieval_score=float(item.get("score") or 0.0),
                    source=extract_source(url),
                    clue=clue,
                    branch=branch,
                    published_date=item.get("published_date", "") or "",
                    provider=self.name,
                )
            )
        return hits


class _SerperProvider(_AsyncHTTPProvider):
    def __init__(self, *, api_key: str, max_results: int, timeout_seconds: float) -> None:
        super().__init__("serper", max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.endpoint = "https://google.serper.dev/search"

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        if self._client is None:
            raise TavilyPipelineError("Search provider must be opened before use.")

        response = await self._client.post(
            self.endpoint,
            json={"q": clue, "num": self.max_results},
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise _build_provider_http_error(self.name, response)

        data = response.json()
        results = data.get("organic", [])[: self.max_results]
        return [
            _make_search_hit(
                provider=self.name,
                clue=clue,
                branch=branch,
                index=index,
                total=len(results),
                title=item.get("title", "") or clue,
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
                raw_content=item.get("snippet", "") or "",
                published_date=item.get("date", "") or "",
            )
            for index, item in enumerate(results)
            if item.get("link")
        ]


class _SerpAPIProvider(_AsyncHTTPProvider):
    def __init__(self, *, api_key: str, max_results: int, timeout_seconds: float) -> None:
        super().__init__("serpapi", max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.endpoint = "https://serpapi.com/search"

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        if self._client is None:
            raise TavilyPipelineError("Search provider must be opened before use.")

        response = await self._client.get(
            self.endpoint,
            params={
                "api_key": self.api_key,
                "q": clue,
                "num": self.max_results,
                "engine": "google",
            },
        )
        if response.status_code >= 400:
            raise _build_provider_http_error(self.name, response)

        data = response.json()
        results = data.get("organic_results", [])[: self.max_results]
        return [
            _make_search_hit(
                provider=self.name,
                clue=clue,
                branch=branch,
                index=index,
                total=len(results),
                title=item.get("title", "") or clue,
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
                raw_content=item.get("snippet", "") or "",
                published_date=item.get("date", "") or "",
            )
            for index, item in enumerate(results)
            if item.get("link")
        ]


class _GoogleCustomSearchProvider(_AsyncHTTPProvider):
    def __init__(
        self,
        *,
        api_key: str,
        search_engine_id: str,
        max_results: int,
        timeout_seconds: float,
    ) -> None:
        super().__init__("google_custom_search", max_results=max_results, timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.endpoint = "https://www.googleapis.com/customsearch/v1"

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        if self._client is None:
            raise TavilyPipelineError("Search provider must be opened before use.")

        response = await self._client.get(
            self.endpoint,
            params={
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": clue,
                "num": min(self.max_results, 10),
            },
        )
        if response.status_code >= 400:
            raise _build_provider_http_error(self.name, response)

        data = response.json()
        results = data.get("items", [])[: min(self.max_results, 10)]
        return [
            _make_search_hit(
                provider=self.name,
                clue=clue,
                branch=branch,
                index=index,
                total=len(results),
                title=item.get("title", "") or clue,
                url=item.get("link", ""),
                snippet=item.get("snippet", "") or "",
                raw_content=item.get("snippet", "") or "",
            )
            for index, item in enumerate(results)
            if item.get("link")
        ]


class _DuckDuckGoProvider(_BaseSearchProvider):
    def __init__(self, *, max_results: int, timeout_seconds: float) -> None:
        super().__init__("duckduckgo", max_results=max_results, timeout_seconds=timeout_seconds)

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        results = await asyncio.to_thread(self._search_sync, clue)
        return [
            _make_search_hit(
                provider=self.name,
                clue=clue,
                branch=branch,
                index=index,
                total=len(results),
                title=item.get("title", "") or clue,
                url=item.get("href", "") or item.get("url", ""),
                snippet=item.get("body", "") or item.get("snippet", "") or "",
                raw_content=item.get("body", "") or item.get("snippet", "") or "",
            )
            for index, item in enumerate(results)
            if item.get("href") or item.get("url")
        ]

    def _search_sync(self, clue: str) -> list[dict[str, Any]]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as error:
            raise SearchProviderError(
                self.name,
                "duckduckgo-search package not installed. Add it to enable DuckDuckGo fallback.",
                status_code=500,
            ) from error

        with DDGS() as ddgs:
            return list(ddgs.text(clue, max_results=self.max_results) or [])


class _SeleniumProvider(_BaseSearchProvider):
    def __init__(self, *, engine: str, max_results: int, timeout_seconds: float) -> None:
        super().__init__(f"selenium_{engine}", max_results=max_results, timeout_seconds=timeout_seconds)
        self.engine = engine

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        results = await asyncio.to_thread(self._search_sync, clue)
        return [
            _make_search_hit(
                provider=self.name,
                clue=clue,
                branch=branch,
                index=index,
                total=len(results),
                title=item.get("title", "") or clue,
                url=item.get("url", "") or item.get("link", ""),
                snippet=item.get("snippet", "") or item.get("description", "") or "",
                raw_content=item.get("content", "") or item.get("snippet", "") or item.get("description", "") or "",
            )
            for index, item in enumerate(results)
            if item.get("url") or item.get("link")
        ]

    def _search_sync(self, clue: str) -> list[dict[str, Any]]:
        try:
            from scrapers.selenium_scraper import SeleniumScraper
        except ImportError as error:
            raise SearchProviderError(
                self.name,
                "selenium package not installed. Add it to enable Selenium fallback.",
                status_code=500,
            ) from error

        scraper = SeleniumScraper(engine=self.engine)
        return scraper.search(clue, top_k=self.max_results)


class TavilySearchAgent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_TAVILY_ENDPOINT,
        max_results: int = DEFAULT_PER_CLUE_RESULTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        provider_order: list[str] | None = None,
        provider_strategy: str | None = None,
    ) -> None:
        self.config = load_search_config(
            tavily_api_key=api_key,
            tavily_endpoint=endpoint,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            provider_order=provider_order,
            provider_strategy=provider_strategy,
        )
        self.api_key = self.config.tavily_api_key
        self.endpoint = self.config.tavily_endpoint
        self.max_results = self.config.max_results
        self.timeout_seconds = self.config.timeout_seconds
        self.provider_order = list(self.config.provider_order)
        self.provider_strategy = self.config.provider_strategy
        self._providers: list[_BaseSearchProvider] = []
        self.last_search_report: dict[str, Any] = SearchProviderReport().as_dict()
        self.providers_used_overall: set[str] = set()

    async def __aenter__(self) -> "TavilySearchAgent":
        self._providers = _build_configured_providers(search_config=self.config)
        if not self._providers:
            raise TavilyPipelineError(
                "No search providers are configured. Set one of TAVILY_API_KEY, SERPER_API_KEY, "
                "SERPAPI_API_KEY, GOOGLE_API_KEY with GOOGLE_SEARCH_ENGINE_ID, or install DuckDuckGo/Selenium fallbacks."
            )

        await asyncio.gather(*(provider.open() for provider in self._providers))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._providers:
            await asyncio.gather(*(provider.close() for provider in self._providers), return_exceptions=True)
        self._providers = []

    async def search(self, clue: str, *, branch: str = "") -> list[SearchHit]:
        return await self.search_many([(clue, branch)])

    async def search_many(self, clues: list[tuple[str, str]]) -> list[SearchHit]:
        if not self._providers:
            raise TavilyPipelineError("TavilySearchAgent must be used as an async context manager.")

        remaining_clues = list(clues)
        hits: list[SearchHit] = []
        encountered_errors: list[BaseException] = []
        report = SearchProviderReport()

        candidate_providers = self._candidate_providers()
        if not candidate_providers:
            raise SearchProviderError(
                "search_pool",
                "All configured search providers are temporarily unavailable or on quota cooldown.",
                status_code=503,
            )

        for provider in candidate_providers:
            if not remaining_clues:
                break

            report.attempted_providers.append(provider.name)
            batch_results = await asyncio.gather(
                *(provider.search(clue, branch=branch) for clue, branch in remaining_clues),
                return_exceptions=True,
            )

            next_remaining: list[tuple[str, str]] = []
            provider_had_hits = False
            provider_had_error = False
            provider_quota_error = False

            for clue_branch, result in zip(remaining_clues, batch_results):
                if isinstance(result, BaseException):
                    provider_had_error = True
                    encountered_errors.append(result)
                    if _is_quota_error(result):
                        provider_quota_error = True
                    if _should_retry_with_next_provider(result):
                        next_remaining.append(clue_branch)
                    continue

                if result:
                    provider_had_hits = True
                    hits.extend(result)

            if provider_had_hits:
                report.providers_used.append(provider.name)
                self.providers_used_overall.add(provider.name)
            if provider_quota_error:
                report.quota_exhausted.append(provider.name)
                _mark_provider_cooldown(
                    provider.name,
                    self.config.quota_cooldown_seconds,
                    "quota_exhausted",
                )
            elif provider_had_error and not provider_had_hits:
                report.provider_failures.append(provider.name)
                _mark_provider_cooldown(
                    provider.name,
                    self.config.failure_cooldown_seconds,
                    "provider_failure",
                )

            if next_remaining:
                report.retried_clues += len(next_remaining)
            remaining_clues = next_remaining

            if hits and not remaining_clues and self.provider_strategy == "fallback":
                break

        report.remaining_clues = len(remaining_clues)
        self.last_search_report = report.as_dict()

        if hits:
            return hits

        if remaining_clues:
            provider_summary = ", ".join(
                report.attempted_providers or [provider.name for provider in candidate_providers]
            )
            quota_summary = ", ".join(report.quota_exhausted)
            detail = (
                f"All configured search providers failed for this request. Tried: {provider_summary}."
                + (f" Quota exhausted: {quota_summary}." if quota_summary else "")
            )
            raise SearchProviderError("search_pool", detail, status_code=503)

        if encountered_errors:
            first_error = encountered_errors[0]
            if isinstance(first_error, SearchProviderError):
                raise first_error
            raise TavilyPipelineError("All search provider requests failed.") from first_error
        return []

    def available_provider_names(self) -> list[str]:
        return [provider.name for provider in self._candidate_providers()]

    def _candidate_providers(self) -> list[_BaseSearchProvider]:
        healthy = [provider for provider in self._providers if not _provider_on_cooldown(provider.name)]
        if not healthy:
            return []

        if self.provider_strategy == "round_robin":
            return _rotate_providers(healthy)
        return healthy


def score_fetch_priority(query: str, hit: SearchHit) -> float:
    combined_text = f"{hit.title} {hit.snippet} {hit.raw_content[:500]}"
    combined_lower = combined_text.lower()
    intent = analyze_query_intent(query)
    score = 0.25
    score += 0.25 * max(0.0, min(1.0, hit.retrieval_score))
    score += 0.2 * lexical_relevance_score(query, combined_text)
    score += 0.15 * domain_reliability_score(hit.url)
    score += 0.15 * document_type_score(hit.url, hit.title, combined_text)
    if intent.prefers_encyclopedic_sources and any(
        hint in f"{hit.url.lower()} {combined_text.lower()}" for hint in ENCYCLOPEDIC_SOURCE_HINTS
    ):
        score += 0.08
    if intent.prefers_character_sources and any(hint in combined_text.lower() for hint in CHARACTER_QUERY_HINTS):
        score += 0.06
    if intent.targets_count and any(hint in combined_text.lower() for hint in ABILITY_QUERY_HINTS):
        score += 0.05
    if intent.prefers_event_sources:
        event_score = event_page_score(hit.url, hit.title, combined_text)
        event_specificity = specificity_overlap_score(query, combined_text)
        score += 0.16 * event_score
        if any(term in combined_lower for term in ("winner", "won", "pageant", "contest", "queen")):
            score += 0.08
        if is_generic_event_topic_page(hit.url, hit.title, combined_text):
            score -= 0.28
        if is_recipe_food_page(hit.url, hit.title, combined_text) and event_score < 0.55:
            score -= 0.32
        if intent.needs_event_discovery_hop and event_specificity < 0.1 and event_score < 0.55:
            score -= 0.24
        if (
            intent.needs_event_discovery_hop
            and any(term in combined_lower for term in ("beauty pageant", "pageant", "festival queen", "beauty queen"))
            and not any(
                term in combined_lower
                for term in (
                    "festival",
                    "celebration",
                    "anniversary",
                    "official",
                    "tourism",
                    "township",
                    "municipality",
                    "municipal",
                    "province",
                    "provincial",
                    "stew",
                    "condiment",
                    "ingredient",
                )
            )
        ):
            score -= 0.14
        if is_low_trust_social_page(hit.url, hit.title, combined_text) and event_score < 0.55:
            score -= 0.22
    if intent.is_open_domain_browsecomp:
        media_score = media_page_score(hit.url, hit.title, combined_text)
        score += 0.12 * specificity_overlap_score(query, combined_text)
        score += 0.14 * media_score
        if intent.prefers_event_sources:
            score += 0.12 * event_page_score(hit.url, hit.title, combined_text)
        if is_broad_overview_page(hit.url, hit.title):
            score -= 0.12
        if is_aggregate_listing_page(hit.url, hit.title, hit.snippet):
            score -= 0.18
        if is_wiki_meta_page(hit.url, hit.title, hit.snippet):
            score -= 0.22
        if is_non_english_wiki_page(hit.url):
            score -= 0.18
        if is_generic_media_topic_page(hit.url, hit.title, hit.snippet):
            score -= 0.22
        if is_generic_event_topic_page(hit.url, hit.title, hit.snippet):
            score -= 0.28
        if is_forum_discussion_page(hit.url, hit.title, hit.snippet):
            score -= 0.2
        if is_low_trust_social_page(hit.url, hit.title, combined_text):
            score -= 0.18
        if intent.is_media_query and media_score < 0.18:
            score -= 0.28
        if is_grounded_browsecomp_page(
            query,
            hit.url,
            hit.title,
            combined_text,
            require_media=intent.is_media_query,
        ):
            score += 0.16
        else:
            score -= 0.18
    if intent.answer_type == "year":
        structural_score, structural_matches, structural_contradictions = historical_year_structural_assessment(
            query,
            hit.url,
            hit.title,
            combined_text,
        )
        if is_specific_historical_year_page(query, hit.url, hit.title, combined_text):
            score += 0.28
        else:
            score -= 0.12
        score += 0.22 * structural_score
        if query_requires_bosnia_top_city(query):
            if any(term in combined_lower for term in BOSNIA_TOP_FOUR_CITY_TERMS):
                score += 0.22
            else:
                score -= 0.3
        if structural_contradictions > 0:
            score -= 0.45
        if (
            historical_year_has_structural_constraints(query)
            and structural_matches == 0
            and not historical_year_trusted_memorial_source(hit.url)
        ):
            score -= 0.32
        if any(term in combined_lower for term in ("victims", "massacre", "battle", "killed", "died", "detachment")):
            score += 0.08
        if any(term in combined_lower for term in ("born in", "awards", "prize", "competition", "salon", "triennale", "biennale")):
            score -= 0.18
    return max(0.0, min(1.0, score))


async def retrieve_documents_async(
    clues: list[str],
    *,
    api_key: str | None = None,
    endpoint: str = DEFAULT_TAVILY_ENDPOINT,
    max_results: int = DEFAULT_PER_CLUE_RESULTS,
) -> list[Document]:
    unique_clues = unique_preserve_order(clues)
    if not unique_clues:
        return []

    async with TavilySearchAgent(
        api_key=api_key,
        endpoint=endpoint,
        max_results=max_results,
    ) as agent:
        hits = await agent.search_many([(clue, f"branch-{index + 1}") for index, clue in enumerate(unique_clues)])

    return [
        Document(
            title=hit.title,
            url=hit.url,
            content=" ".join(part for part in (hit.title, hit.raw_content, hit.snippet) if part).strip(),
            source=hit.source,
            matched_clues=(hit.clue,),
            retrieval_score=hit.retrieval_score,
            raw_content=hit.raw_content,
            branch=hit.branch,
            content_type="search-hit",
        )
        for hit in hits
    ]


def retrieve_documents(clues: list[str]) -> list[Document]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        config = load_search_config()
        return asyncio.run(
            retrieve_documents_async(
                clues,
                api_key=config.tavily_api_key,
                endpoint=config.tavily_endpoint,
                max_results=DEFAULT_PER_CLUE_RESULTS,
            )
        )

    raise TavilyPipelineError(
        "retrieve_documents() cannot be called from an async context. Use retrieve_documents_async() instead."
    )


def _build_configured_providers(search_config: SearchConfig) -> list[_BaseSearchProvider]:
    providers: list[_BaseSearchProvider] = []

    for provider_name in search_config.provider_order:
        normalized_name = provider_name.strip().lower()
        if not normalized_name:
            continue
        provider: _BaseSearchProvider | None = None

        if normalized_name == "tavily" and search_config.tavily_api_key:
            provider = _TavilyProvider(
                api_key=search_config.tavily_api_key,
                endpoint=search_config.tavily_endpoint,
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )
        elif normalized_name == "serper" and search_config.serper_api_key:
            provider = _SerperProvider(
                api_key=search_config.serper_api_key,
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )
        elif normalized_name == "serpapi" and search_config.serpapi_api_key:
            provider = _SerpAPIProvider(
                api_key=search_config.serpapi_api_key,
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )
        elif (
            normalized_name == "google_custom_search"
            and search_config.google_api_key
            and search_config.google_search_engine_id
        ):
            provider = _GoogleCustomSearchProvider(
                api_key=search_config.google_api_key,
                search_engine_id=search_config.google_search_engine_id,
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )
        elif normalized_name == "duckduckgo" and _duckduckgo_available():
            provider = _DuckDuckGoProvider(
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )
        elif normalized_name.startswith("selenium") and _selenium_available():
            engine = normalized_name.partition("_")[2] or search_config.selenium_engine or DEFAULT_SELENIUM_ENGINE
            provider = _SeleniumProvider(
                engine=engine or "google",
                max_results=search_config.max_results,
                timeout_seconds=search_config.timeout_seconds,
            )

        if provider is not None:
            providers.append(provider)

    return providers


def _make_search_hit(
    *,
    provider: str,
    clue: str,
    branch: str,
    index: int,
    total: int,
    title: str,
    url: str,
    snippet: str,
    raw_content: str,
    published_date: str = "",
) -> SearchHit:
    return SearchHit(
        title=title,
        url=url,
        snippet=snippet,
        raw_content=raw_content,
        retrieval_score=_rank_to_score(index, total),
        source=extract_source(url),
        clue=clue,
        branch=branch,
        published_date=published_date,
        provider=provider,
    )


def _rank_to_score(index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(max(0.1, 1.0 - (index / (total + 1))), 4)


def _build_provider_http_error(provider: str, response: httpx.Response) -> SearchProviderError:
    detail = _extract_error_detail(response)
    if _error_looks_like_quota_issue(response.status_code, detail):
        return SearchProviderQuotaError(provider, detail, status_code=response.status_code)
    return SearchProviderError(provider, detail, status_code=response.status_code)


def _extract_error_detail(response: httpx.Response) -> str:
    text = response.text
    try:
        data = response.json()
    except Exception:
        return text
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("error") or detail)
        if detail:
            return str(detail)
        error = data.get("error")
        if error:
            return str(error)
    return text


def _error_looks_like_quota_issue(status_code: int, detail: str) -> bool:
    detail_lower = (detail or "").lower()
    return status_code in {402, 429} or any(
        phrase in detail_lower
        for phrase in (
            "usage limit",
            "plan's set usage limit",
            "quota",
            "credit",
            "credits",
            "rate limit",
            "payment required",
            "insufficient balance",
            "exceeded your plan",
        )
    )


def _should_retry_with_next_provider(error: BaseException) -> bool:
    return isinstance(
        error,
        (SearchProviderError, SearchProviderQuotaError, httpx.HTTPError, RuntimeError, ImportError),
    )


def _is_quota_error(error: BaseException) -> bool:
    if isinstance(error, SearchProviderQuotaError):
        return True
    if isinstance(error, SearchProviderError):
        return _error_looks_like_quota_issue(error.status_code, error.detail)
    return False


def _get_provider_order() -> list[str]:
    return load_search_config().provider_order


def _get_provider_strategy() -> str:
    return load_search_config().provider_strategy


def _get_provider_quota_cooldown_seconds() -> int:
    return load_search_config().quota_cooldown_seconds


def _get_provider_failure_cooldown_seconds() -> int:
    return load_search_config().failure_cooldown_seconds


def _provider_on_cooldown(provider_name: str) -> bool:
    cooldown_until = _PROVIDER_COOLDOWNS.get(provider_name, 0.0)
    if cooldown_until <= time.monotonic():
        _PROVIDER_COOLDOWNS.pop(provider_name, None)
        _PROVIDER_COOLDOWN_REASONS.pop(provider_name, None)
        return False
    return True


def _mark_provider_cooldown(provider_name: str, cooldown_seconds: int, reason: str) -> None:
    _PROVIDER_COOLDOWNS[provider_name] = time.monotonic() + cooldown_seconds
    _PROVIDER_COOLDOWN_REASONS[provider_name] = reason
    logger.warning(
        "Search provider '%s' placed on cooldown for %ss due to %s.",
        provider_name,
        cooldown_seconds,
        reason,
    )


def _rotate_providers(providers: list[_BaseSearchProvider]) -> list[_BaseSearchProvider]:
    global _PROVIDER_ROTATION_CURSOR
    if not providers:
        return []
    offset = _PROVIDER_ROTATION_CURSOR % len(providers)
    _PROVIDER_ROTATION_CURSOR += 1
    return providers[offset:] + providers[:offset]


def _duckduckgo_available() -> bool:
    try:
        import duckduckgo_search  # noqa: F401
    except ImportError:
        return False
    return True


def _selenium_available() -> bool:
    try:
        import selenium  # noqa: F401
    except ImportError:
        return False
    return True
