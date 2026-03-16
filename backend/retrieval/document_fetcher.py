from __future__ import annotations

import asyncio
import time

import httpx

from agent.models import Document, SearchHit
from config.retrieval_config import DEFAULT_FETCH_TIMEOUT_SECONDS, load_document_fetch_config
from parsing.html_parser import parse_html_document
from parsing.pdf_parser import parse_pdf_document
from utils.text_utils import is_low_trust_social_page, normalize_document_text, normalize_whitespace


class DocumentFetcher:
    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        fetch_config = load_document_fetch_config()
        self.timeout_seconds = timeout_seconds or fetch_config.timeout_seconds
        self.user_agent = fetch_config.user_agent
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, Document] = {}

    async def __aenter__(self) -> "DocumentFetcher":
        self._client = httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_hit(self, hit: SearchHit) -> tuple[Document, float, float]:
        if self._client is None:
            raise RuntimeError("DocumentFetcher must be used as an async context manager.")

        cache_key = hit.url
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached, 0.0, 0.0

        download_started = time.perf_counter()
        fallback_text = normalize_whitespace(" ".join(part for part in (hit.raw_content, hit.snippet, hit.title) if part))
        if is_low_trust_social_page(hit.url, hit.title, hit.snippet):
            document = Document(
                title=hit.title,
                url=hit.url,
                content=fallback_text,
                source=hit.source,
                matched_clues=(hit.clue,),
                retrieval_score=hit.retrieval_score,
                raw_content=hit.raw_content,
                content_type="search-hit",
                branch=hit.branch,
                fetched=False,
                fetch_error="Skipped low-value social page fetch.",
                metadata={"published_date": hit.published_date},
            )
            self._cache[cache_key] = document
            return document, 0.0, 0.0
        try:
            response = await self._client.get(hit.url)
            download_time = time.perf_counter() - download_started
            response.raise_for_status()
            final_url = str(response.url)
            content_type = response.headers.get("content-type", "").lower()
            parse_started = time.perf_counter()
            if "pdf" in content_type or final_url.lower().endswith(".pdf"):
                text, metadata = await asyncio.to_thread(parse_pdf_document, response.content)
                document_type = "pdf"
            else:
                text, metadata = await asyncio.to_thread(parse_html_document, response.text, final_url)
                document_type = "html"
            parse_time = time.perf_counter() - parse_started

            content = normalize_document_text(text) if text else fallback_text
            document = Document(
                title=str(metadata.get("title") or hit.title),
                url=final_url,
                content=content,
                source=hit.source,
                matched_clues=(hit.clue,),
                retrieval_score=hit.retrieval_score,
                raw_content=hit.raw_content,
                content_type=document_type,
                branch=hit.branch,
                fetched=True,
                metadata={"published_date": hit.published_date, **metadata},
            )
        except Exception as error:
            download_time = time.perf_counter() - download_started
            parse_time = 0.0
            document = Document(
                title=hit.title,
                url=hit.url,
                content=fallback_text,
                source=hit.source,
                matched_clues=(hit.clue,),
                retrieval_score=hit.retrieval_score,
                raw_content=hit.raw_content,
                content_type="search-hit",
                branch=hit.branch,
                fetched=False,
                fetch_error=str(error),
                metadata={"published_date": hit.published_date},
            )

        self._cache[cache_key] = document
        return document, download_time, parse_time

    async def fetch_many(self, hits: list[SearchHit]) -> tuple[list[Document], float, float]:
        tasks = [self.fetch_hit(hit) for hit in hits]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: list[Document] = []
        total_download_time = 0.0
        total_parse_time = 0.0
        for hit, result in zip(hits, results):
            if isinstance(result, BaseException):
                documents.append(
                    Document(
                        title=hit.title,
                        url=hit.url,
                        content=normalize_whitespace(" ".join(part for part in (hit.raw_content, hit.snippet, hit.title) if part)),
                        source=hit.source,
                        matched_clues=(hit.clue,),
                        retrieval_score=hit.retrieval_score,
                        raw_content=hit.raw_content,
                        content_type="search-hit",
                        branch=hit.branch,
                        fetched=False,
                        fetch_error=str(result),
                        metadata={"published_date": hit.published_date},
                    )
                )
                continue
            document, download_time, parse_time = result
            documents.append(document)
            total_download_time += download_time
            total_parse_time += parse_time
        return documents, total_download_time, total_parse_time
