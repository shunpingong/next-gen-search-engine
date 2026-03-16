from __future__ import annotations

from agent.models import Document, SearchHit
from utils.text_utils import canonicalize_url, normalize_whitespace


class AgentMemory:
    def __init__(self, query: str) -> None:
        self.query = query
        self.visited_urls: set[str] = set()
        self.queued_urls: set[str] = set()
        self.queries_used: set[str] = set()
        self.task_keys: set[str] = set()
        self.search_hits_seen: dict[str, SearchHit] = {}
        self.documents_seen: dict[str, Document] = {}
        self.evidence_collected: list[str] = []

    def remember_query(self, query: str) -> bool:
        normalized = normalize_whitespace(query).lower()
        if not normalized or normalized in self.queries_used:
            return False
        self.queries_used.add(normalized)
        return True

    def remember_task(self, task_key: str) -> bool:
        if task_key in self.task_keys:
            return False
        self.task_keys.add(task_key)
        return True

    def remember_search_hit(self, hit: SearchHit) -> bool:
        key = canonicalize_url(hit.url) or hit.url
        if not key or key in self.search_hits_seen:
            return False
        self.search_hits_seen[key] = hit
        return True

    def queue_url(self, url: str) -> bool:
        key = canonicalize_url(url) or url
        if not key or key in self.queued_urls or key in self.visited_urls:
            return False
        self.queued_urls.add(key)
        return True

    def remember_document(self, document: Document) -> bool:
        key = canonicalize_url(document.url) or document.url or document.title
        if not key:
            return False
        self.visited_urls.add(key)
        self.queued_urls.discard(key)
        existing = self.documents_seen.get(key)
        if existing is None:
            self.documents_seen[key] = document
            return True

        merged_clues = list(existing.matched_clues)
        for clue in document.matched_clues:
            if clue not in merged_clues:
                merged_clues.append(clue)
        existing.matched_clues = tuple(merged_clues)
        existing.retrieval_score = max(existing.retrieval_score, document.retrieval_score)
        existing.rank_score = max(existing.rank_score, document.rank_score)
        if len(document.content) > len(existing.content):
            existing.content = document.content
            existing.raw_content = document.raw_content or existing.raw_content
            existing.metadata.update(document.metadata)
            existing.content_type = document.content_type
            existing.fetch_error = document.fetch_error
            existing.fetched = document.fetched or existing.fetched
            if document.acknowledgement_section:
                existing.acknowledgement_section = document.acknowledgement_section
        if document.entities:
            existing.entities = tuple(dict.fromkeys(existing.entities + document.entities))
        return False

    def add_evidence(self, evidence: str) -> None:
        cleaned = normalize_whitespace(evidence)
        if cleaned:
            self.evidence_collected.append(cleaned)

    def documents(self) -> list[Document]:
        return list(self.documents_seen.values())
