from __future__ import annotations

import networkx as nx

from agent.models import Document


class EvidenceGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_document(self, document: Document) -> None:
        self.graph.add_node(
            f"doc:{document.url}",
            kind="document",
            title=document.title,
            url=document.url,
            source=document.source,
        )

    def add_entity_mention(self, document: Document, entity: str, evidence: str) -> None:
        doc_node = f"doc:{document.url}"
        entity_node = f"entity:{entity}"
        self.graph.add_node(entity_node, kind="entity", value=entity)
        self.graph.add_edge(doc_node, entity_node, relation="mentions", evidence=evidence)

    def add_fact_support(self, document: Document, fact: str, evidence: str) -> None:
        doc_node = f"doc:{document.url}"
        fact_node = f"fact:{fact}"
        self.graph.add_node(fact_node, kind="fact", value=fact)
        self.graph.add_edge(doc_node, fact_node, relation="supports", evidence=evidence)

    def summary(self) -> dict[str, int]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }
