import os
from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

fake_textgrad = types.ModuleType("textgrad")


class _FakeVariable:
    def __init__(self, value, requires_grad=False, role_description=""):
        self.value = value
        self.requires_grad = requires_grad
        self.role_description = role_description

    def backward(self):
        return None


class _FakeOptimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        return None

    def step(self):
        return None


fake_textgrad.Variable = _FakeVariable
fake_textgrad.TextualGradientDescent = _FakeOptimizer
fake_textgrad.BlackboxLLM = lambda *args, **kwargs: None
fake_textgrad.TextLoss = lambda *args, **kwargs: None
fake_textgrad.sum = lambda items: items[0] if items else _FakeVariable("")
fake_textgrad.set_backward_engine = lambda *args, **kwargs: None

sys.modules["textgrad"] = fake_textgrad

os.environ["TAVILY_USE_LLM_DECOMPOSITION"] = "0"
os.environ["TAVILY_USE_LLM_FOLLOW_UP"] = "0"

import main

client = TestClient(main.app)


def test_pagerank_extracts_refined_query_from_string_payload():
    query = '{"refined_query": "monument or memorial constructed before 1970 in former Yugoslavia"}'
    response = client.post(
        "/pagerank",
        json={
            "documents": [
                {
                    "id": "spomenik",
                    "title": "Memorial complex in former Yugoslavia",
                    "content": (
                        "This monument was constructed in 1968 in former Yugoslavia and "
                        "commemorates civilians killed during the war."
                    ),
                }
            ],
            "query": query,
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["effective_query"] == "monument or memorial constructed before 1970 in former Yugoslavia"
    assert payload["scores"]["spomenik"] > 0.0
    assert payload["relevance_scores"]["spomenik"] > 0.0
    assert payload["ranked_documents"][0]["id"] == "spomenik"


def test_pagerank_ranks_documents_by_content_against_query():
    response = client.post(
        "/pagerank",
        json={
            "documents": [
                {
                    "id": "relevant-doc",
                    "title": "Yugoslav memorial site",
                    "content": (
                        "A memorial in former Yugoslavia was built in 1967. "
                        "The monument honors local victims and remains a historic landmark."
                    ),
                },
                {
                    "id": "irrelevant-doc",
                    "title": "Football match archive",
                    "content": (
                        "This page lists league tables, team formations, and match results "
                        "from a regional football tournament."
                    ),
                },
            ],
            "query": {
                "refined_query": "monument or memorial constructed before 1970 in former Yugoslavia"
            },
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["ranked_documents"][0]["id"] == "relevant-doc"
    assert payload["scores"]["relevant-doc"] > payload["scores"]["irrelevant-doc"]
    assert payload["relevance_scores"]["relevant-doc"] > payload["relevance_scores"]["irrelevant-doc"]
