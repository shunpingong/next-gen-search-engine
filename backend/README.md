# Backend

This folder contains the FastAPI service used by the search functions in `frontend/`. It provides live web search, document ranking, and TextGrad refinement endpoints for the chatbot search workflows.

## Folder Structure

```text
backend/
|-- main.py
|-- pyproject.toml
|-- .env.template
|-- tavily_pipeline.py
|-- agent/
|-- config/
|-- extraction/
|-- memory/
|-- parsing/
|-- planner/
|-- ranking/
|-- reflection/
|-- retrieval/
|-- scrapers/
|-- search/
|-- tests/
`-- utils/
```

## Main Modules

`main.py` defines the FastAPI application, request models, route handlers, PageRank ranking endpoint, and TextGrad endpoints.

`tavily_pipeline.py` coordinates the main live-search pipeline used by `POST /tavily`.

`config/` contains environment-backed configuration loaders for search providers, planning, retrieval, ranking, reflection, extraction, and TextGrad settings.

`planner/`, `search/`, `retrieval/`, `ranking/`, `extraction/`, and `reflection/` contain the main search pipeline stages.

`scrapers/` contains provider integrations and fallbacks for Tavily, Serper, SerpAPI, Google Custom Search, DuckDuckGo, Selenium, and shared scraper utilities.

`tests/` contains backend tests for the Tavily pipeline and PageRank behavior.

## Requirements

- Python 3.10 or newer
- A configured search provider key for live search, with `TAVILY_API_KEY` as the standard local setup
- `OPENAI_API_KEY` for TextGrad endpoints and the OpenAI-backed Tavily decomposition, answer extraction, and follow-up steps

## Install

```powershell
cd backend
pip install -e .
```

For test and formatting tools:

```powershell
cd backend
pip install -e ".[dev]"
```

## Environment Variables

Create a local environment file from the template:

```powershell
cd backend
copy .env.template .env
```

Minimal setup for the primary Tavily flow:

```env
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

If OpenAI-backed Tavily decomposition and follow-up should be disabled, keep `TAVILY_API_KEY` and set:

```env
TAVILY_USE_LLM_DECOMPOSITION=0
TAVILY_USE_LLM_FOLLOW_UP=0
```

The template also includes optional provider keys and model settings:

- `GOOGLE_API_KEY`
- `GOOGLE_SEARCH_ENGINE_ID`
- `SERPER_API_KEY`
- `SERPAPI_API_KEY`
- `FIRECRAWL_API_KEY`
- `OPENROUTER_API_KEY`
- `TAVILY_DECOMPOSITION_MODEL`
- `TAVILY_FOLLOW_UP_MODEL`

## Run

```powershell
cd backend
python main.py
```

The API starts on:

```text
http://localhost:8000
```

Useful local URLs:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## API Routes

`GET /`
Returns basic API information and the main available routes.

`GET /health`
Returns the current backend health status.

`POST /tavily`
Runs the live search pipeline. The route accepts a query, retrieves web results, gathers evidence, ranks sources, and returns an answer with supporting source data.

`POST /pagerank`
Ranks a provided list of documents with document similarity and optional query relevance. The response includes score maps and, when `top_k` is provided, ranked documents.

`POST /textgrad/refine-query`
Improves a search query over multiple refinement rounds.

`POST /textgrad/refine-answer`
Improves an answer using a question and supporting context.

`POST /textgrad/refine-plan`
Improves a tool or execution plan using the user query and execution feedback.

`POST /textgrad/optimize-prompt`
Optimizes the shared TextGrad system prompt from evaluation inputs and desired behavior.

All `/textgrad/*` routes require `OPENAI_API_KEY`.

## Request Examples

### `POST /tavily`

```json
{
  "query": "What are the strongest use cases for retrieval-augmented generation?",
  "max_results": 5
}
```

### `POST /pagerank`

```json
{
  "documents": [
    {
      "id": "doc-1",
      "title": "Example document",
      "url": "https://example.com",
      "content": "Document text goes here",
      "score": 0.82
    }
  ],
  "query": "retrieval augmented generation use cases",
  "top_k": 3
}
```

### `POST /textgrad/refine-query`

```json
{
  "query": "cheap gaming mouse near me",
  "max_iterations": 3
}
```

### `POST /textgrad/refine-answer`

```json
{
  "question": "What does the context say about RAG accuracy?",
  "context": "Paste the supporting context here",
  "initial_answer": "Optional starting answer",
  "max_iterations": 3
}
```

### `POST /textgrad/refine-plan`

```json
{
  "user_query": "Find recent benchmarks for open-source rerankers",
  "execution_feedback": "The first plan used too many vague search steps",
  "max_iterations": 3
}
```

### `POST /textgrad/optimize-prompt`

```json
{
  "eval_inputs": [
    "Summarize the query intent",
    "Rewrite the answer using only the provided context"
  ],
  "desired_behavior": "Accurate, concise, and grounded answers",
  "max_iterations": 10
}
```

## Tests

```powershell
cd backend
pytest
```

The backend tests cover the Tavily pipeline and PageRank ranking behavior.

## Frontend Integration

The frontend function package calls this backend at `http://localhost:8000`.

Relevant frontend callers:

- `frontend/playground/LLMRag/generateResponse.js`
- `frontend/functions/webSearchBasic/preprocess.js`
- `frontend/functions/webSearchDetailed/preprocess.js`

Keep the backend running before testing those frontend workflows.
