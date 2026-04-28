# Next-Gen Search Engine

This project contains a FastAPI search backend and a frontend function package for building chatbot search flows. The backend exposes live search, ranking, and TextGrad refinement endpoints. The frontend folder contains the function definitions and playground response handlers that call those backend endpoints from a chatbot platform.

## Repository Structure

```text
.
|-- backend/
|   |-- main.py
|   |-- pyproject.toml
|   |-- .env.template
|   |-- agent/
|   |-- config/
|   |-- extraction/
|   |-- memory/
|   |-- parsing/
|   |-- planner/
|   |-- ranking/
|   |-- reflection/
|   |-- retrieval/
|   |-- scrapers/
|   |-- search/
|   |-- tests/
|   `-- utils/
`-- frontend/
    |-- functions/
    |   |-- refineSearchQuery/
    |   |-- webSearchBasic/
    |   `-- webSearchDetailed/
    `-- playground/
        |-- LLM/
        |-- LLMRag/
        `-- LLMTag/
```

## Backend

The backend is a Python FastAPI service. It provides the local API used by the search functions in `frontend/functions/` and the RAG playground in `frontend/playground/LLMRag/`.

Start with the backend README for setup, environment variables, API routes, and request examples:

```text
backend/README.md
```

Main local endpoints:

- `GET /health` returns the backend health status.
- `POST /tavily` runs the live search pipeline.
- `POST /pagerank` ranks provided documents with PageRank and query relevance.
- `POST /textgrad/refine-query` improves a search query over refinement rounds.
- `POST /textgrad/refine-answer` improves an answer using supplied context.
- `POST /textgrad/refine-plan` improves a tool or execution plan.
- `POST /textgrad/optimize-prompt` optimizes a TextGrad system prompt from evaluation inputs.

## Frontend Function Package

The frontend folder is not a standalone web application. It is a structured package of chatbot platform functions, setup notes, and playground handlers.

Start with the frontend README for folder purpose, setup order, function dependencies, and runtime assumptions:

```text
frontend/README.md
```

The frontend package includes three reusable functions:

- `frontend/functions/refineSearchQuery/` normalizes a raw query and returns a structured search query.
- `frontend/functions/webSearchBasic/` performs live search and returns a concise summary.
- `frontend/functions/webSearchDetailed/` performs live search, optional query refinement, PageRank ranking, and detailed synthesis.

It also includes three playground configurations:

- `frontend/playground/LLM/` streams a direct chat response.
- `frontend/playground/LLMRag/` retrieves context from the local backend before streaming the chat response.
- `frontend/playground/LLMTag/` exposes the basic and detailed web search functions as tools.

## Quick Start

Install and run the backend first:

```powershell
cd backend
pip install -e .
copy .env.template .env
python main.py
```

The API starts on:

```text
http://localhost:8000
```

Open the interactive API docs at:

```text
http://localhost:8000/docs
```

After the backend is running, use the setup files inside `frontend/functions/` and `frontend/playground/` to recreate the required functions and chat dependencies in the target chatbot platform.

## Runtime Dependencies Between Folders

The frontend function code expects the backend to be available at `http://localhost:8000`.

The main runtime dependencies are:

- `frontend/playground/LLMRag/generateResponse.js` calls `POST /tavily`.
- `frontend/functions/webSearchBasic/preprocess.js` calls `POST /tavily` and can call `POST /textgrad/refine-query`.
- `frontend/functions/webSearchDetailed/preprocess.js` calls `POST /tavily`, `POST /pagerank`, and can call `POST /textgrad/refine-query`.
- `frontend/playground/LLMTag/generateResponse.js` expects `environment.llmFunctions.webSearchBasic` and `environment.llmFunctions.webSearchDetailed`.
- Both web search functions expect the query-refinement function alias to be `refineSearchQuery`.

## Documentation Map

- `backend/README.md` explains backend setup, configuration, routes, and request examples.
- `frontend/README.md` explains the frontend function package and platform setup order.
- Each function and playground folder contains its own README with copy-ready setup notes for that specific component.
