# Frontend Function Package

This folder contains the chatbot platform function source code, setup notes, and playground response handlers for the search workflows. It is organized as a deployment package for platform functions rather than a standalone browser application.

The backend API must be running locally before the search and RAG flows can be tested.

```text
http://localhost:8000
```

## Folder Structure

```text
frontend/
|-- functions/
|   |-- refineSearchQuery/
|   |   |-- README.md
|   |   |-- preprocess.js
|   |   |-- postprocess.js
|   |   |-- LLM/
|   |   `-- information/
|   |-- webSearchBasic/
|   |   |-- README.md
|   |   |-- preprocess.js
|   |   |-- postprocess.js
|   |   |-- LLM/
|   |   |-- dependencies/
|   |   `-- information/
|   `-- webSearchDetailed/
|       |-- README.md
|       |-- preprocess.js
|       |-- postprocess.js
|       |-- LLM/
|       |-- dependencies/
|       `-- information/
`-- playground/
    |-- LLM/
    |   |-- README.md
    |   |-- generateResponse.js
    |   |-- dependencies/
    |   `-- information/
    |-- LLMRag/
    |   |-- README.md
    |   |-- generateResponse.js
    |   |-- dependencies/
    |   `-- information/
    `-- LLMTag/
        |-- README.md
        |-- generateResponse.js
        |-- dependencies/
        `-- information/
```

## Function Folders

`functions/refineSearchQuery/` converts a raw or conversational query into a structured search query. It returns `refined_query`, `intent`, and `confidence`.

`functions/webSearchBasic/` performs live search through the local backend, prepares a compact search context, and returns a concise `summary`.

`functions/webSearchDetailed/` performs a deeper search flow. It refines the query, can call TextGrad query optimization, retrieves live search results, ranks documents through PageRank, and returns `topKSummary`.

Each function folder contains:

- `README.md` with the component-specific setup guide.
- `information/information.md` with Basic Information tab values.
- `LLM/LLM.md` with prompt, argument, model, and output setup where the function uses an LLM step.
- `dependencies/` with dependency setup values where the function calls another function.
- `preprocess.js` and `postprocess.js` with the runtime source code.

## Playground Folders

`playground/LLM/` streams a direct chat response through the `chat` dependency.

`playground/LLMRag/` calls the local backend at `POST http://localhost:8000/tavily`, passes the retrieved context into the `chat` dependency, and streams the grounded response.

`playground/LLMTag/` exposes `webSearchBasic` and `webSearchDetailed` as tools, then streams the final response through the `chat` dependency.

Each playground folder contains:

- `README.md` with the component-specific setup guide.
- `generateResponse.js` with the response handler called by the platform.
- `information/information.md` with Basic Information tab values.
- `dependencies/chat.md` with the required chat-completion dependency setup.

## Setup Order

1. Start the backend from `backend/README.md` and confirm that `GET http://localhost:8000/health` returns a healthy response.
2. Create the `refineSearchQuery` function from `functions/refineSearchQuery/`.
3. Create `webSearchBasic` and `webSearchDetailed` from their folders.
4. Keep the function dependency alias as `refineSearchQuery` for both web search functions.
5. Create the chat-completion dependency named `chat` for the playground configuration being used.
6. Add the required function aliases for `playground/LLMTag/`: `webSearchBasic` and `webSearchDetailed`.
7. Test the selected playground flow from the chatbot interface.

## Required Runtime Names

The JavaScript source files expect these runtime references to exist:

```text
environment.llmChatCompletions.chat
environment.llmFunctions.refineSearchQuery
environment.llmFunctions.webSearchBasic
environment.llmFunctions.webSearchDetailed
```

The `chat` dependency name must stay as `chat`.

The `refineSearchQuery` dependency alias must stay as `refineSearchQuery`.

For `playground/LLMTag/`, the tool aliases must stay as `webSearchBasic` and `webSearchDetailed`.

## Backend Endpoints Used By The Frontend Code

`POST /tavily`
Used by `playground/LLMRag/`, `webSearchBasic`, and `webSearchDetailed` for live search.

`POST /pagerank`
Used by `webSearchDetailed` to rank retrieved documents.

`POST /textgrad/refine-query`
Used by `webSearchBasic` and `webSearchDetailed` when query confidence is low and TextGrad refinement is available.

## Important Argument Names

`playground/LLM` passes:

```text
userMessage
```

`playground/LLMRag` passes:

```text
userMessage
retrievedContext
```

`playground/LLMTag` passes:

```text
userMessage
```

Keep these argument names unchanged because the runtime JavaScript sends those exact keys to the chat dependency.

## Component Guides

Use the README inside the relevant component folder for detailed setup:

```text
functions/refineSearchQuery/README.md
functions/webSearchBasic/README.md
functions/webSearchDetailed/README.md
playground/LLM/README.md
playground/LLMRag/README.md
playground/LLMTag/README.md
```
