# webSearchBasic Function Guide

This folder contains the source code and setup notes for the basic web search LLM function.

## Files

- `information/information.md` contains the copy-ready setup values for the Basic Information tab.
- `dependencies/refineSearchQuery.md` contains the copy-ready setup values for the Dependencies tab.
- `LLM/LLM.md` contains the copy-ready setup values for the LLM tab.
- `preprocess.js` prepares the user's string input for the LLM by refining the query, calling Tavily, normalizing basic search results, and building the search-result context.
- `postprocess.js` cleans the LLM output and returns the final `summary` object.

## Runtime Flow

1. The function receives a string input query.
2. `preprocess.js` infers the query context and domain.
3. The query is refined through `environment.llmFunctions.refineSearchQuery`.
4. Low-confidence refined queries are optimized through `POST http://localhost:8000/textgrad/refine-query`.
5. The final query is sent to `POST http://localhost:8000/tavily`.
6. Tavily answer, source, and generic result documents are normalized into `searchResult`.
7. The search context is passed into the function's LLM step.
8. `postprocess.js` returns the final object with `summary`.

## Input And Output

The function input and output schemas are documented in:

```text
functions/webSearchBasic/information/information.md
```

## LLM Setup

The LLM prompt, arguments, parameters, and external-tool status are documented in:

```text
functions/webSearchBasic/LLM/LLM.md
```

## Dependencies Setup

The LLM function dependency setup is documented in:

```text
functions/webSearchBasic/dependencies/refineSearchQuery.md
```

## Notes

- Input type is a single string query.
- Output type is an object with required `summary` and optional `error`.
- The LLM function dependency alias must stay as `refineSearchQuery`.
- The function depends on the local Tavily backend endpoint and optional TextGrad query optimization.
