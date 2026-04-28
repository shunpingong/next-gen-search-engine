# Dependency: refineSearchQuery

Use this file to recreate the Dependencies tab entry for the `webSearchBasic` function.

## Dependency Type

```text
LLM Function
```

## Dependency Details

| Field    | Value               |
| -------- | ------------------- |
| Alias    | `refineSearchQuery` |
| Function | `refineSearchQuery` |

Description:

```text
Transforms a raw or conversational user query into a structured, search-optimized representation suitable for web search, vector retrieval, RAG, or TAG pipelines. The function normalizes user input, infers search intent, removes conversational noise, and outputs a refined keyword query along with lightweight metadata useful for downstream retrieval and ranking.
```

## Source Code Usage

`preprocess.js` calls this dependency with this runtime reference:

```js
environment.llmFunctions.refineSearchQuery(
  { query: question, context, domain },
  environment,
);
```

The dependency alias must stay exactly as `refineSearchQuery`.
