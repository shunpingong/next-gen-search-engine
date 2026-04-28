# Basic Information Tab: webSearchDetailed

Use this file to recreate the Basic Information tab for the `webSearchDetailed` LLM function.

## Basic Information

| Field | Value |
| --- | --- |
| Name | `webSearchDetailed` |
| Description | This function performs a comprehensive, multi-stage web search optimized for accuracy and relevance. It refines the user's query using a language model, retrieves live search results via the Tavily API, normalizes and aggregates the retrieved documents, and applies an external PageRank-based ranking to prioritize the most authoritative sources. The ranked search content is then passed to the language model, which synthesizes a detailed, evidence-based response grounded strictly in the retrieved information. |

## Input Format

JSON Schema:

```json
{
  "type": "string"
}
```

Typedef:

```js
/**
 * The input of the function.
 * @typedef {string} Input
 */
```

## Output Format

JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "topKSummary": {
      "type": "string"
    },
    "error": {
      "type": ["string", "null"]
    }
  },
  "required": ["topKSummary"]
}
```

Typedef:

```js
/**
 * The output of the function.
 * @typedef {object} Output
 * @property {string} topKSummary
 * @property {?string} [error]
 */
```
