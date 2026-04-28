# Basic Information Tab: webSearchBasic

Use this file to recreate the Basic Information tab for the `webSearchBasic` LLM function.

## Basic Information

| Field       | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Name        | `webSearchBasic`                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Description | This function performs a live, LLM-enhanced web search using the Tavily API. It first refines the user's initial query through a language model to ensure clarity and relevance. The refined query is then sent to the Tavily API to retrieve real-world search results. Finally, the raw search data is passed back into the LLM, which summarizes and synthesizes the key information before returning a concise, human-readable answer to the orchestrator. |

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
    "summary": {
      "type": "string"
    },
    "error": {
      "type": ["string", "null"]
    }
  },
  "required": ["summary"]
}
```

Typedef:

```js
/**
 * The output of the function.
 * @typedef {object} Output
 * @property {string} summary
 * @property {?string} [error]
 */
```
