# Basic Information Tab: refineSearchQuery

Use this file to recreate the Basic Information tab for the `refineSearchQuery` LLM function.

## Basic Information

| Field       | Value                                                                                                                                                                                                                                                                                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Name        | `refineSearchQuery`                                                                                                                                                                                                                                                                                                                                                         |
| Description | Transforms a raw or conversational user query into a structured, search-optimized representation suitable for web search, vector retrieval, RAG, or TAG pipelines. The function normalizes user input, infers search intent, removes conversational noise, and outputs a refined keyword query along with lightweight metadata useful for downstream retrieval and ranking. |

## Input Format

JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Raw user query, possibly conversational or vague."
    },
    "context": {
      "type": "string",
      "description": "Optional contextual information such as task goal, domain, or prior conversation."
    },
    "domain": {
      "type": "string",
      "description": "Optional domain hint (e.g. 'software', 'research', 'travel', 'finance')."
    }
  },
  "required": ["query"]
}
```

Typedef:

```js
/**
 * The input of the function.
 * @typedef {object} Input
 * @property {string} query Raw user query, possibly conversational or vague.
 * @property {string} [context] Optional contextual information such as task goal, domain, or prior conversation.
 * @property {string} [domain] Optional domain hint (e.g. 'software', 'research', 'travel', 'finance').
 */
```

## Output Format

JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "refined_query": {
      "type": "string",
      "description": "Concise, search-engine-optimized query."
    },
    "intent": {
      "type": "string",
      "enum": [
        "factual",
        "comparison",
        "how-to",
        "troubleshooting",
        "exploratory"
      ],
      "description": "Inferred primary search intent."
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score (0-1) indicating how confident the system is about the refinement."
    }
  },
  "required": ["refined_query", "intent", "confidence"]
}
```

Typedef:

```js
/**
 * The output of the function.
 * @typedef {object} Output
 * @property {string} refined_query Concise, search-engine-optimized query.
 * @property {"factual"|"comparison"|"how-to"|"troubleshooting"|"exploratory"} intent Inferred primary search intent.
 * @property {number} confidence Confidence score (0-1) indicating how confident the system is about the refinement.
 */
```
