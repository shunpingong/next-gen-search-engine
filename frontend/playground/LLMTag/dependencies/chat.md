# LLM Dependency: chat

Use this file to recreate the LLMTAG `chat` dependency without screenshots. Copy each value into the matching field in the Dependencies tab.

## Dependencies Tab

### Dependencies

| Type                 | Function Name or Alias | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM Chat Completions | `chat`                 | A chatbot that replies according to the previous conversation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| LLM Function         | `webSearchBasic`       | This function performs a live, LLM-enhanced web search using the Tavily API. It first refines the user's initial query through a language model to ensure clarity and relevance. The refined query is then sent to the Tavily API to retrieve real-world search results. Finally, the raw search data is passed back into the LLM, which summarizes and synthesizes the key information before returning a concise, human-readable answer to the orchestrator.                                                                   |
| LLM Function         | `webSearchDetailed`    | This function performs a comprehensive, multi-stage web search optimized for accuracy and relevance. It refines the user's query using a language model, retrieves live search results via the Tavily API, normalizes and aggregates the retrieved documents, and applies an external PageRank-based ranking to prioritize the most authoritative sources. The ranked search content is then passed to the language model, which synthesizes a detailed, evidence-based response grounded strictly in the retrieved information. |

### Information

| Field         | Value                                                          |
| ------------- | -------------------------------------------------------------- |
| Function Name | `chat`                                                         |
| Description   | A chatbot that replies according to the previous conversation. |

### Prompt Template

#### Arguments

Separated by comma:

```text
userMessage
```

#### Messages

Enable memory/history messages from the chat before the latest user message.

```text
system
You are an intelligent information retrieval and reasoning assistant.

---

## Core Behavior
- Provide clear, factual, and accurate answers.
- If retrieved sources are available, prioritize them.
- If no sources are retrieved, answer using reliable general knowledge.
- Do not refuse to answer solely because no tool was used.
- Internally reason step by step, but never reveal reasoning or intermediate thoughts.
- Output only the final answer in Markdown format.
- Be concise, structured, and easy to read.
- Highlight important names, dates, numbers, metrics, or model names.

---

## Tool Usage Policy
- Tools are **optional** but can be used as needed.
- Prefer answering directly when the question can be answered with general knowledge.
- Use tools when:
  - Up-to-date information is required.
  - Verification or ranking is needed.
  - Specific sources are requested.
- Multiple tools **can be called** if needed to answer the question fully.

---

## Tool Output Handling
- Treat tool output as authoritative.
- Preserve formatting if the output is structured.
- Include links to all sources mentioned in tool outputs.
- Use all information from the tool response to answer the question.
- Confidence scores are optional but recommended.

---

## Tool Usage Termination
- Tools may continue to be used until sufficient information is gathered.
- Only stop calling tools when the final answer can be produced.

---

## Output Rules
- Always return a **final answer in Markdown format**.
- If tools were used, include the sources.
- If tools were not used, answer using general knowledge.
- If information is genuinely unknown, state that clearly.



Include Memory (history messages from the chat)


user
{{userMessage}}
```

### Parameters

Tune the parameters for the LLM.

| Parameter         | Value  |
| ----------------- | ------ |
| Max Response      | `5000` |
| Temperature       | `0`    |
| Top P             | `0.95` |
| Frequency Penalty | `0`    |
| Presence Penalty  | `0`    |

## External Tools

### Tool: get_current_datetime

| Field       | Value                          |
| ----------- | ------------------------------ |
| Name        | `get_current_datetime`         |
| Description | Get the current date and time. |

Parameter JSON Schema:

```json
{
  "type": "object",
  "properties": {}
}
```

Typedef:

```js
/**
 * @typedef {object} get_current_datetime
 */
```

### Tool: web_search_basic

| Field       | Value                                                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| Name        | `web_search_basic`                                                                                                        |
| Description | Use this tool for complex, ambiguous, or research-oriented questions that require high accuracy and structured reasoning. |

Full tool guidance:

```text
## High-Accuracy Research Tool Guidelines

Use this tool for **complex, ambiguous, or research-oriented questions** that require **high accuracy and structured reasoning**.

### When to Call This Tool

- The question requires **analysis, comparison, or evaluation**.
- The user asks for **best options, rankings, or trade-offs**.
- **Evidence quality and relevance** are important.
- The query is **ambiguous or multi-faceted**.
- The answer benefits from **ranking, filtering, or selecting top-K sources**.

### Tool Capabilities

This tool performs a **multi-stage retrieval pipeline**, including:

- **Query refinement**
- **Web search**
- **Document cleaning and normalization**
- **Graph-based ranking** (e.g., PageRank)
- **Top-K selection**

### Usage Notes

- This tool has **higher latency** but produces **more reliable, ranked results**.
- For **quick factual answers**, prefer **web_search_basic**.
- Always **pass the entire user question** as the query.
```

Parameter JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The user's full search query"
    }
  },
  "required": ["query"]
}
```

Typedef:

```js
/**
 * @typedef {object} web_search_basic
 * @property {string} query The user's full search query
 */
```

### Tool: web_search_detailed

| Field       | Value                                                                                                                               |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Name        | `web_search_detailed`                                                                                                               |
| Description | Use web_search_detailed for complex, ambiguous, or research-oriented questions that require high accuracy and structured reasoning. |

Full tool guidance:

```text
## Web Search Detailed Tool Guidelines

Use **web_search_detailed** for **complex, ambiguous, or research-oriented questions** that require **high accuracy and structured reasoning**.

### When to Use

- The question requires **analysis, comparison, or evaluation**.
- The user asks for **best options, rankings, or trade-offs**.
- **Evidence quality and relevance** are important.
- The query is **ambiguous or multi-faceted**.
- The answer benefits from **ranking, filtering, or selecting top-K sources**.

### Tool Capabilities

This tool performs a **multi-stage retrieval pipeline**, including:

- **Query refinement**
- **Web search**
- **Document cleaning and normalization**
- **Graph-based ranking** (e.g., PageRank)
- **Top-K selection**

### Usage Notes

- This tool has **higher latency** but produces **more reliable, ranked results**.
- For **quick factual answers**, prefer **web_search_basic**.
- Always **pass the entire user question** as the query.
```

Parameter JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The user's full search query"
    }
  },
  "required": ["query"]
}
```

Typedef:

```js
/**
 * @typedef {object} web_search_detailed
 * @property {string} query The user's full search query
 */
```

## LLM Function Dependencies

### Dependency: webSearchBasic

| Field    | Value            |
| -------- | ---------------- |
| Alias    | `webSearchBasic` |
| Function | `webSearchBasic` |

Description:

```text
This function performs a live, LLM-enhanced web search using the Tavily API. It first refines the user's initial query through a language model to ensure clarity and relevance. The refined query is then sent to the Tavily API to retrieve real-world search results. Finally, the raw search data is passed back into the LLM, which summarizes and synthesizes the key information before returning a concise, human-readable answer to the orchestrator.
```

### Dependency: webSearchDetailed

| Field    | Value               |
| -------- | ------------------- |
| Alias    | `webSearchDetailed` |
| Function | `webSearchDetailed` |

Description:

```text
This function performs a comprehensive, multi-stage web search optimized for accuracy and relevance. It refines the user's query using a language model, retrieves live search results via the Tavily API, normalizes and aggregates the retrieved documents, and applies an external PageRank-based ranking to prioritize the most authoritative sources. The ranked search content is then passed to the language model, which synthesizes a detailed, evidence-based response grounded strictly in the retrieved information.
```

## Source Code Usage

`generateResponse.js` sends the latest user message into this dependency with this argument shape:

```js
args: {
  userMessage,
}
```

The dependency must keep the argument name exactly as `userMessage`, because the function passes that key at runtime.
