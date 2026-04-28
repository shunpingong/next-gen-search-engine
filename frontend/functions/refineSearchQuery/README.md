# refineSearchQuery Function Guide

This folder contains the source code and setup notes for the query refinement LLM function.

## Files

- `information/information.md` contains the copy-ready setup values for the Basic Information tab.
- `LLM/LLM.md` contains the copy-ready setup values for the LLM tab.
- `preprocess.js` normalizes the raw input object and returns `query`, `context`, and `domain` for the LLM prompt.
- `postprocess.js` parses and validates the LLM JSON output, then returns `refined_query`, `intent`, and `confidence`.

## Runtime Flow

1. The function receives an input object containing `query`, with optional `context` and `domain`.
2. `preprocess.js` validates and normalizes the query, context, and domain.
3. The LLM receives `query`, `context`, and `domain`.
4. The LLM returns a single JSON object containing `refined_query`, `intent`, and `confidence`.
5. `postprocess.js` extracts valid JSON, validates fields, applies fallbacks if needed, and returns the final output object.

## Input And Output

The function input and output schemas are documented in:

```text
functions/refineSearchQuery/information/information.md
```

## LLM Setup

The LLM prompt, arguments, parameters, and external-tool status are documented in:

```text
functions/refineSearchQuery/LLM/LLM.md
```

## Dependencies Setup

No LLM function dependencies are configured for this function.

## Notes

- Input type is an object with required `query`.
- Output type is an object with required `refined_query`, `intent`, and `confidence`.
- The output must be valid JSON so `postprocess.js` can parse it reliably.
