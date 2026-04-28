# LLMTag Playground Guide

This folder contains the source code and LLM dependency setup for the tool-augmented information retrieval chatbot function.

## Files

- `generateResponse.js` contains the runtime function called when the user sends a message.
- `information/information.md` contains the copy-ready setup values for the Basic Information tab.
- `dependencies/chat.md` contains the copy-ready setup values for the Dependencies tab.

## Runtime Flow

1. The platform calls `generateResponse(content, chat, environment)`.
2. The function reads the user message from `content.text`.
3. The function creates a response stream named `chat`.
4. The function registers external tools for current datetime, basic web search, and detailed web search.
5. The function calls `environment.llmChatCompletions.chat.stream`.
6. The LLM dependency receives `userMessage` as its argument.
7. The LLM may call the configured external tools when needed.
8. The streamed LLM tokens are sent back to the chat UI.

## Required Dependencies

The function expects this chat-completion dependency to exist:

```text
environment.llmChatCompletions.chat
```

The function also expects these LLM function dependencies to exist:

```text
environment.llmFunctions.webSearchBasic
environment.llmFunctions.webSearchDetailed
```

If you recreate the project in another workspace, create an LLM chat-completion dependency named `chat`, copy the Basic Information tab values from `information/information.md`, copy the Dependencies tab values from `dependencies/chat.md`, then add the two LLM function dependencies with the aliases `webSearchBasic` and `webSearchDetailed`.

## External Tools

The chat dependency exposes these tools to the LLM:

```text
get_current_datetime
web_search_basic
web_search_detailed
```

## Notes

- The dependency name must stay as `chat`.
- The argument name must stay as `userMessage`.
- The LLM function aliases must stay as `webSearchBasic` and `webSearchDetailed`.
- Chat memory should be enabled in the LLM dependency so previous conversation messages are available.
- The function has a reentrancy guard through `environment.temporaryStorage.started` to avoid duplicate streaming calls.
