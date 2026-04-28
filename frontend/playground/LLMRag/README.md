# LLMRag Playground Guide

This folder contains the source code and LLM dependency setup for the RAG-based chatbot response function.

## Files

- `generateResponse.js` contains the runtime function called when the user sends a message.
- `information/information.md` contains the copy-ready setup values for the Basic Information tab.
- `dependencies/chat.md` contains the copy-ready setup values for the Dependencies tab.

## Runtime Flow

1. The platform calls `generateResponse(content, chat, environment)`.
2. The function reads the user message from `content.text`.
3. The function creates a response stream named `chat`.
4. The function calls the local Tavily backend endpoint at `http://localhost:8000/tavily`.
5. The function normalizes the Tavily answer and search results into `retrievedContext`.
6. The function calls `environment.llmChatCompletions.chat.stream`.
7. The LLM dependency receives `userMessage` and `retrievedContext` as arguments.
8. The streamed LLM tokens are sent back to the chat UI.

## Required Dependency

The function expects this dependency to exist:

```text
environment.llmChatCompletions.chat
```

If you recreate the project in another workspace, create an LLM chat-completion dependency named `chat`, copy the Basic Information tab values from `information/information.md`, and copy the Dependencies tab values from `dependencies/chat.md`.

The function also expects a local backend endpoint to be available:

```text
POST http://localhost:8000/tavily
```

## Notes

- Retrieval happens before the LLM call by sending the user message to the local Tavily endpoint.
- No LLM external tool is configured inside the dependency itself.
- Chat memory should be enabled in the LLM dependency so previous conversation messages are available.
- The argument names must stay as `userMessage` and `retrievedContext`.
- The function has a reentrancy guard through `environment.temporaryStorage.started` to avoid duplicate streaming calls.
