# LLM Playground Guide

This folder contains the source code and LLM dependency setup for the chatbot response function.

## Files

- `generateResponse.js` contains the runtime function called when the user sends a message.
- `information/information.md` contains the copy-ready setup values for the Information tab.
- `dependencies/chat.md` contains the copy-ready setup values for the Dependencies tab.

## Runtime Flow

1. The platform calls `generateResponse(content, chat, environment)`.
2. The function reads the user message from `content.text`.
3. The function creates a response stream named `chat`.
4. The function calls `environment.llmChatCompletions.chat.stream`.
5. The LLM dependency receives `userMessage` as its argument.
6. The streamed LLM tokens are sent back to the chat UI.

## Required Dependency

The function expects this dependency to exist:

```text
environment.llmChatCompletions.chat
```

If you recreate the project in another workspace, create an LLM chat-completion dependency named `chat`, copy the Information tab values from `information/information.md`, and copy the Dependencies tab values from `dependencies/chat.md`.

## Notes

- No external tools are used by this function.
- Chat memory should be enabled in the LLM dependency so previous conversation messages are available.
- The function has a reentrancy guard through `environment.temporaryStorage.started` to avoid duplicate streaming calls.
