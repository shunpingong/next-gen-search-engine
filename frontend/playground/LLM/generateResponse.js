/**
 * Generate responses from the user message.
 *
 * To send a message back to the user, use the {@link Chat.reply} function.
 * To stream a message, use the {@link Chat.stream} function to create a message stream.
 *
 * @param {Content} content the message sent by user
 * @param {Chat} chat the ongoing chat
 * @param {Environment} environment the state of the app
 */
async function generateResponse(content, chat, environment) {
  const userMessage = content.text.trim();
  const stream = chat.createStream("chat");

  // Main signpost for overall response duration
  const mainSignpost = chat.createSignpost("generate_response_duration");
  mainSignpost.emitEvent("start");

  // ----------------------------
  // Prevent recursive/double execution
  // ----------------------------
  if (environment.temporaryStorage.started) {
    console.warn("Already generating a response. Skipping duplicate call.");
    return;
  }
  environment.temporaryStorage.started = true;

  try {
    // Stream LLM tokens (NO external tools)
    await environment.llmChatCompletions.chat.stream({
      onToken: (token) => stream.streamToken(token),
      args: { userMessage },
    });
  } catch (err) {
    console.error("LLM streaming failed:", err);
  } finally {
    // Release the reentrancy flag
    environment.temporaryStorage.started = false;
    mainSignpost.emitEvent("end");
  }
}
