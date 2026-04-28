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
  environment.temporaryStorage.started = true; // mark as generating

  // Sub-operation signposts
  const getTimeSignpost = chat.createSignpost("get_current_datetime");
  const webSearchBasicSignpost = chat.createSignpost("web_search_basic");
  const detailedWebSearchSignpost = chat.createSignpost("detailed_web_search");

  // External tools exposed to the LLM (safe: no nested streaming)
  const externalTools = {
    get_current_datetime: async (_argument) => {
      getTimeSignpost.emitEvent("start");
      const time = new Date().toLocaleString();
      getTimeSignpost.emitEvent("finish");
      return time;
    },

    web_search_basic: async (query) => {
      webSearchBasicSignpost.emitEvent("start");
      // Only fetch summary, do NOT call generateResponse or chat.stream
      const result = await environment.llmFunctions.webSearchBasic(
        query,
        environment,
      );
      webSearchBasicSignpost.emitEvent("finish");
      return result.summary;
    },

    web_search_detailed: async (query) => {
      detailedWebSearchSignpost.emitEvent("start");
      // Only fetch preprocessed & ranked top-K results
      const result = await environment.llmFunctions.webSearchDetailed(
        query,
        environment,
      );
      detailedWebSearchSignpost.emitEvent("finish");
      return result.topKSummary;
    },
  };

  try {
    // Stream LLM tokens for response (top-level only)
    await environment.llmChatCompletions.chat.stream({
      onToken: (token) => stream.streamToken(token),
      args: { userMessage },
      options: {
        externalTool: externalTools,
      },
    });
  } catch (err) {
    console.error("LLM streaming failed:", err);
  } finally {
    // Release the reentrancy flag
    environment.temporaryStorage.started = false;
    mainSignpost.emitEvent("end");
  }
}
