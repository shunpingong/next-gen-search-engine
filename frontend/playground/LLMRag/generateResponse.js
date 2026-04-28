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

  const mainSignpost = chat.createSignpost("generate_response_duration");
  const tavilySignpost = chat.createSignpost("tavily_retrieval_duration");

  mainSignpost.emitEvent("start");

  if (environment.temporaryStorage.started) return;
  environment.temporaryStorage.started = true;

  try {
    /* =====================
           Call Local Tavily Endpoint
        ====================== */
    const tavilyEndpoint = "http://localhost:8000/tavily";

    // Create and start the Tavily-specific signpost
    tavilySignpost.emitEvent("start");

    const response = await fetch(tavilyEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: userMessage }),
    });

    // End the signpost as soon as the response is received
    tavilySignpost.emitEvent("end");

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Tavily proxy failed: ${errorText}`);
    }

    const responseData = await response.json();

    /* =====================
           Normalize Search Results
        ====================== */
    const documents = [];

    // Prefer Tavily direct answer
    if (typeof responseData.answer === "string" && responseData.answer.trim()) {
      documents.push({
        id: "tavily-answer",
        title: "Tavily Summary",
        url: "",
        score: 1.0,
        text: responseData.answer.trim(),
      });
    }

    if (Array.isArray(responseData.results)) {
      responseData.results.forEach((r, idx) => {
        const cleanedContent = String(r.content || "")
          .replace(/<[^>]+>/g, "")
          .replace(/\s+/g, " ")
          .trim()
          .slice(0, 1000);

        documents.push({
          id: idx,
          title: r.title || "",
          url: r.url || "",
          score: typeof r.score === "number" ? r.score : 0,
          text: cleanedContent,
        });
      });
    }

    console.log("Document rerieved:", documents.length);
    console.log(documents);

    /* =====================
           Format Retrieved Context for Prompt
        ====================== */
    const retrievedContext = documents
      .map((doc, i) => {
        return `### Source ${i + 1}
                        Title: ${doc.title}
                        URL: ${doc.url}
                        Content:
                        ${doc.text}`;
      })
      .join("\n");

    /* =====================
           Call LLM with Injected Context
        ====================== */
    await environment.llmChatCompletions.chat.stream({
      onToken: (token) => stream.streamToken(token),
      args: {
        userMessage,
        retrievedContext,
      },
    });
  } catch (err) {
    console.error("RAG failed:", err);
    stream.streamToken("An error occurred while retrieving information.");
  } finally {
    environment.temporaryStorage.started = false;
    mainSignpost.emitEvent("end");
  }
}
