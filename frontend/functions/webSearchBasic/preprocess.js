/**
 * Preprocess the input.
 * @param {Input} input - The input of the function.
 * @param {Environment} environment - The environment of the function.
 * @returns {Promise<LLMInput>} The preprocessed input that matches the argument list.
 */
async function preprocess(input, environment) {
  /* =====================
       Helper utilities (deterministic rule-based)
    ====================== */
  function inferQueryContext(query) {
    if (/compare|vs|difference/i.test(query))
      return "Comparative informational query";
    if (/how to|steps|guide|tutorial/i.test(query))
      return "Procedural or instructional query";
    if (/why|explain|theory/i.test(query))
      return "Conceptual or explanatory query";
    return "General informational query";
  }

  function inferDomain(query) {
    if (/paper|journal|doi|arxiv/i.test(query)) return "academic";
    if (/error|bug|stack trace|exception|code/i.test(query))
      return "programming";
    if (/price|buy|review/i.test(query)) return "shopping";
    if (/news|latest|today/i.test(query)) return "news";
    return "general";
  }
  /* =====================
       Normalize input
    ====================== */
  const question = String(input || "").trim();
  console.log("=== Preprocessing Basic Web Search ===");
  console.log("Original question:", question);
  console.log(question);

  environment.temporaryStorage ??= {};
  environment.temporaryStorage.webSearch ??= {};

  const context = inferQueryContext(question);
  const domain = inferDomain(question);

  // Save preprocess input
  environment.temporaryStorage.webSearch.input = {
    query: question,
    context,
    domain,
  };

  /* =====================
       Query refinement via LLM
    ====================== */
  let refinement;
  try {
    console.log("Refining query via LLM...");
    refinement = await environment.llmFunctions.refineSearchQuery(
      { query: question, context, domain },
      environment,
    );
  } catch (err) {
    console.warn("Query refinement failed. Using fallback.", err);
    refinement = {
      refined_query: question,
      intent: "unknown",
      confidence: 0.0,
    };
  }

  // Save refinement output
  environment.temporaryStorage.webSearch.refined = refinement;

  let refinedQuery = refinement.refined_query;
  console.log("Refined query:");
  console.log(refinedQuery);

  /* =====================
       TextGrad query optimization
    ====================== */
  async function callTextGrad(text, numIterations = 3) {
    console.log("[TextGrad] Starting optimization...");
    console.log("[TextGrad] Input text:", text);
    console.log("[TextGrad] Iterations:", numIterations);
    try {
      const response = await fetch(
        "http://localhost:8000/textgrad/refine-query",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: text,
            max_iterations: numIterations,
          }),
        },
      );
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`TextGrad request failed: ${errorText}`);
      }

      const data = await response.json();
      console.log("[TextGrad] Optimization successful");
      console.log("[TextGrad] Optimized response:");
      console.log(data);
      return data;
    } catch (err) {
      console.warn(
        "[TextGrad] Optimization failed. Falling back to refined query.",
        err,
      );
      return text;
    }
  }

  if (refinement.confidence < 0.9) {
    console.log("Confidence below 0.90, applying TextGrad optimization...");

    let refinedQueryResponse = await callTextGrad(refinedQuery, 3);

    // handle case where TextGrad returns string fallback
    if (typeof refinedQueryResponse === "string") {
      refinedQuery = refinedQueryResponse;
    } else {
      refinedQuery = refinedQueryResponse.refined_query;
    }
  } else {
    console.log("Confidence >= 0.90, skipping TextGrad optimization.");
  }

  console.log("Final query used for Tavily:");
  console.log(refinedQuery);

  // Save optimized query
  environment.temporaryStorage.webSearch.optimizedQuery = refinedQuery;

  /* =====================
       Tavily API search (basic, raw)
    ====================== */
  const tavilyEndpoint = "http://localhost:8000/tavily";

  const tavilyStart = performance.now();

  const response = await fetch(tavilyEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: refinedQuery }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Tavily proxy failed: ${errorText}`);
  }
  const data = await response.json();

  const tavilyEnd = performance.now();
  console.log(
    "Retrieval time:",
    ((tavilyEnd - tavilyStart) / 1000).toFixed(3),
    " seconds",
  );

  /* =====================
       Normalize search results
    ====================== */
  const documents = [];
  // Use Tavily direct answer if available
  if (typeof data.answer === "string" && data.answer.trim()) {
    documents.push({
      id: "tavily-answer",
      type: "answer",
      title: "Tavily Answer",
      url: "",
      score: 1.0,
      content: data.answer.trim(),
      evidence: data.evidence || "",
    });
  }

  // Process sources array if present
  if (Array.isArray(data.sources)) {
    data.sources.forEach((s, idx) => {
      documents.push({
        id: `source-${idx}`,
        type: "source",
        title: s.title || "",
        url: s.url || "",
        score: s.score ?? 0,
        content: s.snippet || "",
      });
    });
  }

  // Fallback: process generic results array
  if (Array.isArray(data.results)) {
    data.results.forEach((r, idx) => {
      documents.push({
        id: `doc-${idx}`,
        type: "result",
        title: r.title || "",
        url: r.url || "",
        score: r.score ?? 0,
        content: String(r.content || "")
          .replace(/<[^>]+>/g, "")
          .trim()
          .slice(0, 1000),
      });
    });
  }

  environment.temporaryStorage.webSearch.documents = documents;

  // Prepare single string for LLM consumption
  const searchResult = documents
    .map(
      (d) =>
        `Type: ${d.type}\nTitle: ${d.title}\nURL: ${d.url}\nContent: ${d.content}${d.evidence ? `\nEvidence: ${d.evidence}` : ""}`,
    )
    .join("\n\n---\n\n");

  return { question, refinedQuery, searchResult };
}
