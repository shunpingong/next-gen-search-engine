/**
 * Preprocess the input.
 * @param {Input} input - The input of the function.
 * @param {Environment} environment - The environment of the function.
 * @returns {Promise<LLMInput>} The preprocessed input that matches the argument list.
 */
async function preprocess(input, environment) {
  /* =====================
       Helper utilities
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
    if (/error|bug|exception|stack trace|code/i.test(query))
      return "programming";
    if (/price|buy|review/i.test(query)) return "shopping";
    if (/news|latest|today/i.test(query)) return "news";
    return "general";
  }

  function cleanText(text, maxLen = 2000) {
    return String(text || "")
      .replace(/<[^>]+>/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, maxLen);
  }

  /* =====================
       Normalize input
    ====================== */
  const question = String(input || "").trim();
  console.log("=== Preprocessing Detailed Web Search ===");
  console.log("Original question:", question);

  environment.temporaryStorage ??= {};
  environment.temporaryStorage.webSearch ??= {};

  const context = inferQueryContext(question);
  const domain = inferDomain(question);

  environment.temporaryStorage.webSearch.input = {
    query: question,
    context,
    domain,
  };

  /* =====================
       Query refinement (LLM)
    ====================== */
  let refinement;
  try {
    refinement = await environment.llmFunctions.refineSearchQuery(
      { query: question, context, domain },
      environment,
    );
  } catch (err) {
    console.warn("Query refinement failed, using fallback.", err);
    refinement = {
      refined_query: question,
      intent: "unknown",
      confidence: 0.0,
    };
  }

  // Save refinement output
  environment.temporaryStorage.webSearch.refined = refinement;

  console.log("Refinement is:");
  console.log(refinement);

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
       Tavily search
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
  const responseData = await response.json();

  const tavilyEnd = performance.now();
  console.log(
    "Retrieval time:",
    ((tavilyEnd - tavilyStart) / 1000).toFixed(3),
    " seconds",
  );

  /* =====================
       Normalize documents
    ====================== */
  const documents = [];

  /* ---------- Answer ---------- */

  if (typeof responseData.answer === "string" && responseData.answer.trim()) {
    documents.push({
      id: "tavily-answer",
      title: "Tavily Answer",
      url: responseData.source || "",
      content: cleanText(responseData.answer, 2500),
      score: 1.0,
    });
  }

  /* ---------- Evidence ---------- */

  if (
    typeof responseData.evidence === "string" &&
    responseData.evidence.trim()
  ) {
    documents.push({
      id: "tavily-evidence",
      title: "Supporting Evidence",
      url: responseData.source || "",
      content: cleanText(responseData.evidence, 2000),
      score: 0.95,
    });
  }

  /* ---------- Sources ---------- */

  if (Array.isArray(responseData.sources)) {
    responseData.sources.forEach((src, idx) => {
      documents.push({
        id: `source-${idx}`,
        title: src.title || "",
        url: src.url || "",
        content: cleanText(src.snippet || src.content || "", 2000),
        score: typeof src.score === "number" ? src.score : 0.7,
      });
    });
  }

  /* ---------- Context block ---------- */

  if (typeof responseData.context === "string" && responseData.context.trim()) {
    const chunks = responseData.context.split(/\n{2,}/);

    chunks.forEach((chunk, idx) => {
      if (chunk.trim().length < 40) return;

      documents.push({
        id: `context-${idx}`,
        title: "Context Extract",
        url: "",
        content: cleanText(chunk, 1800),
        score: 0.6,
      });
    });
  }

  console.log("Total normalized documents:", documents.length);

  environment.temporaryStorage.webSearch.rawDocuments = documents;

  /* =====================
        PageRank Ranking
        ====================== */

  let pagerankScores = {};
  const topK = environment.topK || 5;

  const pagerankStart = performance.now();

  try {
    const pagerankResponse = await fetch("http://localhost:8000/pagerank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        documents,
        query: refinedQuery,
        top_k: topK,
      }),
    });

    if (!pagerankResponse.ok) throw new Error("Pagerank endpoint failed");

    const pagerankData = await pagerankResponse.json();

    pagerankScores = pagerankData.scores || {};

    if (Array.isArray(pagerankData.ranked_documents)) {
      documents.length = 0;

      pagerankData.ranked_documents.forEach((doc) => {
        doc.pagerank = doc.pagerank_score ?? pagerankScores[doc.id] ?? 0;

        documents.push(doc);
      });
    } else {
      documents.forEach((doc) => {
        doc.pagerank = pagerankScores[doc.id] ?? 0;
      });
    }

    documents.sort((a, b) => {
      if (b.pagerank !== a.pagerank) return b.pagerank - a.pagerank;

      return (b.score || 0) - (a.score || 0);
    });
  } catch (err) {
    console.warn("PageRank failed", err);
  }

  environment.temporaryStorage.webSearch.rankedDocuments = documents;

  const pagerankEnd = performance.now();

  console.log(
    "PageRank time:",
    ((pagerankEnd - pagerankStart) / 1000).toFixed(3),
    "seconds",
  );

  /* =====================
        Build LLM input
        ====================== */

  const searchResult = documents
    .map((d, idx) => {
      return (
        `Result ${idx + 1}\n` +
        `Title: ${d.title}\n` +
        `URL: ${d.url}\n` +
        `PageRank Score: ${d.pagerank?.toFixed(4) || 0}\n` +
        `Content: ${d.content}`
      );
    })
    .join("\n\n---\n\n");

  console.log("Formatted search result length:", searchResult.length);

  return {
    question,
    refinedQuery,
    searchResult,
  };
}
