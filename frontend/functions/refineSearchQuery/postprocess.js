/**
 * Preprocess the input.
 * @param {string} llmOutput - The input of the function.
 * @param {Environment} environment - The environment of the function.
 * @returns {Promise<Output>} The final output of this function.
 */
async function postprocess(llmOutput, environment) {
  console.log("LLM RAW:", JSON.stringify(llmOutput));

  // Retrieve original query from environment as fallback
  const originalQuery =
    environment?.temporaryStorage?.refineSearchQuery?.preprocessed?.query || "";

  // Fallback if LLM output is empty
  if (!llmOutput || llmOutput.trim().length === 0) {
    return {
      refined_query: originalQuery.trim(),
      intent: "exploratory",
      confidence: 0.5,
    };
  }

  let parsed;
  try {
    // Try to extract JSON from LLM output
    const match = llmOutput.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("No JSON found");
    parsed = JSON.parse(match[0]);
  } catch {
    // If parsing fails, fallback to using the raw output as query
    return {
      refined_query:
        llmOutput.trim().replace(/\s+/g, " ") || originalQuery.trim(),
      intent: "exploratory",
      confidence: 0.5,
    };
  }

  // Destructure and validate fields
  let { refined_query, intent, confidence } = parsed;

  refined_query =
    typeof refined_query === "string"
      ? refined_query
          .trim()
          .replace(/^["'`]+|["'`]+$/g, "")
          .replace(/\s+/g, " ")
      : originalQuery.trim();

  const validIntents = [
    "factual",
    "comparison",
    "how-to",
    "troubleshooting",
    "exploratory",
  ];
  intent = validIntents.includes(intent) ? intent : "exploratory";

  confidence =
    typeof confidence === "number"
      ? Math.min(Math.max(confidence, 0), 1)
      : 0.75;

  // Save final output in temporary storage
  environment.temporaryStorage ??= {};
  environment.temporaryStorage.refineSearchQuery ??= {};
  environment.temporaryStorage.refineSearchQuery.postprocessed = {
    refined_query,
    intent,
    confidence,
  };

  return { refined_query, intent, confidence };
}
