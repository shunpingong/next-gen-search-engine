/**
 * Preprocess the input.
 * @param {Input} input - The input of the function.
 * @param {Environment} environment - The environment of the function.
 * @returns {Promise<LLMInput>} The preprocessed input that matches the argument list.
 */
async function preprocess(input, environment) {
  let { query, context, domain } = input;

  if (typeof query !== "string" || query.trim().length === 0) {
    throw new Error("query must be a non-empty string");
  }

  // Normalize query text
  query = query
    .trim()
    .replace(/\s+/g, " ")
    .replace(
      /\b(can you|could you|please|i want to know|tell me about|what is|how do i)\b/gi,
      "",
    )
    .trim();

  // Normalize context
  context =
    typeof context === "string" ? context.trim().replace(/\s+/g, " ") : "";

  // Normalize domain
  domain = typeof domain === "string" ? domain.trim().toLowerCase() : "";

  // Save temporary storage for debugging
  environment.temporaryStorage ??= {};
  environment.temporaryStorage.refineSearchQuery ??= {};
  environment.temporaryStorage.refineSearchQuery.preprocessed = {
    query,
    context,
    domain,
  };

  return { query, context, domain };
}
