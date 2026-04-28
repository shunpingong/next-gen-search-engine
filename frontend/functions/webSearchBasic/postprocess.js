/**
 * Preprocess the input.
 * @param {string} llmOutput - The input of the function.
 * @param {Environment} environment - The environment of the function.
 * @returns {Promise<Output>} The final output of this function.
 */
async function postprocess(llmOutput, environment) {
  console.log("llm output");
  console.log(llmOutput);
  if (!llmOutput || typeof llmOutput !== "string") {
    return {
      summary: "",
      error: "No output from LLM or invalid type.",
    };
  }

  /* =====================
       Cleanup & normalization
    ====================== */
  let cleaned = llmOutput
    .trim()
    .replace(/^\s*["'`]/, "") // remove starting quotes
    .replace(/["'`]\s*$/, "") // remove ending quotes
    .replace(/\n{3,}/g, "\n\n") // collapse excessive newlines
    .replace(/[ \t]+$/gm, ""); // remove trailing spaces per line

  /* =====================
       LaTeX-safe sanitization
       (replace Unicode characters with ASCII equivalents)
    ====================== */
  cleaned = cleaned
    .replace(/[\u2013\u2014]/g, "-") // en-dash & em-dash → hyphen
    .replace(/[\u2018\u2019\u201A\u201B]/g, "'") // fancy single quotes
    .replace(/[\u201C\u201D\u201E\u201F]/g, '"') // fancy double quotes
    .replace(/\u2026/g, "...") // ellipsis
    .replace(/•/g, "*"); // bullets → asterisk

  console.log("Postprocessed summary (preview 500 chars):");
  console.log(cleaned.slice(0, 500));

  return { summary: cleaned };
}
