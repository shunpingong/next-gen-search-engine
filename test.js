// server.js
import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
const TAVILY_ENDPOINT = process.env.TAVILY_ENDPOINT;

app.get("/", (req, res) => {
  res.send("Proxy server is running");
});

// POST endpoint for Nemobot
app.post("/tavily", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) {
      return res.status(400).json({ error: "Missing query parameter" });
    }

    console.log("Received query:", query);

    const MAX_QUERY_LENGTH = 400;
    const trimmedQuery =
      query.length > MAX_QUERY_LENGTH
        ? query.slice(0, MAX_QUERY_LENGTH)
        : query;

    const response = await fetch(TAVILY_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        Authorization: `Bearer ${TAVILY_API_KEY}`,
      },
      body: JSON.stringify({
        query: trimmedQuery,
        search_depth: "advanced",
        max_results: 5,
        include_answer: true,
        include_raw_content: false,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Tavily API Error:", errorText);
      return res.status(response.status).json({ error: errorText });
    }

    const data = await response.json();
    console.log("Tavily API Success");
    console.log(data);
    res.json(data);
  } catch (error) {
    console.error("Server Error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Tavily Proxy running at http://localhost:${PORT}`);
});
