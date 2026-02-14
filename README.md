# Tool-Augmented Generative (TAG) Search Engine

**Final Year Project (FYP) - Future Search Engine with LLM Tool Calling**

A locally-hosted API service for an intelligent search engine that combines Large Language Models (LLMs) with external tools to overcome the limitations of traditional keyword-based search and pure LLM hallucinations.

## Project Overview

With the rapid rise in Artificial Intelligence (AI), there has been a shift from traditional keyword-based search engines towards search systems powered by Large Language Models (LLMs), such as ChatGPT. This shift is driven by conventional search engines' limitations in understanding user intent and performing multi-step information retrieval.

However, LLM-based search systems suffer from inherent hallucination issues and may retrieve irrelevant or inaccurate information. This project implements a **Tool-Augmented Generative (TAG)** search engine where LLMs serve as orchestrators that plan and decide when to use external tools.

### Key Features

- **Tool-Augmented LLM Search**: LLMs with tool-calling capabilities for intelligent information retrieval
- **Web Search Integration**: Tavily API integration for real-time web search results
- **PageRank Ranking**: Advanced document ranking using PageRank algorithm with query personalization
- **TextGrad Optimization**: Iterative query, answer, and prompt refinement using LLM-as-judge methodology
- **RESTful API Backend**: FastAPI service providing endpoints for integration with Nemobot web application

### System Architecture

The system addresses three key challenges:

1. **Limited Intent Understanding** (Traditional Search) → Solved with LLM natural language understanding
2. **Hallucination Issues** (Pure LLM) → Mitigated with Retrieval-Augmented Generation (RAG)
3. **Relevance & Multi-step Reasoning** → Enhanced with Tool-Augmented Generation (TAG)

## Project Structure

```
backend/
├── main.py                  # Main API server with all endpoints
├── scrapers/                # Web scraping modules (DuckDuckGo, Google, Tavily, etc.)
│   ├── base_scraper.py
│   ├── tavily_scraper.py
│   └── ...
├── logs/                    # TextGrad optimization logs
├── decrypted_datasets/      # BrowseComp evaluation datasets
└── pyproject.toml           # Python dependencies
```

**Note**: This backend API integrates with the Nemobot web application (developed by another team member).

## API Endpoints

All endpoints are hosted locally for development and testing.

### Core Search Endpoints

#### `POST /tavily`

Web search using Tavily API with advanced search depth and raw content extraction.

**Request:**

```json
{
  "query": "What is quantum computing?"
}
```

**Response:** Returns Tavily search results including URLs, snippets, and full content.

---

#### `POST /pagerank`

Compute PageRank scores for a list of documents with optional query-based personalization.

**Request:**

```json
{
  "documents": [
    { "id": "doc1", "title": "...", "content": "...", "url": "..." }
  ],
  "query": "quantum computing",
  "top_k": 5
}
```

**Response:** Returns PageRank scores and ranked documents based on relevance and inter-document relationships.

---

### TextGrad Optimization Endpoints

These endpoints use TextGrad's LLM-as-judge methodology to iteratively improve various components of the search pipeline.

#### `POST /textgrad/refine-query`

Optimize search queries for better search engine compatibility and clarity.

**Request:**

```json
{
  "query": "best restaurants near me",
  "max_iterations": 3
}
```

**Response:**

```json
{
  "original_query": "best restaurants near me",
  "refined_query": "highly rated restaurants near my location",
  "system_prompt_snapshot": "..."
}
```

**Use Cases:**

- Improving vague or conversational queries
- Enhancing search engine compatibility
- Preserving user intent while improving clarity

---

#### `POST /textgrad/refine-answer`

Improve answer quality, factual accuracy, and eliminate hallucinations based on provided context.

**Request:**

```json
{
  "question": "What is the capital of France?",
  "context": "France is a country in Europe. Paris is its largest city and capital.",
  "initial_answer": "The capital is Paris",
  "max_iterations": 3
}
```

**Response:**

```json
{
  "original_answer": "The capital is Paris",
  "refined_answer": "The capital of France is Paris, which is also its largest city.",
  "system_prompt_snapshot": "..."
}
```

**Use Cases:**

- Ensuring factual accuracy against retrieved context
- Eliminating hallucinations
- Improving reasoning and completeness

---

#### `POST /textgrad/refine-plan`

Optimize multi-step tool execution plans based on execution feedback.

**Request:**

```json
{
  "user_query": "Find recent AI research papers and summarize key findings",
  "execution_feedback": "Step 2 failed: API rate limit exceeded",
  "max_iterations": 3
}
```

**Response:**

```json
{
  "original_plan": "1. Search papers 2. Extract data 3. Summarize",
  "refined_plan": "1. Search papers with backoff 2. Extract data in batches 3. Summarize",
  "system_prompt_snapshot": "..."
}
```

**Use Cases:**

- Improving tool orchestration strategies
- Handling execution failures gracefully
- Optimizing multi-step reasoning chains

---

#### `POST /textgrad/optimize-prompt`

Offline system prompt tuning based on evaluation inputs and desired behavior.

**Request:**

```json
{
  "eval_inputs": ["What is machine learning?", "Explain neural networks"],
  "desired_behavior": "Provide concise, accurate technical explanations suitable for beginners",
  "max_iterations": 10
}
```

**Response:**

```json
{
  "original_prompt": "You are a helpful assistant.",
  "optimized_prompt": "You are a technical educator...",
  "evaluation_results": [...],
  "iterations": 10
}
```

**Use Cases:**

- Meta-optimization of system prompts
- Adapting LLM behavior to specific domains
- Improving response quality across evaluation sets

---

### Health Check Endpoints

#### `GET /`

API root with endpoint listing and version information.

#### `GET /health`

Simple health check endpoint.

---

## Quick Start

### Prerequisites

- Python 3.8+
- API Keys (stored in `.env` file):
  - `OPENROUTER_API_KEY` - For LLM access via OpenRouter
  - `TAVILY_API_KEY` - For web search

### Backend Setup

```bash
cd backend
pip install -e .
python main.py
```

API server will start at `http://localhost:8000`

### Integration with Nemobot

This backend API is designed to be consumed by the Nemobot web application (developed separately). The web interface makes HTTP requests to these endpoints to provide an interactive search experience.

---

## Technology Stack

- **FastAPI**: High-performance async web framework
- **TextGrad**: LLM optimization and gradient computation
- **OpenRouter**: Multi-model LLM access (GPT-4, Claude, Llama, etc.)
- **Tavily API**: Advanced web search
- **NumPy**: PageRank computation
- **Python 3.8+**: Core backend language

---

## Evaluation & Benchmarks

This system is evaluated using:

1. **BrowseComp**: OpenAI's benchmark for browsing agents
2. **Custom Queries**: 50-150 real-world user queries
3. **Baseline Comparisons**:
   - Pure LLM-based search
   - LLMs with RAG
   - LLMs with TAG (this system)

---

## Model Configuration

The system supports multiple LLM models via OpenRouter. To change models, edit the model configuration in [backend/main.py](backend/main.py):

```python
# For backward engine (evaluation/feedback):
tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True)

# For forward engine (content generation):
MODEL = tg.BlackboxLLM("experimental:openrouter/openai/gpt-3.5-turbo")
```

**Supported Models:**

- `openai/gpt-4o` (most capable)
- `openai/gpt-4-turbo`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-70b-instruct`

See [OpenRouter Models](https://openrouter.ai/models) for full list.

---

## Research Context

This project serves as the implementation for a Final Year Project on next-generation search systems. Key research contributions:

- **Tool-Augmented Generation (TAG)**: Novel pipeline combining LLM reasoning with external tools
- **Multi-Model Evaluation**: Comparative analysis across different LLM architectures
- **Real-World Applicability**: Benchmarking against realistic user queries
- **API-First Architecture**: RESTful backend enabling integration with various frontend applications

The findings demonstrate the feasibility of TAG approaches and provide a foundation for future intelligent search engines. The backend API architecture allows flexible integration with different user interfaces, including the Nemobot web application.

---

## Future Directions

- Multi-modal search (images, videos, documents)
- Advanced tool libraries beyond web search
- Personalized ranking based on user history
- Real-time streaming responses
- Production deployment with caching and rate limiting

---

## Documentation

- **Backend API**: See [backend/README.md](backend/README.md)
- **TextGrad Endpoints**: See [backend/TEXTGRAD_ENDPOINTS.md](backend/TEXTGRAD_ENDPOINTS.md)
- **Scrapers**: See [backend/scrapers/](backend/scrapers/)

---

## License

This project is developed as part of academic research for educational purposes.

---

## Acknowledgments

- **OpenAI**: BrowseComp benchmark dataset
- **TextGrad**: LLM optimization framework
- **OpenRouter**: Multi-model LLM access
- **Tavily**: Advanced web search API
