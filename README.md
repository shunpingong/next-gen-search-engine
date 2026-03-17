# Tool-Augmented Generative Search Engine

Backend and analysis code for a Final Year Project comparing `LLM-only`, `RAG`, and `TAG` search pipelines.

The backend exposes a FastAPI service for:

- deep web research with retrieval, reranking, extraction, and reflection
- PageRank-style reranking for document sets
- optional TextGrad-based query, answer, plan, and prompt refinement

The `analysis/` folder contains the workbook, scripts, and generated figures used for evaluation.

## Repository Layout

- `backend/` FastAPI backend and TAG pipeline
- `backend/main.py` API entrypoint
- `backend/tests/` backend tests
- `analysis/generate_figures.py` figure generation script
- `analysis/fyp_analysis.py` exploratory analysis script
- `analysis/FYP Results.xlsx` evaluation workbook used by the analysis scripts

## Requirements

- Python 3.10+
- `TAVILY_API_KEY` for web search
- `OPENAI_API_KEY` for the default `/tavily` decomposition and follow-up flow, and for all `/textgrad/*` endpoints

Optional provider keys:

- `SERPER_API_KEY`
- `SERPAPI_API_KEY`
- `GOOGLE_API_KEY`
- `GOOGLE_SEARCH_ENGINE_ID`

If you want to run the retrieval pipeline without OpenAI-backed decomposition and follow-up, set:

```env
TAVILY_USE_LLM_DECOMPOSITION=0
TAVILY_USE_LLM_FOLLOW_UP=0
```

## Setup

```bash
cd backend
pip install -e .
```

If you plan to run the analysis scripts, make sure your environment also has `pandas` and `seaborn` available.

Create a `.env` file in the project root or `backend/` with the keys you need, for example:

```env
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
```

## Run the API

```bash
cd backend
python main.py
```

The API starts on `http://localhost:8000`. FastAPI docs are available at `http://localhost:8000/docs`.

## Main Endpoints

- `POST /tavily` runs the research pipeline and returns an answer, evidence, sources, reasoning trace, and timing stats
- `POST /pagerank` reranks a list of documents using query-aware graph scoring
- `POST /textgrad/refine-query` refines search queries
- `POST /textgrad/refine-answer` improves answer drafts against provided context
- `POST /textgrad/refine-plan` revises multi-step plans using execution feedback
- `POST /textgrad/optimize-prompt` performs offline prompt optimization

## Testing

```bash
cd backend
pytest
```

## Analysis

Regenerate the report figures with:

```bash
python analysis/generate_figures.py
```

Generated figures are written to `analysis/*.png`.

## Notes

- This backend was built to integrate with a separate frontend application.
- The repository is primarily an academic project rather than a production deployment.
