# NextGen Web Search API

This backend provides a powerful REST API for web search with advanced ranking using PageRank and Tavily integration.

## Features

- **Tavily Search Integration**: Access web search results via Tavily API
- **PageRank Algorithm**: Advanced result ranking based on content similarity and relevance
- **Personalized Ranking**: Query-aware PageRank for improved relevance scoring

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

- **TAVILY_API_KEY** (Required): Get from https://tavily.com/
  - Needed for web search functionality

### 3. Run the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

#### Health Check

```
GET /health
```

Check status of the API.

### Tavily Search

#### Search Web

```
POST /tavily
```

Perform a web search using Tavily API.

**Request Body:**

```json
{
  "query": "machine learning best practices"
}
```

**Response:**

```json
{
  "query": "machine learning best practices",
  "answer": "Summary answer from Tavily...",
  "results": [
    {
      "url": "https://example.com",
      "title": "Article Title",
      "content": "Article content...",
      "score": 0.95
    }
  ]
}
```

### PageRank

#### Compute PageRank

```
POST /pagerank
```

Compute PageRank scores for documents with optional personalization.

**Request Body:**

```json
{
  "documents": [
    {
      "id": "doc1",
      "title": "First Document",
      "content": "Content of first document...",
      "score": 0.8
    },
    {
      "id": "doc2",
      "title": "Second Document",
      "content": "Content of second document...",
      "score": 0.6
    }
  ],
  "query": "optional query for personalization",
  "top_k": 5
}
```

**Response:**

```json
{
  "scores": {
    "doc1": 0.55,
    "doc2": 0.45
  },
  "ranked_documents": [
    {
      "id": "doc1",
      "title": "First Document",
      "content": "Content of first document...",
      "pagerank_score": 0.55
    }
  ],
  "total_documents": 2,
  "iterations": 20,
  "personalized_by_query": true
}

## Architecture

```

backend/
├── main.py # FastAPI app with all endpoints
├── scrapers/ # Web scraper modules
├── services/ # Service modules
└── pyproject.toml # Project dependencies

````

## Example Usage

### Example 1: Tavily Search

```python
import requests

# Search the web using Tavily
response = requests.post(
    "http://localhost:8000/tavily",
    json={
        "query": "machine learning best practices"
    }
)

result = response.json()
print(f"Query: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Found {len(result['results'])} results")
for r in result['results']:
    print(f"  - {r['title']}: {r['url']}")
````

### Example 2: PageRank Document Ranking

```python
# Rank documents using PageRank
response = requests.post(
    "http://localhost:8000/pagerank",
    json={
        "documents": [
            {
                "id": "doc1",
                "title": "Machine Learning Guide",
                "content": "Comprehensive guide to machine learning...",
                "score": 0.8
            },
            {
                "id": "doc2",
                "title": "Deep Learning Basics",
                "content": "Introduction to deep learning...",
                "score": 0.6
            }
        ],
        "query": "machine learning best practices",
        "top_k": 5
    }
)

result = response.json()
print(f"Total documents: {result['total_documents']}")
for doc in result['ranked_documents']:
    print(f"  - {doc['title']}: {doc['pagerank_score']:.4f}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
ruff check .
```

## Dependencies

- FastAPI: Web framework
- aiohttp: Async HTTP client
- NumPy & SciPy: Numerical computing and graph analysis
- NetworkX: Graph algorithms
- Pydantic: Data validation

## API Rate Limits

- **Tavily**: Check Tavily API documentation for rate limits

## Future Enhancements

1. **Caching**: Add Redis caching for API responses
2. **Authentication**: Add API key authentication
3. **WebSocket**: Real-time updates for long-running queries
4. **Database**: Store search results and analysis data
5. **Docker**: Containerization for easy deployment

## References

- [Tavily API Documentation](https://tavily.com/docs)
- [PageRank Algorithm](https://en.wikipedia.org/wiki/PageRank)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
