# NextGen Web Search API - Quick Reference

## Current API Structure

### Imports

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import aiohttp, numpy
```

### Request Models

```python
class TavilySearchRequest(BaseModel):
    query: str

class PageRankRequest(BaseModel):
    documents: List[Dict[str, Any]]
    query: Optional[str] = None
    top_k: Optional[int] = None
```

## Endpoints Overview

### 1. Root & Health

```
GET /
GET /health
```

### 2. Web Search

```
POST /tavily
- Searches the web using Tavily API
- Input: { "query": "your search query" }
- Output: Tavily search results with answer and URLs
```

### 3. Document Ranking

```
POST /pagerank
- Ranks documents using PageRank algorithm
- Input: documents[], optional query, optional top_k
- Output: PageRank scores, ranked documents
```

## Utility Functions

### Text Similarity

```python
def compute_text_similarity(text1: str, text2: str) -> float
- Uses Jaccard similarity (word overlap)
- Returns: 0.0 to 1.0
```

## Algorithm Details

### PageRank Implementation

- **Damping Factor**: 0.85
- **Iterations**: 20 (or convergence)
- **Personalization**: Based on query relevance (if provided)
- **Time Complexity**: O(n² + 20\*n²) = O(n²)

## Dependencies in Use

```python
# Web framework
fastapi, uvicorn, pydantic

# Async HTTP
aiohttp

# Numerical computing
numpy, scipy, networkx

# Configuration
python-dotenv
```

## Integration with External Services

### Tavily API

- **Purpose**: Web search
- **Endpoint**: https://api.tavily.com/search
- **Auth**: Bearer token (TAVILY_API_KEY)
- **Parameters**: query, search_depth, max_results, include_answer

## Configuration Required

```bash
# .env file needed
TAVILY_API_KEY=your_api_key_here
```

## File Size Summary

```
main.py: ~263 lines of code
- Imports and config: ~30 lines
- Models: ~10 lines
- Endpoints: ~150 lines
- Helper functions: ~30 lines
- Main execution: ~3 lines
```

## Code Statistics

| Component         | Lines | Status     |
| ----------------- | ----- | ---------- |
| Tavily Endpoint   | ~60   | ✅ Active  |
| PageRank Endpoint | ~90   | ✅ Active  |
| Text Similarity   | ~15   | ✅ Used    |
| Health/Root       | ~10   | ✅ Active  |
| Removed Code      | -545  | ❌ Deleted |

**Previous main.py**: 808 lines  
**New main.py**: 263 lines  
**Reduction**: 67% smaller, 100% focused
