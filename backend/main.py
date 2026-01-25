from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import aiohttp
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NextGen Web Search API",
    description="Advanced web search engine with PageRank-based result ranking and Tavily integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class TavilySearchRequest(BaseModel):
    query: str


class PageRankRequest(BaseModel):
    documents: List[Dict[str, Any]]
    query: Optional[str] = None  # Optional: query for personalizing PageRank
    top_k: Optional[int] = None  # Optional: return only top-k results

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "NextGen Web Search API",
        "version": "1.0.0",
        "endpoints": ["POST /tavily", "POST /pagerank"]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ============================================================================
# Tavily Search Endpoint
# ============================================================================

@app.post("/tavily")
async def tavily_search(request: TavilySearchRequest):
    """
    Tavily search endpoint - proxy for Tavily API.
    Converts query to Tavily API format and returns results.
    """
    try:
        query = request.query
        
        logger.info(f"Received query: {query}")
        
        # Trim query if too long
        MAX_QUERY_LENGTH = 400
        trimmed_query = query[:MAX_QUERY_LENGTH] if len(query) > MAX_QUERY_LENGTH else query
        
        # Get Tavily API credentials from environment
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        TAVILY_ENDPOINT = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")
        
        if not TAVILY_API_KEY:
            raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured")
        
        # Make request to Tavily API
        async with aiohttp.ClientSession() as session:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TAVILY_API_KEY}"
            }
            
            payload = {
                "query": trimmed_query,
                "search_depth": "advanced",
                "max_results": 5,
                "include_answer": True,
                "include_raw_content": True
            }
            
            async with session.post(TAVILY_ENDPOINT, json=payload, headers=headers) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Tavily API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)
                
                data = await response.json()
                
                logger.info("Tavily API Success")
                logger.info(data)
                return data
                
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Server Error: {str(error)}")
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================================
# Utility Functions
# ============================================================================

def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple similarity score between two texts.
    Uses word overlap as a basic similarity metric.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# PageRank Endpoint
# ============================================================================

@app.post("/pagerank")
async def pagerank_endpoint(request: PageRankRequest):
    """
    Compute PageRank scores for a list of documents.
    
    Input: 
    - documents: Array of documents with fields like id, title, url, content, score
    - query (optional): Query string for personalizing PageRank based on relevance
    - top_k (optional): Return only top-k results
    
    Output: Dictionary of document IDs mapped to their PageRank scores
    
    Algorithm:
    1. Build a graph where nodes are documents
    2. Edge weights are based on content similarity
    3. Create relevance scores:
       - If query provided: relevance = query similarity
       - Otherwise: use document scores field
    4. Apply PageRank with personalization based on relevance
    5. Return scores indexed by document ID
    """
    try:
        documents = request.documents
        query = request.query
        top_k = request.top_k
        
        logger.info(f"PageRank computation for {len(documents)} documents" + 
                   (f" with query: {query}" if query else ""))
        
        if not documents:
            return {"scores": {}, "ranked_documents": []}
        
        n = len(documents)
        
        # Extract document IDs
        doc_ids = [doc.get("id", f"doc-{i}") for i, doc in enumerate(documents)]
        
        # Create relevance scores
        relevance_scores = []
        if query:
            # If query provided, score based on query similarity
            for doc in documents:
                text = f"{doc.get('title', '')} {doc.get('content', '')}"
                similarity = compute_text_similarity(query, text)
                relevance_scores.append(similarity)
        else:
            # Otherwise use document scores field
            for doc in documents:
                score = doc.get("score", 0.5)
                if isinstance(score, (int, float)):
                    relevance_scores.append(float(score))
                else:
                    relevance_scores.append(0.5)
        
        # Build adjacency matrix based on content similarity between documents
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    text_i = f"{documents[i].get('title', '')} {documents[i].get('content', '')}"
                    text_j = f"{documents[j].get('title', '')} {documents[j].get('content', '')}"
                    similarity = compute_text_similarity(text_i, text_j)
                    adjacency[i][j] = similarity
        
        # Normalize adjacency matrix (column stochastic)
        col_sums = adjacency.sum(axis=0)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        adjacency = adjacency / col_sums
        
        # PageRank algorithm with personalization
        damping = 0.85
        pagerank = np.ones(n) / n
        relevance_array = np.array(relevance_scores)
        relevance_array = relevance_array / (relevance_array.sum() + 1e-8)  # Normalize with epsilon
        
        # Iterate PageRank
        for _ in range(20):  # 20 iterations
            new_pagerank = (1 - damping) * relevance_array + damping * adjacency.dot(pagerank)
            if np.allclose(new_pagerank, pagerank):
                break
            pagerank = new_pagerank
        
        # Create scores dictionary indexed by document ID
        scores = {doc_ids[i]: float(pagerank[i]) for i in range(n)}
        
        # If top_k is specified, also return ranked documents
        ranked_documents = []
        if top_k is not None:
            ranked_indices = np.argsort(pagerank)[::-1][:top_k]
            for idx in ranked_indices:
                doc = documents[int(idx)].copy()
                doc['pagerank_score'] = float(pagerank[idx])
                ranked_documents.append(doc)
        
        logger.info(f"PageRank computation complete. Top score: {max(pagerank):.4f}")
        
        return {
            "scores": scores,
            "ranked_documents": ranked_documents if top_k else None,
            "total_documents": n,
            "iterations": 20,
            "personalized_by_query": query is not None
        }
        
    except Exception as e:
        logger.error(f"Error in PageRank computation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
