from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
import aiohttp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import service modules (to be created)
from services.github_service import GitHubService
from services.osv_service import OSVService
from services.nvd_service import NVDService
from services.graph_service import GraphService
from services.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ghost Network API",
    description="Tool-Augmented Generation API for Linux Kernel Security and Healthcare Research",
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

# Initialize services
github_service = GitHubService()
osv_service = OSVService()
nvd_service = NVDService()
graph_service = GraphService()
llm_service = LLMService()


# ============================================================================
# Request/Response Models
# ============================================================================

class SecurityQueryRequest(BaseModel):
    query: str
    domain: str = "linux-kernel"  # or "networking", "filesystem", etc.
    max_results: int = 10
    include_vulnerabilities: bool = True
    compute_graph: bool = True


class HealthcareQueryRequest(BaseModel):
    query: str
    specialty: Optional[str] = None
    max_results: int = 10
    include_local_docs: bool = False


class GraphBuildRequest(BaseModel):
    repo_owner: str
    repo_name: str
    file_pattern: Optional[str] = "*.c"
    compute_pagerank: bool = True
    compute_cheirank: bool = True


class VulnerabilityCheckRequest(BaseModel):
    packages: List[str]
    ecosystem: str = "Linux"


class TavilySearchRequest(BaseModel):
    query: str


class EnhancedSearchRequest(BaseModel):
    query: str
    top_k: int = 5  # Number of top results to return after ranking

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "Ghost Network API - Tool-Augmented Generation for Security Research",
        "version": "1.0.0",
        "domains": ["linux-kernel-security", "healthcare-search"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "github": github_service.is_configured(),
            "osv": True,  # Public API
            "nvd": True,  # Public API
            "llm": llm_service.is_available()
        }
    }


# ============================================================================
# DOMAIN 1: Linux Kernel Security Endpoints
# ============================================================================

@app.post("/api/security/query")
async def security_query(request: SecurityQueryRequest):
    """
    Main TAG endpoint for Linux kernel security research.
    
    Workflow:
    1. Search GitHub for relevant repos/issues
    2. Query OSV/NVD for vulnerabilities
    3. Build call graph if requested
    4. Compute PageRank/CheiRank
    5. Use local LLM to synthesize results
    """
    try:
        logger.info(f"Security query: {request.query}")
        
        # Step 1: GitHub Search
        github_results = await github_service.search_security_issues(
            query=request.query,
            domain=request.domain,
            max_results=request.max_results
        )
        
        # Step 2: Vulnerability Check
        vulnerabilities = []
        if request.include_vulnerabilities:
            # Extract package names from GitHub results
            packages = github_service.extract_packages(github_results)
            vulnerabilities = await osv_service.check_vulnerabilities(packages)
            
        # Step 3: Build and analyze graph
        graph_analysis = None
        if request.compute_graph and github_results.get("repos"):
            top_repo = github_results["repos"][0]
            graph_analysis = await graph_service.build_and_analyze(
                owner=top_repo["owner"],
                name=top_repo["name"],
                file_pattern="*.c"
            )
        
        # Step 4: LLM Synthesis
        llm_summary = await llm_service.synthesize_security_results(
            query=request.query,
            github_data=github_results,
            vulnerabilities=vulnerabilities,
            graph_analysis=graph_analysis
        )
        
        return {
            "query": request.query,
            "github_results": github_results,
            "vulnerabilities": vulnerabilities,
            "graph_analysis": graph_analysis,
            "llm_synthesis": llm_summary,
            "recommendations": llm_summary.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error in security query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/github/search")
async def github_search(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("repositories", description="repositories, issues, or code"),
    max_results: int = Query(10, ge=1, le=100),
    language: str = Query("en", description="Language filter (en for English)")  # Add parameter
):
    """Search GitHub for Linux kernel related content"""
    try:
        results = await github_service.search(
            query=query,
            search_type=search_type,
            max_results=max_results,
            language=language  # Pass to service
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/github/repo/{owner}/{repo}/issues")
async def get_repo_issues(
    owner: str,
    repo: str,
    labels: Optional[str] = Query(None, description="Comma-separated labels"),
    state: str = Query("open", description="open, closed, or all")
):
    """Get issues from a specific repository"""
    try:
        results = await github_service.get_repo_issues(
            owner=owner,
            repo=repo,
            labels=labels.split(",") if labels else None,
            state=state
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/github/repo/{owner}/{repo}/files")
async def get_repo_files(
    owner: str,
    repo: str,
    path: str = Query("", description="Path within repo"),
    pattern: Optional[str] = Query(None, description="File pattern (e.g., *.c)")
):
    """Get file list from repository"""
    try:
        results = await github_service.get_repo_files(
            owner=owner,
            repo=repo,
            path=path,
            pattern=pattern
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/security/vulnerabilities/check")
async def check_vulnerabilities(request: VulnerabilityCheckRequest):
    """Check vulnerabilities using OSV.dev API"""
    try:
        results = await osv_service.check_vulnerabilities(
            packages=request.packages,
            ecosystem=request.ecosystem
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/vulnerabilities/cve/{cve_id}")
async def get_cve_details(cve_id: str):
    """Get CVE details from NVD"""
    try:
        results = await nvd_service.get_cve(cve_id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/vulnerabilities/search")
async def search_vulnerabilities(
    keyword: str = Query(..., description="Search keyword"),
    max_results: int = Query(20, ge=1, le=100)
):
    """Search vulnerabilities in NVD"""
    try:
        results = await nvd_service.search_vulnerabilities(
            keyword=keyword,
            max_results=max_results
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/security/graph/build")
async def build_call_graph(request: GraphBuildRequest):
    """Build call graph for a repository and compute PageRank/CheiRank"""
    try:
        results = await graph_service.build_and_analyze(
            owner=request.repo_owner,
            name=request.repo_name,
            file_pattern=request.file_pattern,
            compute_pagerank=request.compute_pagerank,
            compute_cheirank=request.compute_cheirank
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/fuzzing/suggestions")
async def get_fuzzing_suggestions(
    repo_owner: str,
    repo_name: str,
    focus_area: Optional[str] = Query(None, description="e.g., networking, filesystem")
):
    """
    Get AI-generated fuzzing suggestions based on call graph analysis
    and known vulnerabilities
    """
    try:
        # Build graph
        graph_analysis = await graph_service.build_and_analyze(
            owner=repo_owner,
            name=repo_name,
            file_pattern="*.c"
        )
        
        # Check vulnerabilities
        vulnerabilities = await osv_service.check_vulnerabilities(
            packages=[f"{repo_owner}/{repo_name}"]
        )
        
        # Generate suggestions using LLM
        suggestions = await llm_service.generate_fuzzing_strategy(
            graph_analysis=graph_analysis,
            vulnerabilities=vulnerabilities,
            focus_area=focus_area
        )
        
        return suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LLM/AI Endpoints
# ============================================================================

@app.post("/api/llm/summarize")
async def llm_summarize(
    text: str = Query(..., description="Text to summarize"),
    max_length: int = Query(200, description="Maximum summary length")
):
    """Summarize text using local LLM"""
    try:
        summary = await llm_service.summarize(text, max_length)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/llm/qa")
async def llm_question_answering(
    question: str = Query(..., description="Question"),
    context: str = Query(..., description="Context to answer from")
):
    """Answer question using local LLM with provided context"""
    try:
        answer = await llm_service.answer_question(question, context)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Graph Analysis Endpoints
# ============================================================================

@app.get("/api/graph/pagerank")
async def compute_pagerank(
    nodes: str = Query(..., description="Comma-separated node IDs"),
    edges: str = Query(..., description="Comma-separated edges (from-to pairs)")
):
    """Compute PageRank for a given graph"""
    try:
        node_list = nodes.split(",")
        edge_list = [tuple(e.split("-")) for e in edges.split(",")]
        
        results = graph_service.compute_pagerank(node_list, edge_list)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/cheirank")
async def compute_cheirank(
    nodes: str = Query(..., description="Comma-separated node IDs"),
    edges: str = Query(..., description="Comma-separated edges (from-to pairs)")
):
    """Compute CheiRank for a given graph"""
    try:
        node_list = nodes.split(",")
        edge_list = [tuple(e.split("-")) for e in edges.split(",")]
        
        results = graph_service.compute_cheirank(node_list, edge_list)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/api/domains")
async def get_supported_domains():
    """Get list of supported research domains"""
    return {
        "domains": [
            {
                "id": "linux-kernel-security",
                "name": "Linux Kernel Security",
                "description": "Security research, fuzzing, and vulnerability analysis for Linux kernel",
                "tools": ["GitHub", "OSV.dev", "NVD", "Local LLM"],
                "use_cases": ["Vulnerability discovery", "Fuzzing strategy", "Code hotspot analysis"]
            }
        ]
    }


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
# Enhanced Tavily Search with PageRank and LLM Summarization
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


def rank_search_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rank search results using a PageRank-inspired algorithm.
    
    Strategy:
    1. Build a graph where nodes are search results
    2. Edge weights are based on:
       - Content similarity to query
       - Original Tavily score
       - Cross-references between results
    3. Apply PageRank to get final ranking
    4. Return top-k results
    """
    if not results:
        return []
    
    n = len(results)
    if n <= top_k:
        return results
    
    # Initialize adjacency matrix for PageRank
    # Start with uniform distribution
    import numpy as np
    
    # Create relevance scores based on query similarity
    relevance_scores = []
    for result in results:
        # Combine title and content for relevance scoring
        text = f"{result.get('title', '')} {result.get('content', '')}"
        similarity = compute_text_similarity(query, text)
        
        # Combine with original Tavily score if available
        tavily_score = result.get('score', 0.5)
        combined_score = 0.6 * similarity + 0.4 * tavily_score
        relevance_scores.append(combined_score)
    
    # Build adjacency matrix based on content similarity
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                text_i = f"{results[i].get('title', '')} {results[i].get('content', '')}"
                text_j = f"{results[j].get('title', '')} {results[j].get('content', '')}"
                similarity = compute_text_similarity(text_i, text_j)
                adjacency[i][j] = similarity
    
    # Normalize adjacency matrix (column stochastic)
    col_sums = adjacency.sum(axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    adjacency = adjacency / col_sums
    
    # PageRank algorithm
    damping = 0.85
    pagerank = np.ones(n) / n
    relevance_array = np.array(relevance_scores)
    relevance_array = relevance_array / relevance_array.sum()  # Normalize
    
    # Iterate PageRank with personalization (query relevance)
    for _ in range(20):  # 20 iterations
        new_pagerank = (1 - damping) * relevance_array + damping * adjacency.dot(pagerank)
        if np.allclose(new_pagerank, pagerank):
            break
        pagerank = new_pagerank
    
    # Sort results by PageRank score
    ranked_indices = np.argsort(pagerank)[::-1][:top_k]
    
    # Add ranking scores to results
    ranked_results = []
    for idx in ranked_indices:
        result = results[int(idx)].copy()
        result['pagerank_score'] = float(pagerank[idx])
        result['relevance_score'] = float(relevance_scores[idx])
        ranked_results.append(result)
    
    return ranked_results


@app.post("/api/search/enhanced")
async def enhanced_search(request: EnhancedSearchRequest):
    """
    Enhanced search endpoint that:
    1. Queries Tavily API
    2. Extracts query, answer, title, and content
    3. Ranks results using PageRank based on query relevance
    4. Returns top-k results
    5. Optionally generates LLM summary
    """
    try:
        query = request.query
        top_k = request.top_k
        
        logger.info(f"Enhanced search query: {query}")
        
        # Step 1: Query Tavily API
        MAX_QUERY_LENGTH = 400
        trimmed_query = query[:MAX_QUERY_LENGTH] if len(query) > MAX_QUERY_LENGTH else query
        
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        TAVILY_ENDPOINT = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")
        
        if not TAVILY_API_KEY:
            raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured")
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TAVILY_API_KEY}"
            }
            
            payload = {
                "query": trimmed_query,
                "search_depth": "advanced",
                "max_results": 10,  # Get more results than top_k for better ranking
                "include_answer": True,
                "include_raw_content": False
            }
            
            async with session.post(TAVILY_ENDPOINT, json=payload, headers=headers) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Tavily API Error: {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)
                
                tavily_data = await response.json()
        
        # Step 2: Extract relevant fields
        extracted_query = tavily_data.get("query", query)
        tavily_answer = tavily_data.get("answer", "")
        results = tavily_data.get("results", [])
        
        # Extract title and content from each result
        extracted_results = []
        for result in results:
            extracted_results.append({
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0)
            })
        
        logger.info(f"Extracted {len(extracted_results)} results from Tavily")
        
        # Step 3: Rank results using PageRank
        ranked_results = rank_search_results(extracted_query, extracted_results, top_k)
        
        logger.info(f"Ranked top {len(ranked_results)} results using PageRank")
        
        # Step 4: Generate LLM summary if requested
        if ranked_results:
            # Prepare context from top-k results
            context_parts = [f"Query: {extracted_query}\n"]
            
            if tavily_answer:
                context_parts.append(f"Initial Answer: {tavily_answer}\n")
            
            context_parts.append("\nTop Ranked Sources:\n")
            for i, result in enumerate(ranked_results, 1):
                context_parts.append(
                    f"\n{i}. {result['title']}\n"
                    f"   URL: {result['url']}\n"
                    f"   Content: {result['content'][:500]}...\n"
                    f"   PageRank Score: {result['pagerank_score']:.4f}\n"
                )
            
            context = "\n".join(context_parts)
        
        
        # Step 5: Return comprehensive response
        return {
            "query": extracted_query,
            "tavily_answer": tavily_answer,
            "top_k": top_k,
            "ranked_results": ranked_results,
            "context": context,
            "total_results_analyzed": len(extracted_results),
            "ranking_method": "PageRank with query relevance personalization"
        }
        
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Enhanced search error: {str(error)}")
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
