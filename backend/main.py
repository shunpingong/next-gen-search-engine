from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Import service modules (to be created)
from services.github_service import GitHubService
from services.osv_service import OSVService
from services.nvd_service import NVDService
from services.pubmed_service import PubMedService
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
pubmed_service = PubMedService()
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
            "pubmed": True,  # Public API
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
    max_results: int = Query(10, ge=1, le=100)
):
    """Search GitHub for Linux kernel related content"""
    try:
        results = await github_service.search(
            query=query,
            search_type=search_type,
            max_results=max_results
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
# DOMAIN 2: Healthcare Search Endpoints
# ============================================================================

@app.post("/api/healthcare/query")
async def healthcare_query(request: HealthcareQueryRequest):
    """
    Main TAG endpoint for healthcare research.
    
    Workflow:
    1. Search PubMed for relevant literature
    2. Search local documents (if enabled)
    3. Build citation graph
    4. Compute PageRank/CheiRank on papers
    5. Use local LLM for clinician-friendly summary
    """
    try:
        logger.info(f"Healthcare query: {request.query}")
        
        # Step 1: PubMed Search
        pubmed_results = await pubmed_service.search(
            query=request.query,
            specialty=request.specialty,
            max_results=request.max_results
        )
        
        # Step 2: Local document search (placeholder)
        local_docs = []
        if request.include_local_docs:
            # TODO: Implement local document search
            pass
        
        # Step 3: Build citation graph
        citation_graph = await graph_service.build_citation_graph(
            papers=pubmed_results.get("articles", [])
        )
        
        # Step 4: LLM Synthesis
        llm_summary = await llm_service.synthesize_healthcare_results(
            query=request.query,
            pubmed_data=pubmed_results,
            local_docs=local_docs,
            citation_graph=citation_graph
        )
        
        return {
            "query": request.query,
            "pubmed_results": pubmed_results,
            "local_documents": local_docs,
            "citation_analysis": citation_graph,
            "llm_synthesis": llm_summary,
            "clinical_recommendations": llm_summary.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error in healthcare query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/healthcare/pubmed/search")
async def pubmed_search(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=100),
    sort: str = Query("relevance", description="relevance, date, or citations")
):
    """Search PubMed literature"""
    try:
        results = await pubmed_service.search(
            query=query,
            max_results=max_results,
            sort=sort
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/healthcare/pubmed/article/{pmid}")
async def get_pubmed_article(pmid: str):
    """Get detailed article information from PubMed"""
    try:
        results = await pubmed_service.get_article(pmid)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/healthcare/pubmed/related/{pmid}")
async def get_related_articles(
    pmid: str,
    max_results: int = Query(10, ge=1, le=50)
):
    """Get related articles from PubMed"""
    try:
        results = await pubmed_service.get_related(pmid, max_results)
        return results
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
            },
            {
                "id": "healthcare-search",
                "name": "Healthcare Literature Search",
                "description": "Medical literature and clinical guideline search for healthcare professionals",
                "tools": ["PubMed", "OpenAlex", "Local Documents", "Local LLM"],
                "use_cases": ["Clinical decision support", "Literature review", "Treatment protocols"]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
