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

# TextGrad imports
try:
    import textgrad as tg
    TEXTGRAD_AVAILABLE = True
except ImportError:
    TEXTGRAD_AVAILABLE = False
    logger.warning("TextGrad not available. Install with: pip install textgrad")

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
# Use OpenRouter with litellm (experimental engine)
tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)

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


class TextGradOptimizeRequest(BaseModel):
    """Request model for text optimization using TextGrad."""
    text: str  # The text to optimize (e.g., prompt, instruction, or content)
    objective: str  # The optimization objective or goal
    num_iterations: Optional[int] = 3  # Number of optimization iterations
    model: Optional[str] = None  # LLM model to use (defaults to gpt-3.5-turbo)


class TextGradSolutionRequest(BaseModel):
    """Request model for solution optimization using TextGrad."""
    problem: str  # The problem statement
    initial_solution: str  # The initial solution to optimize
    evaluation_criteria: List[str]  # List of criteria to evaluate against
    num_iterations: Optional[int] = 3  # Number of optimization iterations
    model: Optional[str] = None  # LLM model to use


class TextGradGenerateRequest(BaseModel):
    """Request model for generate-and-optimize using TextGrad."""
    task_description: str  # Description of the task to solve
    num_candidates: Optional[int] = 3  # Number of initial candidates
    optimization_rounds: Optional[int] = 2  # Optimization rounds for best candidate
    model: Optional[str] = None  # LLM model to use

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "NextGen Web Search API",
        "version": "1.0.0",
        "endpoints": [
            "POST /tavily",
            "POST /pagerank",
            "POST /textgrad/optimize",
            "POST /textgrad/solution",
            "POST /textgrad/generate"
        ]
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


# ============================================================================
# TextGrad Endpoints - Automatic "Differentiation" via Text
# ============================================================================

@app.post("/textgrad/optimize")
async def textgrad_optimize(request: TextGradOptimizeRequest):
    """
    Optimize text using TextGrad's automatic differentiation via text.
    
    This endpoint uses the TextGrad framework to iteratively improve text prompts,
    instructions, or content based on a specified objective. TextGrad employs
    textual gradients - natural language feedback from LLMs - to optimize the input.
    
    **Reference:**
    - Paper: "TextGrad: Automatic 'Differentiation' via Text" (Yuksekgonul et al., 2024)
    - ArXiv: https://arxiv.org/abs/2406.07496
    - Website: https://textgrad.com/
    
    **Request Body:**
    - `text` (str, required): The initial text to optimize. This can be a prompt,
      instruction, piece of content, or any text you want to improve.
    - `objective` (str, required): The optimization objective or goal. Describe what
      you want the optimized text to achieve (e.g., "be more clear and concise",
      "generate better summaries", "be more persuasive").
    - `num_iterations` (int, optional): Number of optimization iterations to perform.
      Default is 3. More iterations may yield better results but take longer.
    - `model` (str, optional): The LLM model to use for optimization. Defaults to
      "gpt-3.5-turbo". Supported models include OpenAI models (gpt-3.5-turbo,
      gpt-4, etc.) and other compatible models.
    
    **Response:**
    Returns a JSON object containing:
    - `optimized_text`: The final optimized version of your text
    - `original_text`: The original input text for comparison
    - `iterations`: Number of iterations performed
    - `improvement_history`: List showing the text evolution at each iteration
    - `objective`: The optimization objective used
    - `model_used`: The LLM model that performed the optimization
    
    **Example Usage:**
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/textgrad/optimize",
        json={
            "text": "Write a summary of the document",
            "objective": "Create clear, concise, and engaging summaries",
            "num_iterations": 3,
            "model": "gpt-3.5-turbo"
        }
    )
    result = response.json()
    print(f"Optimized: {result['optimized_text']}")
    ```
    
    **Use Cases:**
    - Prompt engineering: Optimize prompts for better LLM responses
    - Content improvement: Enhance clarity, engagement, or persuasiveness
    - Instruction refinement: Make instructions clearer and more actionable
    - Template optimization: Improve email templates, messages, or documents
    
    **Notes:**
    - Requires OpenAI API key set in OPENAI_API_KEY environment variable
    - Processing time increases with number of iterations
    - Results depend on the quality of the objective description
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        model = request.model or "experimental:openrouter/openai/gpt-3.5-turbo"
        logger.info(f"TextGrad optimize request: text length={len(request.text)}, "
                   f"objective='{request.objective[:50]}...', iterations={request.num_iterations}")
        
        # Set up TextGrad backward engine (for generating feedback/gradients)
        tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)
        
        # Create a Variable for the text to optimize
        text_var = tg.Variable(
            request.text,
            requires_grad=True,
            role_description="text content to be optimized"
        )
        
        # Create optimizer
        optimizer = tg.TGD(parameters=[text_var])
        
        # Create loss function based on the objective
        loss_fn = tg.TextLoss(request.objective)
        
        # Optimization loop
        optimization_history = []
        for i in range(request.num_iterations):
            loss = loss_fn(text_var)
            loss.backward()
            optimizer.step()
            
            optimization_history.append({
                "iteration": i + 1,
                "feedback": loss.value if hasattr(loss, 'value') else str(loss),
                "text_length": len(text_var.value)
            })
            
            logger.info(f"Iteration {i + 1}/{request.num_iterations} complete")
        
        result = {
            "original_text": request.text,
            "optimized_text": text_var.value,
            "objective": request.objective,
            "iterations": request.num_iterations,
            "optimization_history": optimization_history,
            "model_used": model
        }
        
        logger.info(f"TextGrad optimization complete. Original length: {len(request.text)}, "
                   f"Optimized length: {len(text_var.value)}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TextGrad optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/textgrad/solution")
async def textgrad_optimize_solution(request: TextGradSolutionRequest):
    """
    Optimize a solution to a problem using TextGrad with multiple evaluation criteria.
    
    This endpoint is designed for complex problem-solving scenarios where you have an
    initial solution that needs iterative improvement based on multiple criteria.
    TextGrad uses textual gradients to refine the solution across all specified criteria.
    
    **Reference:**
    - Paper: "TextGrad: Automatic 'Differentiation' via Text" (Yuksekgonul et al., 2024)
    - ArXiv: https://arxiv.org/abs/2406.07496
    - Website: https://textgrad.com/
    
    **Request Body:**
    - `problem` (str, required): The problem statement or description of what needs
      to be solved.
    - `initial_solution` (str, required): Your initial solution or approach to the
      problem. This will be iteratively improved.
    - `evaluation_criteria` (List[str], required): List of criteria to evaluate and
      optimize the solution against. Examples: ["correctness", "efficiency",
      "readability", "maintainability", "performance"].
    - `num_iterations` (int, optional): Number of optimization iterations. Default is 3.
    - `model` (str, optional): The LLM model to use. Defaults to "gpt-3.5-turbo".
    
    **Response:**
    Returns a JSON object containing:
    - `optimized_solution`: The final optimized solution
    - `original_solution`: The initial solution for comparison
    - `problem`: The problem statement
    - `iterations`: Number of iterations performed
    - `evaluation_criteria`: The criteria used for optimization
    - `improvement_log`: Detailed log of improvements at each iteration
    - `model_used`: The LLM model used
    
    **Example Usage:**
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/textgrad/solution",
        json={
            "problem": "Create a function to find all prime numbers up to n",
            "initial_solution": "def find_primes(n):\n    return [x for x in range(2, n+1) if all(x % y != 0 for y in range(2, x))]",
            "evaluation_criteria": ["correctness", "efficiency", "readability"],
            "num_iterations": 3
        }
    )
    result = response.json()
    print(f"Optimized solution: {result['optimized_solution']}")
    ```
    
    **Use Cases:**
    - Code optimization: Improve code quality, performance, and readability
    - Algorithm refinement: Enhance algorithmic solutions
    - System design: Iteratively improve architecture and design decisions
    - Problem-solving: Refine solutions to complex technical or business problems
    - Multi-objective optimization: Balance trade-offs across multiple criteria
    
    **Notes:**
    - Requires OpenAI API key in OPENAI_API_KEY environment variable
    - More evaluation criteria and iterations increase processing time
    - The quality of criteria descriptions affects optimization results
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        model = request.model or "experimental:openrouter/openai/gpt-3.5-turbo"
        logger.info(f"TextGrad solution optimization: problem length={len(request.problem)}, "
                   f"criteria={request.evaluation_criteria}, iterations={request.num_iterations}")
        
        # Set up TextGrad
        tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)
        
        # Create a Variable for the solution to optimize
        solution_var = tg.Variable(
            request.initial_solution,
            requires_grad=True,
            role_description="solution to the given problem"
        )
        
        # Create a Variable for the problem (non-trainable)
        problem_var = tg.Variable(
            request.problem,
            requires_grad=False,
            role_description="problem statement"
        )
        
        # Create optimizer
        optimizer = tg.TGD(parameters=[solution_var])
        
        # Optimization loop with multiple criteria
        improvement_log = []
        for i in range(request.num_iterations):
            iteration_feedback = []
            
            for criterion in request.evaluation_criteria:
                # Create loss function for each criterion
                evaluation_instruction = (
                    f"Evaluate the following solution to this problem: {request.problem}\n\n"
                    f"Focus on: {criterion}\n"
                    f"Provide specific, actionable feedback for improvement."
                )
                loss_fn = tg.TextLoss(evaluation_instruction)
                
                # Compute loss and collect feedback
                loss = loss_fn(solution_var)
                iteration_feedback.append({
                    "criterion": criterion,
                    "feedback": loss.value if hasattr(loss, 'value') else str(loss)
                })
                
                # Backward pass (accumulates gradients)
                loss.backward()
            
            # Update the solution
            optimizer.step()
            
            improvement_log.append({
                "iteration": i + 1,
                "criteria_feedback": iteration_feedback,
                "solution_length": len(solution_var.value)
            })
            
            logger.info(f"Iteration {i + 1}/{request.num_iterations} complete")
        
        result = {
            "problem": request.problem,
            "original_solution": request.initial_solution,
            "optimized_solution": solution_var.value,
            "evaluation_criteria": request.evaluation_criteria,
            "iterations": request.num_iterations,
            "improvement_log": improvement_log,
            "model_used": model
        }
        
        logger.info(f"TextGrad solution optimization complete")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TextGrad solution optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/textgrad/generate")
async def textgrad_generate_and_optimize(request: TextGradGenerateRequest):
    """
    Generate multiple candidate solutions and optimize the best one using TextGrad.
    
    This endpoint combines solution generation with optimization. It's useful when you
    want to explore multiple approaches to a task and then refine the most promising one.
    TextGrad helps both in the exploration and refinement phases.
    
    **Reference:**
    - Paper: "TextGrad: Automatic 'Differentiation' via Text" (Yuksekgonul et al., 2024)
    - ArXiv: https://arxiv.org/abs/2406.07496
    - Website: https://textgrad.com/
    
    **Request Body:**
    - `task_description` (str, required): Description of the task to solve. Be specific
      about what you want to accomplish.
    - `num_candidates` (int, optional): Number of initial candidate solutions to generate.
      Default is 3. More candidates increase exploration but take longer.
    - `optimization_rounds` (int, optional): Number of optimization rounds to apply to
      the selected best candidate. Default is 2.
    - `model` (str, optional): The LLM model to use. Defaults to "gpt-3.5-turbo".
    
    **Response:**
    Returns a JSON object containing:
    - `task_description`: The original task description
    - `candidates`: List of all initial candidate solutions generated
    - `best_candidate`: The candidate selected for optimization
    - `optimized_result`: The final optimized solution with full optimization details
    - `selection_rationale`: Explanation of why this candidate was chosen
    - `model_used`: The LLM model used
    
    **Example Usage:**
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/textgrad/generate",
        json={
            "task_description": "Write a function to efficiently check if a string is a palindrome",
            "num_candidates": 5,
            "optimization_rounds": 3,
            "model": "gpt-4"
        }
    )
    result = response.json()
    print(f"Best optimized solution: {result['optimized_result']['optimized_text']}")
    ```
    
    **Use Cases:**
    - Creative problem solving: Explore multiple approaches before settling on one
    - Code generation: Generate and refine code solutions
    - Content creation: Create multiple drafts and optimize the best one
    - Strategy development: Generate strategic options and refine the top choice
    - A/B testing preparation: Generate variants for testing
    
    **Workflow:**
    1. Generate `num_candidates` initial solutions based on task description
    2. Evaluate candidates (currently selects the first; can be extended with ranking)
    3. Apply TextGrad optimization to the best candidate for `optimization_rounds`
    4. Return all candidates plus the optimized best solution
    
    **Notes:**
    - Requires OpenAI API key in OPENAI_API_KEY environment variable
    - Processing time scales with num_candidates × optimization_rounds
    - Consider starting with fewer candidates and rounds, then increasing as needed
    - The current implementation uses a simplified candidate selection; production
      versions could implement more sophisticated ranking
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        model = request.model or "experimental:openrouter/openai/gpt-3.5-turbo"
        logger.info(f"TextGrad generate-and-optimize: task='{request.task_description[:50]}...', "
                   f"candidates={request.num_candidates}, rounds={request.optimization_rounds}")
        
        # Set up TextGrad
        tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)
        
        # Create model for generation using the model string directly
        generation_model = tg.BlackboxLLM(model)
        
        # Generate multiple candidates
        candidates = []
        task_var = tg.Variable(
            request.task_description,
            requires_grad=False,
            role_description="task description"
        )
        
        for i in range(request.num_candidates):
            candidate_prompt = f"Provide a solution for the following task:\n{request.task_description}\n\nProvide solution {i+1}:"
            candidate_var = tg.Variable(
                candidate_prompt,
                requires_grad=False,
                role_description="generation prompt"
            )
            
            response = generation_model(candidate_var)
            candidates.append({
                "candidate_id": i + 1,
                "solution": response.value
            })
            logger.info(f"Generated candidate {i + 1}/{request.num_candidates}")
        
        # Select best candidate (simplified: use first candidate)
        # In production, you might want to rank them first
        best_candidate = candidates[0]
        selection_rationale = "Selected first candidate for optimization (can be extended with ranking)"
        
        # Optimize the best candidate
        solution_var = tg.Variable(
            best_candidate["solution"],
            requires_grad=True,
            role_description="solution to optimize"
        )
        
        optimizer = tg.TGD(parameters=[solution_var])
        
        # Create evaluation instruction
        evaluation_instruction = (
            f"Evaluate the following solution for this task: {request.task_description}\n"
            f"Provide specific, actionable feedback for improvement. "
            f"Focus on correctness, clarity, and completeness."
        )
        loss_fn = tg.TextLoss(evaluation_instruction)
        
        # Optimization loop
        optimization_history = []
        for i in range(request.optimization_rounds):
            loss = loss_fn(solution_var)
            loss.backward()
            optimizer.step()
            
            optimization_history.append({
                "round": i + 1,
                "feedback": loss.value if hasattr(loss, 'value') else str(loss),
                "solution_length": len(solution_var.value)
            })
            
            logger.info(f"Optimization round {i + 1}/{request.optimization_rounds} complete")
        
        result = {
            "task_description": request.task_description,
            "candidates": candidates,
            "best_candidate": best_candidate,
            "selection_rationale": selection_rationale,
            "optimized_result": {
                "original_text": best_candidate["solution"],
                "optimized_text": solution_var.value,
                "optimization_history": optimization_history
            },
            "model_used": model
        }
        
        logger.info(f"TextGrad generate-and-optimize complete")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TextGrad generate-and-optimize: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
