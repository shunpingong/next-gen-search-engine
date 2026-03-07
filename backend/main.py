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

# Configure logging (avoid duplicate records via root/uvicorn propagation)
logger = logging.getLogger("main")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# TextGrad imports
try:
    import textgrad as tg
    TEXTGRAD_AVAILABLE = True
except ImportError:
    TEXTGRAD_AVAILABLE = False
    logger.warning("TextGrad not available. Install with: pip install textgrad")

# ============================================================================
# OpenAI API Configuration
# ============================================================================
# Set OpenAI API key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
elif TEXTGRAD_AVAILABLE:
    logger.warning("OPENAI_API_KEY not configured. TextGrad endpoints will fail until it is set.")

BACKWARD_MODEL = os.getenv("TEXTGRAD_BACKWARD_MODEL", "gpt-4o")
ADVANCED_MODEL_NAME = os.getenv("TEXTGRAD_ADVANCED_MODEL", "gpt-4o")
FALLBACK_MODEL = "gpt-4o"
ACTIVE_BACKWARD_MODEL = BACKWARD_MODEL
ACTIVE_ADVANCED_MODEL_NAME = ADVANCED_MODEL_NAME


def set_backward_engine_safe():
    """Set backward engine and gracefully fallback if configured engine is unsupported."""
    global ACTIVE_BACKWARD_MODEL
    if not TEXTGRAD_AVAILABLE:
        return
    try:
        tg.set_backward_engine(ACTIVE_BACKWARD_MODEL, override=True)
    except Exception as e:
        if ACTIVE_BACKWARD_MODEL != FALLBACK_MODEL:
            logger.warning(
                f"Backward engine '{ACTIVE_BACKWARD_MODEL}' unsupported ({e}). "
                f"Falling back to '{FALLBACK_MODEL}'."
            )
            ACTIVE_BACKWARD_MODEL = FALLBACK_MODEL
            tg.set_backward_engine(ACTIVE_BACKWARD_MODEL, override=True)
        else:
            raise


if TEXTGRAD_AVAILABLE:
    set_backward_engine_safe()

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
    max_results: Optional[int] = 10


def _has_usable_answer(answer: Any) -> bool:
    """Heuristic to decide if Tavily returned a useful direct answer."""
    if not isinstance(answer, str):
        return False
    normalized = answer.strip().lower()
    if not normalized:
        return False
    weak_answers = {
        "i don't know",
        "i do not know",
        "not enough information",
        "no answer found",
        "unknown",
    }
    return normalized not in weak_answers


def _build_retrieval_plan(requested_max: int) -> List[int]:
    """
    Build an increasing retrieval plan that minimizes result count first,
    but can scale up to the requested maximum.
    """
    stages = [3, 5, 10]
    plan = [stage for stage in stages if stage <= requested_max]
    if not plan or plan[-1] != requested_max:
        plan.append(requested_max)
    return plan


class PageRankRequest(BaseModel):
    documents: List[Dict[str, Any]]
    query: Optional[str] = None  # Optional: query for personalizing PageRank
    top_k: Optional[int] = None  # Optional: return only top-k results


class QueryRequest(BaseModel):
    """Request model for query refinement."""
    query: str
    max_iterations: Optional[int] = 3


class AnswerRequest(BaseModel):
    """Request model for answer refinement."""
    question: str
    context: str
    initial_answer: Optional[str] = None  # Optional: if provided, refine this answer instead of generating one
    max_iterations: Optional[int] = 3


class PlanRequest(BaseModel):
    """Request model for plan refinement."""
    user_query: str
    execution_feedback: str
    max_iterations: Optional[int] = 3


class PromptOptimizeRequest(BaseModel):
    """Request model for prompt optimization."""
    eval_inputs: List[str]
    desired_behavior: str
    max_iterations: Optional[int] = 10

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "NextGen Web Search API with TextGrad Optimization",
        "version": "2.0.0",
        "endpoints": [
            "POST /tavily",
            "POST /pagerank",
            "POST /textgrad/refine-query - Optimize search queries",
            "POST /textgrad/refine-answer - Improve reasoning and correctness",
            "POST /textgrad/refine-plan - Improve multi-step tool plans",
            "POST /textgrad/optimize-prompt - Offline system prompt tuning"
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
        
        # Clamp max_results to Tavily limit requested by user requirements
        requested_max_results = request.max_results if request.max_results is not None else 10
        requested_max_results = max(1, min(requested_max_results, 10))
        retrieval_plan = _build_retrieval_plan(requested_max_results)
        logger.info(
            f"Tavily retrieval plan: {retrieval_plan} (requested max={requested_max_results})"
        )

        # Make staged requests to Tavily API: retrieve minimally, expand only if needed
        async with aiohttp.ClientSession() as session:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TAVILY_API_KEY}"
            }

            latest_data = None
            for stage_max_results in retrieval_plan:
                payload = {
                    "query": trimmed_query,
                    "search_depth": "advanced",
                    "max_results": stage_max_results,
                    "include_answer": True,
                    "include_raw_content": True
                }

                async with session.post(TAVILY_ENDPOINT, json=payload, headers=headers) as response:
                    if not response.ok:
                        error_text = await response.text()
                        logger.error(f"Tavily API Error: {error_text}")
                        raise HTTPException(status_code=response.status, detail=error_text)

                    data = await response.json()
                    latest_data = data

                results = data.get("results", [])
                answer = data.get("answer")

                logger.info(
                    f"Tavily stage complete: max_results={stage_max_results}, "
                    f"retrieved_results={len(results)}, has_answer={_has_usable_answer(answer)}"
                )
                for idx, result in enumerate(results, start=1):
                    logger.info(
                        "Tavily result stage=%s idx=%s title=%s url=%s score=%s",
                        stage_max_results,
                        idx,
                        result.get("title", ""),
                        result.get("url", ""),
                        result.get("score", "n/a"),
                    )

                if _has_usable_answer(answer):
                    logger.info(
                        f"Stopping retrieval early at max_results={stage_max_results} "
                        "because a usable answer was found."
                    )
                    return data

            logger.info(
                f"Reached max retrieval stage ({requested_max_results}). Returning latest results."
            )
            return latest_data
                
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
# TextGrad Endpoints - LLM-as-Judge Optimization for FYP
# ============================================================================

# Global system prompt for advanced endpoints
if TEXTGRAD_AVAILABLE:
    SYSTEM_PROMPT = tg.Variable(
        "You are an AI search and reasoning assistant for a final-year project focused on retrieval quality. "
        "Your objective is to maximize factual correctness, query intent preservation, and evidence grounding. "
        "When refining queries, keep the exact user intent, preserve location/personal context, and avoid adding new constraints or invented details. "
        "When refining answers, use only information explicitly present in the provided context and include all relevant facts needed for completeness. "
        "Never hallucinate facts, sources, locations, dates, or numbers. "
        "If context is missing or insufficient, state the limitation explicitly instead of guessing. "
        "Prefer concise, structured outputs with clear reasoning and actionable content. "
        "Write in a neutral, professional tone optimized for reliability over creativity. "
        "Do not reveal chain-of-thought; provide brief justification only when it improves verifiability.",
        requires_grad=True,
        role_description="System prompt to the language model"
    )
    ADVANCED_OPTIMIZER = tg.TextualGradientDescent(parameters=[SYSTEM_PROMPT])
else:
    SYSTEM_PROMPT = None
    ADVANCED_OPTIMIZER = None

# This model is used for query/answer/plan refinement and prompt optimization
ADVANCED_MODEL = None  # Will be initialized on first use

def get_advanced_model():
    """Lazy initialization of advanced model."""
    global ADVANCED_MODEL, ACTIVE_ADVANCED_MODEL_NAME
    if not TEXTGRAD_AVAILABLE:
        raise RuntimeError("TextGrad not available")
    if ADVANCED_MODEL is None:
        try:
            ADVANCED_MODEL = tg.BlackboxLLM(
                ACTIVE_ADVANCED_MODEL_NAME,
                system_prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            if ACTIVE_ADVANCED_MODEL_NAME != FALLBACK_MODEL:
                logger.warning(
                    f"Advanced model '{ACTIVE_ADVANCED_MODEL_NAME}' unsupported ({e}). "
                    f"Falling back to '{FALLBACK_MODEL}'."
                )
                ACTIVE_ADVANCED_MODEL_NAME = FALLBACK_MODEL
                ADVANCED_MODEL = tg.BlackboxLLM(
                    ACTIVE_ADVANCED_MODEL_NAME,
                    system_prompt=SYSTEM_PROMPT
                )
            else:
                raise
    return ADVANCED_MODEL

    
@app.post("/textgrad/refine-query")
async def refine_query(request: QueryRequest):
    """
    Refine a search query using TextGrad optimization with LLM-as-judge.
    
    This endpoint uses an iterative optimization process to improve search queries by:
    - Making them more specific and clear
    - Better suited for web search
    - More likely to return relevant results
    
    The optimization uses an LLM-as-judge approach to evaluate and improve the query.
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        logger.info(f"Query refinement request: '{request.query[:50]}...', iterations={request.max_iterations}")
        
        # CHANGE BACKWARD ENGINE HERE: Set model for query evaluation
        set_backward_engine_safe()
        
        model = get_advanced_model()
        original_query = request.query
        
        # Create a variable for the query
        query_var = tg.Variable(
            original_query,
            requires_grad=True,
            role_description="search query to be refined"
        )
        
        optimizer = tg.TextualGradientDescent(parameters=[query_var])
        
        # Optimization loop following paper pattern
        for iteration in range(request.max_iterations):
            optimizer.zero_grad()
            
            # Define the loss function (evaluation criteria)
            loss_instruction = f"""
            Compare the refined query against the original query: "{original_query}"
            
            CRITICAL CONSTRAINTS:
            1. Do NOT add information, details, or specifics not present in the original
            2. Do NOT add command words like "show", "find", "get", "list" unless already present
            3. Do NOT remove or weaken location context (e.g., "near me", "my area", "local")
            4. Do NOT hallucinate locations, dates, activities, or specifics
            5. Keep the query natural for search engines (avoid conversational prefixes)
            
            PRESERVE from original:
            - Location indicators (near me, nearby, local, in my area)
            - Scope and intent
            - Key search terms
            - Personal context (me, my, our)
            
            Score higher if the refined query:
            - improves clarity using ONLY original information
            - enhances search engine compatibility
            - fixes grammar or structure issues
            - maintains or strengthens location context
            - stays concise and searchable
            
            Score lower if the refined query:
            - adds/removes information
            - adds unnecessary command words
            - weakens location or personal context
            - becomes less searchable
            - changes the user's intent
            
            Provide specific feedback for improvement.
            """
            loss_fn = tg.TextLoss(loss_instruction)
            
            # Do the forward pass and compute loss
            loss = loss_fn(query_var)
            
            # Perform the backward pass and compute gradients
            loss.backward()
            
            # Update the query variable
            optimizer.step()
            
            # Log the progress after each iteration
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration + 1}/{request.max_iterations} - Query Refinement")
            logger.info(f"Original: {original_query}")
            logger.info(f"Current:  {query_var.value}")
            logger.info(f"{'='*80}\n")
        
        return {
            "original_query": original_query,
            "refined_query": query_var.value,
            "system_prompt_snapshot": SYSTEM_PROMPT.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in query refinement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/textgrad/refine-answer")
async def refine_answer(request: AnswerRequest):
    """
    Refine an answer using TextGrad optimization with context-aware evaluation.
    
    This endpoint improves answers by:
    - Ensuring factual accuracy against provided context
    - Eliminating hallucinations
    - Improving logical clarity and reasoning
    - Making answers more complete and helpful
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        logger.info(f"Answer refinement request: question='{request.question[:50]}...', iterations={request.max_iterations}")
        
        # CHANGE BACKWARD ENGINE HERE: Set model for answer evaluation
        set_backward_engine_safe()
        
        model = get_advanced_model()
        
        # If initial_answer is provided (from your LLM Synthesis stage), use it
        # Otherwise, generate an initial answer (for standalone testing)
        if request.initial_answer:
            logger.info("Using provided initial answer from LLM Synthesis stage")
            initial_response_value = request.initial_answer
        else:
            logger.info("Generating initial answer using MODEL")
            # Generate initial answer (forward pass)
            question_prompt = f"{request.question}\n\nContext:\n{request.context}"
            question_var = tg.Variable(
                question_prompt,
                requires_grad=False,
                role_description="question with context"
            )
            
            # Get initial answer from the model
            initial_response = model(question_var)
            initial_response_value = initial_response.value
        
        # Initialize the variable to optimize
        answer_var = tg.Variable(
            initial_response_value,
            requires_grad=True,
            role_description="answer to be refined"
        )
        
        # Set up the optimizer
        optimizer = tg.TextualGradientDescent(parameters=[answer_var])
        
        # Optimization loop following paper pattern
        for iteration in range(request.max_iterations):
            optimizer.zero_grad()
            
            # Define the loss function (evaluation criteria)
            loss_instruction = f"""
            Given the evidence context: {request.context}
            Evaluate the answer to: {request.question}
            
            CRITICAL REQUIREMENTS:
            - Answer MUST be fully supported by the provided context
            - Do NOT add facts, details, or information not present in the context
            - Do NOT make assumptions beyond what's explicitly stated
            - Include ALL relevant information from the context (be complete, not minimal)
            
            Penalize severely:
            - hallucinations or fabricated facts not in the context
            - information not supported by the context
            - speculation or assumptions beyond the context
            - being incomplete when context provides more relevant details
            
            Reward:
            - accurate use of ALL relevant context-provided information
            - completeness: include all pertinent facts from context
            - logical clarity and clear reasoning
            - proper structure and coherence
            - staying within context bounds while being thorough
            
            BALANCE: Be complete (use all relevant context) but never hallucinate (add nothing beyond context).
            
            Provide specific feedback for improvement.
            """
            loss_fn = tg.TextLoss(loss_instruction)
            
            # Do the forward pass and compute loss
            loss = loss_fn(answer_var)
            
            # Perform the backward pass and compute gradients
            loss.backward()
            
            # Update the answer variable
            optimizer.step()
            
            # Log the progress after each iteration
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration + 1}/{request.max_iterations} - Answer Refinement")
            logger.info(f"Question: {request.question}")
            logger.info(f"Initial:  {initial_response_value[:100]}...")
            logger.info(f"Current:  {answer_var.value[:100]}...")
            logger.info(f"{'='*80}\n")
        
        return {
            "question": request.question,
            "context": request.context,
            "initial_answer": initial_response_value,
            "refined_answer": answer_var.value,
            "system_prompt_snapshot": SYSTEM_PROMPT.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in answer refinement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/textgrad/refine-plan")
async def refine_plan(request: PlanRequest):
    """
    Refine a tool execution plan using TextGrad optimization.
    
    This endpoint optimizes plans by:
    - Ensuring logical ordering of steps
    - Minimizing unnecessary steps
    - Avoiding problematic tool choices
    - Learning from execution feedback
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        logger.info(f"Plan refinement request: query='{request.user_query[:50]}...', iterations={request.max_iterations}")
        
        # CHANGE BACKWARD ENGINE HERE: Set model for plan evaluation
        set_backward_engine_safe()
        
        model = get_advanced_model()
        
        # Generate initial plan
        plan_prompt = f"User query: {request.user_query}\nGenerate a detailed tool execution plan."
        plan_var_init = tg.Variable(
            plan_prompt,
            requires_grad=False,
            role_description="plan generation prompt"
        )
        
        initial_response = model(plan_var_init)
        
# Create plan variable for optimization
        plan_var = tg.Variable(
            initial_response.value,
            requires_grad=True,
            role_description="execution plan to be refined"
        )
        
        # Set up the optimizer
        optimizer = tg.TextualGradientDescent(parameters=[plan_var])
        
        # Optimization loop following paper pattern
        for iteration in range(request.max_iterations):
            optimizer.zero_grad()
            
            # Define the loss function (evaluation criteria)
            loss_instruction = f"""
            Evaluate the tool execution plan for: {request.user_query}
            
            Consider this execution feedback: {request.execution_feedback}
            
            CRITICAL REQUIREMENTS:
            - Plan must directly address the user query requirements
            - Steps must be realistic and executable with available tools
            - Do NOT add unnecessary or speculative steps
            - Do NOT assume tools or capabilities not mentioned
            
            Evaluate based on:
            - logical ordering of steps (dependent steps after prerequisites)
            - minimal and efficient steps (no redundancy)
            - avoidance of unnecessary or unavailable tools
            - direct response to execution feedback issues
            - feasibility and practicality
            
            Penalize:
            - illogical step ordering
            - redundant or unnecessary steps
            - ignoring execution feedback
            - adding steps not relevant to the query
            
            Provide specific feedback for improvement.
            """
            loss_fn = tg.TextLoss(loss_instruction)
            
            # Do the forward pass and compute loss
            loss = loss_fn(plan_var)
            
            # Perform the backward pass and compute gradients
            loss.backward()
            
            # Update the plan variable
            optimizer.step()
            
            # Log the progress after each iteration
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration + 1}/{request.max_iterations} - Plan Refinement")
            logger.info(f"User Query: {request.user_query}")
            logger.info(f"Initial Plan:\n{initial_response.value[:200]}...")
            logger.info(f"Current Plan:\n{plan_var.value[:200]}...")
            logger.info(f"{'='*80}\n")
        
        return {
            "user_query": request.user_query,
            "initial_plan": initial_response.value,
            "refined_plan": plan_var.value,
            "execution_feedback_used": request.execution_feedback,
            "system_prompt_snapshot": SYSTEM_PROMPT.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in plan refinement: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/textgrad/optimize-prompt")
async def optimize_prompt(request: PromptOptimizeRequest):
    """
    Optimize the system prompt itself using TextGrad.
    
    This endpoint optimizes the global system prompt by:
    - Evaluating responses on test inputs
    - Iteratively improving the system prompt
    - Ensuring desired behavior (helpful, correct, concise, safe)
    
    This is meta-optimization: optimizing the prompt that guides all other responses.
    """
    try:
        if not TEXTGRAD_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="TextGrad not available. Install with: pip install textgrad"
            )
        
        logger.info(f"Prompt optimization request: {len(request.eval_inputs)} inputs, iterations={request.max_iterations}")
        
        # CHANGE BACKWARD ENGINE HERE: Set model for prompt evaluation
        set_backward_engine_safe()
        
        model = get_advanced_model()
        
        # Store original system prompt
        original_prompt = SYSTEM_PROMPT.value
        
        # Optimization loop following paper pattern exactly
        for iteration in range(request.max_iterations):
            # Prepare batch of inputs and desired outputs
            batch_x = request.eval_inputs
            batch_y = [request.desired_behavior] * len(batch_x)  # Same desired behavior for all
            
            ADVANCED_OPTIMIZER.zero_grad()
            
            # Do the forward pass: generate responses for batch of inputs
            responses = []
            for eval_input in batch_x:
                input_var = tg.Variable(
                    eval_input,
                    requires_grad=False,
                    role_description="evaluation input"
                )
                response = model(input_var)
                responses.append(response)
            
            # Compute losses for each (response, desired_behavior) pair
            losses = []
            for response, desired in zip(responses, batch_y):
                loss_instruction = f"""
                Evaluate whether this response matches the desired behavior: {desired}
                
                Response: {response.value}
                
                CRITICAL REQUIREMENTS:
                - Response must align with the desired behavior criteria
                - Evaluate objectively against each criterion
                - Provide actionable feedback for system prompt improvement
                
                Rate based on:
                - helpfulness: Does it provide useful, relevant information?
                - correctness: Is the information accurate and factual?
                - conciseness: Is it clear without unnecessary verbosity?
                - safety: Does it avoid harmful, biased, or inappropriate content?
                
                Penalize:
                - responses that miss the desired behavior
                - verbose or unclear responses
                - incorrect or misleading information
                - unsafe or inappropriate content
                
                Provide specific feedback on how to improve the system prompt to better achieve the desired behavior.
                """
                loss_fn = tg.TextLoss(loss_instruction)
                loss = loss_fn(response)
                losses.append(loss)
            
            # Sum all losses (following paper pattern)
            total_loss = tg.sum(losses)
            
            # Perform the backward pass and compute gradients
            total_loss.backward()
            
            # Update the system prompt
            ADVANCED_OPTIMIZER.step()
            
            # Log the progress after each iteration
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration + 1}/{request.max_iterations} - System Prompt Optimization")
            logger.info(f"Original Prompt: {original_prompt}")
            logger.info(f"Current Prompt:  {SYSTEM_PROMPT.value}")
            logger.info(f"Batch Size: {len(batch_x)} inputs")
            logger.info(f"{'='*80}\n")
        
        return {
            "status": "system prompt optimized",
            "original_system_prompt": original_prompt,
            "final_system_prompt": SYSTEM_PROMPT.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prompt optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
