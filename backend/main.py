import json
import logging
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.app_config import load_application_config
from tavily_pipeline import (
    SearchProviderError,
    TavilyPipelineError,
    TavilySearchError,
    run_tavily_pipeline,
)
from utils.text_utils import lexical_relevance_score, normalize_whitespace, specificity_overlap_score

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

APP_CONFIG = load_application_config()

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
OPENAI_API_KEY = APP_CONFIG.openai_api_key
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
elif TEXTGRAD_AVAILABLE:
    logger.warning("OPENAI_API_KEY not configured. TextGrad endpoints will fail until it is set.")

BACKWARD_MODEL = APP_CONFIG.textgrad_backward_model
ADVANCED_MODEL_NAME = APP_CONFIG.textgrad_advanced_model
FALLBACK_MODEL = APP_CONFIG.textgrad_fallback_model
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
    version="1.1.0"
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


class PageRankRequest(BaseModel):
    documents: List[Dict[str, Any]]
    query: Optional[Any] = None  # Supports plain strings and {"refined_query": "..."} payloads
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
        "version": "2.1.0",
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
    Deep research browsing endpoint.

    Keeps the existing `/tavily` entrypoint, but routes the request through the
    production-style multi-hop research agent so the response includes a grounded
    answer, evidence, reasoning trace, and timing breakdown.
    """
    try:
        requested_max_results = request.max_results if request.max_results is not None else 5
        requested_max_results = max(1, min(requested_max_results, 10))

        logger.info(
            "Received Tavily pipeline query: '%s' (top sources=%s)",
            request.query,
            requested_max_results,
        )

        pipeline_result = await run_tavily_pipeline(
            request.query,
            max_sources=requested_max_results,
        )

        logger.info(
            "Tavily pipeline complete: mode=%s clues=%s retrieved=%s deduplicated=%s returned=%s reranker=%s follow_up=%s answer_found=%s",
            pipeline_result.pipeline_mode,
            len(pipeline_result.clues),
            pipeline_result.retrieved_documents,
            pipeline_result.deduplicated_documents,
            len(pipeline_result.sources),
            pipeline_result.reranker,
            pipeline_result.follow_up_used,
            bool(pipeline_result.answer),
        )

        results = pipeline_result.to_response()
        logger.info("Tavily pipeline response: %s", results)
        return results
    
    except (TavilySearchError, SearchProviderError) as error:
        logger.error("Search provider error: %s", error.detail)
        raise HTTPException(status_code=error.status_code, detail=error.detail)
    except TavilyPipelineError as error:
        logger.error("Tavily pipeline error: %s", str(error))
        raise HTTPException(status_code=500, detail=str(error))
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Server Error: {str(error)}")
        raise HTTPException(status_code=500, detail=str(error))


# ============================================================================
# Utility Functions
# ============================================================================

PAGERANK_MAX_TEXT_CHARS = 6000
PAGERANK_GRAPH_TEXT_CHARS = 2500
PAGERANK_MAX_ITERATIONS = 50
PAGERANK_TOLERANCE = 1e-6


def _extract_query_text(query: Any) -> Optional[str]:
    """Extract a usable query string from raw input or TextGrad-style payloads."""
    if query is None:
        return None

    if isinstance(query, dict):
        for key in ("refined_query", "query", "original_query", "text"):
            value = query.get(key)
            if isinstance(value, str):
                normalized = normalize_whitespace(value)
                if normalized:
                    return normalized

        for value in query.values():
            if isinstance(value, str):
                normalized = normalize_whitespace(value)
                if normalized:
                    return normalized
        return None

    if isinstance(query, list):
        query_parts = []
        for value in query:
            normalized = _extract_query_text(value)
            if normalized:
                query_parts.append(normalized)
        combined_query = normalize_whitespace(" ".join(query_parts))
        return combined_query or None

    if isinstance(query, str):
        normalized_query = normalize_whitespace(query)
        if not normalized_query:
            return None

        if normalized_query.startswith("{") or normalized_query.startswith("["):
            try:
                parsed_query = json.loads(normalized_query)
            except json.JSONDecodeError:
                return normalized_query
            extracted_query = _extract_query_text(parsed_query)
            return extracted_query or normalized_query

        return normalized_query

    normalized_query = normalize_whitespace(str(query))
    return normalized_query or None


def _document_title(document: Dict[str, Any]) -> str:
    title = document.get("title", "")
    return normalize_whitespace(title) if isinstance(title, str) else ""


def _document_content(document: Dict[str, Any], *, max_chars: int) -> str:
    for key in ("content", "raw_content", "snippet", "description", "summary"):
        value = document.get(key, "")
        if isinstance(value, str):
            normalized = normalize_whitespace(value)
            if normalized:
                return normalized[:max_chars]
    return ""


def _compose_pagerank_text(document: Dict[str, Any], *, max_chars: int) -> str:
    title = _document_title(document)
    content = _document_content(document, max_chars=max_chars)
    return normalize_whitespace(" ".join(part for part in (title, content) if part))


def _document_prior_score(document: Dict[str, Any]) -> float:
    for key in ("score", "retrieval_score", "rank_score"):
        value = document.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
    return 0.0


def _normalize_distribution(values: np.ndarray, *, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    clipped_values = np.clip(np.asarray(values, dtype=float), a_min=0.0, a_max=None)
    total = clipped_values.sum()
    if total > 0:
        return clipped_values / total

    if fallback is not None:
        clipped_fallback = np.clip(np.asarray(fallback, dtype=float), a_min=0.0, a_max=None)
        fallback_total = clipped_fallback.sum()
        if fallback_total > 0:
            return clipped_fallback / fallback_total

    if clipped_values.size == 0:
        return clipped_values

    return np.ones(clipped_values.size, dtype=float) / clipped_values.size


def _scale_scores(values: np.ndarray) -> np.ndarray:
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0:
        return numeric_values

    minimum = float(numeric_values.min())
    maximum = float(numeric_values.max())
    spread = maximum - minimum
    if spread <= 1e-12:
        return np.ones_like(numeric_values) if maximum > 0 else np.zeros_like(numeric_values)
    return (numeric_values - minimum) / spread


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute a robust symmetric similarity score between two texts.
    Blends lexical overlap, specificity overlap, and surface similarity.
    """
    normalized_text1 = normalize_whitespace(text1)
    normalized_text2 = normalize_whitespace(text2)

    if not normalized_text1 or not normalized_text2:
        return 0.0

    lexical_score = (
        lexical_relevance_score(normalized_text1, normalized_text2)
        + lexical_relevance_score(normalized_text2, normalized_text1)
    ) / 2.0
    specificity_score = (
        specificity_overlap_score(normalized_text1, normalized_text2)
        + specificity_overlap_score(normalized_text2, normalized_text1)
    ) / 2.0
    sequence_score = SequenceMatcher(
        None,
        normalized_text1[:1200].lower(),
        normalized_text2[:1200].lower(),
    ).ratio()

    return float(min(1.0, (0.4 * lexical_score) + (0.3 * specificity_score) + (0.3 * sequence_score)))


def _compute_query_relevance_score(query: str, document: Dict[str, Any]) -> float:
    document_text = _compose_pagerank_text(document, max_chars=PAGERANK_MAX_TEXT_CHARS)
    if not document_text:
        return 0.0

    title_text = _document_title(document)
    lexical_score = lexical_relevance_score(query, document_text)
    specificity_score = specificity_overlap_score(query, document_text)
    title_score = lexical_relevance_score(query, title_text) if title_text else 0.0

    return float(min(1.0, (0.55 * lexical_score) + (0.30 * specificity_score) + (0.15 * title_score)))


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
        query = _extract_query_text(request.query)
        top_k = request.top_k
        
        logger.info(f"PageRank computation for {len(documents)} documents" + 
                   (f" with query: {query}" if query else ""))
        
        if not documents:
            return {
                "scores": {},
                "pagerank_scores": {},
                "relevance_scores": {},
                "ranked_documents": [] if top_k is not None else None,
                "total_documents": 0,
                "iterations": 0,
                "effective_query": query,
                "personalized_by_query": bool(query),
            }
        
        n = len(documents)
        
        # Extract document IDs
        doc_ids = [doc.get("id", f"doc-{i}") for i, doc in enumerate(documents)]
        graph_texts = [_compose_pagerank_text(doc, max_chars=PAGERANK_GRAPH_TEXT_CHARS) for doc in documents]
        prior_scores = np.array([_document_prior_score(doc) for doc in documents], dtype=float)
        
        # Create relevance scores against the effective query or any existing retrieval scores
        raw_relevance_scores: list[float] = []
        if query:
            for doc in documents:
                raw_relevance_scores.append(_compute_query_relevance_score(query, doc))
        else:
            for doc in documents:
                raw_relevance_scores.append(_document_prior_score(doc))

        # Build adjacency matrix based on content similarity between documents
        adjacency = np.zeros((n, n), dtype=float)
        for source_index in range(n):
            for target_index in range(n):
                if source_index == target_index:
                    continue
                similarity = compute_text_similarity(
                    graph_texts[target_index],
                    graph_texts[source_index],
                )
                adjacency[target_index][source_index] = similarity

        col_sums = adjacency.sum(axis=0)
        dangling_columns = col_sums == 0
        if np.any(dangling_columns):
            adjacency[:, dangling_columns] = 1.0 / n
            col_sums = adjacency.sum(axis=0)
        adjacency = adjacency / col_sums
        
        # PageRank algorithm with personalization from query relevance / prior scores
        damping = 0.85
        pagerank = np.ones(n, dtype=float) / n
        relevance_array = np.array(raw_relevance_scores, dtype=float)
        teleport_vector = _normalize_distribution(relevance_array, fallback=prior_scores)
        iterations_run = 0
        
        # Iterate PageRank
        for iteration in range(1, PAGERANK_MAX_ITERATIONS + 1):
            new_pagerank = (1 - damping) * teleport_vector + damping * adjacency.dot(pagerank)
            new_pagerank = _normalize_distribution(new_pagerank, fallback=teleport_vector)
            iterations_run = iteration
            if np.linalg.norm(new_pagerank - pagerank, ord=1) < PAGERANK_TOLERANCE:
                pagerank = new_pagerank
                break
            pagerank = new_pagerank

        if query:
            graph_boost = _scale_scores(pagerank)
            final_scores_array = np.clip(relevance_array * (0.75 + (0.25 * graph_boost)), 0.0, 1.0)
        else:
            final_scores_array = pagerank
        
        # Create score dictionaries indexed by document ID
        scores = {doc_ids[i]: float(final_scores_array[i]) for i in range(n)}
        pagerank_scores = {doc_ids[i]: float(pagerank[i]) for i in range(n)}
        relevance_scores = {doc_ids[i]: float(relevance_array[i]) for i in range(n)}
        
        ranked_documents = []
        if top_k is not None:
            ranked_indices = np.argsort(final_scores_array)[::-1][:top_k]
            for idx in ranked_indices:
                doc = documents[int(idx)].copy()
                doc['pagerank_score'] = float(pagerank[idx])
                doc['query_relevance_score'] = float(relevance_array[idx])
                doc['final_score'] = float(final_scores_array[idx])
                ranked_documents.append(doc)
        
        logger.info(
            "PageRank computation complete. Top final score: %.4f, top relevance: %.4f, top PageRank: %.4f",
            float(np.max(final_scores_array)),
            float(np.max(relevance_array)),
            float(np.max(pagerank)),
        )
        
        return {
            "scores": scores,
            "pagerank_scores": pagerank_scores,
            "relevance_scores": relevance_scores,
            "ranked_documents": ranked_documents if top_k is not None else None,
            "total_documents": n,
            "iterations": iterations_run,
            "effective_query": query,
            "personalized_by_query": bool(query),
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
