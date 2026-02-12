from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import textgrad as tg
import os
from dotenv import load_dotenv
import logging

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(title="Nemobot TextGrad Optimization Service")

# --------------------------------------------------
# OpenRouter API Configuration
# --------------------------------------------------
# Load environment variables from .env file
load_dotenv()

# Set OpenRouter API key from .env file
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Configure the backward engine (used for generating feedback/gradients in TextGrad)
# CHANGE MODEL HERE: Replace the model string to use different OpenRouter models
# Format: "experimental:openrouter/<provider>/<model-name>"
# Popular options:
#   - "experimental:openrouter/openai/gpt-4o" (most capable)
#   - "experimental:openrouter/openai/gpt-4-turbo"
#   - "experimental:openrouter/anthropic/claude-3.5-sonnet"
#   - "experimental:openrouter/google/gemini-pro-1.5"
#   - "experimental:openrouter/meta-llama/llama-3.1-70b-instruct"
# See all available models at: https://openrouter.ai/models
tg.set_backward_engine("experimental:openrouter/openai/gpt-4o", override=True, cache=False)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Global TextGrad Setup
# --------------------------------------------------

SYSTEM_PROMPT = tg.Variable(
    "You are an AI search assistant specialized in query optimization and information retrieval. "
    "Provide accurate, well-structured responses based on available context. "
    "When refining queries or answers, preserve original intent while improving clarity and precision. "
    "Think step-by-step and never add information not present in the given context.",
    requires_grad=True,
    role_description="System prompt to the language model"
)

# CHANGE MODEL HERE: Set the OpenRouter model for content generation
# This model is used for generating responses and content
# Options: Any OpenRouter model in format "experimental:openrouter/<provider>/<model>"
MODEL = tg.BlackboxLLM(
    "experimental:openrouter/openai/gpt-3.5-turbo",
    system_prompt=SYSTEM_PROMPT
)
OPTIMIZER = tg.TextualGradientDescent(parameters=[SYSTEM_PROMPT])

# --------------------------------------------------
# Request Schemas
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    max_iterations: int = 3

class AnswerRequest(BaseModel):
    question: str
    context: str
    initial_answer: str = None  # Optional: if provided, refine this answer instead of generating one
    max_iterations: int = 3

class PlanRequest(BaseModel):
    user_query: str
    execution_feedback: str
    max_iterations: int = 3

class PromptOptimizeRequest(BaseModel):
    eval_inputs: List[str]
    desired_behavior: str
    max_iterations: int = 10

# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.post("/textgrad/refine-query")
def refine_query(req: QueryRequest):
    """Refine a search query using TextGrad optimization with LLM-as-judge."""
    original = req.query

   # Initialize the variable to optimize
    query_var = tg.Variable(
        original,
        requires_grad=True,
        role_description="search query to be refined"
    )
    
    # Set up the optimizer
    optimizer = tg.TextualGradientDescent(parameters=[query_var])
    
    # Optimization loop following paper pattern
    for iteration in range(req.max_iterations):
        optimizer.zero_grad()
        
        # Define the loss function (evaluation criteria)
        loss_instruction = f"""
        Compare the refined query against the original query: "{original}"
        
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
        logger.info(f"Iteration {iteration + 1}/{req.max_iterations} - Query Refinement")
        logger.info(f"Original: {original}")
        logger.info(f"Current:  {query_var.value}")
        logger.info(f"{'='*80}\n")

    return {
        "refined_query": query_var.value,
        "original_query": original,
        "system_prompt_snapshot": SYSTEM_PROMPT.value
    }


@app.post("/textgrad/refine-answer")
def refine_answer(req: AnswerRequest):
    """Refine an answer using TextGrad optimization with context-aware evaluation."""
    
    # If initial_answer is provided (from your LLM Synthesis stage), use it
    # Otherwise, generate an initial answer (for standalone testing)
    if req.initial_answer:
        logger.info("Using provided initial answer from LLM Synthesis stage")
        initial_response_value = req.initial_answer
    else:
        logger.info("Generating initial answer using MODEL")
        # Generate initial answer (forward pass)
        question_prompt = f"{req.question}\n\nContext:\n{req.context}"
        question_var = tg.Variable(
            question_prompt,
            requires_grad=False,
            role_description="question with context"
        )
        
        # Get initial answer from the model
        initial_response = MODEL(question_var)
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
    for iteration in range(req.max_iterations):
        optimizer.zero_grad()
        
        # Define the loss function (evaluation criteria)
        loss_instruction = f"""
        Given the evidence context: {req.context}
        Evaluate the answer to: {req.question}
        
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
        logger.info(f"Iteration {iteration + 1}/{req.max_iterations} - Answer Refinement")
        logger.info(f"Question: {req.question}")
        logger.info(f"Initial:  {initial_response_value[:100]}...")
        logger.info(f"Current:  {answer_var.value[:100]}...")
        logger.info(f"{'='*80}\n")

    return {
        "refined_answer": answer_var.value,
        "initial_answer": initial_response_value,
        "question": req.question,
        "context": req.context,
        "system_prompt_snapshot": SYSTEM_PROMPT.value
    }


@app.post("/textgrad/refine-plan")
def refine_plan(req: PlanRequest):
    """Refine a tool execution plan using TextGrad optimization."""
    # Generate initial plan (forward pass)
    plan_prompt = f"User query: {req.user_query}\nGenerate a detailed tool execution plan."
    plan_var_init = tg.Variable(
        plan_prompt,
        requires_grad=False,
        role_description="plan generation prompt"
    )
    
    # Get initial plan from the model
    initial_response = MODEL(plan_var_init)
    
    # Initialize the variable to optimize
    plan_var = tg.Variable(
        initial_response.value,
        requires_grad=True,
        role_description="execution plan to be refined"
    )
    
    # Set up the optimizer
    optimizer = tg.TextualGradientDescent(parameters=[plan_var])
    
    # Optimization loop following paper pattern
    for iteration in range(req.max_iterations):
        optimizer.zero_grad()
        
        # Define the loss function (evaluation criteria)
        loss_instruction = f"""
        Evaluate the tool execution plan for: {req.user_query}
        
        Consider this execution feedback: {req.execution_feedback}
        
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
        logger.info(f"Iteration {iteration + 1}/{req.max_iterations} - Plan Refinement")
        logger.info(f"User Query: {req.user_query}")
        logger.info(f"Initial Plan:\n{initial_response.value[:200]}...")
        logger.info(f"Current Plan:\n{plan_var.value[:200]}...")
        logger.info(f"{'='*80}\n")

    return {
        "refined_plan": plan_var.value,
        "initial_plan": initial_response.value,
        "user_query": req.user_query,
        "execution_feedback_used": req.execution_feedback,
        "system_prompt_snapshot": SYSTEM_PROMPT.value
    }


@app.post("/textgrad/optimize-prompt")
def optimize_prompt(req: PromptOptimizeRequest):
    """Optimize the system prompt itself using TextGrad (following paper pattern)."""
    # Store original system prompt
    original_prompt = SYSTEM_PROMPT.value
    
    # Optimization loop following paper pattern exactly
    for iteration in range(req.max_iterations):
        # Prepare batch of inputs and desired outputs
        batch_x = req.eval_inputs
        batch_y = [req.desired_behavior] * len(batch_x)  # Same desired behavior for all
        
        OPTIMIZER.zero_grad()
        
        # Do the forward pass: generate responses for batch of inputs
        responses = []
        for eval_input in batch_x:
            input_var = tg.Variable(
                eval_input,
                requires_grad=False,
                role_description="evaluation input"
            )
            response = MODEL(input_var)
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
        OPTIMIZER.step()
        
        # Log the progress after each iteration
        logger.info(f"\n{'='*80}")
        logger.info(f"Iteration {iteration + 1}/{req.max_iterations} - System Prompt Optimization")
        logger.info(f"Original Prompt: {original_prompt}")
        logger.info(f"Current Prompt:  {SYSTEM_PROMPT.value}")
        logger.info(f"Batch Size: {len(batch_x)} inputs")
        logger.info(f"{'='*80}\n")

    return {
        "status": "system prompt optimized",
        "original_system_prompt": original_prompt,
        "final_system_prompt": SYSTEM_PROMPT.value
    }
