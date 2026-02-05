# TextGrad Endpoints Documentation

## Overview

TextGrad endpoints implement automatic "differentiation" via text as described in the paper ["TextGrad: Automatic 'Differentiation' via Text"](https://arxiv.org/abs/2406.07496) by Yuksekgonul et al. (2024).

These endpoints enable your nemobot application to optimize text, solutions, and generated content using textual gradients provided by LLMs.

**Reference:** https://textgrad.com/

---

## Prerequisites

1. **Install TextGrad:**

   ```bash
   pip install textgrad
   ```

2. **Set OpenAI API Key:**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
   set OPENAI_API_KEY=your-api-key-here       # Windows CMD
   $env:OPENAI_API_KEY="your-api-key-here"    # Windows PowerShell
   ```

3. **Start the backend server:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

---

## Endpoint 1: `/textgrad/optimize`

**Purpose:** Optimize any text based on a specific objective (e.g., improve clarity, persuasiveness, professionalism).

### Request Format

```json
{
  "text": "string (required) - The text to optimize",
  "objective": "string (required) - What you want to achieve",
  "num_iterations": "integer (optional, default: 3) - Number of optimization rounds",
  "model": "string (optional, default: 'gpt-3.5-turbo') - LLM model to use"
}
```

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:8000/textgrad/optimize",
    json={
        "text": "Write a email to my professor asking for extension.",
        "objective": "Make this email more professional, polite, and persuasive while being concise",
        "num_iterations": 3,
        "model": "gpt-3.5-turbo"
    }
)

result = response.json()
print(f"Original: {result['original_text']}")
print(f"Optimized: {result['optimized_text']}")
```

### Response Format

```json
{
  "original_text": "The original text",
  "optimized_text": "The optimized text after iterations",
  "objective": "The optimization objective used",
  "iterations": 3,
  "optimization_history": [
    {
      "iteration": 1,
      "feedback": "Feedback from the LLM",
      "text_length": 150
    }
  ],
  "model_used": "gpt-3.5-turbo"
}
```

### Use Cases

- **Prompt Engineering:** Optimize prompts for better LLM responses
- **Email Writing:** Improve professional communication
- **Content Refinement:** Enhance clarity, engagement, or persuasiveness
- **Instruction Optimization:** Make instructions clearer and more actionable

---

## Endpoint 2: `/textgrad/solution`

**Purpose:** Optimize a solution to a problem using multiple evaluation criteria.

### Request Format

```json
{
  "problem": "string (required) - The problem statement",
  "initial_solution": "string (required) - Your initial solution",
  "evaluation_criteria": [
    "array of strings (required) - Criteria to optimize against"
  ],
  "num_iterations": "integer (optional, default: 3) - Number of optimization rounds",
  "model": "string (optional, default: 'gpt-3.5-turbo') - LLM model to use"
}
```

### Example Request

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
print(f"Optimized Solution:\n{result['optimized_solution']}")
```

### Response Format

```json
{
  "problem": "The original problem statement",
  "original_solution": "The initial solution",
  "optimized_solution": "The optimized solution",
  "evaluation_criteria": ["correctness", "efficiency", "readability"],
  "iterations": 3,
  "improvement_log": [
    {
      "iteration": 1,
      "criteria_feedback": [
        {
          "criterion": "correctness",
          "feedback": "The solution is correct but..."
        }
      ],
      "solution_length": 200
    }
  ],
  "model_used": "gpt-3.5-turbo"
}
```

### Use Cases

- **Code Optimization:** Improve code quality, performance, and readability
- **Algorithm Design:** Enhance algorithmic solutions
- **Problem Solving:** Refine solutions to technical or business problems
- **Multi-objective Optimization:** Balance trade-offs across criteria

---

## Endpoint 3: `/textgrad/generate`

**Purpose:** Generate multiple candidate solutions and optimize the best one.

### Request Format

```json
{
  "task_description": "string (required) - Description of the task",
  "num_candidates": "integer (optional, default: 3) - Number of candidates to generate",
  "optimization_rounds": "integer (optional, default: 2) - Rounds to optimize best candidate",
  "model": "string (optional, default: 'gpt-3.5-turbo') - LLM model to use"
}
```

### Example Request

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
print(f"Best Optimized Solution:\n{result['optimized_result']['optimized_text']}")
```

### Response Format

```json
{
  "task_description": "The task description",
  "candidates": [
    {
      "candidate_id": 1,
      "solution": "First candidate solution"
    }
  ],
  "best_candidate": {
    "candidate_id": 1,
    "solution": "The selected best candidate"
  },
  "selection_rationale": "Why this candidate was chosen",
  "optimized_result": {
    "original_text": "Best candidate before optimization",
    "optimized_text": "Best candidate after optimization",
    "optimization_history": []
  },
  "model_used": "gpt-3.5-turbo"
}
```

### Use Cases

- **Creative Problem Solving:** Explore multiple approaches
- **Code Generation:** Generate and refine code solutions
- **Content Creation:** Create multiple drafts and optimize the best
- **Strategy Development:** Generate strategic options

---

## Usage in Your Nemobot Application

### JavaScript/TypeScript Example

```typescript
// Optimize text
async function optimizeText(text: string, objective: string) {
  const response = await fetch("http://localhost:8000/textgrad/optimize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: text,
      objective: objective,
      num_iterations: 3,
    }),
  });

  const result = await response.json();
  return result.optimized_text;
}

// Optimize a solution
async function optimizeSolution(
  problem: string,
  solution: string,
  criteria: string[],
) {
  const response = await fetch("http://localhost:8000/textgrad/solution", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      problem: problem,
      initial_solution: solution,
      evaluation_criteria: criteria,
      num_iterations: 3,
    }),
  });

  const result = await response.json();
  return result.optimized_solution;
}

// Generate and optimize
async function generateAndOptimize(task: string) {
  const response = await fetch("http://localhost:8000/textgrad/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      task_description: task,
      num_candidates: 3,
      optimization_rounds: 2,
    }),
  });

  const result = await response.json();
  return result.optimized_result.optimized_text;
}
```

### Python Example

```python
import requests

# Optimize text
def optimize_text(text, objective):
    response = requests.post(
        "http://localhost:8000/textgrad/optimize",
        json={
            "text": text,
            "objective": objective,
            "num_iterations": 3
        }
    )
    return response.json()["optimized_text"]

# Optimize solution
def optimize_solution(problem, solution, criteria):
    response = requests.post(
        "http://localhost:8000/textgrad/solution",
        json={
            "problem": problem,
            "initial_solution": solution,
            "evaluation_criteria": criteria,
            "num_iterations": 3
        }
    )
    return response.json()["optimized_solution"]

# Generate and optimize
def generate_and_optimize(task):
    response = requests.post(
        "http://localhost:8000/textgrad/generate",
        json={
            "task_description": task,
            "num_candidates": 3,
            "optimization_rounds": 2
        }
    )
    return response.json()["optimized_result"]["optimized_text"]
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200**: Success
- **503**: TextGrad not available (install with `pip install textgrad`)
- **500**: Server error (check logs for details)

Example error response:

```json
{
  "detail": "TextGrad not available. Install with: pip install textgrad"
}
```

---

## Performance Considerations

- **API Costs:** Each iteration calls GPT-4o for feedback (can be expensive)
- **Processing Time:** More iterations = longer processing time
- **Rate Limits:** Subject to OpenAI API rate limits

### Recommendations

- Start with 2-3 iterations and increase if needed
- Use `gpt-3.5-turbo` for faster, cheaper optimization
- Use `gpt-4` or `gpt-4o` for higher quality results
- Cache results when appropriate

---

## Testing

Run the test suite to verify all endpoints:

```bash
cd backend
python test_textgrad_endpoints.py
```

Make sure the server is running before testing!

---

## References

- **Paper:** [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)
- **Website:** https://textgrad.com/
- **GitHub:** https://github.com/zou-group/textgrad
- **Publication:** Nature (March 2025)

---

## Support

For issues or questions:

1. Check the logs in your terminal
2. Verify OpenAI API key is set correctly
3. Ensure TextGrad is installed: `pip install textgrad`
4. Refer to the test script for working examples
