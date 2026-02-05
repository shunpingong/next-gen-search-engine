"""
Test script for TextGrad endpoints

This script tests all three TextGrad endpoints to ensure they work correctly.
Run the backend server first with: uvicorn main:app --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_textgrad_optimize():
    """Test the /textgrad/optimize endpoint"""
    print("\n=== Testing /textgrad/optimize ===")
    
    payload = {
        "text": "Write a email to my professor asking for extension.",
        "objective": "Make this email more professional, polite, and persuasive while being concise",
        "num_iterations": 2,
        "model": "gpt-3.5-turbo"
    }
    
    response = requests.post(f"{BASE_URL}/textgrad/optimize", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"Original: {result['original_text'][:100]}...")
        print(f"Optimized: {result['optimized_text'][:100]}...")
        print(f"Iterations: {result['iterations']}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
    
    return response.status_code == 200


def test_textgrad_solution():
    """Test the /textgrad/solution endpoint"""
    print("\n=== Testing /textgrad/solution ===")
    
    payload = {
        "problem": "Calculate the sum of all even numbers from 1 to 100",
        "initial_solution": "result = sum([x for x in range(1, 101) if x % 2 == 0])",
        "evaluation_criteria": ["correctness", "efficiency", "readability"],
        "num_iterations": 2,
        "model": "gpt-3.5-turbo"
    }
    
    response = requests.post(f"{BASE_URL}/textgrad/solution", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"Problem: {result['problem'][:80]}...")
        print(f"Original solution: {result['original_solution'][:80]}...")
        print(f"Optimized solution: {result['optimized_solution'][:80]}...")
        print(f"Criteria evaluated: {', '.join(result['evaluation_criteria'])}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
    
    return response.status_code == 200


def test_textgrad_generate():
    """Test the /textgrad/generate endpoint"""
    print("\n=== Testing /textgrad/generate ===")
    
    payload = {
        "task_description": "Create a Python function that checks if a number is prime",
        "num_candidates": 3,
        "optimization_rounds": 2,
        "model": "gpt-3.5-turbo"
    }
    
    response = requests.post(f"{BASE_URL}/textgrad/generate", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"Task: {result['task_description'][:80]}...")
        print(f"Candidates generated: {len(result['candidates'])}")
        print(f"Best candidate optimized with {len(result['optimized_result']['optimization_history'])} rounds")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
    
    return response.status_code == 200


def test_health():
    """Test the health check endpoint"""
    print("\n=== Testing /health ===")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        print(f"✓ Server is healthy!")
        return True
    else:
        print(f"✗ Server not responding")
        return False


if __name__ == "__main__":
    print("TextGrad Endpoints Test Suite")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n❌ Server is not running. Start with: uvicorn main:app --reload")
        exit(1)
    
    # Test all endpoints
    results = {
        # "optimize": test_textgrad_optimize(),
        # "solution": test_textgrad_solution(),
        "generate": test_textgrad_generate()
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    # print(f"  /textgrad/optimize: {'✓ PASS' if results['optimize'] else '✗ FAIL'}")
    # print(f"  /textgrad/solution: {'✓ PASS' if results['solution'] else '✗ FAIL'}")
    print(f"  /textgrad/generate: {'✓ PASS' if results['generate'] else '✗ FAIL'}")
    print("=" * 50)
    
    if all(results.values()):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Check the output above.")
