#!/usr/bin/env python3
"""
Quick test script for NextGen Web Search API
Run this after starting the backend: python main.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_root():
    """Test root endpoint"""
    print("=" * 60)
    print("TEST: Root Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_tavily():
    """Test Tavily search endpoint"""
    print("=" * 60)
    print("TEST: Tavily Search")
    print("=" * 60)
    
    payload = {
        "query": "machine learning best practices"
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/tavily", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Query: {data.get('query')}")
        print(f"Answer: {data.get('answer')[:200]}...")
        print(f"Results Count: {len(data.get('results', []))}")
        if data.get('results'):
            print(f"First Result: {data['results'][0].get('title')}")
    else:
        print(f"Error: {response.text}")
    print()

def test_pagerank():
    """Test PageRank endpoint"""
    print("=" * 60)
    print("TEST: PageRank Document Ranking")
    print("=" * 60)
    
    payload = {
        "documents": [
            {
                "id": "doc1",
                "title": "Machine Learning Guide",
                "content": "Comprehensive guide to machine learning techniques and best practices",
                "score": 0.8
            },
            {
                "id": "doc2",
                "title": "Deep Learning Basics",
                "content": "Introduction to deep learning neural networks",
                "score": 0.6
            },
            {
                "id": "doc3",
                "title": "ML Best Practices",
                "content": "Best practices for machine learning model development",
                "score": 0.7
            }
        ],
        "query": "machine learning best practices",
        "top_k": 2
    }
    
    print(f"Documents: {len(payload['documents'])}")
    print(f"Query: {payload['query']}")
    print(f"Top K: {payload['top_k']}")
    
    response = requests.post(f"{BASE_URL}/pagerank", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total Documents: {data['total_documents']}")
        print(f"Personalized by Query: {data['personalized_by_query']}")
        print(f"\nScores:")
        for doc_id, score in data['scores'].items():
            print(f"  {doc_id}: {score:.4f}")
        
        if data['ranked_documents']:
            print(f"\nTop {len(data['ranked_documents'])} Ranked Documents:")
            for i, doc in enumerate(data['ranked_documents'], 1):
                print(f"  {i}. {doc['title']} (Score: {doc['pagerank_score']:.4f})")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NextGen Web Search API - Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test basic endpoints
        test_health()
        test_root()
        
        # Test Tavily (requires TAVILY_API_KEY)
        print("Note: Tavily test will fail without TAVILY_API_KEY")
        try:
            test_tavily()
        except Exception as e:
            print(f"Tavily test skipped: {e}\n")
        
        # Test PageRank (should always work)
        test_pagerank()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: Could not connect to API at {BASE_URL}")
        print(f"Make sure to start the backend first: python main.py")
        print(f"Details: {e}")
