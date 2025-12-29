"""
Test script for the enhanced search endpoint with PageRank ranking
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_basic_tavily_search():
    """Test basic Tavily search endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Basic Tavily Search")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/tavily",
        json={"query": "Linux kernel security vulnerabilities"}
    )
    
    if response.ok:
        data = response.json()
        print(f"✓ Query: {data.get('query')}")
        print(f"✓ Answer: {data.get('answer', 'N/A')[:100]}...")
        print(f"✓ Results: {len(data.get('results', []))} found")
        return True
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
        return False


def test_enhanced_search():
    """Test enhanced search with PageRank ranking"""
    print("\n" + "="*60)
    print("TEST 2: Enhanced Search with PageRank")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/api/search/enhanced",
        json={
            "query": "Linux kernel security vulnerabilities",
            "top_k": 5
        }
    )
    
    if response.ok:
        data = response.json()
        print(f"✓ Query: {data.get('query')}")
        print(f"✓ Tavily Answer: {data.get('tavily_answer', 'N/A')[:100]}...")
        print(f"✓ Total Results Analyzed: {data.get('total_results_analyzed')}")
        print(f"✓ Ranking Method: {data.get('ranking_method')}")
        
        print(f"\n📊 Top {data.get('top_k')} Ranked Results:")
        for i, result in enumerate(data.get('ranked_results', []), 1):
            print(f"\n  {i}. {result.get('title', 'No Title')}")
            print(f"     URL: {result.get('url', 'N/A')}")
            print(f"     PageRank Score: {result.get('pagerank_score', 0):.4f}")
            print(f"     Relevance Score: {result.get('relevance_score', 0):.4f}")
            print(f"     Original Score: {result.get('score', 0):.4f}")
            print(f"     Content: {result.get('content', '')[:150]}...")

        
        return True
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
        return False


def test_different_queries():
    """Test with different queries to verify ranking"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Queries Comparison")
    print("="*60)
    
    queries = [
        "machine learning algorithms",
        "web security best practices",
        "Python data science libraries"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        response = requests.post(
            f"{BASE_URL}/api/search/enhanced",
            json={
                "query": query,
                "top_k": 3
            }
        )
        
        if response.ok:
            data = response.json()
            print(f"   ✓ Got {len(data.get('ranked_results', []))} ranked results")
            
            # Show top result
            if data.get('ranked_results'):
                top = data['ranked_results'][0]
                print(f"   🥇 Top Result: {top.get('title', 'N/A')}")
                print(f"      PageRank: {top.get('pagerank_score', 0):.4f}")
        else:
            print(f"   ✗ Error: {response.status_code}")


def test_top_k_parameter():
    """Test different top_k values"""
    print("\n" + "="*60)
    print("TEST 4: Top-K Parameter Testing")
    print("="*60)
    
    query = "artificial intelligence applications"
    
    for k in [3, 5, 7]:
        print(f"\n📊 Testing top_k={k}")
        
        response = requests.post(
            f"{BASE_URL}/api/search/enhanced",
            json={
                "query": query,
                "top_k": k
            }
        )
        
        if response.ok:
            data = response.json()
            results = data.get('ranked_results', [])
            print(f"   ✓ Returned {len(results)} results")
            
            if results:
                avg_pagerank = sum(r.get('pagerank_score', 0) for r in results) / len(results)
                print(f"   📈 Average PageRank: {avg_pagerank:.4f}")
        else:
            print(f"   ✗ Error: {response.status_code}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Enhanced Search API Test Suite")
    print("="*60)
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/health")
        if not response.ok:
            print("❌ Server is not running! Start with: uvicorn main:app --reload")
            return
        
        print("✓ Server is running")
        
        # Run tests
        tests = [
            ("Basic Tavily Search", test_basic_tavily_search),
            ("Enhanced Search with PageRank", test_enhanced_search),
            ("Multiple Queries", test_different_queries),
            ("Top-K Parameter", test_top_k_parameter)
        ]
        
        results = []
        for name, test_func in tests:
            try:
                success = test_func()
                results.append((name, success))
            except Exception as e:
                print(f"❌ Test '{name}' failed with exception: {e}")
                results.append((name, False))
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        for name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {name}")
        
        passed = sum(1 for _, s in results if s)
        total = len(results)
        print(f"\nTotal: {passed}/{total} tests passed")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server!")
        print("Make sure the server is running:")
        print("  cd backend")
        print("  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
