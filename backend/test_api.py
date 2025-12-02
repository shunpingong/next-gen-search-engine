"""
Test script for Ghost Network API
Run this to verify all endpoints are working
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_health():
    """Test health endpoint"""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Services: {json.dumps(result['services'], indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_domains():
    """Test domains endpoint"""
    print_section("2. Supported Domains")
    try:
        response = requests.get(f"{BASE_URL}/api/domains")
        result = response.json()
        for domain in result['domains']:
            print(f"\n📁 {domain['name']}")
            print(f"   ID: {domain['id']}")
            print(f"   Tools: {', '.join(domain['tools'])}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_github_search():
    """Test GitHub search"""
    print_section("3. GitHub Search")
    try:
        response = requests.get(
            f"{BASE_URL}/api/security/github/search",
            params={
                "query": "linux kernel fuzzing",
                "search_type": "repositories",
                "max_results": 5
            }
        )
        result = response.json()
        
        if 'error' in result:
            print(f"⚠️ GitHub search returned error (may need GITHUB_TOKEN)")
            print(f"   {result['error']}")
            return False
        
        print(f"✅ Found {result['total_count']} repositories")
        for repo in result['items'][:3]:
            print(f"   - {repo['full_name']} ({repo.get('stargazers_count', 0)} ⭐)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_osv_vulnerabilities():
    """Test OSV vulnerability check"""
    print_section("4. OSV Vulnerability Check")
    try:
        response = requests.post(
            f"{BASE_URL}/api/security/vulnerabilities/check",
            json={
                "packages": ["linux"],
                "ecosystem": "Linux"
            }
        )
        result = response.json()
        print(f"✅ Vulnerability check completed")
        print(f"   Found {len(result)} vulnerabilities")
        if result:
            print(f"   Example: {result[0].get('id', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_graph_pagerank():
    """Test PageRank computation"""
    print_section("5. PageRank Computation")
    try:
        response = requests.get(
            f"{BASE_URL}/api/graph/pagerank",
            params={
                "nodes": "A,B,C,D",
                "edges": "A-B,B-C,C-D,D-A,A-C"
            }
        )
        result = response.json()
        print("✅ PageRank scores:")
        for node, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            print(f"   {node}: {score:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_graph_cheirank():
    """Test CheiRank computation"""
    print_section("6. CheiRank Computation")
    try:
        response = requests.get(
            f"{BASE_URL}/api/graph/cheirank",
            params={
                "nodes": "A,B,C,D",
                "edges": "A-B,B-C,C-D,D-A,A-C"
            }
        )
        result = response.json()
        print("✅ CheiRank scores:")
        for node, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
            print(f"   {node}: {score:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pubmed_search():
    """Test PubMed search"""
    print_section("7. PubMed Search")
    try:
        response = requests.get(
            f"{BASE_URL}/api/healthcare/pubmed/search",
            params={
                "query": "diabetes treatment",
                "max_results": 5
            }
        )
        result = response.json()
        
        if 'error' in result:
            print(f"⚠️ PubMed search error")
            print(f"   {result['error']}")
            return False
        
        print(f"✅ Found {result['total_count']} articles")
        for article in result['articles'][:3]:
            print(f"   - PMID {article['pmid']}: {article['title'][:50]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_security_query():
    """Test main security TAG endpoint"""
    print_section("8. Security Query (TAG)")
    try:
        response = requests.post(
            f"{BASE_URL}/api/security/query",
            json={
                "query": "networking vulnerabilities",
                "domain": "networking",
                "max_results": 3,
                "include_vulnerabilities": True,
                "compute_graph": False  # Skip graph for faster test
            },
            timeout=30
        )
        result = response.json()
        
        if 'error' in result:
            print(f"⚠️ Security query error")
            print(f"   {result['error']}")
            return False
        
        print(f"✅ Security analysis completed")
        print(f"   Repos: {len(result.get('github_results', {}).get('repos', []))}")
        print(f"   Issues: {len(result.get('github_results', {}).get('issues', []))}")
        print(f"   Vulnerabilities: {len(result.get('vulnerabilities', []))}")
        
        if result.get('llm_synthesis'):
            summary = result['llm_synthesis'].get('summary', '')
            print(f"   Summary: {summary[:80]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_healthcare_query():
    """Test main healthcare TAG endpoint"""
    print_section("9. Healthcare Query (TAG)")
    try:
        response = requests.post(
            f"{BASE_URL}/api/healthcare/query",
            json={
                "query": "COVID-19 treatment",
                "max_results": 5,
                "include_local_docs": False
            },
            timeout=30
        )
        result = response.json()
        
        if 'error' in result:
            print(f"⚠️ Healthcare query error")
            print(f"   {result['error']}")
            return False
        
        print(f"✅ Healthcare analysis completed")
        print(f"   Articles: {result.get('pubmed_results', {}).get('total_count', 0)}")
        
        if result.get('llm_synthesis'):
            summary = result['llm_synthesis'].get('summary', '')
            print(f"   Summary: {summary[:80]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    Ghost Network API Test Suite                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print("\n❌ ERROR: API server is not running!")
        print("   Please start the server first:")
        print("   python main.py")
        print("   or: uvicorn main:app --reload")
        sys.exit(1)
    
    tests = [
        ("Health Check", test_health),
        ("Supported Domains", test_domains),
        ("GitHub Search", test_github_search),
        ("OSV Vulnerabilities", test_osv_vulnerabilities),
        ("PageRank", test_graph_pagerank),
        ("CheiRank", test_graph_cheirank),
        ("PubMed Search", test_pubmed_search),
        ("Security Query", test_security_query),
        ("Healthcare Query", test_healthcare_query),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Your Ghost Network API is fully functional.")
    elif passed > total / 2:
        print("\n⚠️  Some tests failed. Check configuration (.env file).")
        print("   Common issues:")
        print("   - Missing GITHUB_TOKEN for GitHub tests")
        print("   - Missing PUBMED_EMAIL for PubMed tests")
        print("   - Network connectivity issues")
    else:
        print("\n❌ Many tests failed. Please check:")
        print("   1. Server is running (python main.py)")
        print("   2. Dependencies installed (pip install -e .)")
        print("   3. .env file configured")
    
    print("\n" + "=" * 70 + "\n")
    
    # Exit code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
