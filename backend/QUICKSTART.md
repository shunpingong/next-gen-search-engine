# Ghost Network API - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites

- Python 3.10+
- pip
- Git

### Step 1: Clone and Navigate

```bash
cd c:\Users\shunp\OneDrive\Desktop\next-gen-search-engine\backend
```

### Step 2: Install Dependencies

```bash
pip install -e .
```

Or install specific packages:

```bash
pip install fastapi uvicorn aiohttp networkx python-dotenv pydantic
```

### Step 3: Configure Environment

Create a `.env` file from the template:

```bash
copy .env.template .env
```

**Minimum Configuration** - Edit `.env`:

```env
# Required for Linux Kernel Security
GITHUB_TOKEN=ghp_your_token_here

# Required for Healthcare Domain
PUBMED_EMAIL=your-email@example.com

# Optional (but recommended)
NVD_API_KEY=your_nvd_key_here
```

**Get Your Tokens:**

- GitHub Token: https://github.com/settings/tokens (select `public_repo` scope)
- NVD API Key: https://nvd.nist.gov/developers/request-an-api-key

### Step 4: Run the Server

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload
```

The API will be running at:

- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs 👈 Open this in your browser!
- **Alternative Docs**: http://localhost:8000/redoc

---

## 🧪 Test the API

### Option 1: Interactive Swagger UI

1. Open http://localhost:8000/docs
2. Try the `/health` endpoint first
3. Expand any endpoint
4. Click "Try it out"
5. Fill in parameters
6. Click "Execute"

### Option 2: cURL Commands

#### Health Check

```bash
curl http://localhost:8000/health
```

#### GitHub Search

```bash
curl "http://localhost:8000/api/security/github/search?query=linux+kernel+fuzzing&search_type=repositories&max_results=5"
```

#### Security Query (POST)

```bash
curl -X POST "http://localhost:8000/api/security/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"networking vulnerabilities\",\"domain\":\"linux-kernel\",\"max_results\":5}"
```

#### PubMed Search

```bash
curl "http://localhost:8000/api/healthcare/pubmed/search?query=diabetes+treatment&max_results=5"
```

### Option 3: Python Script

Create `test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_github_search():
    """Test GitHub search"""
    response = requests.get(
        f"{BASE_URL}/api/security/github/search",
        params={
            "query": "linux kernel fuzzing",
            "search_type": "repositories",
            "max_results": 5
        }
    )
    result = response.json()
    print(f"\nFound {result['total_count']} repositories")
    for repo in result['items'][:3]:
        print(f"  - {repo['full_name']} ({repo['stargazers_count']} stars)")

def test_security_query():
    """Test main security TAG endpoint"""
    response = requests.post(
        f"{BASE_URL}/api/security/query",
        json={
            "query": "race conditions in networking",
            "domain": "networking",
            "max_results": 5,
            "include_vulnerabilities": True,
            "compute_graph": True
        }
    )
    result = response.json()
    print(f"\nSecurity Analysis:")
    print(f"  Repos found: {len(result.get('github_results', {}).get('repos', []))}")
    print(f"  Vulnerabilities: {len(result.get('vulnerabilities', []))}")
    if result.get('llm_synthesis'):
        print(f"  Summary: {result['llm_synthesis'].get('summary', '')[:100]}...")

def test_pubmed_search():
    """Test PubMed search"""
    response = requests.get(
        f"{BASE_URL}/api/healthcare/pubmed/search",
        params={
            "query": "COVID-19 treatment",
            "max_results": 5
        }
    )
    result = response.json()
    print(f"\nPubMed Articles: {result['total_count']} found")
    for article in result['articles'][:3]:
        print(f"  - {article['title'][:60]}...")

def test_graph_analysis():
    """Test PageRank computation"""
    response = requests.get(
        f"{BASE_URL}/api/graph/pagerank",
        params={
            "nodes": "A,B,C,D",
            "edges": "A-B,B-C,C-D,D-A,A-C"
        }
    )
    result = response.json()
    print("\nPageRank scores:")
    for node, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {score:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Ghost Network API Tests")
    print("=" * 60)

    test_health()
    test_github_search()
    test_security_query()
    test_pubmed_search()
    test_graph_analysis()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
```

Run it:

```bash
python test_api.py
```

---

## 📊 Example Workflows

### Workflow 1: Linux Kernel Security Research

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Search for security issues
github_response = requests.get(
    f"{BASE_URL}/api/security/github/search",
    params={
        "query": "linux kernel use-after-free",
        "search_type": "issues",
        "max_results": 10
    }
)

# 2. Check for vulnerabilities in top repo
repos = github_response.json()["items"]
if repos:
    owner, repo = repos[0]["repository"]["full_name"].split("/")

    vuln_response = requests.post(
        f"{BASE_URL}/api/security/vulnerabilities/check",
        json={
            "packages": [f"{owner}/{repo}"],
            "ecosystem": "Linux"
        }
    )

    # 3. Build call graph
    graph_response = requests.post(
        f"{BASE_URL}/api/security/graph/build",
        json={
            "repo_owner": owner,
            "repo_name": repo,
            "file_pattern": "*.c",
            "compute_pagerank": True,
            "compute_cheirank": True
        }
    )

    # 4. Get fuzzing suggestions
    fuzz_response = requests.get(
        f"{BASE_URL}/api/security/fuzzing/suggestions",
        params={
            "repo_owner": owner,
            "repo_name": repo,
            "focus_area": "memory-management"
        }
    )

    print("Fuzzing Strategy:", fuzz_response.json())
```

### Workflow 2: Healthcare Literature Review

```python
# 1. Search PubMed
pubmed_response = requests.get(
    f"{BASE_URL}/api/healthcare/pubmed/search",
    params={
        "query": "hypertension treatment guidelines",
        "max_results": 20,
        "sort": "relevance"
    }
)

articles = pubmed_response.json()["articles"]

# 2. Get related articles for top result
if articles:
    pmid = articles[0]["pmid"]

    related_response = requests.get(
        f"{BASE_URL}/api/healthcare/pubmed/related/{pmid}",
        params={"max_results": 10}
    )

    # 3. Use TAG endpoint for comprehensive analysis
    tag_response = requests.post(
        f"{BASE_URL}/api/healthcare/query",
        json={
            "query": "hypertension treatment guidelines",
            "specialty": "cardiology",
            "max_results": 20,
            "include_local_docs": False
        }
    )

    print("Clinical Summary:", tag_response.json()["llm_synthesis"])
```

---

## 🎯 Key Endpoints to Try

### 1. Health Check

```
GET /health
```

✅ No authentication needed

### 2. GitHub Search

```
GET /api/security/github/search?query=fuzzing&search_type=repositories
```

✅ Works with or without token (better with token)

### 3. Security Query (Main TAG)

```
POST /api/security/query
Body: {"query": "vulnerability analysis", "domain": "linux-kernel"}
```

🔑 Requires GitHub token

### 4. PubMed Search

```
GET /api/healthcare/pubmed/search?query=cancer+treatment&max_results=10
```

✅ Works with email configured

### 5. Graph Analysis

```
GET /api/graph/pagerank?nodes=A,B,C&edges=A-B,B-C,C-A
```

✅ No authentication needed

### 6. CVE Lookup

```
GET /api/security/vulnerabilities/cve/CVE-2024-1234
```

✅ Works better with NVD API key

---

## 🐛 Troubleshooting

### Issue: "Import error"

**Solution**: Install dependencies

```bash
pip install fastapi uvicorn aiohttp networkx python-dotenv pydantic
```

### Issue: "GitHub API rate limit"

**Solution**: Add GITHUB_TOKEN to `.env`

### Issue: "NVD API slow"

**Solution**: Get an NVD API key and add to `.env`

### Issue: "Port 8000 already in use"

**Solution**: Use different port

```bash
uvicorn main:app --reload --port 8001
```

### Issue: "CORS error from frontend"

**Solution**: Add your frontend URL to CORS_ORIGINS in `.env`

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## 📚 Next Steps

1. ✅ Test all endpoints using Swagger UI (http://localhost:8000/docs)
2. 📖 Read full API documentation in `API_DOCUMENTATION.md`
3. 🔧 Configure additional services (local LLM, etc.)
4. 🚀 Build your frontend to consume these APIs
5. 📊 Implement caching for better performance
6. 🔐 Add authentication for production use

---

## 💡 Pro Tips

1. **Use Swagger UI**: The interactive docs at `/docs` are the easiest way to test
2. **Start Simple**: Test `/health` and `/api/domains` endpoints first
3. **Check Logs**: Look at terminal output for debugging
4. **Rate Limits**: Be mindful of API rate limits (especially GitHub and NVD)
5. **Mock Data**: The system uses sample data for graph analysis in MVP mode
6. **LLM Optional**: The API works without a local LLM (uses rule-based synthesis)

---

## 📞 Support

For issues or questions:

1. Check the logs in terminal
2. Review `API_DOCUMENTATION.md` for detailed endpoint info
3. Test with Swagger UI to isolate issues
4. Verify environment variables in `.env`

Happy coding! 🎉
