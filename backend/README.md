# Ghost Network - Tool-Augmented Generation API

This backend provides REST API endpoints for the "Ghost Network" project - a Tool-Augmented Generation (TAG) system for Linux Kernel Security research and Healthcare literature search.

## Features

### Domain 1: Linux Kernel Security

- **GitHub Integration**: Search repositories, issues, and code
- **Vulnerability Detection**: Integration with OSV.dev and NVD APIs
- **Call Graph Analysis**: Build and analyze code call graphs
- **PageRank/CheiRank**: Identify critical code paths and functions
- **Fuzzing Strategy Generation**: AI-assisted fuzzing recommendations

### Domain 2: Healthcare Literature Search

- **PubMed Integration**: Search medical literature
- **Citation Analysis**: Build and analyze citation graphs
- **Clinical Synthesis**: Clinician-friendly summaries

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

- **GITHUB_TOKEN** (Required): Get from https://github.com/settings/tokens
  - Needed for GitHub API access (higher rate limits)
- **NVD_API_KEY** (Optional): Register at https://nvd.nist.gov/developers/request-an-api-key
  - Improves rate limits for CVE queries
- **PUBMED_EMAIL** (Required): Your email address
  - Required by NCBI for PubMed API usage

### 3. Run the Server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

#### Health Check

```
GET /health
```

Check status of all integrated services.

### Linux Kernel Security Domain

#### Main Security Query (TAG)

```
POST /api/security/query
```

Main tool-augmented generation endpoint. Combines GitHub search, vulnerability detection, graph analysis, and LLM synthesis.

**Request Body:**

```json
{
  "query": "networking vulnerabilities in Linux kernel",
  "domain": "linux-kernel",
  "max_results": 10,
  "include_vulnerabilities": true,
  "compute_graph": true
}
```

#### GitHub Search

```
GET /api/security/github/search?query=fuzzing&search_type=repositories
```

#### Repository Issues

```
GET /api/security/github/repo/{owner}/{repo}/issues
```

#### Vulnerability Check

```
POST /api/security/vulnerabilities/check
```

Check packages against OSV.dev vulnerability database.

#### CVE Details

```
GET /api/security/vulnerabilities/cve/{cve_id}
```

#### Build Call Graph

```
POST /api/security/graph/build
```

Build call graph and compute PageRank/CheiRank.

**Request Body:**

```json
{
  "repo_owner": "torvalds",
  "repo_name": "linux",
  "file_pattern": "*.c",
  "compute_pagerank": true,
  "compute_cheirank": true
}
```

#### Fuzzing Suggestions

```
GET /api/security/fuzzing/suggestions?repo_owner=torvalds&repo_name=linux
```

Get AI-generated fuzzing strategy recommendations.

### Healthcare Domain

#### Main Healthcare Query (TAG)

```
POST /api/healthcare/query
```

Main tool-augmented generation endpoint for medical literature.

**Request Body:**

```json
{
  "query": "treatment protocols for hypertension",
  "specialty": "cardiology",
  "max_results": 10,
  "include_local_docs": false
}
```

#### PubMed Search

```
GET /api/healthcare/pubmed/search?query=cancer+treatment&max_results=20
```

#### Article Details

```
GET /api/healthcare/pubmed/article/{pmid}
```

#### Related Articles

```
GET /api/healthcare/pubmed/related/{pmid}
```

### Graph Analysis

#### Compute PageRank

```
GET /api/graph/pagerank?nodes=A,B,C&edges=A-B,B-C,C-A
```

#### Compute CheiRank

```
GET /api/graph/cheirank?nodes=A,B,C&edges=A-B,B-C,C-A
```

### LLM Services

#### Summarize Text

```
POST /api/llm/summarize?text=<long_text>&max_length=200
```

#### Question Answering

```
POST /api/llm/qa?question=<question>&context=<context>
```

## Architecture

```
backend/
├── main.py                 # FastAPI app with all endpoints
├── services/               # Service modules
│   ├── github_service.py   # GitHub API integration
│   ├── osv_service.py      # OSV.dev vulnerability API
│   ├── nvd_service.py      # NVD/CVE API
│   ├── pubmed_service.py   # PubMed E-utilities API
│   ├── graph_service.py    # PageRank/CheiRank computation
│   └── llm_service.py      # Local LLM integration
└── scrapers/               # Existing scraper modules
```

## Example Usage

### Example 1: Security Analysis

```python
import requests

# Main TAG query for Linux kernel security
response = requests.post(
    "http://localhost:8000/api/security/query",
    json={
        "query": "race conditions in network drivers",
        "domain": "networking",
        "max_results": 10,
        "include_vulnerabilities": true,
        "compute_graph": true
    }
)

result = response.json()
print(f"Found {len(result['github_results']['repos'])} repositories")
print(f"Identified {len(result['vulnerabilities'])} vulnerabilities")
print(f"Top fuzzing targets: {result['recommendations']}")
```

### Example 2: Healthcare Search

```python
# Search medical literature
response = requests.post(
    "http://localhost:8000/api/healthcare/query",
    json={
        "query": "COVID-19 treatment protocols",
        "specialty": "infectious disease",
        "max_results": 20
    }
)

result = response.json()
print(f"Found {result['total_count']} relevant articles")
print(f"Clinical recommendations: {result['clinical_recommendations']}")
```

### Example 3: Graph Analysis

```python
# Build call graph for a repository
response = requests.post(
    "http://localhost:8000/api/security/graph/build",
    json={
        "repo_owner": "torvalds",
        "repo_name": "linux",
        "file_pattern": "*.c",
        "compute_pagerank": true,
        "compute_cheirank": true
    }
)

graph = response.json()
print(f"Critical nodes (high PageRank + CheiRank):")
for node in graph['critical_nodes'][:5]:
    print(f"  - {node['node']}: {node['criticality']}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
ruff check .
```

## API Rate Limits

- **GitHub**: 5,000 requests/hour (authenticated), 60/hour (unauthenticated)
- **NVD**: 50 requests/30 seconds (with API key), 5/30 seconds (without)
- **PubMed**: 3 requests/second (with API key), 3/second (without, but limited)

## Future Enhancements

1. **Local LLM Integration**: Add Ollama/llama.cpp integration for synthesis
2. **Code Parsing**: Implement actual C/C++ parsing for call graph extraction
3. **Caching**: Add Redis caching for API responses
4. **Authentication**: Add API key authentication
5. **WebSocket**: Real-time updates for long-running queries
6. **Database**: Store analysis results and graph data
7. **Docker**: Containerization for easy deployment

## References

- [GitHub REST API](https://docs.github.com/en/rest)
- [OSV.dev API](https://osv.dev/docs/)
- [NVD API](https://nvd.nist.gov/developers)
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [PageRank Algorithm](https://en.wikipedia.org/wiki/PageRank)
- [CheiRank Algorithm](https://arxiv.org/abs/1003.1925)
