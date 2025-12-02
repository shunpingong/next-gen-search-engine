# Ghost Network API - Comprehensive Documentation

## Overview

The Ghost Network API is a Tool-Augmented Generation (TAG) system designed for two primary domains:

1. **Linux Kernel Security**: Security research, vulnerability analysis, and fuzzing strategy generation
2. **Healthcare Literature Search**: Medical literature search and clinical decision support

## Table of Contents

1. [API Endpoints Reference](#api-endpoints-reference)
2. [Service Integrations](#service-integrations)
3. [Usage Examples](#usage-examples)
4. [Architecture](#architecture)

---

## API Endpoints Reference

### 1. Linux Kernel Security Endpoints

#### 1.1 Main Security Query (TAG)

**Endpoint:** `POST /api/security/query`

This is the primary TAG endpoint that orchestrates multiple tools to provide comprehensive security analysis.

**Request Body:**

```json
{
  "query": "race conditions in Linux kernel networking",
  "domain": "linux-kernel",
  "max_results": 10,
  "include_vulnerabilities": true,
  "compute_graph": true
}
```

**Parameters:**

- `query` (string, required): Security research query
- `domain` (string): Specific domain (linux-kernel, networking, filesystem, etc.)
- `max_results` (int): Maximum results per search type
- `include_vulnerabilities` (bool): Whether to check for known CVEs
- `compute_graph` (bool): Whether to build and analyze call graph

**Response:**

```json
{
  "query": "race conditions in Linux kernel networking",
  "github_results": {
    "repos": [...],
    "issues": [...],
    "code": [...]
  },
  "vulnerabilities": [...],
  "graph_analysis": {
    "nodes": 20,
    "edges": 35,
    "pagerank": {...},
    "cheirank": {...},
    "critical_nodes": [...]
  },
  "llm_synthesis": {
    "summary": "...",
    "key_findings": [...],
    "recommendations": [...]
  }
}
```

#### 1.2 GitHub Search

**Endpoint:** `GET /api/security/github/search`

**Query Parameters:**

- `query` (string, required): Search query
- `search_type` (string): repositories, issues, or code
- `max_results` (int): Maximum results (1-100)

**Example:**

```
GET /api/security/github/search?query=syzkaller+fuzzing&search_type=repositories&max_results=10
```

#### 1.3 Repository Issues

**Endpoint:** `GET /api/security/github/repo/{owner}/{repo}/issues`

**Path Parameters:**

- `owner` (string): Repository owner
- `repo` (string): Repository name

**Query Parameters:**

- `labels` (string): Comma-separated labels (e.g., "security,bug")
- `state` (string): open, closed, or all

**Example:**

```
GET /api/security/github/repo/torvalds/linux/issues?labels=security&state=open
```

#### 1.4 Repository Files

**Endpoint:** `GET /api/security/github/repo/{owner}/{repo}/files`

**Query Parameters:**

- `path` (string): Path within repository
- `pattern` (string): File pattern (e.g., "\*.c")

**Example:**

```
GET /api/security/github/repo/torvalds/linux/files?path=net&pattern=*.c
```

#### 1.5 Check Vulnerabilities

**Endpoint:** `POST /api/security/vulnerabilities/check`

Check packages against OSV.dev vulnerability database.

**Request Body:**

```json
{
  "packages": ["torvalds/linux", "systemd/systemd"],
  "ecosystem": "Linux"
}
```

**Response:**

```json
[
  {
    "id": "OSV-2024-1234",
    "package": "torvalds/linux",
    "summary": "...",
    "severity": "HIGH",
    "published": "2024-01-15",
    "references": [...]
  }
]
```

#### 1.6 Get CVE Details

**Endpoint:** `GET /api/security/vulnerabilities/cve/{cve_id}`

**Example:**

```
GET /api/security/vulnerabilities/cve/CVE-2024-1234
```

**Response:**

```json
{
  "id": "CVE-2024-1234",
  "description": "...",
  "cvss_v3": {
    "score": 7.5,
    "severity": "HIGH",
    "vector": "..."
  },
  "published": "2024-01-15",
  "references": [...]
}
```

#### 1.7 Search Vulnerabilities

**Endpoint:** `GET /api/security/vulnerabilities/search`

**Query Parameters:**

- `keyword` (string, required): Search keyword
- `max_results` (int): Maximum results (1-100)

**Example:**

```
GET /api/security/vulnerabilities/search?keyword=linux+kernel+network&max_results=20
```

#### 1.8 Build Call Graph

**Endpoint:** `POST /api/security/graph/build`

Build call graph for a repository and compute PageRank/CheiRank.

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

**Response:**

```json
{
  "repository": "torvalds/linux",
  "nodes": 150,
  "edges": 320,
  "graph_metrics": {
    "density": 0.015
  },
  "pagerank": {
    "function_a": 0.025,
    "function_b": 0.018,
    ...
  },
  "cheirank": {
    "function_a": 0.032,
    ...
  },
  "top_pagerank_nodes": [...],
  "top_cheirank_nodes": [...],
  "critical_nodes": [
    {
      "node": "tcp_v4_connect",
      "combined_score": 0.028,
      "pagerank": 0.025,
      "cheirank": 0.032,
      "criticality": "HIGH"
    }
  ]
}
```

#### 1.9 Fuzzing Suggestions

**Endpoint:** `GET /api/security/fuzzing/suggestions`

Generate AI-powered fuzzing strategy recommendations.

**Query Parameters:**

- `repo_owner` (string, required): Repository owner
- `repo_name` (string, required): Repository name
- `focus_area` (string): e.g., networking, filesystem

**Example:**

```
GET /api/security/fuzzing/suggestions?repo_owner=torvalds&repo_name=linux&focus_area=networking
```

**Response:**

```json
{
  "focus_area": "networking",
  "approach": "multi-target",
  "tools": ["syzkaller", "AFL++", "libFuzzer"],
  "targets": [
    {
      "function": "tcp_v4_connect",
      "score": 0.032,
      "reason": "High CheiRank - calls many functions",
      "priority": "HIGH"
    }
  ],
  "harness_suggestions": [...],
  "recommendation": "..."
}
```

---

### 2. Healthcare Literature Endpoints

#### 2.1 Main Healthcare Query (TAG)

**Endpoint:** `POST /api/healthcare/query`

**Request Body:**

```json
{
  "query": "treatment protocols for type 2 diabetes",
  "specialty": "endocrinology",
  "max_results": 20,
  "include_local_docs": false
}
```

**Response:**

```json
{
  "query": "...",
  "pubmed_results": {
    "total_count": 20,
    "articles": [...]
  },
  "local_documents": [],
  "citation_analysis": {
    "nodes": 20,
    "edges": 45,
    "pagerank": {...},
    "top_papers": [...]
  },
  "llm_synthesis": {
    "summary": "...",
    "clinical_recommendations": [...]
  }
}
```

#### 2.2 PubMed Search

**Endpoint:** `GET /api/healthcare/pubmed/search`

**Query Parameters:**

- `query` (string, required): Search query
- `max_results` (int): Maximum results (1-100)
- `sort` (string): relevance, date, or citations

**Example:**

```
GET /api/healthcare/pubmed/search?query=hypertension+treatment&max_results=20&sort=relevance
```

**Response:**

```json
{
  "total_count": 20,
  "query": "hypertension treatment",
  "articles": [
    {
      "pmid": "12345678",
      "title": "...",
      "abstract": "...",
      "authors": ["John Doe", "Jane Smith"],
      "journal": "New England Journal of Medicine",
      "publication_date": "2024-01-15",
      "mesh_terms": ["Hypertension", "Drug Therapy"],
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    }
  ]
}
```

#### 2.3 Get Article Details

**Endpoint:** `GET /api/healthcare/pubmed/article/{pmid}`

**Example:**

```
GET /api/healthcare/pubmed/article/12345678
```

#### 2.4 Get Related Articles

**Endpoint:** `GET /api/healthcare/pubmed/related/{pmid}`

**Query Parameters:**

- `max_results` (int): Maximum related articles (1-50)

**Example:**

```
GET /api/healthcare/pubmed/related/12345678?max_results=10
```

---

### 3. Graph Analysis Endpoints

#### 3.1 Compute PageRank

**Endpoint:** `GET /api/graph/pagerank`

**Query Parameters:**

- `nodes` (string, required): Comma-separated node IDs
- `edges` (string, required): Comma-separated edges (from-to pairs, e.g., "A-B,B-C")

**Example:**

```
GET /api/graph/pagerank?nodes=A,B,C,D&edges=A-B,B-C,C-D,D-A
```

**Response:**

```json
{
  "A": 0.25,
  "B": 0.3,
  "C": 0.25,
  "D": 0.2
}
```

#### 3.2 Compute CheiRank

**Endpoint:** `GET /api/graph/cheirank`

Same parameters as PageRank.

---

### 4. LLM/AI Endpoints

#### 4.1 Summarize Text

**Endpoint:** `POST /api/llm/summarize`

**Query Parameters:**

- `text` (string, required): Text to summarize
- `max_length` (int): Maximum summary length

**Example:**

```
POST /api/llm/summarize?text=<long_text>&max_length=200
```

#### 4.2 Question Answering

**Endpoint:** `POST /api/llm/qa`

**Query Parameters:**

- `question` (string, required): Question to answer
- `context` (string, required): Context to answer from

---

### 5. Utility Endpoints

#### 5.1 Health Check

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "services": {
    "github": true,
    "osv": true,
    "nvd": true,
    "pubmed": true,
    "llm": false
  }
}
```

#### 5.2 Supported Domains

**Endpoint:** `GET /api/domains`

**Response:**

```json
{
  "domains": [
    {
      "id": "linux-kernel-security",
      "name": "Linux Kernel Security",
      "description": "...",
      "tools": ["GitHub", "OSV.dev", "NVD", "Local LLM"],
      "use_cases": [...]
    }
  ]
}
```

---

## Service Integrations

### GitHub API

- **Purpose**: Code search, repository analysis, issue tracking
- **Authentication**: Bearer token (GITHUB_TOKEN)
- **Rate Limits**: 5,000/hour (authenticated), 60/hour (unauthenticated)

### OSV.dev API

- **Purpose**: Open-source vulnerability database
- **Authentication**: None required
- **Rate Limits**: Reasonable use

### NVD API

- **Purpose**: CVE/vulnerability details
- **Authentication**: API key (optional, improves rate limits)
- **Rate Limits**: 50/30s (with key), 5/30s (without)

### PubMed E-utilities

- **Purpose**: Medical literature search
- **Authentication**: Email required
- **Rate Limits**: 3/second

---

## Usage Examples

### Example 1: Complete Security Analysis

```python
import requests

# Step 1: Main TAG query
response = requests.post(
    "http://localhost:8000/api/security/query",
    json={
        "query": "use-after-free vulnerabilities in Linux kernel",
        "domain": "linux-kernel",
        "max_results": 10,
        "include_vulnerabilities": true,
        "compute_graph": true
    }
)

result = response.json()

# Step 2: Get detailed fuzzing strategy
if result.get("github_results", {}).get("repos"):
    top_repo = result["github_results"]["repos"][0]

    fuzz_response = requests.get(
        f"http://localhost:8000/api/security/fuzzing/suggestions",
        params={
            "repo_owner": top_repo["owner"],
            "repo_name": top_repo["name"],
            "focus_area": "memory-management"
        }
    )

    strategy = fuzz_response.json()
    print("Fuzzing targets:", strategy["targets"])
```

### Example 2: Healthcare Literature Review

```python
# Step 1: Search PubMed
response = requests.post(
    "http://localhost:8000/api/healthcare/query",
    json={
        "query": "machine learning for cancer diagnosis",
        "specialty": "oncology",
        "max_results": 30
    }
)

result = response.json()

# Step 2: Get related articles for top paper
if result["pubmed_results"]["articles"]:
    top_pmid = result["pubmed_results"]["articles"][0]["pmid"]

    related = requests.get(
        f"http://localhost:8000/api/healthcare/pubmed/related/{top_pmid}",
        params={"max_results": 10}
    )

    print("Related papers:", related.json())
```

### Example 3: Custom Graph Analysis

```python
# Build your own graph
nodes = ["func_a", "func_b", "func_c", "func_d"]
edges = [
    ("func_a", "func_b"),
    ("func_b", "func_c"),
    ("func_c", "func_d"),
    ("func_d", "func_a"),
    ("func_a", "func_c")
]

# Compute PageRank
pagerank = requests.get(
    "http://localhost:8000/api/graph/pagerank",
    params={
        "nodes": ",".join(nodes),
        "edges": ",".join([f"{e[0]}-{e[1]}" for e in edges])
    }
)

# Compute CheiRank
cheirank = requests.get(
    "http://localhost:8000/api/graph/cheirank",
    params={
        "nodes": ",".join(nodes),
        "edges": ",".join([f"{e[0]}-{e[1]}" for e in edges])
    }
)

print("PageRank:", pagerank.json())
print("CheiRank:", cheirank.json())
```

---

## Architecture

### Service Layer

```
services/
├── github_service.py      # GitHub REST API client
├── osv_service.py         # OSV.dev API client
├── nvd_service.py         # NVD API client
├── pubmed_service.py      # PubMed E-utilities client
├── graph_service.py       # NetworkX-based graph analysis
└── llm_service.py         # Local LLM integration
```

### Workflow: Security Query

1. **GitHub Search**: Find relevant repos/issues/code
2. **Vulnerability Check**: Query OSV.dev and NVD for known CVEs
3. **Graph Building**: Parse code to build call graph
4. **PageRank/CheiRank**: Identify critical nodes
5. **LLM Synthesis**: Generate human-readable summary and recommendations

### Workflow: Healthcare Query

1. **PubMed Search**: Find relevant medical literature
2. **Local Search**: Search hospital documents (if enabled)
3. **Citation Graph**: Build paper citation network
4. **PageRank**: Identify influential papers
5. **LLM Synthesis**: Generate clinical summary

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:

```json
{
  "detail": "Error message here"
}
```

---

## Rate Limiting Recommendations

1. **Cache responses** where possible
2. **Batch requests** for efficiency
3. **Use API keys** for higher rate limits
4. **Implement exponential backoff** for retries
5. **Monitor rate limit headers** in responses
