# Ghost Network - Complete Setup & Development Roadmap

## 🎯 Current Status

### ✅ Phase 1: Backend API (COMPLETED)

- [x] FastAPI server with 30+ endpoints
- [x] 6 service integrations (GitHub, OSV, NVD, PubMed, Graph, LLM)
- [x] PageRank & CheiRank computation
- [x] Tool-Augmented Generation pipeline
- [x] Comprehensive documentation
- [x] Test suite

**Files Created:**

```
backend/
├── main.py                    # Main FastAPI app (500+ lines)
├── services/
│   ├── github_service.py      # GitHub API integration
│   ├── osv_service.py         # OSV vulnerability API
│   ├── nvd_service.py         # NVD/CVE API
│   ├── pubmed_service.py      # PubMed E-utilities
│   ├── graph_service.py       # PageRank/CheiRank
│   └── llm_service.py         # LLM integration
├── pyproject.toml             # Dependencies
├── .env.template              # Environment config template
├── README.md                  # Setup guide
├── API_DOCUMENTATION.md       # Complete API reference
├── QUICKSTART.md              # 5-minute setup
└── test_api.py                # Automated tests
```

---

## 🚀 Setup Instructions (Start Here!)

### Step 1: Install Dependencies

```cmd
cd backend
pip install fastapi uvicorn aiohttp networkx python-dotenv pydantic
```

Or install all at once:

```cmd
pip install -e .
```

### Step 2: Configure Environment

```cmd
copy .env.template .env
```

Edit `.env` file:

```env
# Minimum Required Configuration
GITHUB_TOKEN=ghp_your_github_token_here
PUBMED_EMAIL=your-email@example.com

# Optional (but recommended)
NVD_API_KEY=your_nvd_key
```

**Get Your API Keys:**

1. **GitHub Token**: https://github.com/settings/tokens

   - Click "Generate new token (classic)"
   - Select scopes: `public_repo`, `read:org`
   - Copy token to `.env`

2. **NVD API Key** (Optional): https://nvd.nist.gov/developers/request-an-api-key
   - Improves rate limits for CVE queries

### Step 3: Run the Server

```cmd
python main.py
```

Server will start at:

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs 👈 **Open this!**
- **Alternative Docs**: http://localhost:8000/redoc

### Step 4: Test the API

```cmd
python test_api.py
```

Or open Swagger UI at http://localhost:8000/docs and try the endpoints interactively!

---

## 📋 Phase 2: Frontend Development (TODO)

### Recommended Tech Stack

- **Framework**: React with TypeScript or Vue.js
- **Styling**: Tailwind CSS or Material-UI
- **Graph Viz**: D3.js, vis.js, or Cytoscape.js
- **HTTP Client**: Axios or fetch
- **State Management**: React Context or Zustand

### Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── SearchBar.tsx           # Query input
│   │   ├── DomainSelector.tsx      # Linux/Healthcare toggle
│   │   ├── ResultsDisplay.tsx      # Results grid/list
│   │   ├── GraphVisualization.tsx  # Call graph or citation graph
│   │   ├── VulnerabilityList.tsx   # CVE cards
│   │   ├── RecommendationPanel.tsx # AI suggestions
│   │   └── MetricsDisplay.tsx      # PageRank/CheiRank viz
│   ├── services/
│   │   └── api.ts                  # API client
│   ├── pages/
│   │   ├── SecurityResearch.tsx
│   │   └── HealthcareSearch.tsx
│   ├── App.tsx
│   └── main.tsx
└── package.json
```

### Key Components to Build

#### 1. Search Interface

```typescript
// SearchBar component
interface SearchQuery {
  query: string;
  domain: "linux-kernel" | "healthcare";
  maxResults: number;
  includeVulnerabilities: boolean;
  computeGraph: boolean;
}
```

#### 2. Graph Visualization

```typescript
// Use D3.js or vis.js to visualize call graphs
interface GraphNode {
  id: string;
  label: string;
  pagerank: number;
  cheirank: number;
  criticality: "HIGH" | "MEDIUM" | "LOW";
}

interface GraphEdge {
  from: string;
  to: string;
}
```

#### 3. Results Display

- GitHub repos (cards with stars, description)
- Issues (list with labels, dates)
- Vulnerabilities (cards with severity badges)
- Graph analysis (interactive visualization)
- LLM synthesis (formatted text)

### API Integration Example

```typescript
// api.ts
import axios from "axios";

const API_BASE = "http://localhost:8000";

export const api = {
  async securityQuery(query: SearchQuery) {
    const response = await axios.post(`${API_BASE}/api/security/query`, query);
    return response.data;
  },

  async healthcareQuery(query: HealthcareQuery) {
    const response = await axios.post(
      `${API_BASE}/api/healthcare/query`,
      query
    );
    return response.data;
  },

  async buildGraph(repoOwner: string, repoName: string) {
    const response = await axios.post(`${API_BASE}/api/security/graph/build`, {
      repo_owner: repoOwner,
      repo_name: repoName,
    });
    return response.data;
  },
};
```

---

## 🤖 Phase 3: Local LLM Integration (TODO)

### Option 1: Ollama (Recommended)

#### Installation

1. Download Ollama: https://ollama.ai/
2. Install and start Ollama service
3. Pull a model:

```cmd
ollama pull llama2
```

#### Integration in `llm_service.py`

```python
import aiohttp

class LLMService:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama2"
        self.model_available = True  # Set to True after setup

    async def _call_ollama(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                result = await response.json()
                return result["response"]

    async def synthesize_security_results(
        self,
        query: str,
        github_data: Dict,
        vulnerabilities: List,
        graph_analysis: Dict
    ) -> Dict:
        prompt = f"""
        Analyze this Linux kernel security research query: "{query}"

        GitHub Results: {len(github_data.get('repos', []))} repositories found
        Vulnerabilities: {len(vulnerabilities)} CVEs identified
        Critical Nodes: {graph_analysis.get('critical_nodes', [])}

        Provide:
        1. A concise summary of findings
        2. Top 3 security concerns
        3. Recommended fuzzing targets with rationale
        """

        response = await self._call_ollama(prompt)
        return self._parse_llm_response(response)
```

### Option 2: llama.cpp

```cmd
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download model
# Use with Python bindings
pip install llama-cpp-python
```

### Option 3: Transformers (Hugging Face)

```python
from transformers import pipeline

class LLMService:
    def __init__(self):
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
```

---

## 🔧 Phase 4: Advanced Features (TODO)

### 1. Real Code Parsing

Install tree-sitter for C/C++ parsing:

```cmd
pip install tree-sitter tree-sitter-c
```

Update `graph_service.py`:

```python
import tree_sitter

class GraphService:
    async def parse_c_code(self, file_content: str) -> List[Tuple[str, str]]:
        """Parse C code to extract function calls"""
        parser = tree_sitter.Parser()
        parser.set_language(tree_sitter.Language('build/my-languages.so', 'c'))
        tree = parser.parse(bytes(file_content, "utf8"))

        # Extract function definitions and calls
        # Return list of (caller, callee) tuples
```

### 2. Database Integration

```cmd
pip install sqlalchemy asyncpg
```

Create models:

```python
from sqlalchemy import Column, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True)
    query = Column(String)
    domain = Column(String)
    github_results = Column(JSON)
    vulnerabilities = Column(JSON)
    graph_analysis = Column(JSON)
    created_at = Column(DateTime)
```

### 3. Caching Layer

```cmd
pip install redis aioredis
```

```python
import aioredis

class CacheService:
    def __init__(self):
        self.redis = aioredis.from_url("redis://localhost")

    async def get_cached_result(self, key: str):
        return await self.redis.get(key)

    async def cache_result(self, key: str, value: str, ttl: int = 3600):
        await self.redis.setex(key, ttl, value)
```

### 4. WebSocket Support

```python
from fastapi import WebSocket

@app.websocket("/ws/security/query")
async def websocket_security_query(websocket: WebSocket):
    await websocket.accept()

    data = await websocket.receive_json()
    query = data["query"]

    # Stream progress updates
    await websocket.send_json({"status": "Searching GitHub..."})
    # ... perform analysis ...
    await websocket.send_json({"status": "Computing graph..."})
    # ... send final results ...
```

### 5. Authentication

```cmd
pip install python-jose[cryptography] passlib[bcrypt]
```

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify JWT token
    # Return user
```

### 6. Docker Containerization

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - NVD_API_KEY=${NVD_API_KEY}
      - PUBMED_EMAIL=${PUBMED_EMAIL}
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
```

---

## 📊 Phase 5: Demo & Presentation

### Demo Scenarios

#### Scenario 1: Linux Kernel Security

1. Open frontend
2. Select "Linux Kernel Security" domain
3. Query: "use-after-free vulnerabilities in networking"
4. Show:
   - GitHub repos found
   - CVEs identified
   - Call graph visualization
   - Critical nodes highlighted
   - Fuzzing recommendations

#### Scenario 2: Healthcare Literature

1. Switch to "Healthcare" domain
2. Query: "statin therapy primary prevention"
3. Show:
   - PubMed articles
   - Citation graph
   - Influential papers (high PageRank)
   - Clinical recommendations

### Presentation Slides

1. **Title**: Ghost Network - Future Search Engine for Safety-Critical Domains
2. **Problem**: Traditional search doesn't prioritize by dependency/influence
3. **Solution**: TAG with PageRank/CheiRank
4. **Architecture**: Tool integration diagram
5. **Demo**: Live walkthrough
6. **Results**: Comparison with traditional search
7. **Future Work**: LLM integration, larger graphs

---

## 📝 Phase 6: FYP Report Structure

### Suggested Chapters

1. **Introduction**

   - Motivation (Linux kernel security, healthcare information overload)
   - Research questions
   - Contributions

2. **Literature Review**

   - Tool-Augmented Generation
   - PageRank & CheiRank
   - Linux kernel fuzzing (KernelGPT, AIxCC, syzkaller)
   - Clinical decision support systems

3. **System Design**

   - Architecture
   - Tool integration strategy
   - Graph algorithms
   - API design

4. **Implementation**

   - Technology stack
   - Service modules
   - API endpoints
   - Frontend (if completed)

5. **Evaluation**

   - Performance metrics
   - Accuracy of rankings
   - User study (if conducted)
   - Comparison with baselines

6. **Case Studies**

   - Linux kernel networking vulnerabilities
   - Healthcare literature search

7. **Discussion**

   - Limitations
   - Scalability
   - Privacy considerations
   - Future enhancements

8. **Conclusion**
   - Summary of contributions
   - Impact on security research and healthcare

### Metrics to Include

- API response times
- Graph build performance (nodes, edges, computation time)
- PageRank convergence iterations
- Precision/recall of vulnerability detection
- User satisfaction (if surveyed)

---

## 🎓 Academic Contributions

### Novel Aspects

1. **Dual-domain TAG**: Same architecture for security + healthcare
2. **Graph-based ranking**: PageRank/CheiRank for code and literature
3. **Integrated tool ecosystem**: GitHub + OSV + NVD + PubMed
4. **Actionable outputs**: Fuzzing strategies, not just search results

### Papers to Cite

- PageRank: Brin & Page (1998)
- CheiRank: Ermann & Shepelyansky (2010)
- KernelGPT: Yang et al.
- AIxCC: DARPA Challenge papers
- Tool-Augmented LLMs: Schick et al., Toolformer

---

## ✅ Checklist

### Backend (Done)

- [x] FastAPI server
- [x] Service integrations
- [x] Graph algorithms
- [x] Documentation
- [x] Test suite

### Frontend (TODO)

- [ ] React/Vue setup
- [ ] Search interface
- [ ] Results display
- [ ] Graph visualization
- [ ] Domain switcher

### LLM (TODO)

- [ ] Ollama setup
- [ ] Prompt engineering
- [ ] Response parsing
- [ ] Integration testing

### Advanced (Optional)

- [ ] Code parser (tree-sitter)
- [ ] Database (PostgreSQL)
- [ ] Caching (Redis)
- [ ] WebSocket
- [ ] Authentication
- [ ] Docker

### Demo (TODO)

- [ ] Prepare test queries
- [ ] Record screencast
- [ ] Create slides
- [ ] Practice presentation

### Report (TODO)

- [ ] Write chapters
- [ ] Add diagrams
- [ ] Include screenshots
- [ ] Generate metrics
- [ ] Proofread

---

## 🚦 Priority Roadmap

### Week 1-2: MVP Demo Ready

1. ✅ Backend API running
2. Test all endpoints with Swagger UI
3. Create video demo using Postman/cURL
4. Write intro + design sections of report

### Week 3-4: Frontend

1. Set up React project
2. Build search interface
3. Implement results display
4. Add basic graph visualization

### Week 5-6: LLM Integration

1. Install Ollama
2. Integrate with `llm_service.py`
3. Test prompt engineering
4. Refine outputs

### Week 7-8: Polish & Report

1. Complete frontend features
2. Record final demo
3. Finish report
4. Prepare presentation

---

## 📞 Need Help?

### Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **NetworkX Docs**: https://networkx.org/
- **Ollama**: https://ollama.ai/
- **React**: https://react.dev/

### Testing

```cmd
# Health check
curl http://localhost:8000/health

# Run full test suite
python test_api.py

# Open interactive docs
start http://localhost:8000/docs
```

### Common Issues

1. **Port in use**: Change port in main.py or use `--port 8001`
2. **Import errors**: Run `pip install -e .`
3. **API rate limits**: Add tokens to `.env`
4. **CORS errors**: Add frontend URL to `CORS_ORIGINS`

---

## 🎉 You're Ready!

Your Ghost Network backend is fully functional with:

- ✅ 30+ API endpoints
- ✅ 6 integrated tools
- ✅ PageRank & CheiRank
- ✅ Complete documentation
- ✅ Test suite

**Next step**: Run `python main.py` and open http://localhost:8000/docs

Good luck with your FYP! 🚀
