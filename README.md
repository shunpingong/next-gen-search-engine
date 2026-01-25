# NextGen Web Search Engine

Advanced web search engine with PageRank-based result ranking and Tavily integration.

## Overview

This project provides:

- **Tavily Web Search**: Access to comprehensive web search results via Tavily API
- **PageRank Ranking**: Advanced document ranking using PageRank algorithm with query personalization
- **Modern Web Stack**: FastAPI backend + React/TypeScript frontend

## Folders

- `backend/` : FastAPI backend with search and ranking endpoints
- `frontend/` : React + TypeScript frontend

## Quick Start

### Backend

```bash
cd backend
pip install -e .
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Endpoints

- **POST /tavily** - Web search using Tavily API
- **POST /pagerank** - Compute PageRank scores for documents

## Documentation

See backend README for full API documentation.
