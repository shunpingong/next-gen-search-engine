# Backend Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install numpy
# or install all dependencies
pip install -e .
```

### 2. Configure Environment Variables

1. Copy the example environment file:

   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your API keys:

#### GitHub Token (Required for GitHub API)

- Go to https://github.com/settings/tokens
- Click "Generate new token" → "Generate new token (classic)"
- Select scopes: `public_repo`, `read:user`
- Copy the token and paste it in `.env`:
  ```
  GITHUB_TOKEN=ghp_your_token_here
  ```

#### Optional API Keys

- **NVD API Key**: https://nvd.nist.gov/developers/request-an-api-key
- **PubMed API Key**: https://www.ncbi.nlm.nih.gov/account/

### 3. Run the Server

```bash
cd backend
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## Troubleshooting

### Error: "GitHub API error: 401"

- **Cause**: Missing or invalid GitHub token
- **Fix**:
  1. Create a `.env` file in the `backend` directory
  2. Add your GitHub token: `GITHUB_TOKEN=your_token_here`
  3. Restart the server

### Error: "No module named 'numpy'"

- **Cause**: Missing numpy dependency
- **Fix**: `pip install numpy`

### Error: "No module named 'dotenv'"

- **Cause**: Missing python-dotenv dependency
- **Fix**: `pip install python-dotenv`

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing the API

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:

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
