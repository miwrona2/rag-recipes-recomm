# RAG Recipe Recommendation Microservice

A culinary recipe recommendation microservice built with Python and the **FastAPI** framework. The service is a core component of the RAG (Retrieval-Augmented Generation) architecture, responsible for candidate retrieval, filtering, prompt context assembly, and recipe synthesis.

---

## 🛠 Requirements

- **Python:** `>= 3.10`
- **Virtual environment:** `venv`

---

## 🚀 Getting Started

### 1. Initialize virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables
Copy the `.env.example` file to `.env` and fill in the required keys:
```bash
cp .env.example .env
```

Configuration variables:
- `HOST`: Host address (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `ENVIRONMENT`: Environment mode (`development` / `production`)
- `LOG_LEVEL`: Logging level (`INFO`, `DEBUG`, `WARNING`)
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: LLM model (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `SUPABASE_URL`: Supabase instance URL
- `SUPABASE_KEY`: Supabase access key

### 3. Run development server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
or
```bash
python run.py
```

---

## 📖 API Documentation

Once the server is running, interactive OpenAPI / Swagger UI documentation is available at:
- **Swagger UI:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **ReDoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)
- **OpenAPI JSON:** [http://localhost:8000/api/v1/openapi.json](http://localhost:8000/api/v1/openapi.json)

---

## 📌 Endpoints

### 1. Healthcheck
- `GET /health` and `GET /api/v1/health`
- **Status:** `200 OK`
```json
{
  "status": "ok",
  "service": "RAG Recipe Recommendation Service",
  "version": "0.1.0",
  "environment": "development"
}
```

### 2. Generate Recipe Recommendations
- `POST /api/v1/recipes/generate`
- **Request Body:**
```json
{
  "mood": "quick and nutritious",
  "mealType": "dinner",
  "excludeUrls": ["https://example.com/recipe/1"],
  "limit": 3
}
```
- **Response `200 OK`:**
```json
{
  "recipes": [],
  "trace": {
    "retrievedCandidatesCount": 0,
    "filteredCandidatesCount": 0,
    "llmModelUsed": "gpt-4o-mini",
    "executionTimeMs": 0.45,
    "additionalMetadata": {
      "status": "initialized",
      "subtask": 1
    }
  }
}
```
- **Validation Error `422 Unprocessable Entity`:**
Returned automatically when required fields (`mood`, `mealType`) are missing or parameter values are invalid (`limit` < 1).

---

## 🧪 Testing

Run unit and integration test suite:
```bash
pytest -v
```
