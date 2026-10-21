# RAG Recipe Recommendation Microservice

Mikroserwis rekomendacji przepisów kulinarnych oparty na Pythonie i frameworku **FastAPI**. Serwis stanowi kluczowy komponent architektury RAG (Retrieval-Augmented Generation), odpowiadając za wyszukiwanie kandydatów, filtrowanie, budowanie kontekstu promptu oraz syntezę przepisów.

---

## 🛠 Wymagania

- **Python:** `>= 3.10`
- **Wirtualne środowisko:** `venv`

---

## 🚀 Instalacja i Uruchomienie

### 1. Inicjalizacja środowiska wirtualnego
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Konfiguracja zmiennych środowiskowych
Skopiuj plik `.env.example` do `.env` i uzupełnij klucze:
```bash
cp .env.example .env
```

Zmienne konfiguracyjne:
- `HOST`: Adres hosta (domyślnie `0.0.0.0`)
- `PORT`: Port serwera (domyślnie `8000`)
- `ENVIRONMENT`: Środowisko (`development` / `production`)
- `LOG_LEVEL`: Poziom logowania (`INFO`, `DEBUG`, `WARNING`)
- `OPENAI_API_KEY`: Klucz API do OpenAI
- `OPENAI_MODEL`: Model LLM (domyślnie `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL`: Model embeddingów (domyślnie `text-embedding-3-small`)
- `SUPABASE_URL`: Adres instancji Supabase
- `SUPABASE_KEY`: Klucz dostępu Supabase

### 3. Uruchomienie serwera deweloperskiego
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
lub
```bash
python run.py
```

---

## 📖 Dokumentacja API

Po uruchomieniu serwera interaktywna dokumentacja OpenAPI / Swagger UI jest dostępna pod adresem:
- **Swagger UI:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
- **ReDoc:** [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)
- **OpenAPI JSON:** [http://localhost:8000/api/v1/openapi.json](http://localhost:8000/api/v1/openapi.json)

---

## 📌 Endpointy

### 1. Healthcheck
- `GET /health` oraz `GET /api/v1/health`
- **Status:** `200 OK`
```json
{
  "status": "ok",
  "service": "RAG Recipe Recommendation Service",
  "version": "0.1.0",
  "environment": "development"
}
```

### 2. Generowanie rekomendacji przepisów
- `POST /api/v1/recipes/generate`
- **Request Body:**
```json
{
  "mood": "szybki i pożywny",
  "mealType": "obiad",
  "excludeUrls": ["https://example.com/recipe/1"],
  "limit": 3,
  "availableIngredients": ["kurczak", "czosnek", "oliwa"]
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
- **Błąd walidacji `422 Unprocessable Entity`:**
Zwracany automatycznie w przypadku braku wymaganych pól (`mood`, `mealType`) lub niepoprawnych wartości parametrów (`limit` < 1).

---

## 🧪 Testy

Uruchomienie zestawu testów jednostkowych i integracyjnych:
```bash
pytest -v
```
