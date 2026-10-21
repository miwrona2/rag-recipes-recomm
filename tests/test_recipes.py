import pytest
from fastapi import FastAPI
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_generate_recipes_valid_payload(async_client: AsyncClient):
    payload = {
        "mood": "szybki i pożywny",
        "mealType": "obiad",
        "excludeUrls": ["https://example.com/recipe/1"],
        "limit": 3,
    }
    response = await async_client.post("/api/v1/recipes/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recipes" in data
    assert "trace" in data
    assert isinstance(data["recipes"], list)
    assert data["trace"]["retrievedCandidatesCount"] == 0
    assert data["trace"]["executionTimeMs"] >= 0


@pytest.mark.asyncio
async def test_generate_recipes_missing_mood(async_client: AsyncClient):
    payload = {
        "mealType": "obiad",
    }
    response = await async_client.post("/api/v1/recipes/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert any(err["loc"][-1] == "mood" for err in data["details"])


@pytest.mark.asyncio
async def test_generate_recipes_missing_meal_type(async_client: AsyncClient):
    payload = {
        "mood": "energetyczny",
    }
    response = await async_client.post("/api/v1/recipes/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert any(err["loc"][-1] == "mealType" for err in data["details"])


@pytest.mark.asyncio
async def test_generate_recipes_invalid_limit(async_client: AsyncClient):
    payload = {
        "mood": "lekki",
        "mealType": "kolacja",
        "limit": 0,  # limit must be >= 1
    }
    response = await async_client.post("/api/v1/recipes/generate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert any(err["loc"][-1] == "limit" for err in data["details"])


@pytest.mark.asyncio
async def test_response_headers_middleware(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


@pytest.mark.asyncio
async def test_generate_recipes_with_service_override(app: FastAPI, async_client: AsyncClient):
    from unittest.mock import AsyncMock
    from app.services.recipe_service import RecipeService, get_recipe_service
    from app.core.config import Settings

    mock_search = AsyncMock(return_value=[
        {"id": "1", "title": "Testowa potrawa", "similarity": 0.9}
    ])
    test_service = RecipeService(
        settings=Settings(OPENAI_API_KEY="test", SUPABASE_URL="https://test.supabase.co", SUPABASE_KEY="test"),
        search_fn=mock_search
    )

    app.dependency_overrides[get_recipe_service] = lambda: test_service
    try:
        payload = {
            "mood": "szybki",
            "mealType": "obiad",
            "limit": 2,
        }
        response = await async_client.post("/api/v1/recipes/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["trace"]["retrievedCandidatesCount"] == 1
        assert data["trace"]["additionalMetadata"]["subtask"] == 2
    finally:
        app.dependency_overrides.clear()
