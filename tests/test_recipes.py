import json
import logging
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import Settings
from app.models.recipe import RecipeCandidate, RecipeGenerationRequest
from app.services.recipe_service import (
    RecipeService,
    generate_recipes_with_rag,
    generateRecipesWithRag,
    get_recipe_service,
)
from app.services.recommendation_validator import validate_and_assemble_recommendations


@pytest.fixture
def sample_candidates_list():
    return [
        {
            "id": "rec-1",
            "title": "Kremowe Curry z Ciecierzycą",
            "url": "https://kwestiasmaku.com/przepis/kremowe-curry",
            "cooking_time": "25 min",
            "ingredients": ["ciecierzyca", "mleczko kokosowe", "pasta curry"],
            "steps": ["Podsmaż pastę.", "Dodaj mleczko i ciecierzycę.", "Gotuj 15 minut."],
            "source_name": "Kwestia Smaku",
            "description": "Aromatyczne curry.",
            "similarity": 0.92,
            "mood_tags": ["rozgrzewające", "wegetariańskie"],
            "meal_type_tags": ["obiad"],
        },
        {
            "id": "rec-2",
            "title": "Szybka Sałatka z Awokado",
            "url": "https://aniagotuje.pl/przepis/salatka-awokado",
            "time": "10 min",
            "ingredients": ["awokado", "pomidorki", "rukola"],
            "instructions": ["Pokrój warzywa.", "Wymieszaj z sosem."],
            "source": "Ania Gotuje",
            "similarity": 0.88,
            "mood_tags": ["lekkie", "szybkie"],
            "meal_type_tags": ["obiad"],
        },
    ]


@pytest.fixture
def mock_llm_json_response():
    return json.dumps({
        "recommendations": [
            {
                "recipeId": "rec-1",
                "sourceUrl": "https://kwestiasmaku.com/przepis/kremowe-curry",
                "title": "Kremowe Curry z Ciecierzycą",
                "description": "Pyszne, sycące i rozgrzewające curry w 25 minut.",
                "matchScore": 0.95,
                "matchReasons": ["Idealne na szybki obiad", "Bogate w białko roślinne"],
            }
        ]
    })


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
    mock_search = AsyncMock(return_value=[
        {"id": "1", "title": "Testowa potrawa", "similarity": 0.9, "url": "https://example.com/recipe/1"},
        {"id": "2", "title": "Pyszne spaghetti", "similarity": 0.85, "url": "https://example.com/recipe/2"},
    ])
    test_service = RecipeService(
        settings=Settings(OPENAI_API_KEY="test", SUPABASE_URL="https://test.supabase.co", SUPABASE_KEY="test"),
        search_fn=mock_search,
    )

    app.dependency_overrides[get_recipe_service] = lambda: test_service
    try:
        # Case 1: Exclude recipe 1
        payload = {
            "mood": "szybki",
            "mealType": "obiad",
            "excludeUrls": ["https://example.com/recipe/1"],
            "limit": 2,
        }
        response = await async_client.post("/api/v1/recipes/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["trace"]["retrievedCandidatesCount"] == 2
        assert data["trace"]["filteredCandidatesCount"] == 1
        assert data["trace"]["additionalMetadata"]["subtask"] == 5
        assert data["trace"]["additionalMetadata"]["promptCandidatesCount"] == 1
        assert data["trace"]["additionalMetadata"]["systemPromptLength"] > 0
        assert data["trace"]["additionalMetadata"]["userPromptLength"] > 0
        assert data["trace"]["additionalMetadata"]["status"] == "prompts_built"

        # Case 2: No exclusion
        payload_no_exclude = {
            "mood": "szybki",
            "mealType": "obiad",
            "limit": 2,
        }
        response_no_exclude = await async_client.post("/api/v1/recipes/generate", json=payload_no_exclude)
        assert response_no_exclude.status_code == 200
        data_no_exclude = response_no_exclude.json()
        assert data_no_exclude["trace"]["retrievedCandidatesCount"] == 2
        assert data_no_exclude["trace"]["filteredCandidatesCount"] == 2
        assert data_no_exclude["trace"]["additionalMetadata"]["promptCandidatesCount"] == 2
        assert data_no_exclude["trace"]["additionalMetadata"]["status"] == "prompts_built"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_full_pipeline_success(
    sample_candidates_list, mock_llm_json_response, caplog
):
    """
    Test orchestrator executing the full RAG pipeline with mock search and LLM functions.
    Verifies that results are validated and that OpenAI logs are produced.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock(return_value=mock_llm_json_response)

    with caplog.at_level(logging.INFO):
        results = await generate_recipes_with_rag(
            mood="rozgrzewające",
            meal_type="obiad",
            limit=2,
            search_fn=mock_search,
            llm_call_fn=mock_llm,
        )

    assert len(results) == 1
    recipe = results[0]
    assert recipe["id"] == "rec-1"
    assert recipe["title"] == "Kremowe Curry z Ciecierzycą"
    assert recipe["link"] == "https://kwestiasmaku.com/przepis/kremowe-curry"
    assert recipe["matchScore"] == 0.95
    assert recipe["matchReasons"] == ["Idealne na szybki obiad", "Bogate w białko roślinne"]

    mock_search.assert_awaited_once()
    mock_llm.assert_awaited_once()

    # Verify logs for sending and completing OpenAI request
    log_text = caplog.text
    assert "Wysyłanie zapytania do OpenAI" in log_text
    assert "Pomyślnie zrealizowano zapytanie do OpenAI" in log_text
    assert "RAG Augmentation response" in log_text


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_options_dict_and_alias(
    sample_candidates_list, mock_llm_json_response
):
    """
    Test generateRecipesWithRag alias with options dictionary.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock(return_value=mock_llm_json_response)

    options = {
        "excludeUrls": ["https://kwestiasmaku.com/przepis/kremowe-curry"],
        "limit": 1,
        "searchFn": mock_search,
        "llmCallFn": mock_llm,
    }

    # When rec-1 is excluded, only rec-2 is available
    mock_llm.return_value = json.dumps({
        "recommendations": [
            {
                "recipeId": "rec-2",
                "sourceUrl": "https://aniagotuje.pl/przepis/salatka-awokado",
                "title": "Szybka Sałatka z Awokado",
                "matchScore": 0.9,
            }
        ]
    })

    results = await generateRecipesWithRag("lekkie", "obiad", options)
    assert len(results) == 1
    assert results[0]["id"] == "rec-2"
    assert results[0]["title"] == "Szybka Sałatka z Awokado"


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_missing_parameters():
    """
    Test validation errors when mood or meal_type is missing.
    """
    with pytest.raises(ValueError, match="Parametry mood i mealType są wymagane"):
        await generate_recipes_with_rag("", "obiad")

    with pytest.raises(ValueError, match="Parametry mood i mealType są wymagane"):
        await generate_recipes_with_rag("zdrowe", "")


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_empty_retrieval():
    """
    Test orchestrator when retrieval returns no candidates.
    """
    mock_search = AsyncMock(return_value=[])
    mock_llm = AsyncMock()

    results = await generate_recipes_with_rag(
        mood="egzotyczne",
        meal_type="śniadanie",
        search_fn=mock_search,
        llm_call_fn=mock_llm,
    )

    assert results == []
    mock_search.assert_awaited_once()
    mock_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_all_candidates_excluded(sample_candidates_list):
    """
    Test orchestrator when all candidates are excluded.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock()

    results = await generate_recipes_with_rag(
        mood="rozgrzewające",
        meal_type="obiad",
        exclude_urls=[
            "https://kwestiasmaku.com/przepis/kremowe-curry",
            "https://aniagotuje.pl/przepis/salatka-awokado",
        ],
        search_fn=mock_search,
        llm_call_fn=mock_llm,
    )

    assert results == []
    mock_search.assert_awaited_once()
    mock_llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_retrieval_error():
    """
    Test orchestrator error propagation when retrieval fails.
    """
    mock_search = AsyncMock(side_effect=RuntimeError("Vector DB connection error"))

    with pytest.raises(RuntimeError, match="Vector DB connection error"):
        await generate_recipes_with_rag(
            mood="szybkie",
            meal_type="kolacja",
            search_fn=mock_search,
        )


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_llm_error(sample_candidates_list):
    """
    Test orchestrator error propagation when LLM call fails.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock(side_effect=RuntimeError("OpenAI API rate limit exceeded"))

    with pytest.raises(RuntimeError, match="OpenAI API rate limit exceeded"):
        await generate_recipes_with_rag(
            mood="szybkie",
            meal_type="kolacja",
            search_fn=mock_search,
            llm_call_fn=mock_llm,
        )


@pytest.mark.asyncio
async def test_generate_recipes_with_rag_validation_error(sample_candidates_list):
    """
    Test orchestrator error propagation when LLM returns unparseable content.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock(return_value="NOT_A_VALID_JSON_STRING")

    with pytest.raises(ValueError, match="Błąd formatu odpowiedzi LLM"):
        await generate_recipes_with_rag(
            mood="szybkie",
            meal_type="kolacja",
            search_fn=mock_search,
            llm_call_fn=mock_llm,
        )


@pytest.mark.asyncio
async def test_api_endpoint_full_pipeline_success(app: FastAPI, async_client: AsyncClient, sample_candidates_list, mock_llm_json_response):
    """
    Test full API endpoint execution with overridden service dependencies.
    """
    mock_search = AsyncMock(return_value=sample_candidates_list)
    mock_llm = AsyncMock(return_value=mock_llm_json_response)

    test_service = RecipeService(
        settings=Settings(OPENAI_API_KEY="test", SUPABASE_URL="https://test.supabase.co", SUPABASE_KEY="test"),
        search_fn=mock_search,
        llm_fn=mock_llm,
        validator_fn=validate_and_assemble_recommendations,
    )

    app.dependency_overrides[get_recipe_service] = lambda: test_service
    try:
        payload = {
            "mood": "rozgrzewające",
            "mealType": "obiad",
            "limit": 2,
        }
        response = await async_client.post("/api/v1/recipes/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["recipes"]) == 1
        assert data["recipes"][0]["id"] == "rec-1"
        assert data["recipes"][0]["title"] == "Kremowe Curry z Ciecierzycą"
        assert data["trace"]["retrievedCandidatesCount"] == 2
        assert data["trace"]["filteredCandidatesCount"] == 2
        assert data["trace"]["additionalMetadata"]["status"] == "recommendations_validated"
        assert data["trace"]["additionalMetadata"]["validatedRecipesCount"] == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_endpoint_handles_pipeline_exception(app: FastAPI, async_client: AsyncClient):
    """
    Test that API endpoint returns 500 with descriptive error detail when pipeline fails.
    """
    mock_search = AsyncMock(side_effect=RuntimeError("Supabase connection timeout"))

    test_service = RecipeService(
        settings=Settings(OPENAI_API_KEY="test", SUPABASE_URL="https://test.supabase.co", SUPABASE_KEY="test"),
        search_fn=mock_search,
        validator_fn=validate_and_assemble_recommendations,
    )
    # Force generate_recipes to fail if an unhandled error happens
    async def failing_generate(*args, **kwargs):
        raise RuntimeError("Vector database is unreachable")

    test_service.generate_recipes = failing_generate

    app.dependency_overrides[get_recipe_service] = lambda: test_service
    try:
        payload = {
            "mood": "rozgrzewające",
            "mealType": "obiad",
        }
        response = await async_client.post("/api/v1/recipes/generate", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert "Vector database is unreachable" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_no_fallback_implementation_in_python():
    """
    Verify Acceptance Criteria 4: createFallbackRecommendations is not implemented in Python.
    Fallback remains on the client/caller side.
    """
    import app.services.recipe_service as recipe_service_module
    assert not hasattr(recipe_service_module, "createFallbackRecommendations")
    assert not hasattr(recipe_service_module, "create_fallback_recommendations")
