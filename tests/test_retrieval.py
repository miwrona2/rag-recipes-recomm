import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.embedding_service import get_embedding
from app.services.vector_search import (
    get_supabase_client,
    search_recipes_in_vector_db,
    retrieve_candidates,
)
from app.services.recipe_service import RecipeService
from app.models.recipe import RecipeGenerationRequest
from app.core.config import Settings


@pytest.mark.asyncio
async def test_get_embedding_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_item]

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    embedding = await get_embedding("zdrowy obiad", client=mock_client)
    assert embedding == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_awaited_once_with(
        input="zdrowy obiad",
        model="text-embedding-3-small",
    )


@pytest.mark.asyncio
async def test_get_embedding_invalid_input():
    with pytest.raises(ValueError, match="niepustym ciągiem znaków"):
        await get_embedding("")

    with pytest.raises(ValueError, match="niepustym ciągiem znaków"):
        await get_embedding("   ")


@pytest.mark.asyncio
async def test_get_embedding_missing_key(monkeypatch):
    test_settings = Settings(OPENAI_API_KEY="")
    monkeypatch.setattr("app.services.embedding_service.get_settings", lambda: test_settings)

    with pytest.raises(ValueError, match="Brak klucza OpenAI"):
        await get_embedding("test")


def test_get_supabase_client_missing_config(monkeypatch):
    test_settings = Settings(SUPABASE_URL="", SUPABASE_KEY="")
    monkeypatch.setattr("app.services.vector_search.get_settings", lambda: test_settings)

    with pytest.raises(ValueError, match="Brak konfiguracji Supabase"):
        get_supabase_client()


@pytest.mark.asyncio
async def test_search_recipes_in_vector_db_with_mock_client():
    mock_client = MagicMock()
    mock_rpc = MagicMock()
    mock_rpc.execute.return_value = MagicMock(
        data=[
            {
                "id": "1",
                "title": "Krem z dyni",
                "mood_tags": ["rozgrzewające", "comfort-food"],
                "meal_type_tags": ["obiad"],
                "similarity": 0.85,
            },
            {
                "id": "2",
                "title": "Owsianka owocowa",
                "mood_tags": ["lekkie", "szybkie"],
                "meal_type_tags": ["śniadanie"],
                "similarity": 0.75,
            },
        ]
    )
    mock_client.rpc.return_value = mock_rpc

    mock_embedding = [0.05] * 1536

    # Test retrieval with mood and meal_type filtering
    results = await search_recipes_in_vector_db(
        query="rozgrzewający obiad",
        match_count=10,
        match_threshold=0.35,
        mood="rozgrzewające",
        meal_type="obiad",
        embedding=mock_embedding,
        supabase_client=mock_client,
    )

    mock_client.rpc.assert_called_once_with(
        "match_recipes",
        {
            "query_embedding": mock_embedding,
            "match_threshold": 0.35,
            "match_count": 10,
        },
    )
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["title"] == "Krem z dyni"


@pytest.mark.asyncio
async def test_retrieve_candidates_success():
    fake_candidates = [
        {"id": "101", "title": "Sałatka Cezar", "similarity": 0.88},
        {"id": "102", "title": "Zupa Pomidorowa", "similarity": 0.82},
    ]

    mock_search = AsyncMock(return_value=fake_candidates)

    candidates = await retrieve_candidates(
        mood="szybkie",
        meal_type="lunch",
        match_count=15,
        match_threshold=0.3,
        search_fn=mock_search,
    )

    assert candidates == fake_candidates
    mock_search.assert_awaited_once_with(
        query="szybkie lunch",
        match_count=15,
        match_threshold=0.3,
        mood="szybkie",
        meal_type="lunch",
        supabase_client=None,
        embedding_fn=None,
        embedding=None,
    )


@pytest.mark.asyncio
async def test_retrieve_candidates_empty():
    mock_search = AsyncMock(return_value=[])

    candidates = await retrieve_candidates(
        mood="egzotyczne",
        meal_type="deser",
        search_fn=mock_search,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_recipe_service_retrieval_integration():
    fake_candidates = [
        {"id": "rec-1", "title": "Pikantne Tacos", "similarity": 0.91},
        {"id": "rec-2", "title": "Guacamole z nachos", "similarity": 0.84},
    ]
    mock_search = AsyncMock(return_value=fake_candidates)

    settings = Settings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_KEY="fake-key",
        OPENAI_API_KEY="fake-openai-key",
    )
    service = RecipeService(settings=settings, search_fn=mock_search)

    request = RecipeGenerationRequest(
        mood="pikantne",
        mealType="kolacja",
        limit=3,
    )

    response = await service.generate_recipes(request)

    assert response.trace is not None
    assert response.trace.retrievedCandidatesCount == 2
    assert response.trace.filteredCandidatesCount == 2
    assert response.trace.additionalMetadata["subtask"] == 4
    assert response.trace.additionalMetadata["status"] == "context_prepared"
    assert response.trace.additionalMetadata["promptCandidatesCount"] == 2
