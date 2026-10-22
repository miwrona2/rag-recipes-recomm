"""
Unit tests for execute_llm_call and LLM execution pipeline.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.llm_service import execute_llm_call
from app.services.recipe_service import RecipeService
from app.models.recipe import RecipeGenerationRequest
from app.core.config import Settings


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_1_and_2_openai_call():
    """
    Test AC 1 & AC 2:
    - Default model is 'gpt-4o-mini'
    - Temperature is 0.3
    - response_format is {'type': 'json_object'}
    - Returns raw JSON string from choices[0].message.content
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    expected_json_str = json.dumps({
        "recommendations": [
            {
                "recipeId": "rec-1",
                "title": "Krem z dyni",
                "matchScore": 0.95,
                "matchReasons": ["Rozgrzewające", "Szybkie w przygotowaniu"],
            }
        ]
    })
    mock_message.content = expected_json_str
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    system_prompt = "You are a recipe recommendation assistant."
    user_prompt = "Find a recipe for dinner."

    result = await execute_llm_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        client=mock_client,
        api_key="fake-key",
    )

    assert result == expected_json_str
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )


@pytest.mark.asyncio
async def test_execute_llm_call_custom_model_override():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"recommendations": []}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await execute_llm_call(
        system_prompt="system",
        user_prompt="user",
        model="gpt-4o",
        client=mock_client,
        api_key="fake-key",
    )

    assert result == '{"recommendations": []}'
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        response_format={"type": "json_object"},
    )


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_3_custom_llm_call_fn_async():
    """
    Test AC 3:
    - Custom llm_call_fn bypasses direct API call.
    """
    mock_fn = AsyncMock(return_value='{"recommendations": [{"recipeId": "mock-1"}]}')

    result = await execute_llm_call(
        system_prompt="sys",
        user_prompt="usr",
        model="gpt-4o-mini",
        llm_call_fn=mock_fn,
    )

    assert result == '{"recommendations": [{"recipeId": "mock-1"}]}'
    assert mock_fn.call_count == 1


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_3_custom_llm_call_fn_sync_dict():
    """
    Test AC 3 with synchronous function returning a dict.
    """
    def sync_mock_fn(options_or_sys, *args, **kwargs):
        return {"recommendations": [{"recipeId": "sync-1"}]}

    result = await execute_llm_call(
        system_prompt="sys",
        user_prompt="usr",
        llm_call_fn=sync_mock_fn,
    )

    parsed = json.loads(result)
    assert parsed["recommendations"][0]["recipeId"] == "sync-1"


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_4_missing_api_key(monkeypatch):
    """
    Test AC 4:
    - Raises error when API key is missing.
    """
    test_settings = Settings(OPENAI_API_KEY="")
    monkeypatch.setattr("app.services.llm_service.get_settings", lambda: test_settings)

    with pytest.raises(ValueError, match="Brak klucza OPENAI_API_KEY"):
        await execute_llm_call(
            system_prompt="sys",
            user_prompt="usr",
            api_key="",
        )


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_4_openai_api_error():
    """
    Test AC 4:
    - Raises error when OpenAI API request fails.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("OpenAI API rate limit"))

    with pytest.raises(RuntimeError, match="OpenAI API rate limit"):
        await execute_llm_call(
            system_prompt="sys",
            user_prompt="usr",
            client=mock_client,
            api_key="fake-key",
        )


@pytest.mark.asyncio
async def test_execute_llm_call_acceptance_criteria_4_invalid_choices():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with pytest.raises(ValueError, match="Nieprawidłowa odpowiedź z serwera OpenAI"):
        await execute_llm_call(
            system_prompt="sys",
            user_prompt="usr",
            client=mock_client,
            api_key="fake-key",
        )


@pytest.mark.asyncio
async def test_recipe_service_llm_integration():
    fake_candidates = [
        {"id": "rec-1", "title": "Pikantne Tacos", "similarity": 0.91},
        {"id": "rec-2", "title": "Guacamole z nachos", "similarity": 0.84},
    ]
    mock_search = AsyncMock(return_value=fake_candidates)
    mock_llm = AsyncMock(return_value='{"recommendations": [{"recipeId": "rec-1"}]}')

    settings = Settings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_KEY="fake-key",
        OPENAI_API_KEY="fake-openai-key",
    )
    service = RecipeService(
        settings=settings,
        search_fn=mock_search,
        llm_fn=mock_llm,
    )

    request = RecipeGenerationRequest(
        mood="pikantne",
        mealType="kolacja",
        limit=2,
    )

    response = await service.generate_recipes(request)

    assert response.trace is not None
    assert response.trace.retrievedCandidatesCount == 2
    assert response.trace.filteredCandidatesCount == 2
    assert response.trace.additionalMetadata["subtask"] == 6
    assert response.trace.additionalMetadata["status"] == "llm_called"
    assert response.trace.additionalMetadata["promptCandidatesCount"] == 2
    assert response.trace.additionalMetadata["systemPromptLength"] > 0
    assert response.trace.additionalMetadata["userPromptLength"] > 0
    assert response.trace.additionalMetadata["rawLlmResponseLength"] > 0
    assert mock_llm.call_count == 1
