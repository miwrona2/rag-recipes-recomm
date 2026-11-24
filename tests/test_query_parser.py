import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.tags import merge_tags, sanitize_tags, ALLOWED_MOODS
from app.models.recipe import QUERY_MAX_LENGTH, RecipeGenerationRequest
from app.services.query_parser import ParsedQuery, parse_user_query
from app.services.recipe_service import RecipeService, get_recipe_service
from app.services.vector_search import retrieve_candidates, search_recipes_in_vector_db


# --- request model -----------------------------------------------------------

def test_query_is_optional_and_normalized():
    assert RecipeGenerationRequest(mood="quick", mealType="dinner").query is None
    assert RecipeGenerationRequest(mood="quick", mealType="dinner", query="   ").query is None
    req = RecipeGenerationRequest(mood="quick", mealType="dinner", query="  szybki   makaron \n")
    assert req.query == "szybki makaron"


def test_query_length_limit():
    RecipeGenerationRequest(mood="quick", mealType="dinner", query="a" * QUERY_MAX_LENGTH)
    with pytest.raises(ValidationError):
        RecipeGenerationRequest(mood="quick", mealType="dinner", query="a" * (QUERY_MAX_LENGTH + 1))


# --- tags --------------------------------------------------------------------

def test_sanitize_tags_keeps_only_allowed():
    assert sanitize_tags(["Quick", "keto", "quick", "healthy"], ALLOWED_MOODS) == ["quick", "healthy"]
    assert sanitize_tags("healthy", ALLOWED_MOODS) == ["healthy"]
    assert sanitize_tags(None, ALLOWED_MOODS) == []


def test_merge_tags_tile_first_dedup():
    assert merge_tags("quick", ["healthy", "Quick"]) == ["quick", "healthy"]
    assert merge_tags("quick", None) == ["quick"]


# --- parser ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_user_query_llm_success():
    llm = AsyncMock(return_value=json.dumps({
        "semantic_query": "quick high-protein chicken pasta",
        "mood_tags": ["quick", "keto"],
        "meal_type_tags": ["main-meal", "lunch"],
    }))
    parsed = await parse_user_query("szybki wysokobiałkowy makaron z kurczakiem", llm_fn=llm, model="m")

    assert parsed.source == "llm"
    assert parsed.semantic_query == "quick high-protein chicken pasta"
    assert parsed.mood_tags == ["quick"]
    assert parsed.meal_type_tags == ["main-meal"]
    kwargs = llm.await_args.kwargs
    assert kwargs["model"] == "m"
    assert "szybki wysokobiałkowy makaron z kurczakiem" in kwargs["user_prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not json", json.dumps({"mood_tags": []}), json.dumps([1, 2])])
async def test_parse_user_query_falls_back_on_bad_response(raw):
    parsed = await parse_user_query("jak w KFC", llm_fn=AsyncMock(return_value=raw))
    assert parsed == ParsedQuery(original_query="jak w KFC", semantic_query="jak w KFC", source="fallback")


@pytest.mark.asyncio
async def test_parse_user_query_falls_back_on_llm_error():
    parsed = await parse_user_query("keto", llm_fn=AsyncMock(side_effect=RuntimeError("boom")))
    assert parsed.source == "fallback"
    assert parsed.semantic_query == "keto"


@pytest.mark.asyncio
async def test_parse_user_query_without_llm():
    parsed = await parse_user_query("keto")
    assert parsed.source == "fallback"


# --- retrieval ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_candidates_uses_semantic_query_and_merged_tags():
    search = AsyncMock(return_value=[{"id": "1"}])
    await retrieve_candidates(
        mood="quick",
        meal_type="dinner",
        search_fn=search,
        semantic_query="keto chicken",
        mood_tags=["quick", "healthy"],
        meal_type_tags=["dinner"],
    )
    kwargs = search.await_args.kwargs
    assert kwargs["query"] == "keto chicken (quick dinner)"
    assert kwargs["mood"] == ["quick", "healthy"]
    assert kwargs["meal_type"] == ["dinner"]


@pytest.mark.asyncio
async def test_search_filters_with_tag_lists_any_match():
    class _Resp:
        data = [
            {"id": "a", "mood_tags": ["quick"], "meal_type_tags": ["dinner"]},
            {"id": "b", "mood_tags": ["healthy"], "meal_type_tags": ["dinner"]},
            {"id": "c", "mood_tags": ["exotic"], "meal_type_tags": ["dinner"]},
            {"id": "d", "mood_tags": ["quick"], "meal_type_tags": ["breakfast"]},
        ]

    class _Rpc:
        def execute(self):
            return _Resp()

    class _Client:
        def rpc(self, *_a, **_k):
            return _Rpc()

    results = await search_recipes_in_vector_db(
        query="x",
        embedding=[0.1],
        mood=["quick", "healthy"],
        meal_type=["dinner"],
        supabase_client=_Client(),
    )
    assert [r["id"] for r in results] == ["a", "b"]


# --- service -----------------------------------------------------------------

def _service(**kwargs) -> RecipeService:
    return RecipeService(
        settings=Settings(OPENAI_API_KEY="test", SUPABASE_URL="https://test.supabase.co", SUPABASE_KEY="test"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_service_with_query_merges_tags_and_reports_understanding():
    search = AsyncMock(return_value=[{"id": "1", "title": "Keto chicken", "url": "u1"}])
    parser = AsyncMock(return_value=ParsedQuery(
        original_query="zdrowa kolacja keto",
        semantic_query="healthy keto dinner",
        mood_tags=["healthy"],
        meal_type_tags=["dinner"],
    ))
    prompt_builder_calls = {}

    from app.services.prompt_builder import build_rag_prompts

    def prompt_builder(**kw):
        prompt_builder_calls.update(kw)
        return build_rag_prompts(**kw)

    service = _service(search_fn=search, query_parser_fn=parser, prompt_builder_fn=prompt_builder)
    response = await service.generate_recipes_with_rag(
        RecipeGenerationRequest(mood="quick", mealType="dinner", query="zdrowa kolacja keto")
    )

    parser.assert_awaited_once_with("zdrowa kolacja keto")
    search_kwargs = search.await_args.kwargs
    assert search_kwargs["query"] == "healthy keto dinner (quick dinner)"
    assert search_kwargs["mood"] == ["quick", "healthy"]
    assert search_kwargs["meal_type"] == ["dinner"]
    assert prompt_builder_calls["user_query"] == "zdrowa kolacja keto"

    qu = response.queryUnderstanding
    assert qu is not None
    assert qu.semanticQuery == "healthy keto dinner"
    assert qu.effectiveMoods == ["quick", "healthy"]
    assert qu.effectiveMealTypes == ["dinner"]
    assert qu.source == "llm"
    assert response.trace.additionalMetadata["queryProvided"] is True
    assert response.trace.additionalMetadata["queryParserSource"] == "llm"


@pytest.mark.asyncio
async def test_service_without_query_keeps_previous_behaviour():
    search = AsyncMock(return_value=[])
    parser = AsyncMock()
    service = _service(search_fn=search, query_parser_fn=parser)
    response = await service.generate_recipes_with_rag(RecipeGenerationRequest(mood="quick", mealType="dinner"))

    parser.assert_not_awaited()
    kwargs = search.await_args.kwargs
    assert kwargs["query"] == "quick dinner"
    assert kwargs["mood"] == "quick"
    assert kwargs["meal_type"] == "dinner"
    assert response.queryUnderstanding is None


@pytest.mark.asyncio
async def test_service_parser_exception_falls_back_to_raw_query():
    search = AsyncMock(return_value=[])
    service = _service(search_fn=search, query_parser_fn=AsyncMock(side_effect=RuntimeError("x")))
    response = await service.generate_recipes_with_rag(
        RecipeGenerationRequest(mood="quick", mealType="dinner", query="jak w KFC")
    )
    assert search.await_args.kwargs["query"] == "jak w KFC (quick dinner)"
    assert response.queryUnderstanding.source == "fallback"


@pytest.mark.asyncio
async def test_endpoint_rejects_too_long_query(async_client: AsyncClient):
    response = await async_client.post(
        "/api/v1/recipes/generate",
        json={"mood": "quick", "mealType": "dinner", "query": "a" * (QUERY_MAX_LENGTH + 1)},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_endpoint_returns_query_understanding(app: FastAPI, async_client: AsyncClient):
    service = _service(
        search_fn=AsyncMock(return_value=[]),
        query_parser_fn=AsyncMock(return_value=ParsedQuery(
            original_query="keto", semantic_query="keto", source="llm"
        )),
    )
    app.dependency_overrides[get_recipe_service] = lambda: service
    try:
        response = await async_client.post(
            "/api/v1/recipes/generate",
            json={"mood": "quick", "mealType": "dinner", "query": "keto"},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["queryUnderstanding"]["semanticQuery"] == "keto"


# --- precomputed parse (suggestions served by Node.js) ------------------------

def test_precomputed_query_sanitizes_tags():
    req = RecipeGenerationRequest(
        mood="quick",
        mealType="dinner",
        query="bez mięsa",
        parsedQuery={"semanticQuery": " vegetarian  meal ", "moodTags": ["healthy", "vege"], "mealTypeTags": ["lunch"]},
    )
    assert req.parsedQuery.semanticQuery == "vegetarian meal"
    assert req.parsedQuery.moodTags == ["healthy"]
    assert req.parsedQuery.mealTypeTags == []


def test_precomputed_query_requires_semantic_query():
    with pytest.raises(ValidationError):
        RecipeGenerationRequest(mood="quick", mealType="dinner", parsedQuery={"semanticQuery": ""})


@pytest.mark.asyncio
async def test_service_with_precomputed_query_skips_llm_parser():
    search = AsyncMock(return_value=[])
    parser = AsyncMock()
    service = _service(search_fn=search, query_parser_fn=parser)
    response = await service.generate_recipes_with_rag(
        RecipeGenerationRequest(
            mood="quick",
            mealType="dinner",
            query="szybki makaron z kurczakiem",
            parsedQuery={"semanticQuery": "quick chicken pasta", "moodTags": ["quick"], "mealTypeTags": ["main-meal"]},
        )
    )

    parser.assert_not_awaited()
    kwargs = search.await_args.kwargs
    assert kwargs["query"] == "quick chicken pasta (quick dinner)"
    assert kwargs["mood"] == ["quick"]
    assert kwargs["meal_type"] == ["dinner", "main-meal"]
    qu = response.queryUnderstanding
    assert qu.source == "precomputed"
    assert qu.originalQuery == "szybki makaron z kurczakiem"
    assert response.trace.additionalMetadata["queryParserSource"] == "precomputed"
