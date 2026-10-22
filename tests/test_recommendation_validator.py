"""
Unit tests for validate_and_assemble_recommendations and recommendation validation pipeline.
"""
import json
import pytest
from unittest.mock import AsyncMock

from app.services.recommendation_validator import (
    validate_and_assemble_recommendations,
    validateAndAssembleRecommendations,
)
from app.services.prompt_context import prepare_prompt_context
from app.services.recipe_service import RecipeService
from app.models.recipe import RecipeGenerationRequest, RecipeCandidate
from app.core.config import Settings


@pytest.fixture
def sample_candidates_map():
    return {
        "rec-1": {
            "candidateId": "rec-1",
            "title": "Kremowe Curry z Ciecierzycą",
            "sourceUrl": "https://kwestiasmaku.com/przepis/kremowe-curry",
            "cooking_time": "25 min",
            "ingredients": ["ciecierzyca", "mleczko kokosowe", "pasta curry"],
            "steps": ["Podsmaż pastę.", "Dodaj mleczko i ciecierzycę.", "Gotuj 15 minut."],
            "source_name": "Kwestia Smaku",
            "description": "Aromatyczne curry.",
            "similarity": 0.92,
            "mood_tags": ["rozgrzewające", "wegetariańskie"],
            "meal_type_tags": ["obiad"],
        },
        "https://kwestiasmaku.com/przepis/kremowe-curry": {
            "candidateId": "rec-1",
            "title": "Kremowe Curry z Ciecierzycą",
            "sourceUrl": "https://kwestiasmaku.com/przepis/kremowe-curry",
            "cooking_time": "25 min",
            "ingredients": ["ciecierzyca", "mleczko kokosowe", "pasta curry"],
            "steps": ["Podsmaż pastę.", "Dodaj mleczko i ciecierzycę.", "Gotuj 15 minut."],
            "source_name": "Kwestia Smaku",
            "description": "Aromatyczne curry.",
            "similarity": 0.92,
            "mood_tags": ["rozgrzewające", "wegetariańskie"],
            "meal_type_tags": ["obiad"],
        },
        "rec-2": {
            "candidateId": "rec-2",
            "title": "Szybka Sałatka z Awokado",
            "sourceUrl": "https://aniagotuje.pl/przepis/salatka-awokado",
            "time": "10 min",
            "ingredients": ["awokado", "pomidorki", "rukola"],
            "instructions": ["Pokrój warzywa.", "Wymieszaj z sosem."],
            "source": "Ania Gotuje",
            "similarity": 0.88,
        },
        "https://aniagotuje.pl/przepis/salatka-awokado": {
            "candidateId": "rec-2",
            "title": "Szybka Sałatka z Awokado",
            "sourceUrl": "https://aniagotuje.pl/przepis/salatka-awokado",
            "time": "10 min",
            "ingredients": ["awokado", "pomidorki", "rukola"],
            "instructions": ["Pokrój warzywa.", "Wymieszaj z sosem."],
            "source": "Ania Gotuje",
            "similarity": 0.88,
        },
        "rec-3": {
            "candidateId": "rec-3",
            "id": "rec-3",
            "title": "Zapiekanka Makaronowa z Warzywami",
            # No explicit URL -> synthetic URL will be used
            "cooking_time": 40,
            "ingredients": ["makaron", "cukinia", "ser"],
            "steps": ["Ugotuj makaron.", "Wymieszaj z warzywami.", "Zapiekaj 25 min."],
            "similarity": 0.79,
        },
    }


def test_validate_and_assemble_valid_json_string(sample_candidates_map):
    llm_payload = {
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
    }
    raw_content = json.dumps(llm_payload)

    result = validate_and_assemble_recommendations(
        raw_response_content=raw_content,
        candidates_map=sample_candidates_map,
        mood="rozgrzewające",
        meal_type="obiad",
        limit=3,
    )

    assert len(result) == 1
    rec = result[0]
    assert rec["id"] == "rec-1"
    assert rec["recipeId"] == "rec-1"
    assert rec["title"] == "Kremowe Curry z Ciecierzycą"
    assert rec["name"] == "Kremowe Curry z Ciecierzycą"
    assert rec["description"] == "Pyszne, sycące i rozgrzewające curry w 25 minut."
    assert rec["time"] == "25 min"
    assert rec["link"] == "https://kwestiasmaku.com/przepis/kremowe-curry"
    assert rec["originalUrl"] == "https://kwestiasmaku.com/przepis/kremowe-curry"
    assert rec["sourceName"] == "Kwestia Smaku"
    assert rec["matchScore"] == 0.95
    assert rec["matchReasons"] == ["Idealne na szybki obiad", "Bogate w białko roślinne"]
    assert rec["ingredients"] == ["ciecierzyca", "mleczko kokosowe", "pasta curry"]
    assert rec["steps"] == ["Podsmaż pastę.", "Dodaj mleczko i ciecierzycę.", "Gotuj 15 minut."]


def test_validate_and_assemble_match_by_url_only(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {
                "sourceUrl": "https://aniagotuje.pl/przepis/salatka-awokado",
                "title": "Sałatka z Awokado",
                "matchScore": 0.90,
            }
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        mood="lekkie",
        meal_type="kolacja",
    )

    assert len(result) == 1
    assert result[0]["id"] == "rec-2"
    assert result[0]["title"] == "Sałatka z Awokado"
    assert result[0]["sourceName"] == "Ania Gotuje"
    assert result[0]["time"] == "10 min"


def test_validate_and_assemble_synthetic_url_generation(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {
                "recipeId": "rec-3",
                "title": "Zapiekanka Makaronowa",
            }
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        mood="sycące",
        meal_type="obiad",
    )

    assert len(result) == 1
    assert result[0]["id"] == "rec-3"
    assert result[0]["link"] == "recipe://rec-3/zapiekanka-makaronowa-z-warzywami"
    assert result[0]["time"] == "40 min"


def test_validate_and_assemble_invalid_json_raises_value_error(sample_candidates_map):
    with pytest.raises(ValueError, match="Błąd formatu odpowiedzi LLM w procesie RAG."):
        validate_and_assemble_recommendations(
            raw_response_content="this is not a valid JSON string {",
            candidates_map=sample_candidates_map,
        )


def test_validate_and_assemble_empty_and_unknown_candidates(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {"recipeId": "unknown-999", "title": "Nieistniejący przepis"},
            {},
            {"title": "Brak ID i URL"},
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
    )

    assert result == []


def test_validate_and_assemble_deduplication(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {"recipeId": "rec-1", "title": "Curry Pierwsze", "matchScore": 0.95},
            {"recipeId": "rec-1", "title": "Curry Duplikat", "matchScore": 0.90},
            {"sourceUrl": "https://kwestiasmaku.com/przepis/kremowe-curry", "title": "Curry URL"},
            {"recipeId": "rec-2", "title": "Sałatka", "matchScore": 0.85},
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        limit=5,
    )

    # Should only return 2 unique items: rec-1 and rec-2
    assert len(result) == 2
    assert [r["id"] for r in result] == ["rec-1", "rec-2"]
    assert result[0]["title"] == "Curry Pierwsze"


def test_validate_and_assemble_match_score_clamping_and_fallback(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {"recipeId": "rec-1", "matchScore": 1.45},  # > 1.0 -> clamp to 1.0
            {"recipeId": "rec-2", "matchScore": -0.5},  # < 0.0 -> clamp to 0.0
            {"recipeId": "rec-3", "matchScore": None},  # None -> fallback to similarity (0.79)
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        limit=3,
    )

    assert len(result) == 3
    assert result[0]["matchScore"] == 1.0
    assert result[1]["matchScore"] == 0.0
    assert result[2]["matchScore"] == 0.79


def test_validate_and_assemble_match_reasons_fallback(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {"recipeId": "rec-1", "matchReasons": []},  # Empty list
            {"recipeId": "rec-2", "matchReasons": ["   ", ""]},  # Empty strings
            {"recipeId": "rec-3", "matchReasons": ["Świetny smak", "Szybki czas"]},
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        mood="lekkie i pożywne",
        meal_type="kolacja",
        limit=3,
    )

    assert result[0]["matchReasons"] == [
        "Dopasowano do nastroju: lekkie i pożywne",
        "Typ posiłku: kolacja",
    ]
    assert result[1]["matchReasons"] == [
        "Dopasowano do nastroju: lekkie i pożywne",
        "Typ posiłku: kolacja",
    ]
    assert result[2]["matchReasons"] == ["Świetny smak", "Szybki czas"]


def test_validate_and_assemble_limit_enforcement(sample_candidates_map):
    llm_payload = {
        "recommendations": [
            {"recipeId": "rec-1"},
            {"recipeId": "rec-2"},
            {"recipeId": "rec-3"},
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=sample_candidates_map,
        limit=1,
    )

    assert len(result) == 1
    assert result[0]["id"] == "rec-1"


def test_validate_and_assemble_with_prompt_context_instance():
    candidates = [
        {
            "recipe_id": "item-10",
            "title": "Zupa Dyniowa",
            "url": "https://example.com/dynia",
            "similarity": 0.89,
        }
    ]
    context = prepare_prompt_context(candidates)

    llm_payload = {
        "recommendations": [
            {"recipeId": "item-10", "title": "Aromatyczna Zupa Dyniowa", "matchScore": 0.92}
        ]
    }

    result = validate_and_assemble_recommendations(
        raw_response_content=json.dumps(llm_payload),
        candidates_map=context,
        mood="jesienne",
        meal_type="obiad",
    )

    assert len(result) == 1
    assert result[0]["id"] == "item-10"
    assert result[0]["title"] == "Aromatyczna Zupa Dyniowa"
    assert result[0]["matchScore"] == 0.92


def test_validate_and_assemble_camel_case_alias(sample_candidates_map):
    llm_payload = {"recipes": [{"recipeId": "rec-1"}]}
    res = validateAndAssembleRecommendations(
        raw_response_content=llm_payload,
        candidates_map=sample_candidates_map,
        options={"mood": "fit", "mealType": "obiad", "limit": 2},
    )
    assert len(res) == 1
    assert res[0]["id"] == "rec-1"


@pytest.mark.asyncio
async def test_recipe_service_full_validation_integration():
    fake_candidates = [
        {
            "id": "rec-101",
            "recipe_id": "rec-101",
            "title": "Pikantne Tacos z Wołowiną",
            "url": "https://example.com/tacos",
            "similarity": 0.93,
            "cooking_time": "20 min",
            "ingredients": ["wołowina", "tortille", "papryczki jalapeno"],
            "instructions": ["Podsmaż mięso.", "Nadziewaj tortille."],
        },
        {
            "id": "rec-102",
            "recipe_id": "rec-102",
            "title": "Guacamole z nachos",
            "url": "https://example.com/guacamole",
            "similarity": 0.85,
        },
    ]

    mock_search = AsyncMock(return_value=fake_candidates)
    mock_llm = AsyncMock(
        return_value=json.dumps({
            "recommendations": [
                {
                    "recipeId": "rec-101",
                    "title": "Wyśmienite Pikantne Tacos z Wołowiną",
                    "description": "Świetny meksykański klasyk.",
                    "matchScore": 0.97,
                    "matchReasons": ["Pikantne i sycące", "Ekspresowy czas przygotowania"],
                }
            ]
        })
    )

    settings = Settings(
        SUPABASE_URL="https://example.supabase.co",
        SUPABASE_KEY="fake-key",
        OPENAI_API_KEY="fake-openai-key",
    )

    service = RecipeService(
        settings=settings,
        search_fn=mock_search,
        llm_fn=mock_llm,
        validator_fn=validate_and_assemble_recommendations,
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
    assert response.trace.additionalMetadata["subtask"] == 7
    assert response.trace.additionalMetadata["status"] == "recommendations_validated"
    assert response.trace.additionalMetadata["validatedRecipesCount"] == 1

    assert len(response.recipes) == 1
    recipe = response.recipes[0]
    assert isinstance(recipe, RecipeCandidate)
    assert recipe.id == "rec-101"
    assert recipe.title == "Wyśmienite Pikantne Tacos z Wołowiną"
    assert recipe.url == "https://example.com/tacos"
    assert recipe.description == "Świetny meksykański klasyk."
    assert recipe.matchScore == 0.97
    assert recipe.reasoning == "Pikantne i sycące, Ekspresowy czas przygotowania"
    assert recipe.ingredients == ["wołowina", "tortille", "papryczki jalapeno"]
    assert recipe.instructions == ["Podsmaż mięso.", "Nadziewaj tortille."]
