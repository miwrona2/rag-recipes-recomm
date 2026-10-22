import pytest
from app.services.prompt_context import prepare_prompt_context, PromptContext
from app.models.recipe import RecipeCandidate


def test_prepare_prompt_context_empty():
    context = prepare_prompt_context([])
    assert isinstance(context, PromptContext)
    assert context.candidates_for_prompt == []
    assert context.candidates_map == {}
    assert context.top_candidates == []


def test_prepare_prompt_context_acceptance_criteria_1_structure_and_access():
    candidates = [
        {
            "recipe_id": "101",
            "title": "Jajecznica ze szczypiorkiem",
            "url": "https://kwestiasmaku.com/przepis/jajecznica",
            "ingredients": ["3 jajka", "szczypiorek", "1 łyżka masła"],
            "instructions": ["1. Rozgrzej masło.", "2. Wbij jajka.", "3. Dodaj szczypiorek."],
        }
    ]

    context = prepare_prompt_context(candidates)

    # 1. Attribute access (snake_case)
    assert len(context.candidates_for_prompt) == 1
    assert len(context.top_candidates) == 1
    assert "101" in context.candidates_map

    # 2. Attribute access (camelCase alias)
    assert context.candidatesForPrompt == context.candidates_for_prompt
    assert context.candidatesMap == context.candidates_map
    assert context.topCandidates == context.top_candidates

    # 3. Tuple unpacking
    for_prompt, c_map, top_c = prepare_prompt_context(candidates)
    assert for_prompt == context.candidates_for_prompt
    assert c_map == context.candidates_map
    assert top_c == context.top_candidates

    # 4. Dictionary index access & get
    assert context["candidates_for_prompt"] == context.candidates_for_prompt
    assert context["candidatesForPrompt"] == context.candidates_for_prompt
    assert context.get("candidates_map") == context.candidates_map
    assert context.get("non_existent", "default") == "default"

    # 5. to_dict()
    d = context.to_dict()
    assert "candidates_for_prompt" in d
    assert "candidates_map" in d
    assert "top_candidates" in d


def test_prepare_prompt_context_acceptance_criteria_2_concise_fields_and_limits():
    # Candidate with more than 15 ingredients and more than 3 steps
    many_ingredients = [f"Składnik {i}" for i in range(1, 25)]
    many_steps = [f"{i}. Krok numer {i}" for i in range(1, 10)]

    candidates = [
        {
            "id": f"rec-{i}",
            "title": f"Przepis {i}",
            "cooking_time": "40 min",
            "ingredients": many_ingredients,
            "instructions": "\n".join(many_steps),
            "mood_tags": ["szybkie", "smaczne"],
            "meal_type_tags": ["obiad"],
            "source": "Kwestia Smaku",
            "url": f"https://example.com/recipe/{i}",
        }
        for i in range(15)
    ]

    # Default max_candidates is 10
    context = prepare_prompt_context(candidates, max_candidates=10)

    assert len(context.top_candidates) == 10
    assert len(context.candidates_for_prompt) == 10

    item = context.candidates_for_prompt[0]
    assert item["recipeId"] == "rec-0"
    assert item["title"] == "Przepis 0"
    assert item["time"] == "40 min"
    # Max 15 ingredients
    assert len(item["ingredients"]) == 15
    assert item["ingredients"][0] == "Składnik 1"
    assert item["ingredients"][-1] == "Składnik 15"
    # First 3 steps joined with space (leading numbers stripped by normalizer)
    assert item["stepsSnippet"] == "Krok numer 1 Krok numer 2 Krok numer 3"
    assert item["tags"]["mood"] == ["szybkie", "smaczne"]
    assert item["tags"]["mealType"] == ["obiad"]
    assert item["sourceUrl"] == "https://example.com/recipe/0"
    assert item["sourceName"] == "Kwestia Smaku"


def test_prepare_prompt_context_acceptance_criteria_3_lookup_by_id_and_url():
    candidates = [
        {
            "recipe_id": "99",
            "title": "Krem pomidorowy",
            "url": "https://mojegotowanie.pl/krem-pomidorowy",
            "ingredients": ["pomidory", "bazylia"],
            "steps": ["Zmiksuj pomidory."],
        }
    ]

    context = prepare_prompt_context(candidates)

    # Lookup by recipeId
    record_by_id = context.candidates_map.get("99")
    assert record_by_id is not None
    assert record_by_id["title"] == "Krem pomidorowy"
    assert record_by_id["candidateId"] == "99"
    assert record_by_id["sourceUrl"] == "https://mojegotowanie.pl/krem-pomidorowy"

    # Lookup by sourceUrl
    record_by_url = context.candidates_map.get("https://mojegotowanie.pl/krem-pomidorowy")
    assert record_by_url is not None
    assert record_by_url == record_by_id


def test_prepare_prompt_context_synthetic_url_generation():
    candidates = [
        {
            "recipe_id": "505",
            "title": "Żurek staropolski z jajkiem",
            # No url provided -> fallback to synthetic url
        }
    ]

    context = prepare_prompt_context(candidates)
    item = context.candidates_for_prompt[0]
    assert item["sourceUrl"] == "recipe://505/zurek-staropolski-z-jajkiem"

    # Can be looked up in map by synthetic URL
    record = context.candidates_map.get("recipe://505/zurek-staropolski-z-jajkiem")
    assert record is not None
    assert record["candidateId"] == "505"


def test_prepare_prompt_context_with_pydantic_model():
    pydantic_candidate = RecipeCandidate(
        id="pyd-1",
        title="Naleśniki francuskie",
        url="https://example.com/nalesniki",
        ingredients=["mąka", "mleko", "jajka"],
        instructions=["Wymieszaj.", "Usmaż."],
        cookTimeMinutes=20,
    )

    context = prepare_prompt_context([pydantic_candidate])
    assert len(context.candidates_for_prompt) == 1
    item = context.candidates_for_prompt[0]
    assert item["recipeId"] == "pyd-1"
    assert item["title"] == "Naleśniki francuskie"
    assert item["time"] == "20 min"
    assert item["sourceUrl"] == "https://example.com/nalesniki"

    record = context.candidates_map.get("pyd-1")
    assert record is not None
    assert record["candidateId"] == "pyd-1"
