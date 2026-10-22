"""
Unit tests for build_rag_prompts and RagPrompts data structure.
"""
import json
import pytest
from pydantic import BaseModel

from app.services.prompt_builder import build_rag_prompts, RagPrompts


class DummyCandidateModel(BaseModel):
    recipeId: str
    title: str
    time: str
    sourceUrl: str


def test_build_rag_prompts_empty_candidates():
    prompts = build_rag_prompts(mood="szybkie", meal_type="kolacja", candidates_for_prompt=[], limit=3)
    assert isinstance(prompts, RagPrompts)
    assert "od 1 do 0 NAJLEPSZYCH przepisów" in prompts.system_prompt
    assert "Nastrój (mood): szybkie" in prompts.user_prompt
    assert "Typ posiłku (mealType): kolacja" in prompts.user_prompt
    assert "[]" in prompts.user_prompt
    assert "Wybierz do 3 najlepszych rekomendacji." in prompts.user_prompt


def test_build_rag_prompts_none_candidates():
    prompts = build_rag_prompts(mood="zdrowe", meal_type="obiad", candidates_for_prompt=None, limit=2)
    assert isinstance(prompts, RagPrompts)
    assert "od 1 do 0 NAJLEPSZYCH przepisów" in prompts.system_prompt
    assert "Wybierz do 2 najlepszych rekomendacji." in prompts.user_prompt


def test_acceptance_criteria_1_structure_and_access():
    candidates = [
        {
            "recipeId": "rec-1",
            "title": "Krem z dyni",
            "time": "25 min",
            "ingredients": ["dynia", "mleko kokosowe"],
            "stepsSnippet": "Ugotuj dynię. Zmiksuj z mlekiem.",
            "tags": {"mood": ["rozgrzewające"], "mealType": ["obiad"]},
            "sourceUrl": "https://example.com/krem-dynia",
            "sourceName": "TheFridge",
        }
    ]

    prompts = build_rag_prompts("rozgrzewające", "obiad", candidates, limit=3)

    # 1. Attribute access (snake_case)
    assert isinstance(prompts.system_prompt, str)
    assert isinstance(prompts.user_prompt, str)

    # 2. Attribute access (camelCase alias)
    assert prompts.systemPrompt == prompts.system_prompt
    assert prompts.userPrompt == prompts.user_prompt

    # 3. Tuple unpacking
    sys_p, usr_p = prompts
    assert sys_p == prompts.system_prompt
    assert usr_p == prompts.user_prompt

    # 4. Dict indexing & get
    assert prompts["system_prompt"] == prompts.system_prompt
    assert prompts["systemPrompt"] == prompts.system_prompt
    assert prompts["user_prompt"] == prompts.user_prompt
    assert prompts["userPrompt"] == prompts.user_prompt
    assert prompts.get("system_prompt") == prompts.system_prompt
    assert prompts.get("non_existent", "default_val") == "default_val"

    with pytest.raises(KeyError):
        _ = prompts["invalid_key"]

    # 5. to_dict()
    d = prompts.to_dict()
    assert d == {
        "system_prompt": prompts.system_prompt,
        "user_prompt": prompts.user_prompt,
    }

    # 6. Equality checks
    assert prompts == RagPrompts(prompts.system_prompt, prompts.user_prompt)
    assert prompts == (prompts.system_prompt, prompts.user_prompt)
    assert prompts == {"system_prompt": prompts.system_prompt, "user_prompt": prompts.user_prompt}
    assert prompts != "not_a_prompt"


def test_acceptance_criteria_2_system_prompt_content_and_rules():
    candidates = [
        {"recipeId": "1", "title": "Jajecznica"},
        {"recipeId": "2", "title": "Owsianka"},
        {"recipeId": "3", "title": "Szakszuka"},
        {"recipeId": "4", "title": "Tosty"},
    ]

    # limit = 3, len(candidates) = 4 -> min is 3
    prompts = build_rag_prompts("szybkie", "śniadanie", candidates, limit=3)
    sys = prompts.system_prompt

    assert "Jesteś kulinarnym asystentem AI w aplikacji TheFridge." in sys
    assert "Twoim zadaniem jest wybranie od 1 do 3 NAJLEPSZYCH przepisów z dostarczonej listy kandydatów" in sys
    assert "KRYTYCZNE ZASADY:" in sys
    assert "1. Wybieraj WYŁĄCZNIE przepisy z dostarczonej listy kandydatów." in sys
    assert "2. NIE WYMYŚLAJ nowych przepisów ani nie modyfikuj ich identyfikatorów." in sys
    assert "3. Dla każdego wybranego przepisu podaj:" in sys
    assert "- recipeId: DOKŁADNE recipeId z listy kandydatów" in sys
    assert "- sourceUrl: DOKŁADNY sourceUrl z listy kandydatów (jeśli dostępny)" in sys
    assert "- title: Oryginalny tytuł przepisu" in sys
    assert "- description: Krótki (1-2 zdania), zachęcający opis w języku polskim" in sys
    assert "- matchScore: Liczba zmiennoprzecinkowa od 0.0 do 1.0 określająca stopień dopasowania do preferencji" in sys
    assert "- matchReasons: Tablica 2-3 konkretnych powodów dopasowania w języku polskim" in sys
    assert "4. NIE dołączaj żadnych spersonalizowanych wskazówek ani porad o zamiennikach składników." in sys
    assert "5. Zwróć WYŁĄCZNIE poprawny format JSON zgodny ze schematem." in sys
    assert "SCHEMAT JSON:" in sys
    assert '"recommendations": [' in sys
    assert '"recipeId": "id_kandydata"' in sys
    assert '"matchScore": 0.95' in sys


def test_acceptance_criteria_3_user_prompt_formatting_and_json():
    candidates = [
        {
            "recipeId": "1001",
            "title": "Żurek staropolski z białą kiełbasą i jajkiem",
            "time": "45 min",
            "ingredients": ["zakwas żytni", "biała kiełbasa", "jajka", "majeranek", "czosnek"],
            "stepsSnippet": "Zagotuj wywar z kiełbasą. Wlej zakwas i dodaj majeranek.",
            "tags": {"mood": ["tradycyjne"], "mealType": ["obiad"]},
            "sourceUrl": "recipe://1001/zurek-staropolski",
            "sourceName": "TheFridge",
        }
    ]

    prompts = build_rag_prompts(
        mood="tradycyjne i sycące",
        meal_type="obiad",
        candidates_for_prompt=candidates,
        limit=3,
    )

    usr = prompts.user_prompt
    assert "Preferencje użytkownika:" in usr
    assert "- Nastrój (mood): tradycyjne i sycące" in usr
    assert "- Typ posiłku (mealType): obiad" in usr
    assert "Lista dostępnych kandydatów z bazy przepisów:" in usr
    assert "Wybierz do 3 najlepszych rekomendacji." in usr

    # Verify JSON block parses correctly and retains Polish characters intact
    json_start = usr.find("[\n")
    json_end = usr.rfind("]\n") + 1
    assert json_start != -1 and json_end != -1
    extracted_json = usr[json_start:json_end]
    parsed = json.loads(extracted_json)

    assert len(parsed) == 1
    assert parsed[0]["recipeId"] == "1001"
    assert parsed[0]["title"] == "Żurek staropolski z białą kiełbasą i jajkiem"
    assert "biała kiełbasa" in parsed[0]["ingredients"]
    assert "Żurek" in usr  # Check non-escaped Unicode characters


def test_build_rag_prompts_limit_calculation():
    candidates = [
        {"recipeId": "1", "title": "A"},
        {"recipeId": "2", "title": "B"},
    ]

    # When limit (5) > len(candidates) (2), max selectable is 2
    prompts = build_rag_prompts("fit", "kolacja", candidates, limit=5)
    assert "od 1 do 2 NAJLEPSZYCH przepisów" in prompts.system_prompt
    assert "Wybierz do 5 najlepszych rekomendacji." in prompts.user_prompt


def test_build_rag_prompts_with_pydantic_models():
    candidates = [
        DummyCandidateModel(
            recipeId="pyd-42",
            title="Pudding chia z malinami",
            time="10 min",
            sourceUrl="https://example.com/chia",
        )
    ]

    prompts = build_rag_prompts("fit", "deser", candidates, limit=1)
    assert "od 1 do 1 NAJLEPSZYCH przepisów" in prompts.system_prompt
    assert '"recipeId": "pyd-42"' in prompts.user_prompt
    assert '"title": "Pudding chia z malinami"' in prompts.user_prompt
