import pytest
from app.services.candidate_filter import (
    generate_synthetic_recipe_url,
    filter_excluded_candidates,
)
from app.models.recipe import RecipeCandidate


def test_generate_synthetic_recipe_url_polish_diacritics():
    recipe = {
        "recipe_id": "1042",
        "title": "Kremowe Curry z Ciecierzycą i Żółtkiem!",
    }
    url = generate_synthetic_recipe_url(recipe)
    assert url == "recipe://1042/kremowe-curry-z-ciecierzyca-i-zoltkiem"


def test_generate_synthetic_recipe_url_special_characters():
    recipe = {
        "id": "55",
        "name": "  Sałatka --- grecka & feta (100% smaku)  ",
    }
    url = generate_synthetic_recipe_url(recipe)
    assert url == "recipe://55/salatka-grecka-feta-100-smaku"


def test_generate_synthetic_recipe_url_id_only():
    recipe = {
        "recipe_id": "999",
    }
    url = generate_synthetic_recipe_url(recipe)
    assert url == "recipe://999"


def test_generate_synthetic_recipe_url_empty_and_none():
    assert generate_synthetic_recipe_url(None) == ""
    assert generate_synthetic_recipe_url({}) == ""
    assert generate_synthetic_recipe_url({"title": "Tylko tytuł bez ID"}) == ""


def test_generate_synthetic_recipe_url_from_object():
    class MockRecipe:
        def __init__(self, recipe_id: str, title: str):
            self.recipe_id = recipe_id
            self.title = title

    obj = MockRecipe("77", "Pyszna Zupa Ogórkowa")
    assert generate_synthetic_recipe_url(obj) == "recipe://77/pyszna-zupa-ogorkowa"


def test_filter_excluded_candidates_empty_or_none():
    candidates = [{"recipe_id": "1", "title": "Jajecznica"}]
    assert filter_excluded_candidates([], ["https://example.com/1"]) == []
    assert filter_excluded_candidates(candidates, None) == candidates
    assert filter_excluded_candidates(candidates, []) == candidates
    assert filter_excluded_candidates(candidates, ["", "   "]) == candidates


def test_filter_excluded_candidates_by_raw_url():
    candidates = [
        {"recipe_id": "1", "url": "https://kwestiasmaku.com/przepis/1", "title": "Danie 1"},
        {"recipe_id": "2", "link": "https://kwestiasmaku.com/przepis/2", "title": "Danie 2"},
        {"recipe_id": "3", "originalUrl": "https://kwestiasmaku.com/przepis/3", "title": "Danie 3"},
        {"recipe_id": "4", "original_url": "https://kwestiasmaku.com/przepis/4", "title": "Danie 4"},
    ]
    exclude = [
        "https://kwestiasmaku.com/przepis/1",
        "HTTPS://KWESTIASMAKU.COM/PRZEPIS/3",
    ]
    filtered = filter_excluded_candidates(candidates, exclude)
    assert len(filtered) == 2
    assert [r["recipe_id"] for r in filtered] == ["2", "4"]


def test_filter_excluded_candidates_by_synthetic_url():
    candidates = [
        {"recipe_id": "101", "title": "Zupa Pomidorowa z Ryżem"},
        {"recipe_id": "102", "title": "Kotlet Schabowy"},
    ]
    exclude = ["RECIPE://101/zupa-pomidorowa-z-ryzem"]
    filtered = filter_excluded_candidates(candidates, exclude)
    assert len(filtered) == 1
    assert filtered[0]["recipe_id"] == "102"


def test_filter_excluded_candidates_by_id():
    candidates = [
        {"recipe_id": "201", "title": "Naleśniki z twarogiem"},
        {"id": "202", "title": "Placki ziemniaczane"},
        {"candidateId": "203", "title": "Racuchy z jabłkami"},
    ]
    exclude = ["201", " 203 "]
    filtered = filter_excluded_candidates(candidates, exclude)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "202"


def test_filter_excluded_candidates_all_excluded(caplog):
    candidates = [
        {"recipe_id": "1", "url": "https://example.com/1", "title": "Danie 1"},
    ]
    exclude = ["https://example.com/1"]
    filtered = filter_excluded_candidates(candidates, exclude)
    assert filtered == []
    assert "Wszyscy pobrani kandydaci zostali wykluczeni" in caplog.text
