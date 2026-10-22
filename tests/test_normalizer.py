import json
import pytest
from app.utils.recipe_normalizer import normalize_ingredients, normalize_steps


def test_normalize_ingredients_from_list():
    raw = [" 2 jajka ", " 1 szklanka mąki ", "", "   ", "szczypta soli"]
    result = normalize_ingredients(raw)
    assert result == ["2 jajka", "1 szklanka mąki", "szczypta soli"]


def test_normalize_ingredients_from_json_string():
    raw_json = json.dumps(["1 kg ziemniaków", "2 łyżki masła", ""])
    result = normalize_ingredients(raw_json)
    assert result == ["1 kg ziemniaków", "2 łyżki masła"]


def test_normalize_ingredients_from_plain_text():
    raw_text = "2 jajka\n 1 szklanka mleka,  50g masła \n\n cukier wanilinowy "
    result = normalize_ingredients(raw_text)
    assert result == ["2 jajka", "1 szklanka mleka", "50g masła", "cukier wanilinowy"]


def test_normalize_ingredients_none_and_empty():
    assert normalize_ingredients(None) == []
    assert normalize_ingredients("") == []
    assert normalize_ingredients("   ") == []
    assert normalize_ingredients([]) == []
    assert normalize_ingredients(123) == []


def test_normalize_steps_from_list():
    raw = [" 1. Wymieszaj składniki. ", " 2. Piecz przez 30 minut. "]
    result = normalize_steps(raw)
    assert result == ["1. Wymieszaj składniki.", "2. Piecz przez 30 minut."]


def test_normalize_steps_from_json_string():
    raw_json = json.dumps(["Posiekaj cebulę.", "Podsmaż na oliwie."])
    result = normalize_steps(raw_json)
    assert result == ["Posiekaj cebulę.", "Podsmaż na oliwie."]


def test_normalize_steps_from_numbered_multiline_text():
    raw_text = (
        "1. Ugotuj makaron al dente.\n"
        "2) Przygotuj sos pomidorowy na patelni.\n"
        "3.  Połącz makaron z sosem i posyp parmezanem.\n\n"
        "4. Podawaj na ciepło."
    )
    result = normalize_steps(raw_text)
    assert result == [
        "Ugotuj makaron al dente.",
        "Przygotuj sos pomidorowy na patelni.",
        "Połącz makaron z sosem i posyp parmezanem.",
        "Podawaj na ciepło.",
    ]


def test_normalize_steps_none_and_empty():
    assert normalize_steps(None) == []
    assert normalize_steps("") == []
    assert normalize_steps("   ") == []
    assert normalize_steps([]) == []
    assert normalize_steps(456) == []
