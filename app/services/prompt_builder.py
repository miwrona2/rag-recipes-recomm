"""
Module for constructing system and user prompts for RAG LLM recipe recommendations.
"""
import json
from typing import Any, Dict, Iterator, List, Optional, Union
from pydantic import BaseModel


class RagPrompts:
    """
    Data structure holding system and user prompts for the RAG LLM pipeline.
    Supports attribute access (snake_case and camelCase), tuple unpacking, and dictionary lookup.
    """

    def __init__(self, system_prompt: str, user_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    @property
    def systemPrompt(self) -> str:
        return self.system_prompt

    @property
    def userPrompt(self) -> str:
        return self.user_prompt

    def __iter__(self) -> Iterator[str]:
        return iter((self.system_prompt, self.user_prompt))

    def __getitem__(self, item: Union[int, str]) -> str:
        if isinstance(item, int):
            return (self.system_prompt, self.user_prompt)[item]
        if item in ("system_prompt", "systemPrompt"):
            return self.system_prompt
        if item in ("user_prompt", "userPrompt"):
            return self.user_prompt
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }

    def __repr__(self) -> str:
        return f"RagPrompts(system_prompt_len={len(self.system_prompt)}, user_prompt_len={len(self.user_prompt)})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RagPrompts):
            return (
                self.system_prompt == other.system_prompt
                and self.user_prompt == other.user_prompt
            )
        if isinstance(other, tuple) and len(other) == 2:
            return (
                self.system_prompt == other[0]
                and self.user_prompt == other[1]
            )
        if isinstance(other, dict):
            sys_val = other.get("system_prompt") or other.get("systemPrompt")
            usr_val = other.get("user_prompt") or other.get("userPrompt")
            return self.system_prompt == sys_val and self.user_prompt == usr_val
        return False


def _serialize_candidate(candidate: Any) -> Any:
    """Helper to convert candidate objects or Pydantic models to dict if needed."""
    if isinstance(candidate, BaseModel):
        return candidate.model_dump(by_alias=True)
    if hasattr(candidate, "to_dict") and callable(candidate.to_dict):
        return candidate.to_dict()
    if isinstance(candidate, dict):
        return candidate
    return candidate


def build_rag_prompts(
    mood: str,
    meal_type: str,
    candidates_for_prompt: Optional[List[Any]] = None,
    limit: int = 3,
) -> RagPrompts:
    """
    Step 4: Build system and user prompts for the RAG LLM completion.

    :param mood: User's mood/style preference (e.g. 'zdrowe', 'szybkie', 'comfort-food').
    :param meal_type: Type of meal (e.g. 'obiad', 'śniadanie', 'kolacja').
    :param candidates_for_prompt: Concise list of candidate recipe summaries.
    :param limit: Maximum number of recipe recommendations to select (default 3).
    :return: RagPrompts instance containing system_prompt and user_prompt.
    """
    if candidates_for_prompt is None:
        candidates_for_prompt = []

    candidates_count = len(candidates_for_prompt)
    max_selectable = min(limit, candidates_count)

    system_prompt = (
        f"Jesteś kulinarnym asystentem AI w aplikacji TheFridge.\n"
        f"Twoim zadaniem jest wybranie od 1 do {max_selectable} NAJLEPSZYCH przepisów z dostarczonej listy kandydatów, "
        f"które idealnie pasują do nastroju (mood) i typu posiłku (mealType) użytkownika.\n\n"
        f"KRYTYCZNE ZASADY:\n"
        f"1. Wybieraj WYŁĄCZNIE przepisy z dostarczonej listy kandydatów.\n"
        f"2. NIE WYMYŚLAJ nowych przepisów ani nie modyfikuj ich identyfikatorów.\n"
        f"3. Dla każdego wybranego przepisu podaj:\n"
        f"   - recipeId: DOKŁADNE recipeId z listy kandydatów\n"
        f"   - sourceUrl: DOKŁADNY sourceUrl z listy kandydatów (jeśli dostępny)\n"
        f"   - title: Oryginalny tytuł przepisu\n"
        f"   - description: Krótki (1-2 zdania), zachęcający opis w języku polskim\n"
        f"   - matchScore: Liczba zmiennoprzecinkowa od 0.0 do 1.0 określająca stopień dopasowania do preferencji\n"
        f"   - matchReasons: Tablica 2-3 konkretnych powodów dopasowania w języku polskim (np. \"Idealne na szybki obiad w 20 minut\", \"Bogate w warzywa i białko\", \"Doskonale wpisuje się w nastrój zdrowe\")\n"
        f"4. NIE dołączaj żadnych spersonalizowanych wskazówek ani porad o zamiennikach składników.\n"
        f"5. Zwróć WYŁĄCZNIE poprawny format JSON zgodny ze schematem.\n\n"
        f"SCHEMAT JSON:\n"
        f"{{\n"
        f'  "recommendations": [\n'
        f"    {{\n"
        f'      "recipeId": "id_kandydata",\n'
        f'      "sourceUrl": "url_kandydata",\n'
        f'      "title": "Tytuł przepisu",\n'
        f'      "description": "Krótki opis...",\n'
        f'      "matchScore": 0.95,\n'
        f'      "matchReasons": [\n'
        f'        "Powód 1",\n'
        f'        "Powód 2"\n'
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}"
    )

    serialized_candidates = [_serialize_candidate(c) for c in candidates_for_prompt]
    candidates_json = json.dumps(serialized_candidates, indent=2, ensure_ascii=False)

    user_prompt = (
        f"Preferencje użytkownika:\n"
        f"- Nastrój (mood): {mood}\n"
        f"- Typ posiłku (mealType): {meal_type}\n\n"
        f"Lista dostępnych kandydatów z bazy przepisów:\n"
        f"{candidates_json}\n\n"
        f"Wybierz do {limit} najlepszych rekomendacji."
    )

    return RagPrompts(system_prompt=system_prompt, user_prompt=user_prompt)
