"""
Module for preparing the concise prompt context and candidate lookup map for RAG LLM prompts.
"""
from typing import Any, Dict, Iterator, List, Optional, Union
from pydantic import BaseModel

from app.services.candidate_filter import generate_synthetic_recipe_url
from app.utils.recipe_normalizer import normalize_ingredients, normalize_steps


class PromptContext:
    """
    Data structure holding the results of prepare_prompt_context.
    Supports attribute access (snake_case and camelCase), tuple unpacking, and dictionary lookup.
    """

    def __init__(
        self,
        candidates_for_prompt: List[Dict[str, Any]],
        candidates_map: Dict[str, Dict[str, Any]],
        top_candidates: List[Any],
    ):
        self.candidates_for_prompt = candidates_for_prompt
        self.candidates_map = candidates_map
        self.top_candidates = top_candidates

    @property
    def candidatesForPrompt(self) -> List[Dict[str, Any]]:
        return self.candidates_for_prompt

    @property
    def candidatesMap(self) -> Dict[str, Dict[str, Any]]:
        return self.candidates_map

    @property
    def topCandidates(self) -> List[Any]:
        return self.top_candidates

    def __iter__(self) -> Iterator[Any]:
        return iter((self.candidates_for_prompt, self.candidates_map, self.top_candidates))

    def __getitem__(self, item: Union[int, str]) -> Any:
        if isinstance(item, int):
            return (self.candidates_for_prompt, self.candidates_map, self.top_candidates)[item]
        if item in ("candidates_for_prompt", "candidatesForPrompt"):
            return self.candidates_for_prompt
        if item in ("candidates_map", "candidatesMap"):
            return self.candidates_map
        if item in ("top_candidates", "topCandidates"):
            return self.top_candidates
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_for_prompt": self.candidates_for_prompt,
            "candidates_map": self.candidates_map,
            "top_candidates": self.top_candidates,
        }

    def __repr__(self) -> str:
        return (
            f"PromptContext(candidates_for_prompt={len(self.candidates_for_prompt)} items, "
            f"candidates_map={len(self.candidates_map)} keys, "
            f"top_candidates={len(self.top_candidates)} items)"
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PromptContext):
            return (
                self.candidates_for_prompt == other.candidates_for_prompt
                and self.candidates_map == other.candidates_map
                and self.top_candidates == other.top_candidates
            )
        if isinstance(other, tuple) and len(other) == 3:
            return (
                self.candidates_for_prompt == other[0]
                and self.candidates_map == other[1]
                and self.top_candidates == other[2]
            )
        return False


def _extract_val(obj: Any, *keys: str, default: Any = None) -> Any:
    """Helper to extract a value from either a dictionary or an object attribute."""
    for key in keys:
        if isinstance(obj, dict):
            if key in obj and obj[key] is not None:
                return obj[key]
        elif hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return default


def prepare_prompt_context(
    candidates: List[Union[Dict[str, Any], Any]],
    max_candidates: int = 10,
) -> PromptContext:
    """
    Step 3: Augmentation - prepares concise candidate context for the LLM prompt
    and builds a lookup dictionary (candidates_map) of full records.

    :param candidates: List of candidate recipe dictionaries or objects.
    :param max_candidates: Maximum number of top candidates to include in prompt context (default 10).
    :return: PromptContext containing candidates_for_prompt, candidates_map, and top_candidates.
    """
    if not candidates:
        return PromptContext(candidates_for_prompt=[], candidates_map={}, top_candidates=[])

    top_candidates = list(candidates[:max_candidates])
    candidates_map: Dict[str, Dict[str, Any]] = {}
    candidates_for_prompt: List[Dict[str, Any]] = []

    for idx, c in enumerate(top_candidates):
        raw_id = _extract_val(c, "recipe_id", "id", "candidateId", "candidate_id")
        candidate_id = (
            str(raw_id).strip()
            if raw_id is not None and str(raw_id).strip()
            else f"candidate-{idx + 1}"
        )

        raw_url = _extract_val(c, "url", "link", "originalUrl", "original_url")
        if raw_url is not None and str(raw_url).strip():
            source_url = str(raw_url).strip()
        else:
            source_url = generate_synthetic_recipe_url(c)

        raw_ingredients = _extract_val(c, "ingredients")
        ingredients = normalize_ingredients(raw_ingredients)

        raw_steps = _extract_val(c, "instructions", "steps", "directions")
        steps = normalize_steps(raw_steps)

        # Build full record for lookup
        if isinstance(c, dict):
            full_record = dict(c)
        elif isinstance(c, BaseModel):
            full_record = c.model_dump()
        elif hasattr(c, "__dict__"):
            full_record = dict(c.__dict__)
        else:
            full_record = {"raw": c}

        full_record.update({
            "candidateId": candidate_id,
            "sourceUrl": source_url,
            "ingredients": ingredients,
            "steps": steps,
        })

        candidates_map[candidate_id] = full_record
        if source_url:
            candidates_map[source_url] = full_record

        # Formatting cooking time
        time_val = _extract_val(
            c,
            "cooking_time",
            "time",
            "prep_time",
            "cookingTime",
            "prepTime",
            "totalTimeMinutes",
            "cookTimeMinutes",
            "prepTimeMinutes",
        )
        if time_val is None or time_val == "":
            formatted_time = "30 min"
        elif isinstance(time_val, (int, float)):
            formatted_time = f"{time_val} min"
        else:
            formatted_time = str(time_val).strip() or "30 min"

        # Formatting tags
        raw_mood = _extract_val(c, "mood_tags", "mood", default=[])
        if isinstance(raw_mood, str):
            mood_tags = [m.strip() for m in raw_mood.split(",") if m.strip()]
        elif isinstance(raw_mood, (list, tuple, set)):
            mood_tags = list(raw_mood)
        else:
            mood_tags = []

        raw_meal_type = _extract_val(c, "meal_type_tags", "meal_type", default=[])
        if isinstance(raw_meal_type, str):
            meal_type_tags = [m.strip() for m in raw_meal_type.split(",") if m.strip()]
        elif isinstance(raw_meal_type, (list, tuple, set)):
            meal_type_tags = list(raw_meal_type)
        else:
            meal_type_tags = []

        title = _extract_val(c, "title", "name", default="Bez tytułu") or "Bez tytułu"
        source_name = _extract_val(c, "source_name", "source", default="TheFridge") or "TheFridge"

        prompt_item = {
            "recipeId": candidate_id,
            "title": str(title),
            "time": formatted_time,
            "ingredients": ingredients[:15],
            "stepsSnippet": " ".join(steps[:3]),
            "tags": {
                "mood": mood_tags,
                "mealType": meal_type_tags,
            },
            "sourceUrl": source_url,
            "sourceName": str(source_name),
        }
        candidates_for_prompt.append(prompt_item)

    return PromptContext(
        candidates_for_prompt=candidates_for_prompt,
        candidates_map=candidates_map,
        top_candidates=top_candidates,
    )
