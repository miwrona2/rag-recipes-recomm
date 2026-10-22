"""
Module for validating and assembling LLM recipe recommendations.
"""
import json
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel

from app.core.logging import logger
from app.services.candidate_filter import generate_synthetic_recipe_url


def _get_field(obj: Any, *keys: str, default: Any = None) -> Any:
    """Helper to retrieve a value from a dictionary or an object attribute."""
    if obj is None:
        return default
    for key in keys:
        if isinstance(obj, dict):
            if key in obj and obj[key] is not None:
                return obj[key]
        elif hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                return val
    return default


def validate_and_assemble_recommendations(
    raw_response_content: Union[str, Dict[str, Any], List[Any]],
    candidates_map: Union[Dict[str, Any], Any],
    options: Optional[Dict[str, Any]] = None,
    mood: Optional[str] = None,
    meal_type: Optional[str] = None,
    mealType: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Step 6: Validation and Assembly - validates LLM response, matches IDs/URLs
    against candidate map, deduplicates entries, normalizes scores/reasons,
    and constructs final enriched recipe recommendation objects.

    :param raw_response_content: Raw JSON string or parsed dictionary/list from LLM.
    :param candidates_map: Dictionary or PromptContext mapping recipe IDs and URLs to full records.
    :param options: Optional configuration dictionary containing mood, mealType, limit.
    :param mood: Target mood string preference.
    :param meal_type: Target meal type string preference (snake_case).
    :param mealType: Target meal type string preference (camelCase).
    :param limit: Maximum number of recommendations to return (default: 3).
    :param kwargs: Additional keyword arguments.
    :return: List of validated and assembled recipe dictionaries.
    """
    opts = options or {}
    effective_mood = mood or opts.get("mood") or kwargs.get("mood") or ""
    effective_meal_type = (
        meal_type
        or mealType
        or opts.get("mealType")
        or opts.get("meal_type")
        or kwargs.get("mealType")
        or kwargs.get("meal_type")
        or ""
    )
    raw_limit = limit if limit is not None else opts.get("limit", kwargs.get("limit", 3))
    try:
        effective_limit = max(1, int(raw_limit))
    except (ValueError, TypeError):
        effective_limit = 3

    # Unwrap candidates_map if PromptContext instance or custom object is provided
    effective_map: Dict[str, Any] = {}
    if isinstance(candidates_map, dict):
        effective_map = candidates_map
    elif hasattr(candidates_map, "candidates_map"):
        effective_map = getattr(candidates_map, "candidates_map")
    elif hasattr(candidates_map, "candidatesMap"):
        effective_map = getattr(candidates_map, "candidatesMap")
    elif candidates_map is not None:
        effective_map = dict(candidates_map)

    # Parse raw LLM response content
    parsed_recommendations: List[Any] = []
    if isinstance(raw_response_content, str):
        try:
            parsed = json.loads(raw_response_content)
        except Exception as err:
            logger.error(f"[RAG_VALIDATION_ERROR] Nie udało się sparsować odpowiedzi JSON z LLM: {err}")
            raise ValueError("Błąd formatu odpowiedzi LLM w procesie RAG.") from err
    else:
        parsed = raw_response_content

    if isinstance(parsed, list):
        parsed_recommendations = parsed
    elif isinstance(parsed, dict):
        parsed_recommendations = (
            parsed.get("recommendations")
            or parsed.get("recipes")
            or []
        )
        if not isinstance(parsed_recommendations, list):
            parsed_recommendations = []
    else:
        parsed_recommendations = []

    validated_recipes: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()

    for item in parsed_recommendations:
        if not item or not isinstance(item, (dict, BaseModel)) and not hasattr(item, "__dict__"):
            continue

        raw_recipe_id = _get_field(item, "recipeId", "recipe_id", "id", "candidateId", "candidate_id")
        raw_source_url = _get_field(item, "sourceUrl", "source_url", "url", "link", "originalUrl", "original_url")

        recipe_id_str = str(raw_recipe_id).strip() if raw_recipe_id is not None and str(raw_recipe_id).strip() else None
        source_url_str = str(raw_source_url).strip() if raw_source_url is not None and str(raw_source_url).strip() else None

        if not recipe_id_str and not source_url_str:
            continue

        matched_candidate = None
        if recipe_id_str and recipe_id_str in effective_map:
            matched_candidate = effective_map[recipe_id_str]
        elif source_url_str and source_url_str in effective_map:
            matched_candidate = effective_map[source_url_str]

        if not matched_candidate:
            logger.warning(
                f"[RAG_VALIDATION_WARNING] Odrzucono rekomendację - recipeId {recipe_id_str} nie istnieje w kandydatach."
            )
            continue

        candidate_id = str(
            _get_field(matched_candidate, "candidateId", "candidate_id", "recipe_id", "id", default=recipe_id_str or "")
        ).strip()

        if candidate_id in used_ids:
            continue
        used_ids.add(candidate_id)

        # Validate matchScore
        raw_score = _get_field(item, "matchScore", "match_score")
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
            match_score = max(0.0, min(1.0, float(raw_score)))
        else:
            cand_similarity = _get_field(matched_candidate, "similarity", default=0.85)
            try:
                match_score = float(cand_similarity)
            except (ValueError, TypeError):
                match_score = 0.85

        # Validate matchReasons
        raw_reasons = _get_field(item, "matchReasons", "match_reasons")
        if isinstance(raw_reasons, (list, tuple, set)):
            filtered_reasons = [
                str(r).strip() for r in raw_reasons if r is not None and str(r).strip()
            ]
        else:
            filtered_reasons = []

        if filtered_reasons:
            match_reasons = filtered_reasons
        else:
            default_reasons = []
            if effective_mood:
                default_reasons.append(f"Dopasowano do nastroju: {effective_mood}")
            if effective_meal_type:
                default_reasons.append(f"Typ posiłku: {effective_meal_type}")
            if not default_reasons:
                default_reasons = ["Dopasowano do preferencji kulinarnych"]
            match_reasons = default_reasons

        # Determine candidate URL
        candidate_url = (
            _get_field(matched_candidate, "sourceUrl", "source_url", "url", "link", "originalUrl", "original_url")
            or generate_synthetic_recipe_url(matched_candidate)
        )

        title = str(
            _get_field(item, "title", "name")
            or _get_field(matched_candidate, "title", "name", default="Bez tytułu")
            or "Bez tytułu"
        ).strip()

        description = str(
            _get_field(item, "description")
            or _get_field(matched_candidate, "description", default="")
            or ""
        ).strip()

        time_val = _get_field(
            matched_candidate,
            "cooking_time",
            "time",
            "prep_time",
            "cookingTime",
            "prepTime",
            "totalTimeMinutes",
            "cookTimeMinutes",
            "prepTimeMinutes",
            default="30 min",
        )
        if isinstance(time_val, (int, float)) and not isinstance(time_val, bool):
            time_str = f"{time_val} min"
        else:
            time_str = str(time_val).strip() if time_val else "30 min"

        ingredients = _get_field(matched_candidate, "ingredients", default=[])
        if not isinstance(ingredients, list):
            ingredients = list(ingredients) if isinstance(ingredients, (tuple, set)) else [str(ingredients)]

        steps = _get_field(matched_candidate, "steps", default=[])
        if not isinstance(steps, list):
            steps = list(steps) if isinstance(steps, (tuple, set)) else [str(steps)]

        instructions = _get_field(matched_candidate, "instructions", default=steps)
        if not isinstance(instructions, list):
            instructions = list(instructions) if isinstance(instructions, (tuple, set)) else [str(instructions)]

        source_name = str(
            _get_field(matched_candidate, "source_name", "source", "sourceName", default="TheFridge")
            or "TheFridge"
        ).strip()

        image_path = _get_field(matched_candidate, "image_path", "imagePath", default=None)
        image_url = _get_field(matched_candidate, "image_url", "imageUrl", default=None)

        raw_tags = _get_field(matched_candidate, "mood_tags", "tags")
        if raw_tags is not None:
            if isinstance(raw_tags, (list, tuple, set)):
                tags = list(raw_tags)
            elif isinstance(raw_tags, str):
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            else:
                tags = [str(raw_tags)]
        else:
            fallback_tags = [t for t in [effective_mood, effective_meal_type] if t]
            tags = fallback_tags if fallback_tags else ["TheFridge"]

        validated_record = {
            "id": candidate_id,
            "recipeId": candidate_id,
            "title": title,
            "name": title,
            "time": time_str,
            "ingredients": ingredients,
            "steps": steps,
            "instructions": instructions,
            "link": candidate_url,
            "originalUrl": candidate_url,
            "sourceName": source_name,
            "description": description,
            "imagePath": image_path,
            "imageUrl": image_url,
            "tags": tags,
            "matchScore": match_score,
            "matchReasons": match_reasons,
        }

        validated_recipes.append(validated_record)

        if len(validated_recipes) >= effective_limit:
            break

    return validated_recipes


# CamelCase alias
validateAndAssembleRecommendations = validate_and_assemble_recommendations
