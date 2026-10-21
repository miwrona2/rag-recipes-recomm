import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Union
from app.core.logging import logger

PL_MAP = {
    "ą": "a",
    "ć": "c",
    "ę": "e",
    "ł": "l",
    "ń": "n",
    "ó": "o",
    "ś": "s",
    "ź": "z",
    "ż": "z",
    "Ą": "a",
    "Ć": "c",
    "Ę": "e",
    "Ł": "l",
    "Ń": "n",
    "Ó": "o",
    "Ś": "s",
    "Ź": "z",
    "Ż": "z",
}
PL_PATTERN = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")


def generate_synthetic_recipe_url(recipe: Optional[Union[Dict[str, Any], Any]]) -> str:
    """
    Generates a synthetic URL for a recipe based on recipe_id and title/name.
    Matches the Node.js implementation: recipe://{recipe_id}/{title_slug} or recipe://{recipe_id}

    :param recipe: Recipe dictionary or object with id/recipe_id and title/name.
    :return: Synthetic recipe URL string, e.g. 'recipe://1042/kremowe-curry-z-ciecierzyca'.
    """
    if not recipe:
        return ""

    if isinstance(recipe, dict):
        recipe_id = recipe.get("recipe_id") or recipe.get("id") or recipe.get("candidateId") or ""
        raw_title = recipe.get("title") or recipe.get("name") or ""
    else:
        recipe_id = (
            getattr(recipe, "recipe_id", None)
            or getattr(recipe, "id", None)
            or getattr(recipe, "candidateId", "")
            or ""
        )
        raw_title = (
            getattr(recipe, "title", None)
            or getattr(recipe, "name", "")
            or ""
        )

    recipe_id = str(recipe_id).strip()
    raw_title = str(raw_title).strip()

    # 1. Polish characters map
    replaced_title = PL_PATTERN.sub(lambda m: PL_MAP.get(m.group(0), m.group(0)), raw_title)
    # 2. Lowercase
    lowered = replaced_title.lower()
    # 3. Unicode normalize NFD & strip diacritics
    normalized = unicodedata.normalize("NFD", lowered)
    stripped = re.sub(r"[\u0300-\u036f]", "", normalized)
    # 4. Replace non-alphanumeric chars with '-'
    slug = re.sub(r"[^a-z0-9]+", "-", stripped)
    # 5. Strip leading and trailing dashes
    slug = slug.strip("-")

    if recipe_id and slug:
        return f"recipe://{recipe_id}/{slug}"
    elif recipe_id:
        return f"recipe://{recipe_id}"
    return ""


def filter_excluded_candidates(
    candidates: List[Union[Dict[str, Any], Any]],
    exclude_urls: Optional[List[str]] = None,
) -> List[Union[Dict[str, Any], Any]]:
    """
    Step 2: Filtering - removes candidates present in exclude_urls list.
    Supports matching against rawUrl (url/link/originalUrl/original_url),
    syntheticUrl (recipe://id/slug), and candidateId / recipe_id / id.

    :param candidates: List of candidate recipe dictionaries or objects.
    :param exclude_urls: List of URL strings or IDs to exclude.
    :return: Filtered list of candidate recipes.
    """
    if not candidates:
        return []

    if not exclude_urls or not isinstance(exclude_urls, (list, tuple, set)):
        return list(candidates)

    excluded_set: Set[str] = {
        str(u).strip().lower()
        for u in exclude_urls
        if u is not None and str(u).strip()
    }

    if not excluded_set:
        return list(candidates)

    filtered: List[Any] = []
    for recipe in candidates:
        if isinstance(recipe, dict):
            raw_url = str(
                recipe.get("url")
                or recipe.get("link")
                or recipe.get("originalUrl")
                or recipe.get("original_url")
                or ""
            ).strip().lower()
            candidate_id = str(
                recipe.get("recipe_id")
                or recipe.get("id")
                or recipe.get("candidateId")
                or ""
            ).strip().lower()
        else:
            raw_url = str(
                getattr(recipe, "url", None)
                or getattr(recipe, "link", None)
                or getattr(recipe, "originalUrl", None)
                or getattr(recipe, "original_url", None)
                or ""
            ).strip().lower()
            candidate_id = str(
                getattr(recipe, "recipe_id", None)
                or getattr(recipe, "id", None)
                or getattr(recipe, "candidateId", None)
                or ""
            ).strip().lower()

        synthetic_url = generate_synthetic_recipe_url(recipe).lower()

        is_excluded = bool(
            (raw_url and raw_url in excluded_set)
            or (synthetic_url and synthetic_url in excluded_set)
            or (candidate_id and candidate_id in excluded_set)
        )

        if not is_excluded:
            filtered.append(recipe)

    if len(filtered) == 0 and len(candidates) > 0:
        logger.warning("[RAG_SERVICE] Wszyscy pobrani kandydaci zostali wykluczeni przez excludeUrls.")

    return filtered
