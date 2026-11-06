import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supabase import Client, create_client
from app.core.config import get_settings


def get_supabase_client(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    client: Optional[Client] = None,
) -> Client:
    if client is not None:
        return client

    settings = get_settings()
    url = supabase_url or settings.SUPABASE_URL
    key = supabase_key or settings.SUPABASE_KEY

    if not url or not key:
        raise ValueError(
            "Brak konfiguracji Supabase. Ustaw SUPABASE_URL oraz SUPABASE_KEY w pliku .env."
        )

    return create_client(url, key)


async def main() -> int:
    client = get_supabase_client()
    recipes_res = client.table("recipes").select("recipe_id, mood_tags, meal_type_tags").execute()
    recipes_data = recipes_res.data or []

    grouped_recipe_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for recipe in recipes_data:
        if not isinstance(recipe, dict):
            continue
        recipe_id = recipe.get("recipe_id") if "recipe_id" in recipe else recipe.get("id")
        if recipe_id is None:
            continue

        mood_tags = recipe.get("mood_tags") or []
        if isinstance(mood_tags, str):
            mood_tags = [mood_tags]

        meal_type_tags = recipe.get("meal_type_tags") or []
        if isinstance(meal_type_tags, str):
            meal_type_tags = [meal_type_tags]

        clean_recipe_id = str(recipe_id).strip()
        for mood in mood_tags:
            clean_mood = str(mood).strip()
            if not clean_mood:
                continue
            for meal_type in meal_type_tags:
                clean_meal_type = str(meal_type).strip()
                if not clean_meal_type:
                    continue
                pair_key = (clean_mood, clean_meal_type)
                if clean_recipe_id not in grouped_recipe_ids[pair_key]:
                    grouped_recipe_ids[pair_key].append(clean_recipe_id)

    sample_file = PROJECT_ROOT / "app" / "data" / "sample.json"
    with open(sample_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in samples:
        sample_input = sample.get("input", {})
        mood = sample_input.get("mood", "").strip()
        meal_type = (sample_input.get("mealType") or sample_input.get("meal_type") or "").strip()
        pair_key = (mood, meal_type)
        sample["relevantRecipeIds"] = grouped_recipe_ids.get(pair_key, [])

    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)