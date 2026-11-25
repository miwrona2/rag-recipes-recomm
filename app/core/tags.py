"""
Allowed tile tags ("Na co mam ochotę") shared by the request model and the query parser.
"""
from typing import Iterable, List, Optional, Sequence

ALLOWED_MOODS: List[str] = ["comfort-food", "healthy", "quick", "exotic", "fridge-rescue"]
ALLOWED_MEAL_TYPES: List[str] = ["breakfast", "main-meal", "dinner", "dessert"]


def sanitize_tags(tags: Optional[Iterable[object]], allowed: Sequence[str]) -> List[str]:
    """Keep only allowed tag values (case-insensitive), deduplicated, in input order."""
    if not tags or isinstance(tags, (str, bytes)):
        tags = [tags] if isinstance(tags, str) else []
    allowed_lookup = {a.lower(): a for a in allowed}
    result: List[str] = []
    for tag in tags:
        canonical = allowed_lookup.get(str(tag).strip().lower())
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def merge_tags(primary: Optional[str], extra: Optional[Iterable[str]]) -> List[str]:
    """Tile tag first, then tags inferred from the text query (deduplicated, case-insensitive)."""
    merged: List[str] = []
    seen = set()
    for tag in [primary, *(extra or [])]:
        if not tag or not str(tag).strip():
            continue
        key = str(tag).strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(str(tag).strip())
    return merged
