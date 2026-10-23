"""
Helper functions for normalizing recipe ingredients and instructions/steps
for the RAG recommendation pipeline.
"""
import json
import re
from typing import Any, List, Optional, Union


def normalize_ingredients(raw_ingredients: Any) -> List[str]:
    """
    Normalizes ingredients to a list of non-empty cleaned strings.

    :param raw_ingredients: Ingredients in form of a list, JSON string array,
                           or newline/comma-separated text.
    :return: List of cleaned ingredient strings.
    """
    if raw_ingredients is None:
        return []

    if isinstance(raw_ingredients, (list, tuple, set)):
        return [
            str(item).strip()
            for item in raw_ingredients
            if item is not None and str(item).strip()
        ]

    if isinstance(raw_ingredients, str):
        trimmed = raw_ingredients.strip()
        if not trimmed:
            return []

        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if item is not None and str(item).strip()
                ]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # If not valid JSON list, split by newlines and commas
        parts = re.split(r"[\n,]+", trimmed)
        return [part.strip() for part in parts if part.strip()]

    return []


def normalize_steps(raw_instructions: Any) -> List[str]:
    """
    Normalizes steps/instructions to a list of non-empty cleaned strings.
    Strips leading step numbers (e.g., '1.', '1)', '1. ').

    :param raw_instructions: Instructions in form of a list, JSON string array,
                             or newline-separated text.
    :return: List of cleaned instruction step strings.
    """
    if raw_instructions is None:
        return []

    if isinstance(raw_instructions, (list, tuple, set)):
        return [
            str(step).strip()
            for step in raw_instructions
            if step is not None and str(step).strip()
        ]

    if isinstance(raw_instructions, str):
        trimmed = raw_instructions.strip()
        if not trimmed:
            return []

        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, list):
                return [
                    str(step).strip()
                    for step in parsed
                    if step is not None and str(step).strip()
                ]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Split by newlines and strip leading numbering (e.g. "1. ", "2) ")
        lines = re.split(r"\n+", trimmed)
        cleaned_steps: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                cleaned_steps.append(cleaned)
        return cleaned_steps

    return []
