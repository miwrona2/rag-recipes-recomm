"""
Module for executing LLM completions using OpenAI Chat Completions API.
"""
import inspect
import json
from typing import Any, Callable, Optional
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import logger


async def execute_llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    llm_call_fn: Optional[Callable[..., Any]] = None,
    client: Optional[AsyncOpenAI] = None,
) -> str:
    """
    Step 5: Execute LLM call using OpenAI Chat Completions or custom callable.

    :param system_prompt: System prompt defining AI instructions and output schema.
    :param user_prompt: User prompt containing recipe candidate details and request preferences.
    :param model: Model identifier (defaults to 'gpt-4o-mini').
    :param api_key: Optional OpenAI API key override.
    :param llm_call_fn: Optional custom callable to execute or mock the LLM call.
    :param client: Optional AsyncOpenAI client instance (useful for testing/mocking).
    :return: Raw LLM response string content (JSON string).
    """
    settings = get_settings()
    used_model = model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    # If custom llm_call_fn is provided, delegate execution
    if llm_call_fn is not None and callable(llm_call_fn):
        result = None
        try:
            result = llm_call_fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=used_model,
            )
        except TypeError:
            try:
                result = llm_call_fn({
                    "systemPrompt": system_prompt,
                    "userPrompt": user_prompt,
                    "model": used_model,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })
            except TypeError:
                result = llm_call_fn(system_prompt, user_prompt, used_model)

        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result) if result is not None else "{}"

    used_api_key = api_key or settings.OPENAI_API_KEY
    if not used_api_key and client is None:
        raise ValueError("Brak klucza OPENAI_API_KEY do wywołania etapu generacji RAG.")

    if client is None:
        client = AsyncOpenAI(api_key=used_api_key)

    try:
        response = await client.chat.completions.create(
            model=used_model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        if not response or not hasattr(response, "choices") or not response.choices:
            raise ValueError("Nieprawidłowa odpowiedź z serwera OpenAI (brak pola choices).")

        message = response.choices[0].message
        content = message.content if message else None
        if content is None:
            return "{}"
        return content
    except Exception as e:
        logger.error(f"[OPENAI_LLM_ERROR] Błąd podczas wywołania modelu LLM: {str(e)}")
        raise
