from typing import List, Optional
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.logging import logger


async def get_embedding(
    text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[AsyncOpenAI] = None,
) -> List[float]:
    """
    Generate embedding vector for a given text using OpenAI API.

    :param text: Text string to embed.
    :param api_key: Optional OpenAI API key override.
    :param model: Optional model override (defaults to settings.OPENAI_EMBEDDING_MODEL).
    :param client: Optional AsyncOpenAI client instance (useful for testing/mocking).
    :return: List of floats representing the embedding vector.
    """
    if not text or not isinstance(text, str) or not text.strip():
        raise ValueError("Tekst do wygenerowania embeddingu musi być niepustym ciągiem znaków.")

    settings = get_settings()
    used_api_key = api_key or settings.OPENAI_API_KEY
    used_model = model or settings.OPENAI_EMBEDDING_MODEL

    if client is None:
        if not used_api_key:
            raise ValueError(
                "Brak klucza OpenAI. Ustaw OPENAI_API_KEY w pliku .env lub przekaż w parametrach."
            )
        client = AsyncOpenAI(api_key=used_api_key)

    try:
        response = await client.embeddings.create(
            input=text.strip(),
            model=used_model,
        )
        if not response.data or not response.data[0].embedding:
            raise ValueError("Nieprawidłowa odpowiedź embeddingu z OpenAI.")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"[OPENAI_EMBEDDING_ERROR] Błąd podczas generowania embeddingu: {str(e)}")
        raise
