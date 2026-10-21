import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from supabase import Client, create_client
from app.core.config import get_settings
from app.core.logging import logger
from app.services.embedding_service import get_embedding


def get_supabase_client(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    client: Optional[Client] = None,
) -> Client:
    """
    Get or create a Supabase client instance.
    """
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


async def search_recipes_in_vector_db(
    query: Union[str, List[str]],
    match_count: int = 20,
    match_threshold: float = 0.35,
    mood: Optional[str] = None,
    meal_type: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    embedding_fn: Optional[Callable[..., Any]] = None,
    supabase_client: Optional[Client] = None,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search recipes in Supabase vector database using embedding and cosine similarity RPC match_recipes.

    :param query: Search query text or list of query strings/ingredients.
    :param match_count: Maximum number of candidate recipes to retrieve.
    :param match_threshold: Cosine similarity threshold (0.0 to 1.0).
    :param mood: Optional mood tag filter.
    :param meal_type: Optional meal type tag filter.
    :param embedding: Optional precomputed embedding vector.
    :param embedding_fn: Optional embedding generator function (defaults to get_embedding).
    :param supabase_client: Optional Supabase client instance.
    :param supabase_url: Optional Supabase URL override.
    :param supabase_key: Optional Supabase key override.
    :return: List of recipe candidate dicts sorted by similarity descending.
    """
    query_text = ""
    if isinstance(query, list):
        query_text = ", ".join([str(item).strip() for item in query if str(item).strip()])
    elif isinstance(query, str):
        query_text = query.strip()

    if not query_text and (embedding is None or not isinstance(embedding, list)):
        raise ValueError(
            'Parametr "query" musi być niepustym ciągiem znaków lub listą składników, '
            'chyba że podano "embedding".'
        )

    # 1. Obtain or generate embedding
    query_embedding = embedding
    if query_embedding is None:
        fn = embedding_fn or get_embedding
        query_embedding = await fn(query_text)

    if not isinstance(query_embedding, list) or len(query_embedding) == 0:
        raise ValueError("Nie udało się wygenerować prawidłowego wektora embeddingu.")

    # 2. Get Supabase client
    client = get_supabase_client(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        client=supabase_client,
    )

    # 3. Call RPC function match_recipes
    rpc_params = {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": match_count,
    }

    logger.info(
        f"Wywołanie funkcji match_recipes w Supabase z parametrami: match_threshold={match_threshold}, match_count={match_count}"
    )

    def _execute_rpc() -> Any:
        return client.rpc("match_recipes", rpc_params).execute()

    try:
        response = await asyncio.to_thread(_execute_rpc)
    except Exception as e:
        logger.error(f"[RAG_RETRIEVAL_ERROR] Błąd podczas wyszukiwania wektorowego w Supabase: {str(e)}")
        raise RuntimeError(f"Błąd wyszukiwania wektorowego: {str(e)}") from e

    results: List[Dict[str, Any]] = response.data if response and response.data else []

    # 4. Optional tag filtering (mood, mealType) matching Node.js implementation
    if mood:
        target_mood = mood.lower()

        def _matches_mood(recipe: Dict[str, Any]) -> bool:
            mood_tags = recipe.get("mood_tags") or recipe.get("mood")
            if not mood_tags:
                return True
            if isinstance(mood_tags, list):
                return any(target_mood in str(t).lower() for t in mood_tags)
            return target_mood in str(mood_tags).lower()

        results = [r for r in results if _matches_mood(r)]

    if meal_type:
        target_meal_type = meal_type.lower()

        def _matches_meal_type(recipe: Dict[str, Any]) -> bool:
            meal_type_tags = (
                recipe.get("meal_type_tags")
                or recipe.get("meal_type")
                or recipe.get("mealType")
            )
            if not meal_type_tags:
                return True
            if isinstance(meal_type_tags, list):
                return any(target_meal_type in str(t).lower() for t in meal_type_tags)
            return target_meal_type in str(meal_type_tags).lower()

        results = [r for r in results if _matches_meal_type(r)]

    return results


async def retrieve_candidates(
    mood: str,
    meal_type: str,
    match_count: int = 20,
    match_threshold: float = 0.35,
    supabase_client: Optional[Client] = None,
    embedding_fn: Optional[Callable[..., Any]] = None,
    embedding: Optional[List[float]] = None,
    search_fn: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Step 1: Retrieval - fetches candidates from the vector database.
    Equivalent of Node.js retrieveCandidates(mood, mealType, options).

    :param mood: User mood preference (e.g. 'zdrowe', 'szybkie', 'comfort-food').
    :param meal_type: User meal type (e.g. 'obiad', 'śniadanie', 'kolacja').
    :param match_count: Target number of vector candidates (default 20).
    :param match_threshold: Minimum cosine similarity threshold (default 0.35).
    :param supabase_client: Optional custom Supabase client.
    :param embedding_fn: Optional embedding generator function.
    :param embedding: Optional precomputed embedding vector.
    :param search_fn: Optional search function (defaults to search_recipes_in_vector_db).
    :return: List of candidate recipe dictionaries.
    """
    query = f"{mood} {meal_type}"
    search = search_fn or search_recipes_in_vector_db

    try:
        candidates = await search(
            query=query,
            match_count=match_count,
            match_threshold=match_threshold,
            mood=mood,
            meal_type=meal_type,
            supabase_client=supabase_client,
            embedding_fn=embedding_fn,
            embedding=embedding,
            **kwargs,
        )
    except Exception as err:
        logger.error(f"[RAG_RETRIEVAL_ERROR] Błąd podczas wyszukiwania wektorowego: {str(err)}")
        raise

    if not candidates:
        logger.warning(f"[RAG_SERVICE] Brak kandydatów z bazy wektorowej dla zapytania: {query}")
        return []

    return candidates
