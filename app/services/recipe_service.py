import time
from typing import Any, Callable, List, Optional
from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.models.recipe import (
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    RecipeCandidate,
    GenerationTrace,
)
from app.services.vector_search import retrieve_candidates


class RecipeService:
    """Service handling recipe recommendation pipeline."""

    def __init__(
        self,
        settings: Settings,
        search_fn: Optional[Callable[..., Any]] = None,
        embedding_fn: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self.search_fn = search_fn
        self.embedding_fn = embedding_fn

    async def generate_recipes(
        self,
        request: RecipeGenerationRequest,
        search_fn: Optional[Callable[..., Any]] = None,
        embedding_fn: Optional[Callable[..., Any]] = None,
    ) -> RecipeGenerationResponse:
        """
        Execute RAG recipe recommendation pipeline.
        Step 1 (Subtask 2): Candidate retrieval from vector database.
        Subsequent subtasks will integrate filtering, prompting, LLM call, and response assembly.
        """
        start_time = time.time()
        logger.info(
            f"Generating recipes for mood='{request.mood}', mealType='{request.mealType}', limit={request.limit}"
        )

        effective_search_fn = search_fn or self.search_fn
        effective_embedding_fn = embedding_fn or self.embedding_fn

        retrieved_candidates = []
        # If custom search_fn is provided OR Supabase & OpenAI are configured, retrieve candidates
        if effective_search_fn or (
            self.settings.SUPABASE_URL and self.settings.SUPABASE_KEY and self.settings.OPENAI_API_KEY
        ):
            try:
                retrieved_candidates = await retrieve_candidates(
                    mood=request.mood,
                    meal_type=request.mealType,
                    match_count=20,
                    match_threshold=0.35,
                    search_fn=effective_search_fn,
                    embedding_fn=effective_embedding_fn,
                )
            except Exception as e:
                logger.error(f"Błąd podczas pobierania kandydatów: {e}")
                retrieved_candidates = []

        recipes: List[RecipeCandidate] = []
        execution_time_ms = (time.time() - start_time) * 1000

        trace = GenerationTrace(
            retrievedCandidatesCount=len(retrieved_candidates),
            filteredCandidatesCount=0,
            llmModelUsed=self.settings.OPENAI_MODEL,
            executionTimeMs=round(execution_time_ms, 2),
            additionalMetadata={
                "status": "candidates_retrieved" if retrieved_candidates else "initialized",
                "subtask": 2,
            },
        )

        return RecipeGenerationResponse(
            recipes=recipes,
            trace=trace,
        )


def get_recipe_service() -> RecipeService:
    return RecipeService(settings=get_settings())
