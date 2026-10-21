import time
from typing import List
from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.models.recipe import (
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    RecipeCandidate,
    GenerationTrace,
)


class RecipeService:
    """Service handling recipe recommendation pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_recipes(self, request: RecipeGenerationRequest) -> RecipeGenerationResponse:
        """
        Execute RAG recipe recommendation pipeline.
        In Subtask 1, returns a structured skeleton response.
        Subsequent subtasks will integrate candidate retrieval, filtering, and LLM synthesis.
        """
        start_time = time.time()
        logger.info(
            f"Generating recipes for mood='{request.mood}', mealType='{request.mealType}', limit={request.limit}"
        )

        # In Podzadanie 1: Structured stub response
        # Future subtasks: retrieveCandidates -> filterExcludedCandidates -> prompt -> llm -> validate
        recipes: List[RecipeCandidate] = []
        
        execution_time_ms = (time.time() - start_time) * 1000
        trace = GenerationTrace(
            retrievedCandidatesCount=0,
            filteredCandidatesCount=0,
            llmModelUsed=self.settings.OPENAI_MODEL,
            executionTimeMs=round(execution_time_ms, 2),
            additionalMetadata={"status": "initialized", "subtask": 1},
        )

        return RecipeGenerationResponse(
            recipes=recipes,
            trace=trace,
        )


def get_recipe_service() -> RecipeService:
    return RecipeService(settings=get_settings())
