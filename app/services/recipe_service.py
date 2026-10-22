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
from app.services.candidate_filter import filter_excluded_candidates
from app.services.prompt_context import prepare_prompt_context, PromptContext
from app.services.prompt_builder import build_rag_prompts, RagPrompts
from app.services.llm_service import execute_llm_call


class RecipeService:
    """Service handling recipe recommendation pipeline."""

    def __init__(
        self,
        settings: Settings,
        search_fn: Optional[Callable[..., Any]] = None,
        embedding_fn: Optional[Callable[..., Any]] = None,
        filter_fn: Optional[Callable[..., Any]] = None,
        context_fn: Optional[Callable[..., Any]] = None,
        prompt_builder_fn: Optional[Callable[..., Any]] = None,
        llm_fn: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self.search_fn = search_fn
        self.embedding_fn = embedding_fn
        self.filter_fn = filter_fn
        self.context_fn = context_fn
        self.prompt_builder_fn = prompt_builder_fn
        self.llm_fn = llm_fn

    async def generate_recipes(
        self,
        request: RecipeGenerationRequest,
        search_fn: Optional[Callable[..., Any]] = None,
        embedding_fn: Optional[Callable[..., Any]] = None,
        filter_fn: Optional[Callable[..., Any]] = None,
        context_fn: Optional[Callable[..., Any]] = None,
        prompt_builder_fn: Optional[Callable[..., Any]] = None,
        llm_fn: Optional[Callable[..., Any]] = None,
    ) -> RecipeGenerationResponse:
        """
        Execute RAG recipe recommendation pipeline.
        Step 1 (Subtask 2): Candidate retrieval from vector database.
        Step 2 (Subtask 3): Filtering excluded candidates.
        Step 3 (Subtask 4): Preparing concise prompt context and candidate lookup map.
        Step 4 (Subtask 5): Constructing system and user prompts for LLM.
        Step 5 (Subtask 6): Executing LLM generation call.
        Subsequent subtasks will integrate response validation and assembly.
        """
        start_time = time.time()
        logger.info(
            f"Generating recipes for mood='{request.mood}', mealType='{request.mealType}', limit={request.limit}"
        )

        effective_search_fn = search_fn or self.search_fn
        effective_embedding_fn = embedding_fn or self.embedding_fn
        effective_filter_fn = filter_fn or self.filter_fn or filter_excluded_candidates
        effective_context_fn = context_fn or self.context_fn or prepare_prompt_context
        effective_prompt_builder_fn = prompt_builder_fn or self.prompt_builder_fn or build_rag_prompts
        effective_llm_fn = llm_fn or self.llm_fn

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

        # Step 2: Filter excluded candidates
        filtered_candidates = effective_filter_fn(
            retrieved_candidates,
            exclude_urls=request.excludeUrls,
        )

        # Step 3: Augmentation - prepare prompt context
        prompt_context = effective_context_fn(
            filtered_candidates,
            max_candidates=10,
        )

        # Step 4: Prompting - build system and user prompts
        prompts = effective_prompt_builder_fn(
            mood=request.mood,
            meal_type=request.mealType,
            candidates_for_prompt=prompt_context.candidates_for_prompt,
            limit=request.limit,
        )

        # Step 5: Generation - execute LLM call if candidates exist and llm_fn is configured
        raw_llm_response = None
        if effective_llm_fn and prompt_context.candidates_for_prompt:
            try:
                raw_llm_response = await effective_llm_fn(
                    system_prompt=prompts.system_prompt,
                    user_prompt=prompts.user_prompt,
                    model=self.settings.OPENAI_MODEL,
                )
            except Exception as e:
                logger.error(f"Błąd podczas wywołania LLM w RecipeService: {e}")
                raw_llm_response = None

        recipes: List[RecipeCandidate] = []
        execution_time_ms = (time.time() - start_time) * 1000

        trace = GenerationTrace(
            retrievedCandidatesCount=len(retrieved_candidates),
            filteredCandidatesCount=len(filtered_candidates),
            llmModelUsed=self.settings.OPENAI_MODEL,
            executionTimeMs=round(execution_time_ms, 2),
            additionalMetadata={
                "status": "llm_called" if raw_llm_response is not None else ("prompts_built" if prompt_context.candidates_for_prompt else ("context_prepared" if filtered_candidates else ("candidates_retrieved" if retrieved_candidates else "initialized"))),
                "subtask": 6 if raw_llm_response is not None else 5,
                "promptCandidatesCount": len(prompt_context.candidates_for_prompt),
                "systemPromptLength": len(prompts.system_prompt) if prompts else 0,
                "userPromptLength": len(prompts.user_prompt) if prompts else 0,
                "rawLlmResponseLength": len(raw_llm_response) if raw_llm_response else 0,
            },
        )

        return RecipeGenerationResponse(
            recipes=recipes,
            trace=trace,
        )


def get_recipe_service() -> RecipeService:
    return RecipeService(settings=get_settings())
