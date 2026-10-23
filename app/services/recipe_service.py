"""
Service and orchestrator module for the complete RAG recipe recommendation pipeline.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Union
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
from app.services.prompt_context import prepare_prompt_context
from app.services.prompt_builder import build_rag_prompts
from app.services.llm_service import execute_llm_call
from app.services.recommendation_validator import validate_and_assemble_recommendations


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
        validator_fn: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings
        self.search_fn = search_fn
        self.embedding_fn = embedding_fn
        self.filter_fn = filter_fn
        self.context_fn = context_fn
        self.prompt_builder_fn = prompt_builder_fn
        self.llm_fn = llm_fn
        self.validator_fn = validator_fn

    async def generate_recipes_with_rag(
        self,
        request: RecipeGenerationRequest,
        search_fn: Optional[Callable[..., Any]] = None,
        embedding_fn: Optional[Callable[..., Any]] = None,
        filter_fn: Optional[Callable[..., Any]] = None,
        context_fn: Optional[Callable[..., Any]] = None,
        prompt_builder_fn: Optional[Callable[..., Any]] = None,
        llm_fn: Optional[Callable[..., Any]] = None,
        validator_fn: Optional[Callable[..., Any]] = None,
    ) -> RecipeGenerationResponse:
        """
        Execute RAG recipe recommendation pipeline.
        Step 1: Candidate retrieval from vector database.
        Step 2: Filtering excluded candidates.
        Step 3: Preparing concise prompt context and candidate lookup map.
        Step 4: Constructing system and user prompts for LLM.
        Step 5: Executing LLM generation call.
        Step 6: Validating and assembling recommendation objects.
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
        effective_validator_fn = validator_fn or self.validator_fn

        retrieved_candidates = []
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
                logger.error(f"[RAG_RETRIEVAL_ERROR] Błąd podczas pobierania kandydatów: {e}")
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
            logger.info(
                f"Wysyłanie zapytania do OpenAI (model: {self.settings.OPENAI_MODEL}, mood: '{request.mood}', mealType: '{request.mealType}')..."
            )
            try:
                raw_llm_response = await effective_llm_fn(
                    system_prompt=prompts.system_prompt,
                    user_prompt=prompts.user_prompt,
                    model=self.settings.OPENAI_MODEL,
                )
                logger.info(
                    f"Pomyślnie zrealizowano zapytanie do OpenAI (model: {self.settings.OPENAI_MODEL}, długość odpowiedzi: {len(raw_llm_response) if raw_llm_response else 0} znaków)."
                )
            except Exception as e:
                logger.error(f"[OPENAI_LLM_ERROR] Błąd podczas wywołania LLM w RecipeService: {e}")
                raw_llm_response = None

        # Step 6: Validation and Assembly
        validated_recipes = []
        if effective_validator_fn and raw_llm_response is not None and prompt_context.candidates_map:
            try:
                validated_recipes = effective_validator_fn(
                    raw_response_content=raw_llm_response,
                    candidates_map=prompt_context.candidates_map,
                    mood=request.mood,
                    meal_type=request.mealType,
                    limit=request.limit,
                )
            except Exception as e:
                logger.error(f"[RAG_VALIDATION_ERROR] Błąd podczas walidacji rekomendacji: {e}")
                validated_recipes = []

        recipes: List[RecipeCandidate] = []
        for r in validated_recipes:
            if isinstance(r, RecipeCandidate):
                recipes.append(r)
            elif isinstance(r, dict):
                reasons = r.get("matchReasons") or []
                reasoning_str = ", ".join(reasons) if isinstance(reasons, list) else str(reasons)
                candidate_obj = RecipeCandidate(
                    id=str(r.get("id") or r.get("recipeId") or ""),
                    title=str(r.get("title") or r.get("name") or "Bez tytułu"),
                    url=r.get("link") or r.get("originalUrl") or r.get("url"),
                    description=r.get("description"),
                    ingredients=r.get("ingredients") or [],
                    instructions=r.get("instructions") or r.get("steps") or [],
                    matchScore=r.get("matchScore"),
                    reasoning=reasoning_str or None,
                )
                recipes.append(candidate_obj)

        execution_time_ms = (time.time() - start_time) * 1000

        subtask_num = (
            7
            if effective_validator_fn and raw_llm_response is not None
            else (
                6
                if raw_llm_response is not None
                else (5 if prompt_context.candidates_for_prompt else (4 if filtered_candidates else (2 if retrieved_candidates else 1)))
            )
        )
        status_str = (
            "recommendations_validated"
            if effective_validator_fn and raw_llm_response is not None
            else (
                "llm_called"
                if raw_llm_response is not None
                else ("prompts_built" if prompt_context.candidates_for_prompt else ("context_prepared" if filtered_candidates else ("candidates_retrieved" if retrieved_candidates else "initialized")))
            )
        )

        trace = GenerationTrace(
            retrievedCandidatesCount=len(retrieved_candidates),
            filteredCandidatesCount=len(filtered_candidates),
            llmModelUsed=self.settings.OPENAI_MODEL,
            executionTimeMs=round(execution_time_ms, 2),
            additionalMetadata={
                "status": status_str,
                "subtask": subtask_num,
                "promptCandidatesCount": len(prompt_context.candidates_for_prompt),
                "systemPromptLength": len(prompts.system_prompt) if prompts else 0,
                "userPromptLength": len(prompts.user_prompt) if prompts else 0,
                "rawLlmResponseLength": len(raw_llm_response) if raw_llm_response else 0,
                "validatedRecipesCount": len(validated_recipes),
            },
        )

        return RecipeGenerationResponse(
            recipes=recipes,
            trace=trace,
        )

    # Alias for backward compatibility
    generate_recipes = generate_recipes_with_rag


def get_recipe_service() -> RecipeService:
    return RecipeService(
        settings=get_settings(),
        llm_fn=execute_llm_call,
        validator_fn=validate_and_assemble_recommendations,
    )
