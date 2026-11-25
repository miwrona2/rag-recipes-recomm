from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.core.tags import ALLOWED_MEAL_TYPES, ALLOWED_MOODS, sanitize_tags

QUERY_MAX_LENGTH = 200


class PrecomputedQuery(BaseModel):
    """Ready-made Step 0 result (e.g. stored with a suggestion in the Node.js DB) - skips the LLM parse."""

    semanticQuery: str = Field(
        ...,
        min_length=1,
        max_length=QUERY_MAX_LENGTH * 2,
        description="English query used for vector search",
        examples=["quick chicken pasta"],
    )
    moodTags: List[str] = Field(default_factory=list, description="Mood tags; values outside the allowed list are dropped")
    mealTypeTags: List[str] = Field(default_factory=list, description="Meal type tags; values outside the allowed list are dropped")

    @field_validator("semanticQuery", mode="before")
    @classmethod
    def _normalize_semantic_query(cls, v: Any) -> Any:
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("moodTags", mode="after")
    @classmethod
    def _sanitize_moods(cls, v: List[str]) -> List[str]:
        return sanitize_tags(v, ALLOWED_MOODS)

    @field_validator("mealTypeTags", mode="after")
    @classmethod
    def _sanitize_meal_types(cls, v: List[str]) -> List[str]:
        return sanitize_tags(v, ALLOWED_MEAL_TYPES)


class RecipeGenerationRequest(BaseModel):
    mood: str = Field(
        ...,
        min_length=1,
        description="The mood, taste preference, or context (e.g. 'quick', 'comfort food', 'light')",
        examples=["quick and healthy"],
    )
    mealType: str = Field(
        ...,
        min_length=1,
        description="The target meal type (e.g. 'breakfast', 'lunch', 'dinner', 'dessert', 'snack')",
        examples=["dinner"],
    )
    excludeUrls: List[str] = Field(
        default_factory=list,
        description="List of recipe URLs to exclude from candidates",
        examples=[["https://example.com/recipes/pasta"]],
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of recipe recommendations to generate",
        examples=[3],
    )
    query: Optional[str] = Field(
        default=None,
        max_length=QUERY_MAX_LENGTH,
        description=(
            "Optional free-text description of what the user is looking for "
            "(any language). Complements mood/mealType; omit or leave empty to skip."
        ),
        examples=["szybki makaron z kurczakiem"],
    )

    parsedQuery: Optional[PrecomputedQuery] = Field(
        default=None,
        description=(
            "Optional ready-made parse of `query` (e.g. for an unedited suggestion). "
            "When set, the LLM query-parsing step is skipped."
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        v = " ".join(v.split())
        return v or None


class RecipeCandidate(BaseModel):
    id: Optional[str] = Field(default=None, description="Recipe unique ID")
    title: str = Field(..., description="Recipe title")
    url: Optional[str] = Field(default=None, description="Original recipe URL")
    description: Optional[str] = Field(default=None, description="Short summary/description")
    ingredients: List[str] = Field(default_factory=list, description="List of required ingredients")
    instructions: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    prepTimeMinutes: Optional[int] = Field(default=None, ge=0, description="Preparation time in minutes")
    cookTimeMinutes: Optional[int] = Field(default=None, ge=0, description="Cooking time in minutes")
    totalTimeMinutes: Optional[int] = Field(default=None, ge=0, description="Total time in minutes")
    servings: Optional[int] = Field(default=None, ge=1, description="Number of servings")
    matchScore: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="RAG/LLM match relevance score")
    reasoning: Optional[str] = Field(default=None, description="Why this recipe fits the request")


class QueryUnderstanding(BaseModel):
    originalQuery: str = Field(..., description="User query as received (whitespace-normalized)")
    semanticQuery: str = Field(..., description="English query used for vector search")
    moodTags: List[str] = Field(default_factory=list, description="Mood tags inferred from the query (allowed values only)")
    mealTypeTags: List[str] = Field(default_factory=list, description="Meal type tags inferred from the query (allowed values only)")
    effectiveMoods: List[str] = Field(default_factory=list, description="Tile mood merged with inferred mood tags, used for filtering")
    effectiveMealTypes: List[str] = Field(default_factory=list, description="Tile mealType merged with inferred meal type tags, used for filtering")
    source: Literal["llm", "precomputed", "fallback"] = Field(
        ...,
        description=(
            "'llm' when parsed by the LLM, 'precomputed' when taken from request.parsedQuery, "
            "'fallback' when the raw query was used as-is"
        ),
    )


class GenerationTrace(BaseModel):
    retrievedCandidatesCount: int = Field(default=0, description="Number of candidates retrieved from vector store")
    filteredCandidatesCount: int = Field(default=0, description="Number of candidates remaining after filtering")
    llmModelUsed: Optional[str] = Field(default=None, description="LLM model identifier used for generation")
    executionTimeMs: Optional[float] = Field(default=None, description="Total pipeline execution time in milliseconds")
    additionalMetadata: Optional[Dict[str, Any]] = Field(default=None, description="Debugging metadata or token usage")


class RecipeGenerationResponse(BaseModel):
    recipes: List[RecipeCandidate] = Field(default_factory=list, description="Recommended recipes")
    queryUnderstanding: Optional[QueryUnderstanding] = Field(
        default=None, description="How the optional text query was understood (null when no query was sent)"
    )
    trace: Optional[GenerationTrace] = Field(default=None, description="Pipeline diagnostic trace metadata")
