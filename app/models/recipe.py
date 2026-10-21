from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


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
    availableIngredients: Optional[List[str]] = Field(
        default=None,
        description="Optional list of ingredients available in the fridge",
        examples=[["chicken breast", "garlic", "olive oil"]],
    )
    dietaryRestrictions: Optional[List[str]] = Field(
        default=None,
        description="Optional dietary restrictions (e.g. 'vegetarian', 'gluten-free')",
        examples=[["gluten-free"]],
    )
    userPreferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional user preference settings",
    )


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


class GenerationTrace(BaseModel):
    retrievedCandidatesCount: int = Field(default=0, description="Number of candidates retrieved from vector store")
    filteredCandidatesCount: int = Field(default=0, description="Number of candidates remaining after filtering")
    llmModelUsed: Optional[str] = Field(default=None, description="LLM model identifier used for generation")
    executionTimeMs: Optional[float] = Field(default=None, description="Total pipeline execution time in milliseconds")
    additionalMetadata: Optional[Dict[str, Any]] = Field(default=None, description="Debugging metadata or token usage")


class RecipeGenerationResponse(BaseModel):
    recipes: List[RecipeCandidate] = Field(default_factory=list, description="Recommended recipes")
    trace: Optional[GenerationTrace] = Field(default=None, description="Pipeline diagnostic trace metadata")
