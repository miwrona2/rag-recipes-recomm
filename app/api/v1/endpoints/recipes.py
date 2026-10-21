from fastapi import APIRouter, Depends, status
from app.models.recipe import RecipeGenerationRequest, RecipeGenerationResponse
from app.services.recipe_service import RecipeService, get_recipe_service

router = APIRouter()


@router.post(
    "/recipes/generate",
    response_model=RecipeGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate recipe recommendations",
    description="Validates input parameters and triggers the RAG recipe generation pipeline.",
)
async def generate_recipes(
    payload: RecipeGenerationRequest,
    recipe_service: RecipeService = Depends(get_recipe_service),
) -> RecipeGenerationResponse:
    return await recipe_service.generate_recipes(payload)
