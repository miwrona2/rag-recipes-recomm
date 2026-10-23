from fastapi import APIRouter, Depends, HTTPException, status
from app.core.logging import logger
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
    try:
        return await recipe_service.generate_recipes_with_rag(payload)
    except ValueError as e:
        logger.warning(f"Validation error during recipe generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error during recipe generation pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas generowania rekomendacji RAG: {str(e)}",
        )
