from typing import Dict, Any
from fastapi import APIRouter, status
from app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns the current operational status of the service",
    response_model=Dict[str, Any],
)
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
