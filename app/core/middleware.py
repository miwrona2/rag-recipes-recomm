import time
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests, execution time, and error capture."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()
        
        path = request.url.path
        method = request.method
        
        logger.info(f"[{request_id}] Incoming request: {method} {path}")
        
        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration:.2f}"
            logger.info(
                f"[{request_id}] Completed {method} {path} - Status: {response.status_code} ({duration:.2f}ms)"
            )
            return response
        except Exception as exc:
            duration = (time.time() - start_time) * 1000
            logger.exception(f"[{request_id}] Unhandled error during {method} {path} ({duration:.2f}ms): {str(exc)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": str(exc) if str(exc) else "An unexpected error occurred while processing the request.",
                    "request_id": request_id,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time-Ms": f"{duration:.2f}",
                },
            )


def setup_middleware(app: FastAPI) -> None:
    """Register custom middlewares to FastAPI application."""
    app.add_middleware(RequestLoggingMiddleware)
