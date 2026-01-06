"""Middleware configuration for dashboard API.

This module sets up middleware for logging, error handling, CORS, and security.
"""

import logging
import time
import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from src.dashboard.api.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    get_rate_limit_config
)

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details.
        
        Parameters:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response: HTTP response with X-Process-Time header
        """
        start_time = time.time()
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add process time header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # Log response
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"{request.method} {request.url.path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s",
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": "An unexpected error occurred"
                }
            )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors globally.
        
        Parameters:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            Response: HTTP response with error details if exception occurred
        """
        try:
            return await call_next(request)
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "detail": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": "An unexpected error occurred. Please check logs for details."
                }
            )


def setup_middleware(app) -> None:
    """Setup application middleware.
    
    Parameters:
        app: FastAPI application instance
        
    Security Impact:
        - Logs all requests for audit trail
        - Handles errors gracefully without exposing sensitive information
        - Rate limiting prevents abuse
        - Security headers protect against common vulnerabilities
        
    Middleware Order (important):
        1. SecurityHeadersMiddleware - Adds security headers
        2. RateLimitMiddleware - Enforces rate limits
        3. ErrorHandlingMiddleware - Handles errors
        4. LoggingMiddleware - Logs requests/responses
    """
    # Security headers (first, so they're always added)
    enable_hsts = os.getenv("ENABLE_HSTS", "false").lower() == "true"
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=enable_hsts)
    
    # Rate limiting (before error handling to catch rate limit errors)
    rate_limit_config = get_rate_limit_config()
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=int(os.getenv("RATE_LIMIT_DEFAULT", "100")),
        default_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        per_endpoint_limits=rate_limit_config
    )
    
    # Error handling (before logging to catch errors)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Logging (last, to log everything including errors)
    app.add_middleware(LoggingMiddleware)

