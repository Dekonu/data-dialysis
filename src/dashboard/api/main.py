"""Main FastAPI application for Data-Dialysis dashboard.

This module sets up the FastAPI application with all routes, middleware,
and configuration for the dashboard API.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard.api.middleware import setup_middleware
from src.dashboard.api.logging_config import setup_logging
from src.dashboard.api.routes import (
    health,
    metrics,
    audit,
    circuit_breaker,
    websocket
)

# Configure structured logging
use_json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
log_level = os.getenv("LOG_LEVEL", "INFO")
setup_logging(use_json=use_json_logs, log_level=log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Data-Dialysis Dashboard API starting up...")
    logger.info("API documentation available at /api/docs")
    logger.info(f"Logging level: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info(f"JSON logs: {os.getenv('JSON_LOGS', 'false')}")
    yield
    # Shutdown
    logger.info("Data-Dialysis Dashboard API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Data-Dialysis Dashboard API",
    description="Health monitoring and metrics API for Data-Dialysis pipeline",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS configuration
# In production, replace with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",  # Alternative port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)

# Setup custom middleware
setup_middleware(app)

# Include routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(audit.router)
app.include_router(circuit_breaker.router)
app.include_router(websocket.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Data-Dialysis Dashboard API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.dashboard.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
