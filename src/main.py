"""
LALS - Likhon Advanced Language System (Multi-API Gateway)
============================================================

A unified API gateway supporting both OpenAI and Anthropic protocols.
Built with FastAPI and llama-cpp-python for efficient local inference.

Main Features:
- OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions, /v1/models)
- Anthropic-compatible endpoints (/v1/messages, /v1/complete)
- Universal auto-detection endpoint (/v1/inference)
- Streaming support for both protocols
- Qwen3-0.6B model (Q4_K_M quantized)

Author: Likhon AI Labs
Version: 2.0.0
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_config, Config
from core.model import get_model_engine, shutdown_model_engine
from routers.openai_routes import router as openai_router
from routers.anthropic_routes import router as anthropic_router
from routers.universal_routes import router as universal_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting LALS Multi-API Gateway...")
    
    # Load configuration
    config = get_config()
    logger.info(f"Model path: {config.model.path}")
    logger.info(f"Context size: {config.model.n_ctx}")
    logger.info(f"Threads: {config.model.n_threads}")
    
    # Note: Model will be loaded lazily on first request
    # This allows the app to start quickly even if model download is needed
    logger.info("LALS v2.0.3 - Multi-API Gateway starting (model loads on first request)")
    logger.info(f"Listening on http://{config.server.host}:{config.server.port}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down LALS...")
    shutdown_model_engine()
    logger.info("Shutdown complete.")


# Create FastAPI application
app = FastAPI(
    title="LALS - Likhon Advanced Language System",
    description=(
        "A unified API gateway supporting both OpenAI and Anthropic protocols. "
        "Provides advanced NLU, automated content generation, and code assistance "
        "capabilities through a transformer-based architecture optimized for "
        "lower latency and efficient token processing."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# Include routers
config = get_config()

if config.api.enable_openai_api:
    app.include_router(openai_router)
    logger.info("Registered OpenAI-compatible routes")

if config.api.enable_anthropic_api:
    app.include_router(anthropic_router)
    logger.info("Registered Anthropic-compatible routes")

if config.api.enable_universal_router:
    app.include_router(universal_router)
    logger.info("Registered universal inference route")


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint - returns service information."""
    return {
        "service": "LALS - Likhon Advanced Language System",
        "version": "2.0.0",
        "model": "Qwen3-0.6B-Q4_K_M",
        "architecture": "Transformer-based, multi-protocol API gateway",
        "capabilities": ["NLU", "Content Generation", "Code Assistance"],
        "protocols": {
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "universal": "/v1/inference"
        },
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "GET /v1/models": "List models (OpenAI)",
            "POST /v1/chat/completions": "Chat completion (OpenAI)",
            "POST /v1/completions": "Legacy completion (OpenAI)",
            "POST /v1/messages": "Claude messages (Anthropic)",
            "POST /v1/complete": "Legacy completion (Anthropic)",
            "POST /v1/inference": "Auto-detected routing"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_model_engine(auto_load=False)
        if engine and engine.is_loaded:
            return {
                "status": "healthy",
                "model": "qwen3-0.6b-q4_k_m.gguf",
                "protocols": {
                    "openai": config.api.enable_openai_api,
                    "anthropic": config.api.enable_anthropic_api,
                    "universal": config.api.enable_universal_router
                }
            }
        else:
            return {
                "status": "initializing",
                "message": "Model not yet loaded, will load on first request"
            }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes/orchestration."""
    return {"status": "ready"}


# Legacy endpoint (backwards compatibility)
@app.get("/v1")
async def api_info():
    """API information endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-0.6b",
                "object": "model",
                "created": 1677610602,
                "owned_by": "likhon-ai-labs"
            }
        ]
    }


def main():
    """Main entry point."""
    import uvicorn
    from core.config import get_config
    
    config = get_config()
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=config.server.workers if not config.server.reload else 1,
        log_level=config.server.log_level.lower()
    )


if __name__ == "__main__":
    main()
