"""
OpenGuard - Main FastAPI Application

This module creates the FastAPI application that handles:
- Health checks
- Model listing
- Chat completions with guard application
- Authentication and request middleware
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src import config, mapper, llm, log as log_module
from src.guards import get_guards
from src.guard_engine import apply_guards, log_audit
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.request_state import RequestStateMiddleware

# Setup logging
logger = log_module.setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="OpenGuard",
    description="OpenAI-compatible guardrail proxy",
    version="0.1.0"
)

# Add middlewares in correct order
app.add_middleware(RequestStateMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.OPENGUARD_CORS_ORIGINS.value,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_auth(request: Request):
    """Verify API key if configured"""
    api_key = config.OPENGUARD_API_KEY.value
    api_keys_list = config.OPENGUARD_API_KEYS.value

    # If no keys configured, allow all requests
    if not api_key and not api_keys_list:
        return True

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # Remove "Bearer "

    # Check against configured keys
    valid_keys = [api_key] if api_key else []
    valid_keys.extend(api_keys_list)

    if token not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "OpenGuard",
        "version": "0.1.0",
        "description": "OpenAI-compatible guardrail proxy",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models",
            "chat": "/v1/chat/completions"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(authorized: bool = Depends(verify_auth)):
    """List all available models from downstream APIs"""
    try:
        models = await mapper.list_downstream()
        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorized: bool = Depends(verify_auth)
):
    """Handle chat completion with guards applied"""
    try:
        # Parse request body
        payload = await request.json()

        # Apply guards
        guards = get_guards()
        modified_payload, audit_logs = apply_guards(payload, guards)
        log_audit(audit_logs)

        # Resolve backend
        try:
            proxy_config = await mapper.resolve_request_config(modified_payload)
        except ValueError as e:
            logger.error(f"Failed to resolve backend: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        # Create LLM proxy instance
        llm_instance = llm.LLM(
            url=proxy_config['url'],
            headers=proxy_config['headers'],
            payload=proxy_config['payload'],
            stream=proxy_config['stream']
        )

        # Proxy request
        if proxy_config['stream']:
            return StreamingResponse(
                llm_instance.serve(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming: collect response and return as JSON
            response_text = None
            async for chunk in llm_instance.serve():
                response_text = chunk
                break  # Only one chunk for non-streaming

            if response_text:
                return JSONResponse(
                    content=json.loads(response_text),
                    status_code=200
                )
            else:
                raise HTTPException(status_code=500, detail="Empty response from downstream")

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def main():
    """Entry point for CLI"""
    import uvicorn

    host = config.OPENGUARD_HOST.value
    port = config.OPENGUARD_PORT.value
    log_level = config.OPENGUARD_LOG_LEVEL.value.lower()

    logger.info(f"Starting OpenGuard on {host}:{port}")
    logger.info(f"Config file: {config.OPENGUARD_CONFIG.value}")

    # Load guards at startup
    guards = get_guards()
    logger.info(f"Loaded {len(guards)} guard rules")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False
    )


if __name__ == "__main__":
    main()
