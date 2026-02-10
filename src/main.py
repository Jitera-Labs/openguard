"""
OpenGuard - Main FastAPI Application

This module creates the FastAPI application that handles:
- Health checks
- Model listing
- Chat completions with guard application
- Authentication and request middleware
"""

import json

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src import config, llm, mapper
from src import log as log_module
from src.guard_engine import apply_guards, log_audit
from src.guards import GuardBlockedError, get_guards
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.request_state import RequestStateMiddleware

# Setup logging
logger = log_module.setup_logger(__name__)

# Create FastAPI app
app = FastAPI(title="OpenGuard", description="OpenAI-compatible guardrail proxy", version="0.1.0")

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
        "endpoints": {"health": "/health", "models": "/v1/models", "chat": "/v1/chat/completions"},
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
async def chat_completions(request: Request, authorized: bool = Depends(verify_auth)):
    """Handle chat completion with guards applied"""
    try:
        # Parse request body
        payload = await request.json()

        # Validate messages
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
             raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")
        for m in messages:
             if not isinstance(m, dict) or "content" not in m or "role" not in m:
                  raise HTTPException(status_code=400, detail="Invalid message format")
             if m.get("content") is None:
                  raise HTTPException(status_code=400, detail="Message content cannot be null")

        # Validate types
        if "max_tokens" in payload and not isinstance(payload["max_tokens"], int):
             raise HTTPException(status_code=400, detail="'max_tokens' must be integer")
        if "temperature" in payload and not isinstance(payload["temperature"], (int, float)):
             raise HTTPException(status_code=400, detail="'temperature' must be number")

        # Resolve backend
        try:
            proxy_config = mapper.resolve_request_config(payload)
        except ValueError as e:
            logger.error(f"Failed to resolve backend: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        # Create LLM proxy instance
        llm_instance = llm.LLM(
            url=proxy_config["url"],
            headers=proxy_config["headers"],
            model=payload.get("model"),
            params=proxy_config["params"],
            messages=payload.get("messages"),
        )

        # Apply guards
        guards = get_guards()
        # Guards modify chat/llm in-place
        _, audit_logs = apply_guards(llm_instance.chat, llm_instance, guards)
        log_audit(audit_logs)

        # Proxy request
        stream = payload.get("stream", False)

        if stream:
            return StreamingResponse(await llm_instance.serve(), media_type="text/event-stream")
        else:
            # Non-streaming: collect response and return as JSON
            full_content = ""
            tool_calls = []
            # Dictionary to aggregate tool calls by index: {index: tool_call_dict}
            pending_tool_calls = {}
            final_data = None
            finish_reason = None

            async for chunk in await llm_instance.serve():
                # Handle error chunks (raw JSON)
                if not chunk.startswith("data: "):
                    try:
                        err = json.loads(chunk)
                        if "error" in err:
                            # Map error code if present, default to 500
                            code = int(err["error"].get("code", 500))
                            if code == 404:
                                code = 400
                            return JSONResponse(content=err, status_code=code)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    continue

                # Handle SSE chunks
                line = chunk.strip()
                if line == "data: [DONE]":
                    continue

                try:
                    if len(line) > 6:
                        data = json.loads(line[6:]) # Skip "data: "

                        # Store metadata from last valid chunk
                        final_data = data

                        # Accumulate content
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content

                            # Accumulate tool calls with merging
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index")
                                    # Fallback: if no index (e.g. from some backends), append to list or treat as new
                                    # But standard OpenAI stream has index.
                                    # If llm.py sends full objects without index, we can just append, but let's try to handle both.
                                    if idx is None:
                                        tool_calls.append(tc)
                                    else:
                                        if idx not in pending_tool_calls:
                                            pending_tool_calls[idx] = tc
                                        else:
                                            current = pending_tool_calls[idx]
                                            # Merge arguments
                                            if "function" in tc:
                                                if "name" in tc["function"] and tc["function"]["name"]:
                                                    if "function" not in current:
                                                        current["function"] = {}
                                                    current["function"]["name"] = tc["function"]["name"]

                                                if "arguments" in tc["function"] and tc["function"]["arguments"]:
                                                    if "function" not in current:
                                                        current["function"] = {"arguments": ""}
                                                    if "arguments" not in current["function"]:
                                                        current["function"]["arguments"] = ""
                                                    current["function"]["arguments"] += tc["function"]["arguments"]

                                            # Merge other fields if present
                                            if "id" in tc and tc["id"]:
                                                current["id"] = tc["id"]
                                            if "type" in tc and tc["type"]:
                                                current["type"] = tc["type"]

                            if "finish_reason" in choices[0]:
                                finish_reason = choices[0]["finish_reason"]

                except json.JSONDecodeError:
                    continue

            if final_data:
                # Add aggregated tool calls to the list
                if pending_tool_calls:
                    # Sort by index to maintain order
                    sorted_idxs = sorted(pending_tool_calls.keys())
                    for idx in sorted_idxs:
                        tool_calls.append(pending_tool_calls[idx])

                # Construct non-streaming response format
                response_obj = final_data.copy()
                response_obj["object"] = "chat.completion"

                # Reconstruct message
                message = {
                    "role": "assistant",
                    "content": full_content if full_content else None
                }

                if tool_calls:
                    message["tool_calls"] = tool_calls

                if "choices" in response_obj and response_obj["choices"]:
                    response_obj["choices"][0]["message"] = message

                    # Set finish reason if captured
                    if finish_reason:
                        response_obj["choices"][0]["finish_reason"] = finish_reason
                    elif tool_calls:
                         # Default to tool_calls if not explicitly set but tools are present
                         response_obj["choices"][0]["finish_reason"] = "tool_calls"

                    # Remove delta if present
                    if "delta" in response_obj["choices"][0]:
                        del response_obj["choices"][0]["delta"]

                return JSONResponse(content=response_obj, status_code=200)
            else:
                raise HTTPException(status_code=500, detail="Empty response from downstream")

    except GuardBlockedError as e:
        return JSONResponse(
            status_code=403,
            content={"error": {"message": str(e), "type": "guard_block", "code": 403}},
        )
    except httpx.HTTPStatusError as e:
        error_body = await e.response.aread()
        logger.error(f"Downstream error {e.response.status_code}: {error_body.decode('utf-8')}")

        try:
            content = json.loads(error_body)
        except json.JSONDecodeError:
            content = {
                "error": {
                    "message": error_body.decode("utf-8"),
                    "type": "downstream_error",
                    "code": e.response.status_code,
                }
            }

        return JSONResponse(status_code=e.response.status_code, content=content)
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Failed to connect to downstream API: {str(e)}",
                    "type": "connection_error",
                    "code": 502,
                }
            },
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error: {e}")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Request to downstream API timed out",
                    "type": "timeout_error",
                    "code": 504,
                }
            },
        )
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except (TypeError, KeyError, AttributeError) as e:
        logger.error(f"Invalid request format: {e}")
        raise HTTPException(status_code=400, detail="Invalid request format")
    except httpx.HTTPStatusError as e:
        logger.error(f"Upstream error: {e}")
        if e.response.status_code == 404:
            raise HTTPException(status_code=400, detail="Model not found or invalid")
        raise HTTPException(status_code=e.response.status_code, detail="Upstream error")
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

    uvicorn.run("src.main:app", host=host, port=port, log_level=log_level, reload=False)


if __name__ == "__main__":
    main()
