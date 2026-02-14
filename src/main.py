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
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src import config, llm, mapper
from src.chat import Chat
from src import log as log_module
from src.guard_engine import apply_guards, log_audit
from src.guards import GuardBlockedError, get_guards
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.request_state import RequestStateMiddleware

# Setup logging
logger = log_module.setup_logger(__name__)

# Create FastAPI app
app = FastAPI(title="OpenGuard", description="guarding proxy for AI", version="0.1.0")

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


def _anthropic_error(status_code: int, message: str, error_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _is_guard_blocked_error(error: Exception):
    return error.__class__.__name__ == "GuardBlockedError"


def _requires_explicit_anthropic_tool_use(payload: dict | None):
    if not isinstance(payload, dict):
        return False

    if payload.get("stream"):
        return False

    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")

    if not isinstance(tools, list) or len(tools) == 0:
        return False

    return isinstance(tool_choice, dict) and tool_choice.get("type") == "tool"


def _anthropic_message_has_tool_use(response_payload: dict | None):
    if not isinstance(response_payload, dict):
        return False

    content_blocks = response_payload.get("content")
    if not isinstance(content_blocks, list):
        return False

    return any(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in content_blocks
    )


def _build_forward_headers(request: Request, provider: str, route_config: dict):
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    mode = route_config.get("mode")
    downstream_key = route_config.get("key")

    if mode in {"internal", "endpoint"} and downstream_key:
        if provider == "anthropic":
            headers.pop("authorization", None)
            headers.pop("Authorization", None)
            headers["x-api-key"] = downstream_key
        else:
            headers.pop("x-api-key", None)
            headers.pop("X-API-Key", None)
            headers["Authorization"] = f"Bearer {downstream_key}"

    return headers


class _GuardContext:
    def __init__(self, model, params, provider, raw_payload):
        self.model = model
        self.params = params
        self.provider = provider
        self.raw_payload = raw_payload


async def _forward_provider_request(
    request: Request,
    provider: str,
    endpoint_path: str,
    body_bytes: bytes | None,
    payload: dict | None,
):
    incoming_key = mapper.extract_api_key(request.headers)
    model = (payload or {}).get("model") if isinstance(payload, dict) else None

    if incoming_key and incoming_key in mapper.get_internal_api_keys() and model:
        model_map = mapper.MODEL_TO_BACKEND.get(provider.lower(), {})
        if model not in model_map:
            try:
                await mapper.list_downstream(provider=provider)
            except Exception as e:
                logger.warning(f"Unable to warm model cache for provider '{provider}': {e}")

    try:
        route_config = mapper.resolve_provider_route(
            provider=provider,
            endpoint_path=endpoint_path,
            body=payload,
            headers=request.headers,
        )
    except ValueError as e:
        return _anthropic_error(500, str(e), error_type="api_error")

    proxy_headers = _build_forward_headers(request, provider, route_config)
    query_params = dict(request.query_params)

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.request(
                method=request.method,
                url=route_config["endpoint"],
                headers=proxy_headers,
                params=query_params,
                content=body_bytes if body_bytes else None,
            )

            if (
                provider == "anthropic"
                and endpoint_path == "/messages/count_tokens"
                and response.status_code == 404
                and isinstance(payload, dict)
            ):
                fallback_payload = dict(payload)
                fallback_payload.setdefault("max_tokens", 1)
                fallback_payload["stream"] = False
                fallback_body = json.dumps(fallback_payload).encode("utf-8")

                fallback_route = mapper.resolve_provider_route(
                    provider="anthropic",
                    endpoint_path="/messages",
                    body=fallback_payload,
                    headers=request.headers,
                )
                fallback_headers = _build_forward_headers(request, "anthropic", fallback_route)

                fallback_response = await client.request(
                    method="POST",
                    url=fallback_route["endpoint"],
                    headers=fallback_headers,
                    params=query_params,
                    content=fallback_body,
                )

                if fallback_response.status_code < 400:
                    try:
                        fallback_json = fallback_response.json()
                    except json.JSONDecodeError:
                        fallback_json = {}

                    usage = fallback_json.get("usage") if isinstance(fallback_json, dict) else None
                    if isinstance(usage, dict) and usage.get("input_tokens") is not None:
                        return JSONResponse(
                            status_code=200,
                            content={"input_tokens": usage["input_tokens"]},
                        )

            if (
                provider == "anthropic"
                and endpoint_path == "/messages"
                and response.status_code < 400
                and _requires_explicit_anthropic_tool_use(payload)
            ):
                try:
                    downstream_payload = response.json()
                except json.JSONDecodeError:
                    downstream_payload = None

                if not _anthropic_message_has_tool_use(downstream_payload):
                    return _anthropic_error(
                        422,
                        (
                            "Downstream model did not return tool_use for explicit tool_choice; "
                            "tool calling may be unsupported by the selected model/backend."
                        ),
                        error_type="invalid_request_error",
                    )

        content_type = response.headers.get("content-type", "application/json")
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection", "content-type"}
        }

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=content_type,
            headers=response_headers,
        )
    except httpx.ConnectError as e:
        logger.error(f"Anthropic proxy connection error: {e}")
        return _anthropic_error(502, f"Failed to connect to downstream API: {str(e)}", "api_error")
    except httpx.TimeoutException as e:
        logger.error(f"Anthropic proxy timeout error: {e}")
        return _anthropic_error(504, "Request to downstream API timed out", "api_error")
    except Exception as e:
        logger.error(f"Anthropic proxy error: {e}", exc_info=True)
        return _anthropic_error(500, "Internal server error", "api_error")


async def _proxy_provider_passthrough(request: Request, provider: str, endpoint_path: str):
    body_bytes = await request.body()
    payload = None

    if body_bytes:
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _anthropic_error(400, "Invalid JSON in request body")

    return await _forward_provider_request(request, provider, endpoint_path, body_bytes, payload)


async def _proxy_anthropic_messages_with_guards(request: Request):
    body_bytes = await request.body()
    payload = None

    if body_bytes:
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _anthropic_error(400, "Invalid JSON in request body")

    if not isinstance(payload, dict):
        return _anthropic_error(400, "Invalid request format")

    try:
        chat = Chat.from_payload(payload, provider="anthropic")
    except Exception as e:
        logger.error(f"Invalid Anthropic payload format: {e}")
        return _anthropic_error(400, "Invalid request format")

    param_keys_excluded = {"model", "messages", "system"}
    guard_params = {
        key: value for key, value in payload.items() if key not in param_keys_excluded
    }

    try:
        route_config = mapper.resolve_provider_route(
            provider="anthropic",
            endpoint_path="/messages",
            body=payload,
            headers=request.headers,
        )
    except ValueError as e:
        return _anthropic_error(500, str(e), error_type="api_error")

    inspection_headers = _build_forward_headers(request, "anthropic", route_config)

    llm_instance = llm.LLM(
        url=route_config["url"],
        headers=inspection_headers,
        query_params=dict(request.query_params),
        model=payload.get("model"),
        params=guard_params,
        chat=chat,
    )
    llm_instance.provider = "anthropic"
    llm_instance.raw_payload = payload

    try:
        guards = get_guards()
        _, audit_logs = apply_guards(chat, llm_instance, guards)
        log_audit(audit_logs)
    except GuardBlockedError as e:
        return _anthropic_error(403, str(e), error_type="invalid_request_error")
    except Exception as e:
        if _is_guard_blocked_error(e):
            return _anthropic_error(403, str(e), error_type="invalid_request_error")
        raise

    forwarded_payload = chat.serialize("anthropic")

    for key, value in llm_instance.params.items():
        if key not in param_keys_excluded:
            forwarded_payload[key] = value

    for key in list(forwarded_payload.keys()):
        if (
            key not in param_keys_excluded
            and key in payload
            and key not in llm_instance.params
        ):
            del forwarded_payload[key]

    forwarded_body = json.dumps(forwarded_payload).encode("utf-8")
    return await _forward_provider_request(
        request,
        provider="anthropic",
        endpoint_path="/messages",
        body_bytes=forwarded_body,
        payload=forwarded_payload,
    )


def _select_models_provider(request: Request):
    incoming_key = mapper.extract_api_key(request.headers)
    if not incoming_key:
        return "openai"

    anthropic_backends = mapper.get_provider_backends("anthropic")
    if any(backend.get("key") and backend["key"] == incoming_key for backend in anthropic_backends):
        return "anthropic"

    return "openai"


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "OpenGuard",
        "version": "0.1.0",
        "description": "guarding proxy for AI",
        "endpoints": {"health": "/health", "models": "/v1/models", "chat": "/v1/chat/completions"},
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request, authorized: bool = Depends(verify_auth)):
    """List all available models from downstream APIs"""
    try:
        provider = _select_models_provider(request)
        models = await mapper.list_downstream(provider=provider)
        return {"object": "list", "data": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.get("/v1/anthropic/models")
async def list_anthropic_models(request: Request):
    """Dedicated Anthropic-compatible models listing endpoint."""
    return await _proxy_provider_passthrough(request, provider="anthropic", endpoint_path="/models")


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Dedicated Anthropic messages endpoint with guard-aware passthrough semantics."""
    return await _proxy_anthropic_messages_with_guards(request)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Dedicated Anthropic count_tokens endpoint if supported by downstream."""
    return await _proxy_provider_passthrough(
        request,
        provider="anthropic",
        endpoint_path="/messages/count_tokens",
    )


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
                        data = json.loads(line[6:])  # Skip "data: "

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
                                    # Fallback: if no index (e.g. from some backends),
                                    # append to list or treat as new
                                    # But standard OpenAI stream has index.
                                    # If llm.py sends full objects without index,
                                    # we can just append, but let's try to handle both.
                                    if idx is None:
                                        tool_calls.append(tc)
                                    else:
                                        if idx not in pending_tool_calls:
                                            pending_tool_calls[idx] = tc
                                        else:
                                            current = pending_tool_calls[idx]
                                            # Merge arguments
                                            if "function" in tc:
                                                if (
                                                    "name" in tc["function"]
                                                    and tc["function"]["name"]
                                                ):
                                                    if "function" not in current:
                                                        current["function"] = {}
                                                    current["function"]["name"] = tc["function"][
                                                        "name"
                                                    ]

                                                if (
                                                    "arguments" in tc["function"]
                                                    and tc["function"]["arguments"]
                                                ):
                                                    if "function" not in current:
                                                        current["function"] = {"arguments": ""}
                                                    if "arguments" not in current["function"]:
                                                        current["function"]["arguments"] = ""
                                                    current["function"]["arguments"] += tc[
                                                        "function"
                                                    ]["arguments"]

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
                message = {"role": "assistant", "content": full_content if full_content else None}

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

        if e.response.status_code == 404:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Model not found or invalid",
                        "type": "invalid_request_error",
                        "code": 400,
                    }
                },
            )

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
    except Exception as e:
        if _is_guard_blocked_error(e):
            return JSONResponse(
                status_code=403,
                content={"error": {"message": str(e), "type": "guard_block", "code": 403}},
            )
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def main():
    """
    Entry point for the application script.
    Delegates to the CLI application.
    """
    from src.cli import app

    app()
