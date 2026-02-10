import json
import uuid
from typing import AsyncGenerator

import httpx

from src import log
from src.llm_registry import llm_registry

logger = log.setup_logger(__name__)


class LLM:
    """
    Simplified LLM proxy class for forwarding requests to downstream APIs.
    No module system - just basic proxying with streaming support.
    """

    def __init__(self, url: str, headers: dict, payload: dict, stream: bool):
        """
        Initialize LLM proxy.

        Args:
          url: Backend URL (without /v1/chat/completions path)
          headers: Auth headers for backend
          payload: Request payload (guards already applied)
          stream: Whether to stream the response
        """
        self.url = url
        self.headers = headers
        self.payload = payload
        self.stream = stream
        self.id = str(uuid.uuid4())

    async def proxy_request(self) -> AsyncGenerator:
        """
        Proxy the request to downstream API.
        Handles both streaming and non-streaming responses.
        """
        endpoint = f"{self.url}/v1/chat/completions"

        logger.debug(f"Proxying request to {endpoint}")
        logger.debug(f"Stream: {self.stream}")
        logger.debug(f"Payload: {json.dumps(self.payload, indent=2)}")

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                if self.stream:
                    # Streaming request
                    async with client.stream(
                        "POST", endpoint, headers=self.headers, json=self.payload
                    ) as response:
                        try:
                            response.raise_for_status()

                            # Stream SSE chunks
                            buffer = b""
                            async for chunk_bytes in response.aiter_bytes():
                                buffer += chunk_bytes

                                # Process complete lines
                                while b"\n" in buffer:
                                    line, buffer = buffer.split(b"\n", 1)
                                    line_str = line.decode("utf-8").strip()

                                    # Skip empty lines and comments
                                    if not line_str or line_str.startswith(":"):
                                        continue

                                    # Check for [DONE] marker
                                    if line_str == "data: [DONE]":
                                        yield "data: [DONE]\n\n"
                                        continue

                                    # Forward SSE lines
                                    if line_str.startswith("data:"):
                                        yield f"{line_str}\n\n"

                        except httpx.HTTPStatusError as e:
                            # Forward downstream error
                            error_body = await e.response.aread()
                            logger.error(
                                f"Downstream error {e.response.status_code}: "
                                f"{error_body.decode('utf-8')}"
                            )

                            # Return error in OpenAI format
                            error_response = {
                                "error": {
                                    "message": f"Downstream API error: {e.response.status_code}",
                                    "type": "downstream_error",
                                    "code": e.response.status_code,
                                }
                            }
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield "data: [DONE]\n\n"

                else:
                    # Non-streaming request
                    response = await client.post(endpoint, headers=self.headers, json=self.payload)

                    try:
                        response.raise_for_status()
                        result = response.json()
                        yield json.dumps(result)

                    except httpx.HTTPStatusError:
                        raise

                    except Exception:
                        # Let other exceptions bubble up to outer handler
                        raise

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            error_response = {
                "error": {
                    "message": f"Failed to connect to downstream API: {str(e)}",
                    "type": "connection_error",
                    "code": 502,
                }
            }
            if self.stream:
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield json.dumps(error_response)

        except httpx.TimeoutException as e:
            logger.error(f"Timeout error: {e}")
            error_response = {
                "error": {
                    "message": "Request to downstream API timed out",
                    "type": "timeout_error",
                    "code": 504,
                }
            }
            if self.stream:
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield json.dumps(error_response)

        except httpx.HTTPStatusError:
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            error_response = {
                "error": {
                    "message": f"Internal proxy error: {str(e)}",
                    "type": "internal_error",
                    "code": 500,
                }
            }
            if self.stream:
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield json.dumps(error_response)

    async def serve(self):
        """
        Main entry point for proxying.
        Registers the LLM instance and yields results.
        """
        logger.debug(f"Serving LLM proxy {self.id}")
        llm_registry.register(self)

        try:
            async for chunk in self.proxy_request():
                yield chunk
        finally:
            llm_registry.unregister(self)
            logger.debug(f"Unregistered LLM proxy {self.id}")
