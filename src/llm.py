import asyncio
import json
import time
import traceback
import uuid
from typing import AsyncGenerator, Optional

import httpx

import src.chat as ch
import src.format as format
import src.log as log
from src.config import BOOST_PUBLIC_URL, EXTRA_LLM_PARAMS, INTERMEDIATE_OUTPUT
from src.events import AsyncEventEmitter
from src.llm_registry import llm_registry

# import src.mods as mods
# import src.tools as tools
# import src.tools.registry as tools_registry

logger = log.setup_logger(__name__)

# Legacy prefix from previous "Boost" product; used to separate internal params from LLM params.
BOOST_PARAM_PREFIX = "@boost_"


class StreamRedactor:
    def __init__(self, patterns, blocks=None, window_size=40):
        self.patterns = patterns
        self.blocks = blocks or []
        self.window_size = window_size
        self.buffer = ""

    def push(self, text: str) -> str:
        from src.guards import GuardBlockedError

        self.buffer += text
        for pattern, kw in self.blocks:
            if pattern.search(self.buffer):
                raise GuardBlockedError(
                    f"Request blocked: found keyword '{kw}' in streaming response"
                )
        for pattern, repl in self.patterns:
            self.buffer = pattern.sub(repl, self.buffer)
        if len(self.buffer) > self.window_size:
            emit_len = len(self.buffer) - self.window_size
            emit_str = self.buffer[:emit_len]
            self.buffer = self.buffer[emit_len:]
            return emit_str
        return ""

    def flush(self) -> str:
        from src.guards import GuardBlockedError

        for pattern, kw in self.blocks:
            if pattern.search(self.buffer):
                raise GuardBlockedError(
                    f"Request blocked: found keyword '{kw}' in streaming response"
                )
        for pattern, repl in self.patterns:
            self.buffer = pattern.sub(repl, self.buffer)
        emit_str = self.buffer
        self.buffer = ""
        return emit_str


class LLM(AsyncEventEmitter):
    url: str
    headers: dict
    query_params: dict

    model: str
    params: dict
    boost_params: dict
    module: str
    provider: str
    raw_payload: Optional[dict]

    queue: asyncio.Queue
    is_streaming: bool
    is_final_stream: bool

    cpl_id: int

    def __init__(self, **kwargs):
        super().__init__()

        self.id = str(uuid.uuid4())
        self.url = kwargs.get("url") or ""
        self.headers = kwargs.get("headers", {})
        self.query_params = kwargs.get("query_params", {})

        self.model = kwargs.get("model") or ""
        self.split_params(kwargs.get("params", {}))

        self.chat = self.resolve_chat(**kwargs)
        self.messages = self.chat.history()

        self.module = kwargs.get("module") or ""
        self.provider = kwargs.get("provider", "openai")
        self.raw_payload = kwargs.get("raw_payload")

        self.queue = asyncio.Queue()
        self.queues = []
        self.is_streaming = False
        self.is_final_stream = False
        self.cpl_id = 0

        self.stream_patterns = []
        self.stream_blocks = []

    @property
    def chat_completion_endpoint(self):
        return f"{self.url}/chat/completions"

    def split_params(self, params: dict):
        self.params = {
            k: v
            for k, v in {**EXTRA_LLM_PARAMS.value, **params}.items()
            if not k.startswith(BOOST_PARAM_PREFIX)
        }
        self.boost_params = {
            k[len(BOOST_PARAM_PREFIX) :]: v
            for k, v in params.items()
            if k.startswith(BOOST_PARAM_PREFIX)
        }

    def generate_system_fingerprint(self):
        return "fp_boost"

    def generate_chunk_id(self):
        self.cpl_id += 1
        return f"chatcmpl-{self.cpl_id}"

    def get_response_content(self, params: dict, response: dict):
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices:
            logger.warning(f"Malformed response: missing choices: {str(response)[:200]}")
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if content is None:
            content = ""

        if "response_format" in params and "type" in params["response_format"]:
            if (
                params["response_format"]["type"] == "json_schema"
                or params["response_format"]["type"] == "json"
            ):
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return content

        return content

    async def inspect_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ):
        provider = getattr(self, "provider", "openai")
        if provider == "anthropic":
            return await self._inspect_completion_anthropic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                response_format=response_format,
            )

        return await self._inspect_completion_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            response_format=response_format,
        )

    # Params stripped from inspection calls to prevent user manipulation
    # of the inspector LLM (e.g. high temperature to randomize output,
    # logit_bias to favor "allow" tokens, stop sequences to truncate decisions).
    _INSPECTION_STRIPPED_PARAMS: frozenset[str] = frozenset(
        {
            "stream",
            "stream_options",
            "messages",
            "temperature",
            "top_p",
            "top_k",
            "logit_bias",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stop",
            "tools",
            "tool_choice",
            "n",
        }
    )

    async def _inspect_completion_openai(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ):
        resolved_model = model or self.model
        if not resolved_model:
            raise RuntimeError("missing model for inspection completion")

        params = {
            k: v
            for k, v in (self.params or {}).items()
            if k not in self._INSPECTION_STRIPPED_PARAMS
        }

        body = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **params,
            "stream": False,
        }

        if response_format is not None:
            body["response_format"] = response_format

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.chat_completion_endpoint,
                headers=self.headers,
                params=self.query_params,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()

        return self._extract_inspection_text(payload)

    async def _inspect_completion_anthropic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ):
        resolved_model = model or self.model
        if not resolved_model:
            raise RuntimeError("missing model for inspection completion")

        body = {
            "model": resolved_model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 256,
        }

        if response_format is not None:
            body["response_format"] = response_format

        endpoint = self._provider_endpoint("/messages")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                headers=self.headers,
                params=self.query_params,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()

        return self._extract_inspection_text(payload)

    def _provider_endpoint(self, path: str):
        base = (self.url or "").rstrip("/")
        normalized = path if path.startswith("/") else f"/{path}"

        if getattr(self, "provider", "openai") == "anthropic":
            if normalized.startswith("/v1/"):
                return f"{base}{normalized}"
            if base.endswith("/v1"):
                return f"{base}{normalized}"
            return f"{base}/v1{normalized}"

        return f"{base}{normalized}"

    def _extract_inspection_text(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("invalid inspector response payload")

        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            content = message.get("content")
            return self._content_to_text(content)

        content_blocks = payload.get("content")
        if isinstance(content_blocks, list):
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
            if text_parts:
                return "\n".join(text_parts).strip()

        raise ValueError("invalid inspector response: missing text content")

    def _content_to_text(self, content):
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "\n".join(text_parts).strip()

        raise ValueError("invalid inspector response: missing text content")

    def get_chunk_content(self, chunk):
        try:
            choices = chunk.get("choices", [])
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {})
            return delta.get("content", "")
        except (KeyError, IndexError):
            logger.error(f"Unexpected chunk format: {chunk}")
            return ""

    def get_chunk_tool_calls(self, chunk):
        try:
            choices = chunk.get("choices", [])
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {})
            return delta.get("tool_calls", [])
        except (KeyError, IndexError):
            logger.error(f"Unexpected chunk format: {chunk}")
            return []

    def parse_chunk(self, chunk):
        if isinstance(chunk, dict):
            return chunk

        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")

        chunk_str = chunk.split("\n")[0]
        if chunk_str.startswith("data: "):
            chunk_str = chunk_str[6:]

        if chunk_str == "[DONE]":
            return self.chunk_from_message("")

        return json.loads(chunk_str)

    def output_from_chunk(self, chunk):
        return {
            "id": chunk["id"],
            "object": "chat.completion",
            "created": chunk["created"],
            "model": self.model,
            "system_fingerprint": self.generate_system_fingerprint(),
            "choices": [
                {
                    "index": choice["index"],
                    "message": {
                        "role": choice["delta"].get("role", "assistant"),
                        "content": choice["delta"].get("content", ""),
                    },
                    "finish_reason": None,
                }
                for choice in chunk["choices"]
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def chunk_from_delta(self, delta: dict):
        now = int(time.time())

        return {
            "id": self.generate_chunk_id(),
            "object": "chat.completion.chunk",
            "created": now,
            "model": self.model,
            "system_fingerprint": self.generate_system_fingerprint(),
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    def chunk_from_message(self, message: str):
        return self.chunk_from_delta({"role": "assistant", "content": message})

    def chunk_from_tool_call(self, tool_call: dict):
        if "index" not in tool_call:
            tool_call["index"] = 0

        return self.chunk_from_delta({"role": "assistant", "tool_calls": [tool_call]})

    def chunk_to_string(self, chunk):
        if isinstance(chunk, dict):
            chunk = f"data: {json.dumps(chunk)}\n\n"

        return chunk

    def is_tool_call(self, chunk):
        choices = chunk.get("choices", [])
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        has_tool_calls = delta.get("tool_calls", [])
        return len(has_tool_calls) > 0

    def event_to_string(self, event, data):
        payload = {"object": "boost.listener.event", "event": event, "data": data}

        return f"data: {json.dumps(payload)}\n\n"

    async def serve(self):
        logger.debug("Serving boosted LLM...")
        llm_registry.register(self)

        async def apply_mod():
            try:
                if self.module is None:
                    logger.debug("No module specified")
                    await self.stream_final_completion()
                    return

                # mod = mods.registry.get(self.module)
                # if mod is None:
                #   logger.error(f"Module '{self.module}' not found.")
                #   return
                logger.warning("Modules are currently disabled.")
                await self.stream_final_completion()
            except httpx.HTTPStatusError as e:
                logger.error(f"Upstream HTTP error: {e}")
                try:
                    content = e.response.content.decode("utf-8")
                    try:
                        error_json = json.loads(content)
                        error_data = error_json.get("error", {"message": content})
                    except Exception:
                        error_data = {"message": content}
                except Exception:
                    error_data = {"message": str(e)}

                await self.emit_data(
                    json.dumps(
                        {
                            "error": {
                                "message": error_data.get("message", str(e)),
                                "type": error_data.get("type", "upstream_error"),
                                "code": e.response.status_code,
                            }
                        }
                    )
                )
            except httpx.ConnectError as e:
                await self.emit_data(
                    json.dumps(
                        {
                            "error": {
                                "message": f"Failed to connect to downstream API: {str(e)}",
                                "type": "connection_error",
                                "code": 502,
                            }
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error in LLM service: {e}")
                # Emit generic error if needed
            finally:
                await self.emit_done()

        task = asyncio.create_task(apply_mod())
        task.add_done_callback(
            lambda t: (
                logger.error("Modifier task failed: %s", t.exception()) if t.exception() else None
            )
        )
        return self.response_stream()

    async def generator(self):
        self.is_streaming = True

        while self.is_streaming or not self.queue.empty():
            chunk = await self.queue.get()

            if chunk is None:
                break

            yield chunk

    async def response_stream(self):
        async for chunk in self.generator():
            # Final stream is always passed back as
            # that's the useful payload of a given iteration
            if INTERMEDIATE_OUTPUT.value or self.is_final_stream:
                yield chunk

    async def listen(self):
        queue = asyncio.Queue()
        self.queues.append(queue)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def emit_status(self, status):
        await self.emit_message(format.format_status(status))

    async def emit_artifact(self, artifact, wait=True):
        artifact = artifact.replace("<<boost_public_url>>", BOOST_PUBLIC_URL.value).replace(
            "<<listener_id>>", self.id
        )
        await self.emit_message(format.format_artifact(artifact))
        if wait:
            await asyncio.sleep(1.0)

    async def emit_message(self, message):
        await self.emit_chunk(self.chunk_from_message(message))

    async def emit_chunk(self, chunk):
        if (
            "choices" not in chunk
            or not chunk["choices"]
            or "delta" not in chunk["choices"][0]
            or "content" not in chunk["choices"][0]["delta"]
        ):
            if "choices" not in chunk or not chunk["choices"]:
                chunk["choices"] = [{}]
            if "delta" not in chunk["choices"][0]:
                chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["delta"]["content"] = ""

        if "choices" not in chunk or not chunk["choices"] or "index" not in chunk["choices"][0]:
            if "choices" not in chunk or not chunk["choices"]:
                chunk["choices"] = [{}]
            chunk["choices"][0]["index"] = 0

        await self.emit_data(self.chunk_to_string(chunk))

    async def emit_data(self, data):
        await self.queue.put(data)
        await self.emit_to_listeners(data)

    async def emit_to_listeners(self, data):
        for queue in self.queues:
            await queue.put(data)

    async def emit_listener_event(self, event, data):
        await self.emit_to_listeners(self.event_to_string(event, data))

    async def emit_done(self):
        await self.emit_data("data: [DONE]")
        await self.emit_data(None)
        await self.remove_all_listeners()
        self.is_streaming = False
        llm_registry.unregister(self)

    async def stream_final_completion(self, **kwargs):
        self.is_final_stream = True
        return await self.stream_chat_completion(**kwargs)

    async def stream_chat_completion(self, **kwargs):
        request = await self.resolve_request(**kwargs)

        chat = request.get("chat", self.chat)
        params = request.get("params", self.params)
        model = request.get("model", self.model)
        url = request.get("url", self.url)
        headers = request.get("headers", self.headers)
        query_params = request.get("query_params", self.query_params)
        should_emit = kwargs.get("emit", True)

        logger.debug(f"Params: {params}")
        logger.debug(f"Chat: {str(chat):.256}")

        result = ""
        pending_tool_calls = {}  # Track tool calls being built, keyed by id
        first_tool_call_id = None

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        ) as client:
            current_stream_content = ""

            # Assistant must remember what it said so far
            if current_stream_content and current_stream_content != "":
                if chat.tail.role == "assistant":
                    chat.tail.content += current_stream_content
                else:
                    chat.assistant(current_stream_content)

            body = {
                "model": model,
                "messages": chat.history(),
                **params,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            }

            logger.debug(body)

            # Flag to determine if we need to execute tool calls
            end_of_stream = False
            current_stream_content = ""
            # Track tool call IDs in order of appearance
            tool_call_order = []

            async with client.stream(
                "POST",
                f"{url}/chat/completions",
                headers=headers,
                params=query_params,
                json=body,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    logger.error(
                        f"Chat completion error {response.status_code}: {body.decode('utf-8')}"
                    )
                    # Ensure we raise an error that will be caught by the serving loop
                    response.raise_for_status()

                buffer = b""

                async for chunk in response.aiter_bytes():
                    buffer += chunk

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.decode("utf-8").strip()

                        if not line or line.startswith(":"):
                            continue

                        if line == "data: [DONE]":
                            end_of_stream = True
                            if hasattr(self, "_stream_redactor"):
                                flushed = self._stream_redactor.flush()
                                if flushed:
                                    current_stream_content += flushed
                                    result += flushed
                                    if should_emit:
                                        await self.emit_chunk(
                                            {
                                                "choices": [
                                                    {"delta": {"content": flushed}, "index": 0}
                                                ]
                                            }
                                        )
                            continue

                        if not line.startswith("data:"):
                            continue

                        try:
                            parsed = self.parse_chunk(line)

                            # Safely check finish_reason
                            choices = parsed.get("choices", [])
                            finish_reason = None
                            if choices:
                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason == "tool_calls":
                                    end_of_stream = True

                            # Extract content for regular text responses
                            content = self.get_chunk_content(parsed)

                            if content:
                                if not hasattr(self, "_stream_redactor") and (
                                    self.stream_patterns or self.stream_blocks
                                ):
                                    self._stream_redactor = StreamRedactor(
                                        self.stream_patterns, self.stream_blocks
                                    )

                                if hasattr(self, "_stream_redactor"):
                                    emitted = self._stream_redactor.push(content)
                                    content = emitted

                                current_stream_content += content
                                result += content

                                # Override content in parsed chunk
                                if (
                                    "choices" in parsed
                                    and parsed["choices"]
                                    and "delta" in parsed["choices"][0]
                                ):
                                    if "content" in parsed["choices"][0]["delta"]:
                                        parsed["choices"][0]["delta"]["content"] = content

                            if finish_reason and hasattr(self, "_stream_redactor"):
                                flushed = self._stream_redactor.flush()
                                if flushed:
                                    current_stream_content += flushed
                                    result += flushed
                                    if (
                                        "choices" in parsed
                                        and parsed["choices"]
                                        and "delta" in parsed["choices"][0]
                                    ):
                                        # Append to whatever is there
                                        existing = parsed["choices"][0]["delta"].get("content", "")
                                        parsed["choices"][0]["delta"]["content"] = (
                                            existing + flushed
                                        )

                            # Process tool call chunks
                            if self.is_tool_call(parsed):
                                # Extract tool call data safely
                                choices = parsed.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                tool_calls_data = delta.get("tool_calls", [])

                                if not tool_calls_data:
                                    continue

                                tool_call = tool_calls_data[0]
                                tool_id = tool_call.get("id")
                                index = tool_call.get("index", 0)

                                # Store the first tool call ID we see
                                if tool_id and not first_tool_call_id:
                                    first_tool_call_id = tool_id

                                # Use tool_id as primary key if available, fall back to index
                                # This handles Ollama bug where multiple calls
                                # have same index but different ids
                                if tool_id:
                                    key = tool_id
                                else:
                                    # For streaming chunks without id,
                                    # find existing call by index
                                    # or use index as key for truly streamed arguments
                                    key = f"idx_{index}"
                                    for (
                                        existing_key,
                                        existing_call,
                                    ) in pending_tool_calls.items():
                                        if existing_call.get("_index") == index:
                                            key = existing_key
                                            break

                                # Initialize tool call if new
                                if key not in pending_tool_calls:
                                    pending_tool_calls[key] = {
                                        "id": tool_id or first_tool_call_id,
                                        "function": {
                                            "name": tool_call.get("function", {}).get("name"),
                                            "arguments": "",
                                        },
                                        "type": tool_call.get("type") or "function",
                                        "_index": index,
                                    }
                                    tool_call_order.append(key)

                                # Update arguments
                                function_args = tool_call.get("function", {}).get("arguments")
                                if key in pending_tool_calls and function_args is not None:
                                    pending_tool_calls[key]["function"]["arguments"] += (
                                        function_args
                                    )

                                logger.debug(f"Tool call chunk: {parsed}")
                            else:
                                if should_emit:
                                    await self.emit_chunk(parsed)

                        except json.JSONDecodeError:
                            logger.error(f'Failed to parse chunk: "{line}"')
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            for line in traceback.format_tb(e.__traceback__):
                                logger.error(line)

            # After stream ends, check if we need to execute tool calls
            if pending_tool_calls and (end_of_stream or not current_stream_content):
                for key in tool_call_order:
                    tool_call = pending_tool_calls.get(key)
                    if not tool_call:
                        continue
                    # Remove internal tracking field before use
                    tool_call.pop("_index", None)

                    logger.info(f"Passing back to API client: {tool_call}")
                    await self.emit_chunk(self.chunk_from_tool_call(tool_call))

                return result

        return result

    async def chat_completion(self, **kwargs):
        chat = self.resolve_chat(**kwargs)
        params = await self.resolve_request_params(**kwargs)
        should_resolve = kwargs.get("resolve", False)

        logger.debug(f"Chat Completion for '{self.chat_completion_endpoint}'")
        logger.debug(f"Params: {params}")
        logger.debug(f"Chat: {str(chat):.256}...")
        if chat is None:
            chat = self.chat

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        ) as client:
            body = {"model": self.model, "messages": chat.history(), **params, "stream": False}
            response = await client.post(
                self.chat_completion_endpoint, headers=self.headers, json=body
            )
            result = response.json()
            if should_resolve:
                return self.get_response_content(params, result)
            return result

    async def consume_stream(self, stream: AsyncGenerator[bytes, None]):
        output_obj = None
        content = ""
        tool_calls = []

        async for chunk_bytes in stream:
            chunk = self.parse_chunk(chunk_bytes)
            if output_obj is None:
                output_obj = self.output_from_chunk(chunk)
            chunk_content = self.get_chunk_content(chunk)
            chunk_tools = self.get_chunk_tool_calls(chunk)

            content += chunk_content
            tool_calls.extend(chunk_tools)

        if output_obj:
            output_obj["choices"][0]["message"]["content"] = content

            if len(tool_calls) > 0:
                output_obj["choices"][0]["message"]["tool_calls"] = tool_calls
                output_obj["choices"][0]["finish_reason"] = "tool_calls"

        return output_obj

    async def resolve_request_params(self, **kwargs):
        params = {
            "model": kwargs.get("model", self.model),
            **self.params,
            **kwargs.get("params", {}),
        }

        if kwargs.get("schema"):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "StructuredResponseSchema",
                    "schema": kwargs["schema"].model_json_schema(),
                },
            }

        return params

    def resolve_chat(
        self,
        messages: Optional[list] = None,
        chat: Optional["ch.Chat"] = None,
        prompt: Optional[str] = None,
        **prompt_kwargs,
    ):
        if chat is not None:
            return chat

        if messages is not None:
            return ch.Chat.from_conversation(messages)

        if prompt is not None:
            message = prompt.format(**prompt_kwargs)
            return ch.Chat.from_conversation([{"role": "user", "content": message}])

        if hasattr(self, "chat"):
            return self.chat

        return ch.Chat()

    async def resolve_model(self, model: Optional[str] = None, **rest) -> str:
        return model or self.model

    async def resolve_headers(self, **kwargs):
        return self.headers

    async def resolve_query_params(self, **kwargs):
        return self.query_params

    async def resolve_url(self, **kwargs):
        return self.url

    async def resolve_request(self, **kwargs):
        logger.debug("resolving")

        tasks = {
            "url": self.resolve_url(**kwargs),
            "headers": self.resolve_headers(**kwargs),
            "params": self.resolve_request_params(**kwargs),
            "model": self.resolve_model(**kwargs),
            "query_params": self.resolve_query_params(**kwargs),
        }

        values = await asyncio.gather(*tasks.values())
        results = {k: v for k, v in zip(tasks.keys(), values)}
        results["chat"] = self.resolve_chat(**kwargs)

        return results

    async def start_thinking(self):
        await self.emit_message("\n<think>\n")

    async def stop_thinking(self):
        await self.emit_message("\n</think>\n")
