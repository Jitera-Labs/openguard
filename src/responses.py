"""
Responses API translation layer.

Handles bidirectional translation between the OpenAI Responses API format
and the Chat Completions API format, plus native passthrough detection.
"""

import json
from typing import AsyncIterator
from urllib.parse import urlparse

# Domains that natively support the Responses API — passthrough without translation.
RESPONSES_API_NATIVE_DOMAINS: frozenset[str] = frozenset(
    {
        "api.openai.com",
    }
)


def upstream_supports_responses_api(base_url: str) -> bool:
    """Return True if the upstream URL natively supports the Responses API."""
    host = urlparse(base_url).hostname or ""
    return host in RESPONSES_API_NATIVE_DOMAINS or host.endswith(".openai.azure.com")


# Responses API built-in tool types that have no Chat Completions equivalent.
_BUILTIN_TOOL_TYPES = frozenset(
    {
        "web_search_preview",
        "web_search",
        "file_search",
        "computer_use_preview",
        "computer",
        "shell",
        "code_interpreter",
    }
)


def responses_to_chat_completions(payload: dict) -> dict:
    """Translate a Responses API request payload to Chat Completions format."""
    messages: list[dict] = []

    # `instructions` → system message prepended
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    # `input` → messages
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")

            if item_type == "function_call":
                # assistant tool call
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                # tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )
            else:
                # type == "message" or no type — passthrough role/content
                role = item.get("role", "user")
                content = item.get("content", "")
                # content may be a list of content parts; map to Chat Completions format
                if isinstance(content, list):
                    cc_content: list[dict] = []
                    for part in content:
                        if isinstance(part, dict):
                            part_type = part.get("type")
                            if part_type in ("input_text", "text"):
                                cc_content.append({"type": "text", "text": part.get("text", "")})
                            elif part_type == "input_image":
                                img_part: dict = {"type": "image_url"}
                                if "image_url" in part:
                                    img_part["image_url"] = part["image_url"]
                                elif "file_id" in part:
                                    # Pass file_id through, though standard CC uses image_url
                                    img_part["image_url"] = {"url": part["file_id"]}
                                cc_content.append(img_part)
                            else:
                                # Pass through other types
                                cc_content.append(part)
                        elif isinstance(part, str):
                            cc_content.append({"type": "text", "text": part})
                    content = cc_content  # type: ignore
                messages.append({"role": role, "content": content})

    result: dict = {"messages": messages}

    # Scalar fields
    for src, dst in [
        ("model", "model"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stream", "stream"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
        ("seed", "seed"),
        ("user", "user"),
        ("tool_choice", "tool_choice"),
        ("parallel_tool_calls", "parallel_tool_calls"),
        ("metadata", "metadata"),
        ("store", "store"),
        ("stream_options", "stream_options"),
        ("service_tier", "service_tier"),
        ("top_logprobs", "top_logprobs"),
    ]:
        if src in payload:
            result[dst] = payload[src]

    # text -> response_format
    if "text" in payload and isinstance(payload["text"], dict):
        if "format" in payload["text"]:
            result["response_format"] = payload["text"]["format"]

    # reasoning -> reasoning_effort
    if "reasoning" in payload and isinstance(payload["reasoning"], dict):
        if "effort" in payload["reasoning"]:
            result["reasoning_effort"] = payload["reasoning"]["effort"]

    # max_output_tokens → max_tokens
    if "max_output_tokens" in payload:
        result["max_tokens"] = payload["max_output_tokens"]

    # Tools: translate function tools, skip built-in types
    raw_tools = payload.get("tools")
    if isinstance(raw_tools, list):
        cc_tools = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "function")
            if tool_type in _BUILTIN_TOOL_TYPES:
                continue
            # Wrap in Chat Completions externally-tagged format
            fn_data = tool.get("function", tool)
            cc_tool: dict = {
                "type": "function",
                "function": {
                    "name": fn_data.get("name", ""),
                    "description": fn_data.get("description", ""),
                    "parameters": fn_data.get("parameters", {}),
                },
            }
            if "strict" in fn_data:
                cc_tool["function"]["strict"] = fn_data["strict"]
            elif "strict" in tool:
                cc_tool["function"]["strict"] = tool["strict"]
            cc_tools.append(cc_tool)
        if cc_tools:
            result["tools"] = cc_tools

    return result


def chat_completions_to_responses(cc_response: dict, original_request: dict) -> dict:
    """Translate a Chat Completions response to Responses API format."""
    resp_id = f"resp_{cc_response.get('id', '')}"
    created_at = cc_response.get("created", 0)
    model = cc_response.get("model", original_request.get("model", ""))

    choices = cc_response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    status = "completed"
    incomplete_details = None
    if finish_reason == "content_filter":
        status = "incomplete"
        incomplete_details = {"reason": "content_filter"}
    elif finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}

    output: list[dict] = []

    # Tool calls
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{tc.get('id', '')}",
                "call_id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            }
        )

    # Text content
    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{resp_id}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )

    # Usage
    cc_usage = cc_response.get("usage", {})
    usage = {
        "input_tokens": cc_usage.get("prompt_tokens", 0),
        "output_tokens": cc_usage.get("completion_tokens", 0),
        "total_tokens": cc_usage.get("total_tokens", 0),
    }

    if "prompt_tokens_details" in cc_usage:
        usage["input_tokens_details"] = cc_usage["prompt_tokens_details"]
    if "completion_tokens_details" in cc_usage:
        usage["output_tokens_details"] = cc_usage["completion_tokens_details"]

    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": status,
        "output": output,
        "usage": usage,
    }
    if incomplete_details:
        resp["incomplete_details"] = incomplete_details
    return resp


async def translate_streaming_response(
    stream: AsyncIterator[bytes],
    original_request: dict,
) -> AsyncIterator[str]:
    """
    Translate a Chat Completions SSE stream to Responses API SSE format.

    Yields SSE-formatted strings.
    """
    resp_id: str = ""
    model: str = original_request.get("model", "")
    created_at: int = 0
    output_item_index = 0
    item_started = False
    accumulated_text = ""
    accumulated_tool_calls: dict[int, dict] = {}
    final_usage = None

    async for raw_chunk in stream:
        chunk_text = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk

        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue

            if line == "data: [DONE]":
                # Emit done events
                output_items = []
                if item_started:
                    if accumulated_text:
                        yield _sse_event(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "output_index": output_item_index,
                                "content_index": 0,
                                "text": accumulated_text,
                            },
                        )
                        output_items.append(
                            {
                                "type": "message",
                                "id": f"msg_{resp_id}",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": accumulated_text}],
                            }
                        )
                        yield _sse_event(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_item_index,
                                "item": output_items[-1],
                            },
                        )
                        output_item_index += 1

                    for tc_idx, tc in sorted(accumulated_tool_calls.items()):
                        tc_item = {
                            "type": "function_call",
                            "id": f"fc_{tc.get('id', '')}",
                            "call_id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", ""),
                        }
                        output_items.append(tc_item)
                        yield _sse_event(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_item_index,
                                "item": tc_item,
                            },
                        )
                        output_item_index += 1

                completed_response = {
                    "id": resp_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": model,
                    "status": "completed",
                    "output": output_items,
                }
                if final_usage:
                    usage = {
                        "input_tokens": final_usage.get("prompt_tokens", 0),
                        "output_tokens": final_usage.get("completion_tokens", 0),
                        "total_tokens": final_usage.get("total_tokens", 0),
                    }
                    if "prompt_tokens_details" in final_usage:
                        usage["input_tokens_details"] = final_usage["prompt_tokens_details"]
                    if "completion_tokens_details" in final_usage:
                        usage["output_tokens_details"] = final_usage["completion_tokens_details"]
                    completed_response["usage"] = usage

                yield _sse_event(
                    "response.completed",
                    {"type": "response.completed", "response": completed_response},
                )
                yield "data: [DONE]\n\n"
                return

            if not line.startswith("data: "):
                continue

            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            # Extract metadata from first chunk
            if not resp_id:
                resp_id = f"resp_{data.get('id', '')}"
                model = data.get("model", model)
                created_at = data.get("created", 0)
                # Emit response.created
                yield _sse_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "created_at": created_at,
                            "model": model,
                            "status": "in_progress",
                            "output": [],
                        },
                    },
                )

            choices = data.get("choices", [])
            if not choices:
                if "usage" in data:
                    final_usage = data["usage"]
                continue
            delta = choices[0].get("delta", {})
            content_delta = delta.get("content")
            tool_calls_delta = delta.get("tool_calls")

            if content_delta is not None:
                if not item_started:
                    item_started = True
                    yield _sse_event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": output_item_index,
                            "item": {
                                "type": "message",
                                "id": f"msg_{resp_id}",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        },
                    )
                    yield _sse_event(
                        "response.content_part.added",
                        {
                            "type": "response.content_part.added",
                            "item_id": f"msg_{resp_id}",
                            "output_index": output_item_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": ""},
                        },
                    )

                accumulated_text += content_delta
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": f"msg_{resp_id}",
                        "output_index": output_item_index,
                        "content_index": 0,
                        "delta": content_delta,
                    },
                )

            if tool_calls_delta:
                if not item_started:
                    item_started = True
                for tc in tool_calls_delta:
                    idx = tc.get("index", 0)
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", ""),
                        }
                    else:
                        if "id" in tc and tc["id"]:
                            accumulated_tool_calls[idx]["id"] = tc["id"]
                        if "function" in tc:
                            fn = tc["function"]
                            if "name" in fn and fn["name"]:
                                accumulated_tool_calls[idx]["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"]:
                                accumulated_tool_calls[idx]["arguments"] += fn["arguments"]


def _sse_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
