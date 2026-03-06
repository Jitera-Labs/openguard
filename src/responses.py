"""
Responses API translation layer.

Handles bidirectional translation between the OpenAI Responses API format
and the Chat Completions API format, plus native passthrough detection.
"""

import json
from copy import deepcopy
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


def apply_guarded_chat_completions_to_responses_request(
    original_request: dict,
    guarded_payload: dict,
) -> dict:
    """Project guarded Chat Completions fields back onto a Responses request."""
    updated_request = deepcopy(original_request)

    messages = guarded_payload.get("messages")
    if isinstance(messages, list):
        instructions, response_input = _chat_messages_to_responses_input(messages)
        if instructions is None:
            updated_request.pop("instructions", None)
        else:
            updated_request["instructions"] = instructions
        updated_request["input"] = response_input

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
        if src in guarded_payload:
            updated_request[dst] = deepcopy(guarded_payload[src])

    if "max_tokens" in guarded_payload:
        updated_request["max_output_tokens"] = guarded_payload["max_tokens"]

    if "response_format" in guarded_payload:
        text_config = deepcopy(updated_request.get("text"))
        if not isinstance(text_config, dict):
            text_config = {}
        text_config["format"] = deepcopy(guarded_payload["response_format"])
        updated_request["text"] = text_config

    if "reasoning_effort" in guarded_payload:
        reasoning_config = deepcopy(updated_request.get("reasoning"))
        if not isinstance(reasoning_config, dict):
            reasoning_config = {}
        reasoning_config["effort"] = guarded_payload["reasoning_effort"]
        updated_request["reasoning"] = reasoning_config

    if "tools" in guarded_payload:
        updated_request["tools"] = _project_function_tools(
            original_request.get("tools"),
            guarded_payload["tools"],
        )

    return updated_request


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
    response_started = False
    accumulated_text = ""
    accumulated_tool_calls: dict[int, dict] = {}
    final_usage = None

    async for raw_chunk in stream:
        chunk_text = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else raw_chunk

        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue

            if not line.startswith("data: "):
                try:
                    error_chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                error = error_chunk.get("error")
                if isinstance(error, dict):
                    yield _sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "message": error.get("message", "Upstream error"),
                                "type": error.get("type", "upstream_error"),
                                "code": error.get("code", 500),
                            },
                        },
                    )
                    return
                continue

            if line == "data: [DONE]":
                if not response_started:
                    yield _sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "message": "Upstream stream ended before starting a response",
                                "type": "invalid_upstream_stream",
                                "code": 502,
                            },
                        },
                    )
                    return

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
                response_started = True
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


def _chat_messages_to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict] | str]:
    instructions: str | None = None
    response_input: list[dict] = []

    start_index = 0
    if messages and messages[0].get("role") == "system":
        instructions = _chat_content_to_text(messages[0].get("content"))
        start_index = 1

    for message in messages[start_index:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")

        if role == "tool":
            response_input.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": _chat_content_to_text(message.get("content")),
                }
            )
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
                response_input.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
                )

            if message.get("content") in (None, "", []):
                continue

        response_input.append(
            {
                "type": "message",
                "role": role,
                "content": _chat_content_to_response_parts(message.get("content")),
            }
        )

    if len(response_input) == 1 and response_input[0].get("role") == "user":
        content = response_input[0].get("content")
        if isinstance(content, str):
            return instructions, content

    return instructions, response_input


def _chat_content_to_response_parts(content: object) -> str | list[dict]:
    if isinstance(content, str):
        return content

    if content is None:
        return []

    if not isinstance(content, list):
        text = _chat_content_to_text(content)
        return text if text else []

    response_parts: list[dict] = []
    for part in content:
        if isinstance(part, str):
            response_parts.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type == "text":
            response_parts.append({"type": "input_text", "text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                response_parts.append({"type": "input_image", "image_url": image_url})
            elif image_url is not None:
                response_parts.append({"type": "input_image", "image_url": image_url})
        else:
            response_parts.append(deepcopy(part))
    return response_parts


def _chat_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _project_function_tools(original_tools: object, guarded_tools: object) -> list[dict]:
    original_list = original_tools if isinstance(original_tools, list) else []
    guarded_list = guarded_tools if isinstance(guarded_tools, list) else []

    projected_tools: list[dict] = []
    guarded_index = 0

    for original_tool in original_list:
        if not isinstance(original_tool, dict):
            continue

        tool_type = original_tool.get("type", "function")
        if tool_type in _BUILTIN_TOOL_TYPES:
            projected_tools.append(deepcopy(original_tool))
            continue

        guarded_tool = guarded_list[guarded_index] if guarded_index < len(guarded_list) else None
        guarded_index += 1 if guarded_tool is not None else 0
        if not isinstance(guarded_tool, dict):
            continue
        projected_tools.append(_merge_function_tool(original_tool, guarded_tool))

    while guarded_index < len(guarded_list):
        guarded_tool = guarded_list[guarded_index]
        guarded_index += 1
        if isinstance(guarded_tool, dict):
            projected_tools.append(_merge_function_tool({}, guarded_tool))

    return projected_tools


def _merge_function_tool(original_tool: dict, guarded_tool: dict) -> dict:
    merged_tool = deepcopy(original_tool)
    merged_tool["type"] = "function"

    original_function: dict[str, object] | None = None
    if isinstance(original_tool.get("function"), dict):
        original_function = original_tool["function"]

    guarded_function: dict[str, object] = {}
    if isinstance(guarded_tool.get("function"), dict):
        guarded_function = guarded_tool["function"]

    if original_function is not None:
        merged_function = deepcopy(original_function)
        merged_function.update(
            {
                "name": guarded_function.get("name", ""),
                "description": guarded_function.get("description", ""),
                "parameters": deepcopy(guarded_function.get("parameters", {})),
            }
        )
        if "strict" in guarded_function:
            merged_function["strict"] = guarded_function["strict"]
        merged_tool["function"] = merged_function
    else:
        merged_tool.update(
            {
                "name": guarded_function.get("name", ""),
                "description": guarded_function.get("description", ""),
                "parameters": deepcopy(guarded_function.get("parameters", {})),
            }
        )
        if "strict" in guarded_function:
            merged_tool["strict"] = guarded_function["strict"]

    return merged_tool


def _sse_event(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
