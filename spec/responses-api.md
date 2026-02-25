# OpenAI Responses API — Implementation Validation

This document compares the official [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) specification against OpenGuard's implementation in `src/responses.py` and `src/main.py`.

OpenAI's Responses API (`/v1/responses`) differs significantly from the Chat Completions API (`/v1/chat/completions`). It introduces request properties like `input`, `instructions`, `text` (structured outputs), and `reasoning`, and returns a response object with fields like `status`, `output`, and `incomplete_details`.

OpenGuard implements this via a translation layer — `responses_to_chat_completions` and `chat_completions_to_responses` — that maps Responses API payloads to standard Chat Completions payloads for routing to non-native LLM providers. When the destination is `api.openai.com` or Azure, payloads are passed through without translation.

---

## Issues Found

### 1. Dropped Request Features (Silent Data Loss)

`responses_to_chat_completions` only maps a small set of scalar fields (`model`, `temperature`, `top_p`, `stream`, `frequency_penalty`, `presence_penalty`, `seed`, `user`). All other Responses API request properties are silently dropped.

Affected fields:

| Field | Purpose |
|---|---|
| `text` | Structured Outputs — triggers JSON schema generation via `text={"format": {"type": "json_schema", ...}}` |
| `reasoning` | Reasoning effort for o1/o3 models — e.g., `reasoning={"effort": "high"}` |
| `tool_choice` | Controls which tool the model calls |
| `parallel_tool_calls` | Whether the model may call multiple tools simultaneously |
| `metadata` | Arbitrary key-value metadata attached to the response |
| `store` | Whether to persist the response for later retrieval |
| `stream_options` | Options passed with streaming requests (e.g., `include_usage`) |
| `service_tier` | Specifies compute tier for the request |
| `top_logprobs` | Number of top log probabilities to return |

Any client setting these fields against a non-native provider backend will receive a silently degraded response with no error.

---

### 2. Broken Multimodal / Vision Support

When the `input` field is a list of content parts, `responses_to_chat_completions` flattens all parts into a single string by joining only items where `type == "input_text"` or `"text"`. Any other content part type is silently discarded.

```python
# src/responses.py — responses_to_chat_completions
if isinstance(content, list):
    text_parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_text":
            text_parts.append(part.get("text", ""))
        elif isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(part.get("text", ""))
        elif isinstance(part, str):
            text_parts.append(part)
    content = "\n".join(text_parts)
```

Any image passed as `{"type": "input_image", "image_url": {...}}` or `{"type": "input_image", "file_id": "..."}` is permanently stripped before the request reaches the downstream model. Vision is completely non-functional for requests translated through this layer.

---

### 3. Missing `strict` Flag on Tool Definitions

When mapping function tools from Responses API format to Chat Completions format, `responses_to_chat_completions` copies `name`, `description`, and `parameters` but omits the `strict` boolean.

```python
# src/responses.py — responses_to_chat_completions
cc_tools.append({
    "type": "function",
    "function": {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {}),
        # "strict" is not forwarded
    },
})
```

This means a client passing `strict: true` on a tool definition will have that constraint silently dropped, disabling strict schema adherence on the downstream call.

---

### 4. Incorrect `status` Mapping and Missing Token Detail Fields

**Status on truncation:**
`chat_completions_to_responses` maps `finish_reason == "content_filter"` to `status="failed"` and all other values (including `"length"`) to `status="completed"`.

```python
# src/responses.py — chat_completions_to_responses
status = "failed" if finish_reason == "content_filter" else "completed"
```

Per the official spec:
- `finish_reason="length"` must produce `status="incomplete"` with `incomplete_details={"reason": "max_output_tokens"}`.
- `finish_reason="content_filter"` must produce `status="incomplete"` with `incomplete_details={"reason": "content_filter"}` — not `status="failed"`.

**Missing deep token usage fields:**
The `usage` block returned by `chat_completions_to_responses` only maps top-level counts:

```python
# src/responses.py — chat_completions_to_responses
usage = {
    "input_tokens": cc_usage.get("prompt_tokens", 0),
    "output_tokens": cc_usage.get("completion_tokens", 0),
    "total_tokens": cc_usage.get("total_tokens", 0),
}
```

The following fields specified by the Responses API are not present:
- `input_tokens_details` (includes `cached_tokens`)
- `output_tokens_details` (includes `reasoning_tokens`)

---

### 5. Streaming: Tool Calls Dropped and Usage Metrics Lost

`translate_streaming_response` only extracts text content from each streaming delta. It does not handle `tool_calls` in deltas and discards the final usage chunk.

**Tool calls lost:**
```python
# src/responses.py — translate_streaming_response
delta = choices[0].get("delta", {})
content_delta = delta.get("content")
# delta.get("tool_calls") is never read
```

Any function call attempted by the model during a streaming response is permanently discarded. The client receives blank content with no indication that a tool call was intended.

**Usage metrics lost:**
The final SSE chunk in an OpenAI streaming response typically contains token usage with an empty `choices` array (`{"choices": [], "usage": {...}}`). The translator exits early when `choices` is empty:

```python
# src/responses.py — translate_streaming_response
choices = data.get("choices", [])
if not choices:
    continue
```

This guarantees that the final `response.completed` event will never carry token usage data.
