import json

import pytest
from fastapi.responses import JSONResponse

from src.responses import translate_streaming_response


def test_responses_unknown_model_returns_404(test_client, monkeypatch):
    """Responses API normalizes unknown models into a deterministic 404."""
    from src import config

    monkeypatch.setattr(
        config.LOUDER_OPENAI_URLS,
        "__value__",
        ["http://downstream.test", "http://secondary.test"],
    )

    response = test_client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "Hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown model: 'nonexistent-model'"}


@pytest.mark.asyncio
async def test_translate_streaming_response_emits_error_event_for_raw_upstream_error():
    """Raw upstream error chunks become SSE error events and stop completion."""

    async def raw_error_stream():
        yield (
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,'
            b'"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},'
            b'"finish_reason":null}]}'
            b"\n\n"
        )
        yield b'{"error":{"message":"Upstream exploded","type":"server_error","code":502}}\n'
        yield b"data: [DONE]\n\n"

    events = []
    async for chunk in translate_streaming_response(raw_error_stream(), {"model": "test-model"}):
        events.append(chunk)

    assert any("event: error" in event for event in events)
    assert not any("event: response.completed" in event for event in events)

    error_event = next(event for event in events if "event: error" in event)
    assert '"message": "Upstream exploded"' in error_event
    assert '"code": 502' in error_event
