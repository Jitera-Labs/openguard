import json

import pytest
from fastapi.responses import JSONResponse

from src.guards import GuardAction, GuardRule
from src.responses import translate_streaming_response


def test_responses_native_passthrough_applies_guards(test_client, monkeypatch):
    """Native Responses passthrough still applies guards before forwarding."""
    from src import main as main_module

    captured: dict[str, object] = {}

    async def fake_forward_provider_request(request, provider, endpoint_path, body_bytes, payload):
        captured["provider"] = provider
        captured["endpoint_path"] = endpoint_path
        captured["body"] = json.loads(body_bytes.decode("utf-8"))
        captured["payload"] = payload
        return JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(
        main_module.responses_module, "upstream_supports_responses_api", lambda url: True
    )
    monkeypatch.setattr(main_module, "_forward_provider_request", fake_forward_provider_request)

    response = test_client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "Please remove badword from this request"},
    )

    assert response.status_code == 200
    assert captured["provider"] == "openai"
    assert captured["endpoint_path"] == "/responses"

    forwarded_payload = captured["payload"]
    assert isinstance(forwarded_payload, dict)
    assert forwarded_payload["input"] == "Please remove [FILTERED] from this request"
    assert "badword" not in forwarded_payload["input"].lower()
    assert captured["body"] == forwarded_payload


def test_responses_unknown_model_returns_404(test_client, monkeypatch):
    """Responses API normalizes unknown models into a deterministic 404."""
    from src import config

    monkeypatch.setattr(
        config.OPENGUARD_OPENAI_URLS,
        "__value__",
        ["http://downstream.test", "http://secondary.test"],
    )

    response = test_client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "Hello"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown model: 'nonexistent-model'"}


def test_responses_guard_block_preempts_unknown_model(test_client, monkeypatch):
    """Guard-blocked Responses requests return 403 before unknown-model resolution."""
    from src import main as main_module

    monkeypatch.setattr(
        main_module,
        "get_guards",
        lambda: [
            GuardRule(
                match={},
                apply=[
                    GuardAction(
                        type="keyword_filter",
                        config={"keywords": ["forbidden"], "action": "block"},
                    )
                ],
            )
        ],
    )

    response = test_client.post(
        "/v1/responses",
        json={"model": "nonexistent-model", "input": "This is forbidden"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "message": "Request blocked: found keyword 'forbidden'",
            "type": "guard_block",
            "code": 403,
        }
    }


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
