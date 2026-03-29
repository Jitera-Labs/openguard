"""
Integration tests for Louder - end-to-end testing of the complete system.

Tests cover:
- API endpoints (health, root, models)
- Chat completions with various guards
- Streaming and non-streaming responses
- Authentication
- Error handling
"""

import json
from importlib.metadata import version as _get_version
from unittest.mock import AsyncMock, MagicMock, patch


def test_health(test_client):
    """Test health endpoint returns OK status"""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root(test_client):
    """Test root endpoint provides API information"""
    response = test_client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Louder"
    assert data["version"] == _get_version("louder")
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
    assert data["endpoints"]["models"] == "/v1/models"
    assert data["endpoints"]["chat"] == "/v1/chat/completions"


def test_list_models(test_client, setup_mock_downstream, mock_downstream_models):
    """Test model listing with mocked downstream API"""
    response = test_client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) == len(mock_downstream_models)

    # Verify model IDs
    model_ids = [m["id"] for m in data["data"]]
    assert "test-model" in model_ids
    assert "unmatched-model" in model_ids


def test_list_models_uses_anthropic_provider_when_anthropic_key(test_client, setup_mock_downstream):
    """Test /v1/models routes to Anthropic model listing when Anthropic key is provided."""
    response = test_client.get("/v1/models", headers={"x-api-key": "anthropic-downstream-key"})

    assert response.status_code == 200
    last_get = setup_mock_downstream["get"][-1]
    assert last_get["url"] == "http://anthropic.test/v1/models"
    assert last_get["headers"]["x-api-key"] == "anthropic-downstream-key"


def test_anthropic_messages_passthrough(test_client, setup_mock_downstream):
    """Test dedicated Anthropic /v1/messages endpoint forwards payload and auth."""
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"custom": "value"},
    }

    response = test_client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"

    forwarded = setup_mock_downstream["request"][-1]
    assert forwarded["method"] == "POST"
    assert forwarded["url"] == "http://anthropic.test/v1/messages"
    assert json.loads(forwarded["content"].decode("utf-8"))["metadata"]["custom"] == "value"





def test_anthropic_messages_explicit_tool_choice_without_tool_use_returns_error(
    test_client, setup_mock_downstream
):
    """Explicit Anthropic tool_choice requires downstream tool_use block or a proxy error."""
    payload = {
        "model": "claude-test-model",
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Call calculate_sum with a=2 and b=3.",
                    }
                ],
            }
        ],
        "tools": [
            {
                "name": "calculate_sum",
                "description": "Calculate the sum of two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "calculate_sum"},
    }

    response = test_client.post("/v1/messages", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "tool_use" in body["error"]["message"]

    forwarded = setup_mock_downstream["request"][-1]
    forwarded_payload = json.loads(forwarded["content"].decode("utf-8"))
    assert forwarded_payload["tools"][0]["name"] == "calculate_sum"
    assert forwarded_payload["tool_choice"]["type"] == "tool"





def test_anthropic_count_tokens_passthrough(test_client, setup_mock_downstream):
    """Test dedicated Anthropic /v1/messages/count_tokens endpoint forwarding."""
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Count these tokens"}],
    }

    response = test_client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    assert response.json()["input_tokens"] == 12

    forwarded = setup_mock_downstream["request"][-1]
    assert forwarded["url"] == "http://anthropic.test/v1/messages/count_tokens"

















def test_streaming_chat_completion(test_client, setup_mock_streaming):
    """Test streaming chat completion"""
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Read and verify streaming response
    content = response.text
    assert "data: " in content
    assert "[DONE]" in content

    # Parse chunks
    lines = [line for line in content.split("\n") if line.strip()]
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) > 0

    # Verify at least one chunk has content
    has_content = False
    for line in data_lines:
        if line == "data: [DONE]":
            continue
        data = json.loads(line[6:])  # Skip "data: "
        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
            has_content = True
            break

    assert has_content

    forwarded = setup_mock_streaming["stream"][0]["json"]
    assert forwarded["stream"] is True


def test_non_streaming_chat_completion(test_client, setup_mock_non_streaming):
    """Test non-streaming chat completion"""
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"]
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    # LLM serve forces streaming, so upstream request has stream=True
    assert forwarded["stream"] is True


def test_authentication_disabled(test_client, setup_mock_non_streaming):
    """Test that requests work without authentication when disabled"""
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Request without Authorization header
    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_authentication_enabled(test_client, setup_mock_non_streaming, monkeypatch):
    """Test authentication when API key is configured"""
    from src import config

    # Mock the value directly
    monkeypatch.setattr(config.LOUDER_API_KEY, "__value__", "test-key-123")

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Request without Authorization header should fail
    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 401

    # Request with wrong key should fail
    headers = {"Authorization": "Bearer wrong-key"}
    response = test_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 403

    # Request with correct key should succeed
    headers = {"Authorization": "Bearer test-key-123"}
    response = test_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200


def test_anthropic_messages_accepts_front_door_x_api_key(
    test_client, setup_mock_downstream, monkeypatch
):
    """Test Anthropic-compatible endpoint accepts x-api-key for proxy auth."""
    from src import config

    monkeypatch.setattr(config.LOUDER_API_KEY, "__value__", "test-key-123")

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = test_client.post(
        "/v1/messages",
        json=payload,
        headers={"x-api-key": "test-key-123"},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "message"

    forwarded = setup_mock_downstream["request"][-1]
    assert forwarded["url"] == "http://anthropic.test/v1/messages"
    assert forwarded["headers"]["x-api-key"] == "anthropic-downstream-key"


def test_invalid_model(test_client, setup_mock_downstream, monkeypatch):
    """Test error handling for unknown model"""
    from src import config

    # Force multiple URLs to disable fallback logic
    monkeypatch.setattr(
        config.LOUDER_OPENAI_URLS,
        "__value__",
        ["http://downstream.test", "http://secondary.test"],
    )

    payload = {
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 404
    assert "Unknown model" in response.json()["detail"]


def test_missing_model(test_client, setup_mock_downstream):
    """Test error handling for missing model field"""
    payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_invalid_json(test_client):
    """Test error handling for invalid JSON"""
    response = test_client.post(
        "/v1/chat/completions", data="not valid json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]

















def test_downstream_timeout_handling(test_client, setup_mock_downstream):
    """Test handling of downstream API timeout"""
    import httpx

    # Mock a timeout
    with patch("httpx.AsyncClient") as mock_client:
        client_instance = MagicMock()

        async def mock_stream(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        stream_context = MagicMock()
        stream_context.__aenter__ = AsyncMock(side_effect=mock_stream)
        stream_context.__aexit__ = AsyncMock()

        client_instance.stream = MagicMock(return_value=stream_context)
        client_instance.__aenter__ = AsyncMock(return_value=client_instance)
        client_instance.__aexit__ = AsyncMock()

        mock_client.return_value = client_instance

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }

        response = test_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 504


def test_models_endpoint_requires_auth_when_enabled(
    test_client, setup_mock_downstream, monkeypatch
):
    """Test that /v1/models endpoint requires authentication when enabled"""
    monkeypatch.setenv("LOUDER_API_KEY", "test-key-123")

    # Reload config
    import importlib

    from src import config

    importlib.reload(config)

    # Request without auth should fail
    response = test_client.get("/v1/models")
    assert response.status_code == 401

    # Request with correct bearer auth should succeed
    headers = {"Authorization": "Bearer test-key-123"}
    response = test_client.get("/v1/models", headers=headers)
    assert response.status_code == 200


def test_models_endpoint_accepts_front_door_x_api_key(
    test_client, setup_mock_downstream, monkeypatch
):
    """Test that /v1/models accepts x-api-key when authentication is enabled."""
    from src import config

    monkeypatch.setattr(config.LOUDER_API_KEY, "__value__", "test-key-123")

    response = test_client.get("/v1/models", headers={"x-api-key": "test-key-123"})

    assert response.status_code == 200
    assert response.json()["object"] == "list"

    forwarded = setup_mock_downstream["get"][-1]
    assert forwarded["url"] == "http://downstream.test/models"
    assert "x-api-key" not in {key.lower() for key in forwarded["headers"]}


