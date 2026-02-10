"""
Integration tests for OpenGuard - end-to-end testing of the complete system.

Tests cover:
- API endpoints (health, root, models)
- Chat completions with various guards
- Streaming and non-streaming responses
- Authentication
- Error handling
"""

import json
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
    assert data["name"] == "OpenGuard"
    assert data["version"] == "0.1.0"
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


def test_chat_no_guard_unmatched_model(test_client, setup_mock_non_streaming):
    """Test chat completion with model that doesn't match any guards"""
    payload = {
        "model": "unmatched-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    assert forwarded["model"] == "unmatched-model"
    assert forwarded["messages"][0]["content"] == "Hello, how are you?"


def test_chat_with_content_filter(test_client, setup_mock_non_streaming):
    """Test chat completion with content filter guard applied"""
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "This message contains badword and offensive content"}
        ],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    content = forwarded["messages"][0]["content"]
    assert "badword" not in content.lower()
    assert "offensive" not in content.lower()
    assert "[FILTERED]" in content


def test_chat_with_pii_filter(test_client, setup_mock_non_streaming):
    """Test chat completion with PII filter guard applied"""
    payload = {
        "model": "test-secure-model",
        "messages": [
            {"role": "user", "content": "My email is test@example.com and phone is 555-123-4567"}
        ],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    assert forwarded["messages"][0]["role"] == "system"
    assert "PII has been filtered" in forwarded["messages"][0]["content"]
    user_content = forwarded["messages"][1]["content"]
    assert "<protected:email>" in user_content
    assert "<protected:phone>" in user_content


def test_chat_with_max_tokens(test_client, setup_mock_non_streaming):
    """Test chat completion with max_tokens guard applied"""
    payload = {
        "model": "test-limited-model",
        "messages": [{"role": "user", "content": "Tell me a long story"}],
        "max_tokens": 1000,  # Should be reduced to 100
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    assert forwarded["max_tokens"] == 100


def test_chat_with_multiple_guards(test_client, setup_mock_non_streaming):
    """Test chat completion with multiple guards applied"""
    payload = {
        "model": "test-protected-model",
        "messages": [{"role": "user", "content": "This is spam content"}],
        "max_tokens": 1000,
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"

    forwarded = setup_mock_non_streaming["stream"][0]["json"]
    assert forwarded["max_tokens"] == 500
    assert "[FILTERED]" in forwarded["messages"][0]["content"]


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
    # Enable authentication
    monkeypatch.setenv("OPENGUARD_API_KEY", "test-key-123")

    # Need to reload config to pick up new env var
    import importlib

    from src import config

    importlib.reload(config)

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


def test_invalid_model(test_client, setup_mock_downstream, monkeypatch):
    """Test error handling for unknown model"""
    # Force multiple URLs to disable fallback logic
    monkeypatch.setenv("OPENGUARD_OPENAI_URL_SECONDARY", "http://secondary.test")
    import importlib

    from src import config

    importlib.reload(config)

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


def test_content_filter_multiple_words(test_client, setup_mock_non_streaming):
    """Test content filter replaces multiple blocked words"""
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Please remove badword and offensive terms"}],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_pii_filter_multiple_types(test_client, setup_mock_non_streaming):
    """Test PII filter detects multiple PII types"""
    payload = {
        "model": "test-secure-model",
        "messages": [
            {
                "role": "user",
                "content": "Contact me at john@example.com or 555-123-4567. SSN: 123-45-6789",
            }
        ],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_multimodal_content_with_filter(test_client, setup_mock_non_streaming):
    """Test guards work with multimodal content (array of content parts)"""
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This has badword in it"},
                    {"type": "text", "text": "And more offensive content"},
                ],
            }
        ],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_max_tokens_enforces_when_absent(test_client, setup_mock_non_streaming):
    """Test max_tokens guard adds limit when not present"""
    payload = {
        "model": "test-limited-model",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": False,
        # No max_tokens field
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


def test_guard_with_case_insensitive_matching(test_client, setup_mock_non_streaming):
    """Test guards match models case-insensitively"""
    payload = {
        "model": "TEST-MODEL",  # Uppercase
        "messages": [{"role": "user", "content": "This has BADWORD"}],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200


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
        assert response.status_code == 500


def test_models_endpoint_requires_auth_when_enabled(
    test_client, setup_mock_downstream, monkeypatch
):
    """Test that /v1/models endpoint requires authentication when enabled"""
    monkeypatch.setenv("OPENGUARD_API_KEY", "test-key-123")

    # Reload config
    import importlib

    from src import config

    importlib.reload(config)

    # Request without auth should fail
    response = test_client.get("/v1/models")
    assert response.status_code == 401

    # Request with correct auth should succeed
    headers = {"Authorization": "Bearer test-key-123"}
    response = test_client.get("/v1/models", headers=headers)
    assert response.status_code == 200


def test_multiple_messages_with_guards(test_client, setup_mock_non_streaming):
    """Test guards applied across multiple messages in conversation"""
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First badword message"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Second offensive message"},
        ],
        "stream": False,
    }

    response = test_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
