"""
Basic tests for proxy infrastructure.
"""

import pytest

from src import llm, mapper


@pytest.mark.asyncio
async def test_list_downstream_no_backend():
    """Test that list_downstream handles missing backends gracefully."""
    models = await mapper.list_downstream()
    # Should return empty list or models if any backends configured
    assert isinstance(models, list)
    # assert isinstance(backends, dict) # Removed because it returns only list


@pytest.mark.asyncio
async def test_resolve_request_config_missing_model():
    """Test that resolve_request_config raises ValueError for missing model."""
    payload = {"messages": [{"role": "user", "content": "test"}]}

    with pytest.raises(ValueError, match="Unable to proxy request without a model specifier"):
        mapper.resolve_request_config(payload)


@pytest.mark.asyncio
async def test_resolve_request_config_unknown_model(monkeypatch):
    """Test that resolve_request_config raises HTTPException for unknown model."""
    import importlib

    from src import config

    # Ensure multiple URLs are configured so fallback logic doesn't trigger
    monkeypatch.setenv("OPENGUARD_OPENAI_URL_PRIMARY", "http://url1")
    monkeypatch.setenv("OPENGUARD_OPENAI_URL_SECONDARY", "http://url2")

    # Reload config to pick up new env vars
    importlib.reload(config)

    payload = {
        "model": "nonexistent-model-12345",
        "messages": [{"role": "user", "content": "test"}],
    }

    try:
        with pytest.raises(ValueError, match="Unknown model"):
            mapper.resolve_request_config(payload)
    finally:
        # Restore config to avoid side effects
        importlib.reload(config)


@pytest.mark.asyncio
async def test_llm_initialization():
    """Test that LLM can be initialized."""
    llm_instance = llm.LLM(
        url="http://localhost:11434",
        headers={"Content-Type": "application/json"},
        messages=[],
        model="test",
        stream=False,
    )

    assert llm_instance.url == "http://localhost:11434"
    assert llm_instance.is_streaming is False
    assert llm_instance.id is not None


@pytest.mark.asyncio
async def test_llm_registry():
    """Test that LLM registry works."""
    from src.llm_registry import llm_registry

    llm_instance = llm.LLM(
        url="http://localhost:11434", headers={}, messages=[], model="test", stream=False
    )

    llm_registry.register(llm_instance)
    assert llm_registry.get(llm_instance.id) == llm_instance

    llm_registry.unregister(llm_instance)
    assert llm_registry.get(llm_instance.id) is None


@pytest.mark.asyncio
async def test_llm_connection_error(monkeypatch):
    """Test that LLM handles connection errors gracefully."""
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    # Mock httpx to raise ConnectError
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def side_effect(*args, **kwargs):
        raise httpx.ConnectError("Connection failed")

    mock_client.post = AsyncMock(side_effect=side_effect)
    mock_client.stream = MagicMock(
        side_effect=side_effect
    )  # stream is sync/async context manager usually

    # Mock stream context manager raising error on enter?
    # stream_chat_completion uses client.stream
    async def stream_side_effect(*args, **kwargs):
        raise httpx.ConnectError("Connection failed")

    context = MagicMock()
    context.__aenter__ = AsyncMock(side_effect=stream_side_effect)
    context.__aexit__ = AsyncMock()

    # client.stream returns context manager
    mock_client.stream = MagicMock(return_value=context)

    # Patch AsyncClient constructor
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))

    llm_instance = llm.LLM(
        url="http://invalid-host-12345:9999",
        headers={},
        model="test",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
    )

    # Should yield error response, not raise exception
    chunks = []
    async for chunk in await llm_instance.serve():
        chunks.append(chunk)

    assert len(chunks) > 0
    # Should be error response
    import json

    response = json.loads(chunks[0])
    assert "error" in response
    assert response["error"]["code"] == 502


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_resolve_provider_route_internal_mode_precedence(monkeypatch):
    """Internal OpenGuard key takes precedence and uses model routing first."""
    import importlib

    from src import config

    monkeypatch.setenv("OPENGUARD_API_KEY", "internal-key")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_URL_A", "http://anthropic-a.test")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_KEY_A", "anth-key-a")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_URL_B", "http://anthropic-b.test")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_KEY_B", "anth-key-b")

    importlib.reload(config)
    importlib.reload(mapper)

    mapper.MODEL_TO_BACKEND["anthropic"]["claude-special"] = "http://anthropic-b.test"

    route = mapper.resolve_provider_route(
        provider="anthropic",
        endpoint_path="/messages",
        body={"model": "claude-special"},
        headers={"Authorization": "Bearer internal-key"},
    )

    assert route["mode"] == "internal"
    assert route["url"] == "http://anthropic-b.test"
    assert route["endpoint"] == "http://anthropic-b.test/v1/messages"


def test_resolve_provider_route_direct_key_precedence(monkeypatch):
    """Downstream provider key routes directly to matching backend."""
    import importlib

    from src import config

    monkeypatch.setenv("OPENGUARD_API_KEY", "")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_URL_A", "http://anthropic-a.test")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_KEY_A", "anth-key-a")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_URL_B", "http://anthropic-b.test")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_KEY_B", "anth-key-b")

    importlib.reload(config)
    importlib.reload(mapper)

    route = mapper.resolve_provider_route(
        provider="anthropic",
        endpoint_path="/messages",
        body={"model": "anything"},
        headers={"x-api-key": "anth-key-a"},
    )

    assert route["mode"] == "direct"
    assert route["url"] == "http://anthropic-a.test"


def test_resolve_provider_route_endpoint_fallback(monkeypatch):
    """Unknown key falls back to endpoint-based provider routing."""
    import importlib

    from src import config

    monkeypatch.setenv("OPENGUARD_API_KEY", "internal-only")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_URL_A", "http://anthropic-a.test")
    monkeypatch.setenv("OPENGUARD_ANTHROPIC_KEY_A", "anth-key-a")

    importlib.reload(config)
    importlib.reload(mapper)

    route = mapper.resolve_provider_route(
        provider="anthropic",
        endpoint_path="/messages",
        body={"model": "anything"},
        headers={"x-api-key": "unknown-key"},
    )

    assert route["mode"] == "endpoint"
    assert route["endpoint"] == "http://anthropic-a.test/v1/messages"


@pytest.mark.asyncio
async def test_forward_provider_request_warms_cache_for_internal_model_routing(monkeypatch):
    """Internal key + model should warm cache and route to mapped backend on first call."""
    from src import main as main_module

    class DummyRequest:
        method = "POST"
        headers = {"Authorization": "Bearer internal-key"}
        query_params = {}

    mapper.MODEL_TO_BACKEND["anthropic"].clear()

    monkeypatch.setattr(
        main_module.mapper,
        "get_provider_backends",
        lambda provider: [
            {"provider": "anthropic", "url": "http://anthropic-a.test", "key": "anth-key-a"},
            {"provider": "anthropic", "url": "http://anthropic-b.test", "key": "anth-key-b"},
        ],
    )
    monkeypatch.setattr(main_module.mapper, "get_internal_api_keys", lambda: ["internal-key"])

    warmed = {"called": False}

    async def fake_list_downstream(provider="openai"):
        warmed["called"] = True
        mapper.MODEL_TO_BACKEND["anthropic"]["claude-special"] = "http://anthropic-b.test"
        return []

    monkeypatch.setattr(main_module.mapper, "list_downstream", fake_list_downstream)

    captured = {"url": None}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, headers=None, params=None, content=None):
            captured["url"] = url
            return type(
                "MockResponse",
                (),
                {
                    "content": b'{"ok":true}',
                    "status_code": 200,
                    "headers": {"content-type": "application/json", "x-test": "1"},
                },
            )()

    monkeypatch.setattr(main_module.httpx, "AsyncClient", lambda timeout=None: MockClient())

    payload = {"model": "claude-special", "messages": [{"role": "user", "content": "hi"}]}
    response = await main_module._forward_provider_request(
        request=DummyRequest(),
        provider="anthropic",
        endpoint_path="/messages",
        body_bytes=b"{}",
        payload=payload,
    )

    assert warmed["called"] is True
    assert captured["url"] == "http://anthropic-b.test/v1/messages"
    assert response.status_code == 200
