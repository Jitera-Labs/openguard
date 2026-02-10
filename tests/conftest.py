"""
Pytest fixtures and configuration for OpenGuard integration tests.

Provides test client, mock downstream APIs, and test guards configuration.
"""

import os
import textwrap
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def guards_yaml(tmp_path):
    """Create temporary guards.yaml for testing."""
    content = textwrap.dedent(
        """
guards:
  - match:
      model:
        _ilike: "%test%"
    apply:
      - type: content_filter
        config:
          blocked_words:
            - "badword"
            - "offensive"

  - match:
      model:
        _ilike: "%secure%"
    apply:
      - type: pii_filter
        config:
          enabled: true

  - match:
      model:
        _ilike: "%limited%"
    apply:
      - type: max_tokens
        config:
          max_tokens: 100

  - match:
      model:
        _ilike: "%protected%"
    apply:
      - type: content_filter
        config:
          blocked_words:
            - "spam"
      - type: max_tokens
        config:
          max_tokens: 500
        """
    ).lstrip()
    path = tmp_path / "guards.yaml"
    path.write_text(content)
    return str(path)


@pytest.fixture
def test_env(guards_yaml):
    """Set up test environment variables with isolation."""
    old_env = os.environ.copy()

    os.environ["OPENGUARD_CONFIG"] = guards_yaml
    os.environ["OPENGUARD_OPENAI_URL_TEST"] = "http://downstream.test"
    os.environ["OPENGUARD_OPENAI_KEY_TEST"] = ""
    os.environ["OPENGUARD_API_KEY"] = ""
    os.environ["OPENGUARD_API_KEYS"] = ""
    os.environ["OPENGUARD_LOG_LEVEL"] = "DEBUG"

    yield

    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture(scope="function")
def mock_httpx(
    mock_downstream_models, mock_non_streaming_response, mock_streaming_response, monkeypatch
):
    """Mock httpx for all requests - must be set up before test_client."""
    captured, _ = _build_httpx_mock(
        mock_downstream_models,
        mock_non_streaming_response,
        mock_streaming_response,
        monkeypatch,
    )
    return captured


@pytest.fixture(scope="function")
def test_client(test_env, mock_httpx):
    """Create FastAPI test client with test config."""
    import importlib

    from src import config as config_module
    from src import guards as guards_module
    from src import main as main_module

    importlib.reload(config_module)
    guards_module._guards_cache = None
    importlib.reload(guards_module)
    importlib.reload(main_module)

    return TestClient(main_module.app)


@pytest.fixture
def mock_downstream_models():
    """Mock downstream model list response."""
    return [
        {
            "id": "test-model",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test",
        },
        {
            "id": "test-secure-model",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test",
        },
        {
            "id": "test-limited-model",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test",
        },
        {
            "id": "test-protected-model",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test",
        },
        {
            "id": "unmatched-model",
            "object": "model",
            "created": 1234567890,
            "owned_by": "test",
        },
    ]


@pytest.fixture
def mock_streaming_response():
    """Mock streaming chat completion response."""
    chunks = [
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',  # noqa: E501
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',  # noqa: E501
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',  # noqa: E501
        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',  # noqa: E501
        "data: [DONE]\n\n",
    ]
    return [chunk.encode("utf-8") for chunk in chunks]


@pytest.fixture
def mock_non_streaming_response():
    """Mock non-streaming chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def _build_httpx_mock(
    mock_downstream_models,
    non_streaming_response,
    streaming_chunks,
    monkeypatch,
):
    captured = {"get": [], "post": [], "stream": []}

    async def mock_get(url, headers=None, **kwargs):
        response = MagicMock()
        response.json.return_value = {"object": "list", "data": mock_downstream_models}
        response.raise_for_status = MagicMock()
        captured["get"].append({"url": url, "headers": headers})
        return response

    async def mock_post(url, headers=None, json=None, **kwargs):
        response = MagicMock()
        response.json.return_value = non_streaming_response
        response.raise_for_status = MagicMock()
        captured["post"].append({"url": url, "headers": headers, "json": json})
        return response

    async def aiter_bytes():
        for chunk in streaming_chunks:
            yield chunk

    def mock_stream(method, url, headers=None, json=None, **kwargs):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.aiter_bytes = aiter_bytes
        captured["stream"].append({"method": method, "url": url, "headers": headers, "json": json})

        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=None)
        return context

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.stream = MagicMock(side_effect=mock_stream)

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    return captured, mock_client


@pytest.fixture
def setup_mock_downstream(mock_httpx):
    """Mock downstream model listing and non-streaming completions."""
    return mock_httpx


@pytest.fixture
def setup_mock_non_streaming(mock_httpx):
    """Mock non-streaming chat completion calls."""
    return mock_httpx


@pytest.fixture
def setup_mock_streaming(mock_httpx):
    """Mock streaming chat completion calls."""
    return mock_httpx


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cached data between tests."""
    from src import guards as guards_module
    from src import mapper as mapper_module

    guards_module._guards_cache = None

    try:
        mapper_module.list_downstream.cache.clear()
    except Exception:
        pass

    yield
