"""
Basic tests for proxy infrastructure.
"""

import pytest
import asyncio
from src import mapper, llm


@pytest.mark.asyncio
async def test_list_downstream_no_backend():
  """Test that list_downstream handles missing backends gracefully."""
  models, backends = await mapper.list_downstream()
  # Should return empty list or models if any backends configured
  assert isinstance(models, list)
  assert isinstance(backends, dict)


@pytest.mark.asyncio
async def test_resolve_request_config_missing_model():
  """Test that resolve_request_config raises ValueError for missing model."""
  payload = {
    "messages": [{"role": "user", "content": "test"}]
  }

  with pytest.raises(ValueError, match="Unable to proxy request without a model specifier"):
    await mapper.resolve_request_config(payload)


@pytest.mark.asyncio
async def test_resolve_request_config_unknown_model():
  """Test that resolve_request_config raises ValueError for unknown model."""
  payload = {
    "model": "nonexistent-model-12345",
    "messages": [{"role": "user", "content": "test"}]
  }

  with pytest.raises(ValueError, match="Unknown model"):
    await mapper.resolve_request_config(payload)


@pytest.mark.asyncio
async def test_llm_initialization():
  """Test that LLM can be initialized."""
  llm_instance = llm.LLM(
    url="http://localhost:11434",
    headers={"Content-Type": "application/json"},
    payload={"model": "test", "messages": []},
    stream=False
  )

  assert llm_instance.url == "http://localhost:11434"
  assert llm_instance.stream is False
  assert llm_instance.id is not None


@pytest.mark.asyncio
async def test_llm_registry():
  """Test that LLM registry works."""
  from src.llm_registry import llm_registry

  llm_instance = llm.LLM(
    url="http://localhost:11434",
    headers={},
    payload={},
    stream=False
  )

  llm_registry.register(llm_instance)
  assert llm_registry.get(llm_instance.id) == llm_instance

  llm_registry.unregister(llm_instance)
  assert llm_registry.get(llm_instance.id) is None


@pytest.mark.asyncio
async def test_llm_connection_error():
  """Test that LLM handles connection errors gracefully."""
  llm_instance = llm.LLM(
    url="http://invalid-host-12345:9999",
    headers={},
    payload={"model": "test", "messages": [{"role": "user", "content": "test"}]},
    stream=False
  )

  # Should yield error response, not raise exception
  chunks = []
  async for chunk in llm_instance.serve():
    chunks.append(chunk)

  assert len(chunks) > 0
  # Should be error response
  import json
  response = json.loads(chunks[0])
  assert "error" in response
  assert response["error"]["code"] == 502


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
