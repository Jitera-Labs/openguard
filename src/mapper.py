from asyncache import cached
from cachetools import TTLCache
from typing import Dict, List, Tuple
import httpx

from src import config
from src import log

logger = log.setup_logger(__name__)


@cached(TTLCache(1024, 60))
async def list_downstream() -> Tuple[List[dict], Dict[str, dict]]:
  """
  Fetch models from all downstream APIs with 60-second caching.
  Returns tuple of (models_list, model_to_backend_mapping).
  """
  logger.debug("Listing downstream models")

  all_models = []
  model_to_backend = {}

  for url, key in zip(
    config.OPENGUARD_OPENAI_URLS.value,
    config.OPENGUARD_OPENAI_KEYS.value,
  ):
    try:
      endpoint = f"{url}/v1/models"
      headers = {}

      if key:
        headers["Authorization"] = f"Bearer {key}"

      headers["Content-Type"] = "application/json"

      logger.debug(f"Fetching models from '{endpoint}'")

      async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        json_data = response.json()
        models = json_data.get("data", [])

        logger.debug(f"Found {len(models)} models at '{endpoint}'")
        all_models.extend(models)

        # Track which backend each model comes from
        for model in models:
          model_id = model.get("id")
          if model_id:
            model_to_backend[model_id] = {"url": url, "key": key}

    except httpx.TimeoutException:
      logger.error(f"Timeout fetching models from {endpoint}")
    except httpx.HTTPStatusError as e:
      logger.error(f"HTTP error fetching models from {endpoint}: {e.response.status_code}")
    except Exception as e:
      logger.error(f"Failed to fetch models from {endpoint}: {e}")

  return all_models, model_to_backend


async def build_model_map() -> Dict[str, dict]:
  """
  Build a mapping of model IDs to their backend configuration.
  Returns: {model_id: {url: str, headers: dict}}
  """
  models, model_to_backend = await list_downstream()
  model_map = {}

  for model in models:
    model_id = model.get("id")
    if not model_id:
      continue

    backend_info = model_to_backend.get(model_id)
    if not backend_info:
      continue

    url = backend_info["url"]
    key = backend_info["key"]

    headers = {"Content-Type": "application/json"}
    if key:
      headers["Authorization"] = f"Bearer {key}"

    model_map[model_id] = {
      "url": url,
      "headers": headers
    }

  logger.debug(f"Built model map with {len(model_map)} models")
  return model_map


async def resolve_request_config(payload: dict) -> dict:
  """
  Resolve request configuration from payload.

  Args:
    payload: Request payload with 'model' field

  Returns:
    dict with:
      - url: Backend URL
      - headers: Auth headers for backend
      - payload: Original payload (guards already applied)
      - stream: Whether to stream the response
  """
  model = payload.get("model")

  if not model:
    raise ValueError("Unable to proxy request without a model specifier")

  model_map = await build_model_map()

  # Case-insensitive model lookup
  model_lower = model.lower()
  backend_config = None
  actual_model_key = None

  for key in model_map:
    if key.lower() == model_lower:
      backend_config = model_map[key]
      actual_model_key = key
      break

  if not backend_config:
    raise ValueError(f"Unknown model: '{model}'")

  backend_config = model_map[actual_model_key]
  stream = payload.get("stream", False)

  resolved_config = {
    "url": backend_config["url"],
    "headers": backend_config["headers"],
    "payload": payload,
    "stream": stream
  }

  logger.debug(f"Resolved model '{model}' to backend '{backend_config['url']}'")

  return resolved_config
