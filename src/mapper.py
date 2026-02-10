from typing import TYPE_CHECKING, Dict, Optional

import httpx
from asyncache import cached
from cachetools import TTLCache
from fastapi import HTTPException

from . import config, log

if TYPE_CHECKING:
  from . import llm

logger = log.setup_logger(__name__)

MODEL_TO_BACKEND: Dict[str, str] = {}

# TODO: Add mods (modules) support when adoption is complete
# import mods


@cached(TTLCache(1024, 60))
async def list_downstream():
  logger.debug("Listing downstream models")

  all_models = []

  # Using OpenGuard configuration
  urls = config.OPENGUARD_OPENAI_URLS.value
  keys = config.OPENGUARD_OPENAI_KEYS.value

  # Ensure keys matches urls length, pad with empty strings if not
  if isinstance(urls, str): urls = [urls]
  if isinstance(keys, str): keys = [keys]

  if len(keys) < len(urls):
      keys.extend([""] * (len(urls) - len(keys)))


  for url, key in zip(urls, keys):
    try:
      endpoint = f"{url}/models"
      headers = {
        "Content-Type": "application/json",
      }
      if key:
         headers["Authorization"] = f"Bearer {key}"

      logger.debug(f"Fetching models from '{endpoint}'")

      async with httpx.AsyncClient() as client:
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        json = response.json()
        models = json.get("data", [])

        logger.debug(f"Found {len(models)} models at '{endpoint}'")
        all_models.extend(models)

        for model in models:
          MODEL_TO_BACKEND[model["id"]] = url

    except Exception as e:
      logger.error(f"Failed to fetch models from {endpoint}: {e}")

  return all_models


def get_proxy_model(module, model: dict) -> Dict:
  return {
    **model,
    "id": f"{module.ID_PREFIX}-{model['id']}",
    "name": f"{module.ID_PREFIX} {model['id']}",
  }


def resolve_proxy_model(model_id: str) -> str:
    # Logic disabled until mods are ported
    # parts = model_id.split("-")
    # if parts[0] in mods.registry:
    #   return "-".join(parts[1:])
    return model_id


def resolve_proxy_module(model_id: str) -> Optional[str]:
    # Logic disabled until mods are ported
    # parts = model_id.split("-")
    # if parts[0] in mods.registry:
    #   return parts[0]
    return None


def resolve_request_config(body: Dict) -> Dict:
  model = body.get("model")
  messages = body.get("messages")
  params = {k: v for k, v in body.items() if k not in ["model", "messages"]}

  if not model:
    raise ValueError("Unable to proxy request without a model specifier")

  proxy_model = resolve_proxy_model(model)
  proxy_module = resolve_proxy_module(model)
  proxy_backend = MODEL_TO_BACKEND.get(proxy_model)

  logger.debug(
    f"Resolved proxy model: {proxy_model}, proxy module: {proxy_module}, proxy backend: {proxy_backend}"
  )

  if not proxy_backend:
    # If not found in cache, it might be that we haven't fetched models yet or it's invalid.
    # For now, if we can't find backend, we might fail or default to first one.
    # Current logic strictly requires backend map.

    # Try to re-populate cache if empty or missing?
    # list_downstream is cached.

    # Fallback: check if we can guess backend from config if only 1 exists
    urls = config.OPENGUARD_OPENAI_URLS.value
    if isinstance(urls, str): urls = [urls]

    if len(urls) == 1:
        proxy_backend = urls[0]
    else:
        # Check if we should raise HTTPException or ValueError based on contract
        # Standard behavior here seems to ideally be 400/404
        raise HTTPException(
        status_code=404,
        detail=f"Unknown model: '{model}'",
        )

  # Find key for backend
  urls = config.OPENGUARD_OPENAI_URLS.value
  keys = config.OPENGUARD_OPENAI_KEYS.value
  if isinstance(urls, str): urls = [urls]
  if isinstance(keys, str): keys = [keys]

  proxy_key = ""
  if proxy_backend in urls:
      idx = urls.index(proxy_backend)
      if idx < len(keys):
          proxy_key = keys[idx]

  proxy_config = {
    "url": proxy_backend,
    "headers":
      {
        "Content-Type": "application/json",
      },
    "model": proxy_model,
    "params": params,
    "messages": messages,
    "module": proxy_module,
  }

  if proxy_key:
      proxy_config["headers"]["Authorization"] = f"Bearer {proxy_key}"

  return proxy_config

def is_title_generation_task(llm: 'llm.LLM'):
  # TODO: Better way to identify?
  return llm.chat.has_substring("3-5 word title")

DIRECT_TASK_PROMPTS = [
    # Open WebUI prompts related to system tasks
  'Generate a concise, 3-5 word title',
  'Based on the chat history, determine whether a search is necessary',
  'Generate 1-3 broad tags categorizing',
  'You are an autocompletion system. Continue the text in `<text>` based on the **completion type**',
  'determine the necessity of generating search queries',
  # Custom for the test
  '[{DIRECT}]'
]

def is_direct_task(llm: 'llm.LLM'):
  return any(llm.chat.has_substring(prompt) for prompt in DIRECT_TASK_PROMPTS)
