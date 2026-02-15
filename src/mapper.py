from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import httpx
from asyncache import cached
from cachetools import TTLCache
from fastapi import HTTPException

from . import config, log

if TYPE_CHECKING:
    from . import llm

logger = log.setup_logger(__name__)

MODEL_TO_BACKEND: Dict[str, Dict[str, str]] = {"openai": {}, "anthropic": {}}

# TODO: Add mods (modules) support when adoption is complete
# import mods


@cached(TTLCache(1024, 60))
async def list_downstream(provider: str = "openai"):
    logger.debug(f"Listing downstream models for provider '{provider}'")

    all_models = []

    for backend in get_provider_backends(provider):
        url = backend["url"]
        key = backend["key"]
        try:
            endpoint = build_provider_endpoint(url, provider, "/models")
            headers = {
                "Content-Type": "application/json",
            }
            if key:
                if provider == "anthropic":
                    headers["x-api-key"] = key
                else:
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
                    model_id = model.get("id")
                    if model_id:
                        MODEL_TO_BACKEND.setdefault(provider, {})[model_id] = url

        except Exception as e:
            logger.error(f"Failed to fetch models from {endpoint}: {e}")

    return all_models


def _as_list(value):
    if isinstance(value, str):
        return [value]
    return list(value or [])


def get_provider_backends(provider: str):
    provider_normalized = provider.lower()
    if provider_normalized == "anthropic":
        urls = _as_list(config.OPENGUARD_ANTHROPIC_URLS.value)
        keys = _as_list(config.OPENGUARD_ANTHROPIC_KEYS.value)
    else:
        urls = _as_list(config.OPENGUARD_OPENAI_URLS.value)
        keys = _as_list(config.OPENGUARD_OPENAI_KEYS.value)

    if len(keys) < len(urls):
        keys.extend([""] * (len(urls) - len(keys)))

    backends = []
    for index, url in enumerate(urls):
        if not url:
            continue
        backends.append({"provider": provider_normalized, "url": url, "key": keys[index]})

    return backends


def get_internal_api_keys():
    api_key = config.OPENGUARD_API_KEY.value
    api_keys_list = _as_list(config.OPENGUARD_API_KEYS.value)

    keys = []
    if api_key:
        keys.append(api_key)
    keys.extend(k for k in api_keys_list if k)
    return keys


def extract_api_key(headers: Optional[Mapping[str, str]]):
    if not headers:
        return ""

    x_api_key = headers.get("x-api-key") or headers.get("X-API-Key")
    if x_api_key:
        return x_api_key

    auth_header = headers.get("authorization") or headers.get("Authorization") or ""
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return ""


def build_provider_endpoint(base_url: str, provider: str, path: str):
    base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"

    if provider.lower() == "anthropic":
        if normalized_path.startswith("/v1/"):
            return f"{base}{normalized_path}"
        if base.endswith("/v1"):
            return f"{base}{normalized_path}"
        return f"{base}/v1{normalized_path}"

    return f"{base}{normalized_path}"


def _resolve_backend_for_model(provider: str, model: Optional[str]):
    if model:
        cached = MODEL_TO_BACKEND.get(provider, {}).get(model)
        if cached:
            for backend in get_provider_backends(provider):
                if backend["url"] == cached:
                    return backend

    backends = get_provider_backends(provider)
    if len(backends) == 1:
        return backends[0]

    return None


def resolve_provider_route(provider: str, endpoint_path: str, body: Optional[Dict], headers):
    provider_normalized = provider.lower()
    backends = get_provider_backends(provider_normalized)
    if not backends:
        raise ValueError(f"No downstream backends configured for provider '{provider_normalized}'")

    incoming_key = extract_api_key(headers)
    internal_keys = get_internal_api_keys()

    if incoming_key and incoming_key in internal_keys:
        internal_backend = _resolve_backend_for_model(
            provider_normalized, (body or {}).get("model")
        )
        if internal_backend is None:
            internal_backend = backends[0]

        return {
            **internal_backend,
            "mode": "internal",
            "endpoint": build_provider_endpoint(
                internal_backend["url"], provider_normalized, endpoint_path
            ),
        }

    if incoming_key:
        for backend in backends:
            if backend["key"] and backend["key"] == incoming_key:
                return {
                    **backend,
                    "mode": "direct",
                    "endpoint": build_provider_endpoint(
                        backend["url"], provider_normalized, endpoint_path
                    ),
                }

    backend = backends[0]
    return {
        **backend,
        "mode": "endpoint",
        "endpoint": build_provider_endpoint(backend["url"], provider_normalized, endpoint_path),
    }


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
    proxy_backend = MODEL_TO_BACKEND.get("openai", {}).get(proxy_model)

    logger.debug(
        f"Resolved proxy model: {proxy_model}, proxy module: {proxy_module}, "
        f"proxy backend: {proxy_backend}"
    )

    if not proxy_backend:
        # If not found in cache, it might be that we haven't fetched models yet or it's invalid.
        # For now, if we can't find backend, we might fail or default to first one.
        # Current logic strictly requires backend map.

        # Try to re-populate cache if empty or missing?
        # list_downstream is cached.

        # Fallback: check if we can guess backend from config if only 1 exists
        urls = _as_list(config.OPENGUARD_OPENAI_URLS.value)

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
    urls = _as_list(config.OPENGUARD_OPENAI_URLS.value)
    keys = _as_list(config.OPENGUARD_OPENAI_KEYS.value)

    proxy_key = ""
    if proxy_backend in urls:
        idx = urls.index(proxy_backend)
        if idx < len(keys):
            proxy_key = keys[idx]

    proxy_config: Dict[str, Any] = {
        "url": proxy_backend,
        "headers": {
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


def is_title_generation_task(llm: "llm.LLM"):
    # TODO: Better way to identify?
    return llm.chat.has_substring("3-5 word title")


DIRECT_TASK_PROMPTS = [
    # Open WebUI prompts related to system tasks
    "Generate a concise, 3-5 word title",
    "Based on the chat history, determine whether a search is necessary",
    "Generate 1-3 broad tags categorizing",
    "You are an autocompletion system. Continue the text in `<text>` based on the "
    "**completion type**",
    "determine the necessity of generating search queries",
    # Custom for the test
    "[{DIRECT}]",
]


def is_direct_task(llm: "llm.LLM"):
    return any(llm.chat.has_substring(prompt) for prompt in DIRECT_TASK_PROMPTS)
