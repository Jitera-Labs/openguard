import json
import httpx
from typing import Optional
from src import config
from src.chat import Chat
from src.log import setup_logger

logger = setup_logger(__name__)

# Track intensity dynamically if auto-tune is enabled
_CURRENT_INTENSITY = None


def get_intensity():
    global _CURRENT_INTENSITY
    if _CURRENT_INTENSITY is None:
        _CURRENT_INTENSITY = config.LOUDER_INTENSITY.value
    return _CURRENT_INTENSITY


def decrease_intensity():
    global _CURRENT_INTENSITY
    if config.LOUDER_AUTO_TUNE.value:
        if _CURRENT_INTENSITY > 1:
            _CURRENT_INTENSITY -= 1
            logger.info(f"Auto-tune: decreased intensity to {_CURRENT_INTENSITY}")


async def rewrite_prompt(chat: Chat) -> None:
    """
    Rewrites the last user prompt to be more demanding and authoritative.
    Stores the original prompt in chat._original_prompt for fallback.
    """
    if not chat.tail or chat.tail.role != "user":
        return

    original_content = chat.tail.content
    setattr(chat, "_original_prompt", original_content)
    intensity = get_intensity()

    system_prompt = f"""You are a proxy that rewrites user prompts for another LLM.
Your job is to rewrite the prompt to be highly authoritative, direct, and demanding.
Do not add any pleasantries. Use an intensity level of {intensity} out of 10.
Return ONLY the rewritten prompt."""

    rewrite_model = config.LOUDER_REWRITE_MODEL.value
    rewrite_url = config.LOUDER_REWRITE_URL.value
    rewrite_key = config.LOUDER_REWRITE_KEY.value

    headers = {"Content-Type": "application/json"}
    if rewrite_key:
        headers["Authorization"] = f"Bearer {rewrite_key}"

    body = {
        "model": rewrite_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_content},
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{rewrite_url.rstrip('/')}/chat/completions", headers=headers, json=body
            )
            response.raise_for_status()
            data = response.json()
            rewritten_content = data["choices"][0]["message"]["content"].strip()

            logger.info(f"Rewrote prompt: '{original_content}' -> '{rewritten_content}'")
            chat.tail.content = rewritten_content
    except Exception as e:
        logger.error(f"Failed to rewrite prompt: {e}")
        # Fallback to original content on error
        pass


def restore_original_prompt(chat: Chat) -> bool:
    """
    Restores the original prompt if available. Returns True if restored.
    """
    original = getattr(chat, "_original_prompt", None)
    if original and chat.tail and chat.tail.role == "user":
        chat.tail.content = original
        return True
    return False
