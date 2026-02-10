"""Max tokens guard - enforces token limits on requests."""

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    Enforce maximum token limit.

    Args:
        chat: Chat object
        llm: LLM object
        config: Guard configuration with 'max_tokens' value

    Returns:
        List of audit logs
    """
    max_tokens_limit = config.get("max_tokens")
    if max_tokens_limit is None:
        return []

    audit_logs = []

    # Check params in LLM
    # params is a dict
    current_max = llm.params.get("max_tokens")

    # Override if present and exceeds limit, or add if absent
    if current_max is None:
        llm.params["max_tokens"] = max_tokens_limit
        audit_logs.append(f"max_tokens: Enforced limit of {max_tokens_limit}")
    elif current_max > max_tokens_limit:
        llm.params["max_tokens"] = max_tokens_limit
        audit_logs.append(f"max_tokens: Enforced limit of {max_tokens_limit} (was {current_max})")

    return audit_logs
