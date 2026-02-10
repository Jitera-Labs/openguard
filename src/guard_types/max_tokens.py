"""Max tokens guard - enforces token limits on requests."""

import copy
from typing import List, Tuple


def apply(payload: dict, config: dict) -> Tuple[dict, List[str]]:
    """
    Enforce maximum token limit.

    Args:
        payload: Request payload
        config: Guard configuration with 'max_tokens' value

    Returns:
        Tuple of (modified_payload, audit_logs)
    """
    max_tokens_limit = config.get("max_tokens")
    if max_tokens_limit is None:
        return payload, []

    # Create a deep copy to avoid modifying original
    modified_payload = copy.deepcopy(payload)
    audit_logs = []

    current_max = modified_payload.get("max_tokens")

    # Override if present and exceeds limit, or add if absent
    if current_max is None:
        modified_payload["max_tokens"] = max_tokens_limit
        audit_logs.append(f"max_tokens: Enforced limit of {max_tokens_limit}")
    elif current_max > max_tokens_limit:
        modified_payload["max_tokens"] = max_tokens_limit
        audit_logs.append(f"max_tokens: Enforced limit of {max_tokens_limit} (was {current_max})")

    return modified_payload, audit_logs
