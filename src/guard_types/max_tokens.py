"""Max tokens guard - enforces token limits on requests."""

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


class Config(BaseModel):
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens allowed for the LLM request. "
        "Overrides any existing limit if it is higher.",
    )


META = GuardMeta(
    name="max_tokens",
    description="Enforces maximum token limit on LLM requests.",
    config_schema=Config,
    docs="""
## Max Tokens Guard

This guard ensures that the `max_tokens` parameter sent to the LLM does not exceed
the configured limit. If the parameter is absent, it adds the limit.
If it's present and exceeds the limit, it overwrites it.

### Configuration
- `max_tokens` (int, optional): The maximum allowed tokens.
""",
    examples=[
        {
            "max_tokens": 1000,
        }
    ],
)


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
    cfg = Config.model_validate(config)

    max_tokens_limit = cfg.max_tokens
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
