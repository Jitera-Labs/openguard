"""Programmatic tool definition filter using Hasura-style matching."""

from typing import TYPE_CHECKING, Any, Dict, List, Literal

from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta
from src.guards import GuardBlockedError
from src.selection import match_filter

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


class Config(BaseModel):
    filter: Dict[str, Any] = Field(
        description="Hasura-style filter applied to each normalized tool definition.",
    )
    scope: Literal["any", "every", "none"] = Field(
        default="any",
        description=(
            "'any': trigger if any tool matches. "
            "'every': trigger if all tools match. "
            "'none': trigger if no tools match (useful for requiring a tool)."
        ),
    )
    action: Literal["block", "strip", "log"] = Field(
        default="block",
        description=(
            "'block': reject the request. "
            "'strip': remove matching tools from the request. "
            "'log': log the finding without blocking."
        ),
    )


META = GuardMeta(
    name="tool_filter",
    description="Filter tool definitions using Hasura-style matching rules.",
    config_schema=Config,
    docs=(
        "Evaluates each tool definition in the request against a Hasura-style filter.\n"
        "Tool definitions are normalized to a flat shape before filtering:\n"
        '`{"name": "...", "description": "...", "parameters": {...}, "type": "..."}`\n'
        "This works identically for OpenAI and Anthropic tool formats.\n\n"
        "Supports three actions: block (reject request), strip (remove matching tools), "
        "or log (audit only)."
    ),
    examples=[
        {
            "name": "Block tools with suspicious descriptions",
            "config": {
                "filter": {
                    "description": {
                        "_iregex": "ignore previous|disregard|system prompt",
                    },
                },
                "action": "block",
            },
        },
        {
            "name": "Strip shell execution tools",
            "config": {
                "filter": {"name": {"_ilike": "%execute%"}},
                "action": "strip",
            },
        },
    ],
)


def _normalize_tool(tool: Any, provider: str) -> dict:
    """Normalize a provider-specific tool definition to a flat dict for filtering."""
    if not isinstance(tool, dict):
        return {"name": "", "description": "", "parameters": {}, "type": "unknown"}

    if provider == "anthropic":
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
            "type": tool.get("type", "custom"),
        }

    fn = tool.get("function")
    if isinstance(fn, dict):
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
            "type": tool.get("type", "function"),
        }

    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", tool.get("input_schema", {})),
        "type": tool.get("type", "function"),
    }


def apply(chat: "Chat", llm: "LLM", config: Dict[str, Any]) -> List[str]:
    """Filter tool definitions using Hasura-style matching."""
    cfg = Config.model_validate(config)

    raw_tools = llm.params.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        return []

    provider = getattr(llm, "provider", "openai")
    pairs = [(_normalize_tool(tool, provider), tool) for tool in raw_tools]
    matching_indices = {i for i, (norm, _) in enumerate(pairs) if match_filter(norm, cfg.filter)}

    triggered = False
    if cfg.scope == "any":
        triggered = len(matching_indices) > 0
    elif cfg.scope == "every":
        triggered = len(matching_indices) == len(pairs)
    elif cfg.scope == "none":
        triggered = len(matching_indices) == 0

    if not triggered:
        return []

    matched_names = [pairs[i][0].get("name", "?") for i in sorted(matching_indices)]

    if cfg.action == "block":
        raise GuardBlockedError(f"Request blocked by tool_filter: matched tools {matched_names}")

    if cfg.action == "strip":
        llm.params["tools"] = [
            original for i, (_, original) in enumerate(pairs) if i not in matching_indices
        ]
        return [f"tool_filter: stripped tools {matched_names}"]

    return [f"tool_filter: matched tools {matched_names}"]
