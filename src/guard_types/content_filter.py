"""Content filtering guard - blocks specific words/phrases."""

import re
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


class Config(BaseModel):
    """Configuration for content filter."""

    blocked_words: List[str] = Field(
        default_factory=list,
        description="List of words or phrases to filter from content.",
    )


META = GuardMeta(
    name="content_filter",
    description="Filters blocked words from message content.",
    config_schema=Config,
    docs="""\
Replaces occurrences of configured blocked words with '[FILTERED]' in message content.
Supports both string content and multimodal list content.
""",
    examples=[
        {
            "name": "basic",
            "config": {
                "blocked_words": ["badword", "secret"],
            },
        }
    ],
)


def apply(chat: "Chat", llm: "LLM", config: Dict[str, Any]) -> List[str]:
    """
    Filter blocked words from message content.

    Args:
        chat: Chat object to filter
        llm: LLM object
        config: Guard configuration with 'blocked_words' list

    Returns:
        List of audit logs
    """
    cfg = Config.model_validate(config)
    blocked_words = cfg.blocked_words
    if not blocked_words:
        return []

    # Compile patterns once before iterating nodes
    compiled_patterns = [
        (word, re.compile(re.escape(word), re.IGNORECASE)) for word in blocked_words
    ]

    audit_logs: List[str] = []

    # Iterate over all messages in the chat
    # chat.plain() returns list of ChatNodes
    for idx, node in enumerate(chat.plain()):
        content = node.content

        if content is None:
            continue

        if isinstance(content, str):
            modified = False
            for word, pattern in compiled_patterns:
                # Case-insensitive search
                if word.lower() in content.lower():
                    content = pattern.sub("[FILTERED]", content)
                    modified = True
                    audit_logs.append(
                        f"content_filter: Replaced '{word}' in message {idx} ({node.role})"
                    )

            if modified:
                node.content = content

        elif isinstance(content, list):
            # Handle multimodal content (list of dicts)
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content = part.get("text", "")
                    modified = False

                    for word, pattern in compiled_patterns:
                        if word.lower() in text_content.lower():
                            text_content = pattern.sub("[FILTERED]", text_content)
                            modified = True
                            audit_logs.append(
                                f"content_filter: Replaced '{word}' in message {idx} "
                                f"({node.role}), part {part_idx}"
                            )

                    if modified:
                        part["text"] = text_content

    return audit_logs
