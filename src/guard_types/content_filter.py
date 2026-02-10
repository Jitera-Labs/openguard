"""Content filtering guard - blocks specific words/phrases."""

import re
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


def apply(chat: "Chat", llm: "LLM", config: Dict) -> List[str]:
    """
    Filter blocked words from message content.

    Args:
        chat: Chat object to filter
        llm: LLM object
        config: Guard configuration with 'blocked_words' list

    Returns:
        List of audit logs
    """
    blocked_words = config.get("blocked_words", [])
    if not blocked_words:
        return []

    audit_logs = []

    # Iterate over all messages in the chat
    # chat.plain() returns list of ChatNodes
    for idx, node in enumerate(chat.plain()):
        content = node.content

        if content is None:
            continue

        if isinstance(content, str):
            modified = False
            for word in blocked_words:
                # Case-insensitive search
                if word.lower() in content.lower():
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    content = pattern.sub("[FILTERED]", content)
                    modified = True
                    audit_logs.append(
                        f"content_filter: Replaced '{word}' in message {idx} ({node.role})"
                    )

            if modified:
                node.content = content

        elif isinstance(content, list):
            # Multimodal content
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and "text" in part:
                    original_text = part["text"]
                    text = original_text
                    modified = False

                    for word in blocked_words:
                        if word.lower() in text.lower():
                            pattern = re.compile(re.escape(word), re.IGNORECASE)
                            text = pattern.sub("[FILTERED]", text)
                            modified = True
                            audit_logs.append(
                                f"content_filter: Replaced '{word}' "
                                f"in message {idx}, part {part_idx} ({node.role})"
                            )

                    if modified:
                        part["text"] = text

    return audit_logs
