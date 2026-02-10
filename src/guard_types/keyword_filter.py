"""Keyword filter guard implementation."""

import copy
import re
from typing import Any, List, Tuple, TYPE_CHECKING

from src.guards import GuardBlockedError

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    Apply keyword based filtering/blocking/logging.

    Args:
        chat: Chat object
        llm: LLM object
        config: Guard configuration

    Returns:
        List of audit logs
    """
    keywords = config.get("keywords", [])
    if not keywords:
        return []

    match_mode = config.get("match_mode", "any")  # "any" or "all"
    case_sensitive = config.get("case_sensitive", False)
    action = config.get("action", "block")  # "block", "sanitize", "log"
    replacement = config.get("replacement", "[REDACTED]")

    audit_logs = []

    flags = 0 if case_sensitive else re.IGNORECASE

    def get_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        return str(content)

    def check_text(text: str, keyword: str) -> bool:
        pattern = re.escape(keyword)
        return bool(re.search(pattern, text, flags))

    # Identify which keywords are present in the payload
    present_keywords = set()

    for node in chat.plain():
        content = node.content
        text = get_text_content(content)
        for keyword in keywords:
            if check_text(text, keyword):
                present_keywords.add(keyword)

    # Determine if the guard should be triggered
    triggered = False
    trigger_reason = ""

    if match_mode == "any":
        if present_keywords:
            triggered = True
            trigger_reason = f"found keyword '{list(present_keywords)[0]}'"
    elif match_mode == "all":
        if set(keywords).issubset(present_keywords):
            triggered = True
            trigger_reason = "found all required keywords"

    if not triggered:
        return []

    # Execute Action
    if action == "block":
        if match_mode == "any":
            first_kw = list(present_keywords)[0]
            raise GuardBlockedError(f"Request blocked: found keyword '{first_kw}'")
        else:
            raise GuardBlockedError(f"Request blocked: {trigger_reason}")

    elif action == "log":
        audit_logs.append(f"Keyword filter triggered: {trigger_reason}")
        return audit_logs

    elif action == "sanitize":
        audit_logs.append(f"Sanitized keywords: {', '.join(present_keywords)}")

        def replace_in_text(text_val: str) -> str:
            new_text = text_val
            for kw in keywords:
                # If match_mode is 'any', we replace any found kw.
                # If match_mode is 'all', we replace all kw
                if match_mode == "any" or (match_mode == "all" and set(keywords).issubset(present_keywords)):
                    # Only replace if kw is in present_keywords?
                    # logic: sanitization should probably remove ALL keywords if trigger condition is met?
                    # Or only the ones found?
                    # If match_mode=all and we found all, we likely want to remove all.
                    if kw in present_keywords:
                         pattern = re.compile(re.escape(kw), flags)
                         new_text = pattern.sub(replacement, new_text)
            return new_text

        for node in chat.plain():
            content = node.content
            if isinstance(content, str):
                new_content = replace_in_text(content)
                if new_content != content:
                    node.content = new_content
            elif isinstance(content, list):
                 for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                         part["text"] = replace_in_text(part["text"])

    return audit_logs

