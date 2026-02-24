"""Keyword filter guard implementation."""

import re
from typing import TYPE_CHECKING, Any, List

from pydantic import BaseModel, Field

from src import log
from src.guard_meta import GuardMeta
from src.guards import GuardBlockedError

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM

logger = log.setup_logger(__name__)


class Config(BaseModel):
    keywords: List[str] = Field(
        default_factory=list,
        description="List of keywords or regular expressions to match.",
    )
    match_mode: str = Field(
        default="any",
        description="Whether to trigger on 'any' matched keyword or require 'all' to match.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the matching should be case-sensitive.",
    )
    action: str = Field(
        default="block",
        description="Action to take when triggered ('block', 'sanitize', 'log').",
    )
    replacement: str = Field(
        default="[REDACTED]",
        description="Text to replace matched keywords with when action is 'sanitize'.",
    )
    use_regex: bool = Field(
        default=False,
        description="Whether the keywords should be treated as regular expressions.",
    )


META = GuardMeta(
    name="Keyword Filter",
    description=(
        "Matches specific keywords or regex patterns in user input and "
        "performs an action (block, sanitize, log)."
    ),
    config_schema=Config,
    docs=(
        "The Keyword Filter guard scans messages for specific words, phrases, "
        "or regular expressions.\n"
        "It supports three modes:\n"
        "- **block**: Rejects the request if a keyword is found.\n"
        "- **sanitize**: Replaces found keywords with the configured replacement text.\n"
        "- **log**: Only logs the finding without altering the request or blocking it.\n"
        "\n"
        "Matching can be case-sensitive or insensitive, and you can require matching "
        "'all' keywords instead of 'any'."
    ),
    examples=[
        {
            "name": "Block Specific Words",
            "config": {
                "keywords": ["secret", "confidential"],
                "match_mode": "any",
                "action": "block",
            },
        },
        {
            "name": "Sanitize API Keys",
            "config": {
                "keywords": ["sk-[a-zA-Z0-9]{48}"],
                "use_regex": True,
                "action": "sanitize",
                "replacement": "[API_KEY_REDACTED]",
            },
        },
    ],
)


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
    cfg = Config.model_validate(config)

    if not cfg.keywords:
        return []

    audit_logs = []

    flags = 0 if cfg.case_sensitive else re.IGNORECASE

    if hasattr(llm, "stream_patterns"):
        for kw in cfg.keywords:
            raw_pattern = kw if cfg.use_regex else re.escape(kw)
            try:
                compiled = re.compile(raw_pattern, flags)
                if cfg.action == "sanitize":
                    llm.stream_patterns.append((compiled, cfg.replacement))
                elif cfg.action == "block":
                    if not hasattr(llm, "stream_blocks"):
                        llm.stream_blocks = []
                    llm.stream_blocks.append((compiled, kw))
            except re.error:
                continue

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
        if cfg.use_regex:
            try:
                compiled = re.compile(keyword, flags)
            except re.error:
                logger.warning("Invalid regex pattern skipped: %r", keyword)
                return False
            return bool(compiled.search(text))
        else:
            return bool(re.search(re.escape(keyword), text, flags))

    # Identify which keywords are present in the payload
    present_keywords = set()

    for node in chat.plain():
        content = node.content
        text = get_text_content(content)
        for keyword in cfg.keywords:
            if check_text(text, keyword):
                present_keywords.add(keyword)

    # Determine if the guard should be triggered
    triggered = False
    trigger_reason = ""

    if cfg.match_mode == "any":
        if present_keywords:
            triggered = True
            trigger_reason = f"found keyword '{list(present_keywords)[0]}'"
    elif cfg.match_mode == "all":
        if set(cfg.keywords).issubset(present_keywords):
            triggered = True
            trigger_reason = "found all required keywords"

    if not triggered:
        return []

    # Execute Action
    if cfg.action == "block":
        if cfg.match_mode == "any":
            first_kw = list(present_keywords)[0]
            raise GuardBlockedError(f"Request blocked: found keyword '{first_kw}'")
        else:
            raise GuardBlockedError(f"Request blocked: {trigger_reason}")

    elif cfg.action == "log":
        audit_logs.append(f"Keyword filter triggered: {trigger_reason}")
        return audit_logs

    elif cfg.action == "sanitize":
        audit_logs.append(f"Sanitized keywords: {', '.join(present_keywords)}")

        def replace_in_text(text_val: str) -> str:
            new_text = text_val
            for kw in cfg.keywords:
                # If match_mode is 'any', we replace any found kw.
                # If match_mode is 'all', we replace all kw
                all_found = cfg.match_mode == "all" and set(cfg.keywords).issubset(present_keywords)
                if cfg.match_mode == "any" or all_found:
                    # Only replace if kw is in present_keywords?
                    # logic: sanitization should probably remove ALL keywords
                    # if trigger condition is met?
                    # Or only the ones found?
                    # If match_mode=all and we found all, we likely want to remove all.
                    if kw in present_keywords:
                        raw_pattern = kw if cfg.use_regex else re.escape(kw)
                        try:
                            compiled = re.compile(raw_pattern, flags)
                        except re.error:
                            logger.warning("Invalid regex pattern skipped: %r", kw)
                            continue
                        new_text = compiled.sub(cfg.replacement, new_text)
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
