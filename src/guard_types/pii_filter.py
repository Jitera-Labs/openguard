"""PII filtering guard - detects and replaces personally identifiable information."""

import re
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


# PII regex patterns — tightened to reduce false positives
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
# Phone: requires separator between groups to reduce false positives on plain digit sequences
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?" r"(?:\(\d{3}\)|\d{3})" r"[-.\s]\d{3}[-.\s]\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Credit card: requires separators between all groups to reduce false positives
CREDITCARD_PATTERN = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b")

PII_PATTERNS = {
    "email": (EMAIL_PATTERN, "<protected:email>"),
    "phone": (PHONE_PATTERN, "<protected:phone>"),
    "ssn": (SSN_PATTERN, "<protected:ssn>"),
    "creditcard": (CREDITCARD_PATTERN, "<protected:creditcard>"),
}


class Config(BaseModel):
    pii_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of PII types to filter (e.g. ['email', 'phone']). "
            "If omitted, all detectors run."
        ),
    )


def _get_active_patterns(cfg: Config) -> dict:
    """Return only the PII patterns that are enabled by config."""
    if cfg.pii_types is None:
        # Default: run all detectors (backward compatible)
        return PII_PATTERNS
    return {k: v for k, v in PII_PATTERNS.items() if k in cfg.pii_types}


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    Filter PII from message content.

    Args:
        chat: Chat object
        llm: LLM object
        config: Guard configuration. Optional 'pii_types' list to select specific
                detectors (e.g. ["email", "phone"]). If omitted, all detectors run.

    Returns:
        List of audit logs
    """
    cfg = Config.model_validate(config)
    active_patterns = _get_active_patterns(cfg)
    if not active_patterns:
        return []

    if hasattr(llm, "stream_patterns"):
        for pii_type, (pattern, replacement) in active_patterns.items():
            llm.stream_patterns.append((pattern, replacement))

    audit_logs = []
    pii_found = False

    for idx, node in enumerate(chat.plain()):
        content = node.content

        if content is None:
            continue

        # Handle string content
        if isinstance(content, str):
            modified_content = content

            for pii_type, (pattern, replacement) in active_patterns.items():
                matches = pattern.findall(modified_content)
                if matches:
                    audit_logs.append(
                        f"pii_filter: Found {pii_type} in message {idx}"
                        f" ({node.role}): {len(matches)} match(es)"
                    )
                    pii_found = True
                    modified_content = pattern.sub(replacement, modified_content)

            if modified_content != content:
                node.content = modified_content

        # Handle array content (multimodal messages)
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and "text" in part:
                    modified_text = part["text"]
                    part_modified = False

                    for pii_type, (pattern, replacement) in active_patterns.items():
                        matches = pattern.findall(modified_text)
                        if matches:
                            audit_logs.append(
                                f"pii_filter: Found {pii_type} "
                                f"in message {idx} ({node.role}),"
                                f" part {part_idx}: {len(matches)} match(es)"
                            )
                            pii_found = True
                            modified_text = pattern.sub(replacement, modified_text)
                            part_modified = True

                    if part_modified:
                        part["text"] = modified_text

    # If any PII was found, inject a system message at the start
    if pii_found:
        chat.system(
            "Note: PII has been filtered from the following conversation. "
            "Protected items appear as <protected:type>."
        )

    return audit_logs


META = GuardMeta(
    name="pii_filter",
    description="Detects and replaces personally identifiable information in message content.",
    config_schema=Config,
    docs=(
        "The `pii_filter` guard scans the chat context for Personally Identifiable\n"
        "Information (PII) such as email addresses, phone numbers, SSNs, and credit card\n"
        "numbers.\n"
        "When PII is found, it is replaced with a masked value like `<protected:email>`\n"
        "before being sent to the LLM.\n"
        "You can configure which types of PII to filter by passing an optional `pii_types`\n"
        'list containing values like `"email"`, `"phone"`, `"ssn"`, or `"creditcard"`.\n'
        "If `pii_types` is omitted, all available PII detectors will run."
    ),
    examples=[
        {"pii_types": ["email", "phone"]},
        {},
    ],
)
