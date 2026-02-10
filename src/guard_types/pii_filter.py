"""PII filtering guard - detects and replaces personally identifiable information."""

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM

# PII regex patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDITCARD_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

PII_PATTERNS = {
    "email": (EMAIL_PATTERN, "<protected:email>"),
    "phone": (PHONE_PATTERN, "<protected:phone>"),
    "ssn": (SSN_PATTERN, "<protected:ssn>"),
    "creditcard": (CREDITCARD_PATTERN, "<protected:creditcard>"),
}


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    Filter PII from message content.

    Args:
        chat: Chat object
        llm: LLM object
        config: Guard configuration

    Returns:
        List of audit logs
    """
    audit_logs = []
    pii_found = False

    for idx, node in enumerate(chat.plain()):
        content = node.content

        if content is None:
            continue

        # Handle string content
        if isinstance(content, str):
            modified_content = content

            for pii_type, (pattern, replacement) in PII_PATTERNS.items():
                matches = pattern.findall(modified_content)
                if matches:
                    for match in matches:
                        audit_logs.append(
                            f"pii_filter: Found {pii_type} in message {idx} ({node.role}): {match}"
                        )
                        pii_found = True
                    modified_content = pattern.sub(replacement, modified_content)

            if modified_content != content:
                node.content = modified_content

        # Handle array content (multimodal messages)
        elif isinstance(content, list):
            # content for list is mutable?
            # Usually chat node content stores the list object.

            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and "text" in part:
                    modified_text = part["text"]
                    part_modified = False

                    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
                        matches = pattern.findall(modified_text)
                        if matches:
                            for match in matches:
                                audit_logs.append(
                                    f"pii_filter: Found {pii_type} "
                                    f"in message {idx} ({node.role}), part {part_idx}: {match}"
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
