"""PII filtering guard - detects and replaces personally identifiable information."""

import copy
import re
from typing import List, Tuple

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


def apply(payload: dict, config: dict) -> Tuple[dict, List[str]]:
    """
    Filter PII from message content.

    Args:
        payload: Request payload with messages
        config: Guard configuration

    Returns:
        Tuple of (modified_payload, audit_logs)
    """
    # Create a deep copy to avoid modifying original
    modified_payload = copy.deepcopy(payload)
    audit_logs = []
    pii_found = False

    messages = modified_payload.get("messages", [])

    for idx, message in enumerate(messages):
        content = message.get("content")

        if content is None:
            continue

        # Handle string content
        if isinstance(content, str):
            modified_content = content

            for pii_type, (pattern, replacement) in PII_PATTERNS.items():
                matches = pattern.findall(modified_content)
                if matches:
                    for match in matches:
                        audit_logs.append(f"pii_filter: Found {pii_type} in message {idx}: {match}")
                        pii_found = True
                    modified_content = pattern.sub(replacement, modified_content)

            message["content"] = modified_content

        # Handle array content (multimodal messages)
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and "text" in part:
                    modified_text = part["text"]

                    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
                        matches = pattern.findall(modified_text)
                        if matches:
                            for match in matches:
                                audit_logs.append(
                                    f"pii_filter: Found {pii_type} "
                                    f"in message {idx}, part {part_idx}: {match}"
                                )
                                pii_found = True
                            modified_text = pattern.sub(replacement, modified_text)

                    part["text"] = modified_text

    # If any PII was found, inject a system message at the start
    if pii_found:
        system_message = {
            "role": "system",
            "content": (
                "Note: PII has been filtered from the following conversation. "
                "Protected items appear as <protected:type>."
            ),
        }
        modified_payload["messages"].insert(0, system_message)

    return modified_payload, audit_logs
