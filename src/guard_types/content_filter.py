"""Content filtering guard - blocks specific words/phrases."""
from typing import Dict, List, Tuple, Any
import copy


def apply(payload: dict, config: dict) -> Tuple[dict, List[str]]:
    """
    Filter blocked words from message content.

    Args:
        payload: Request payload with messages
        config: Guard configuration with 'blocked_words' list

    Returns:
        Tuple of (modified_payload, audit_logs)
    """
    blocked_words = config.get('blocked_words', [])
    if not blocked_words:
        return payload, []

    # Create a deep copy to avoid modifying original
    modified_payload = copy.deepcopy(payload)
    audit_logs = []

    messages = modified_payload.get('messages', [])

    for idx, message in enumerate(messages):
        content = message.get('content')

        if content is None:
            continue

        # Handle string content
        if isinstance(content, str):
            original_content = content
            modified = False

            for word in blocked_words:
                # Case-insensitive search
                if word.lower() in content.lower():
                    # Case-insensitive replace
                    import re
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    content = pattern.sub('[FILTERED]', content)
                    modified = True
                    audit_logs.append(f"content_filter: Replaced '{word}' in message {idx}")

            if modified:
                message['content'] = content

        # Handle array content (multimodal messages)
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and 'text' in part:
                    original_text = part['text']
                    text = original_text
                    modified = False

                    for word in blocked_words:
                        if word.lower() in text.lower():
                            import re
                            pattern = re.compile(re.escape(word), re.IGNORECASE)
                            text = pattern.sub('[FILTERED]', text)
                            modified = True
                            audit_logs.append(f"content_filter: Replaced '{word}' in message {idx}, part {part_idx}")

                    if modified:
                        part['text'] = text

    return modified_payload, audit_logs
