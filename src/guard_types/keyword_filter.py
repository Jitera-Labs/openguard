"""Keyword filter guard implementation."""
from typing import Dict, List, Tuple, Any, Optional
import copy
import re
from src.guards import GuardBlockedError

def apply(payload: dict, config: dict) -> Tuple[dict, List[str]]:
    """
    Apply keyword based filtering/blocking/logging.

    Args:
        payload: Request payload
        config: Guard configuration

    Returns:
        Tuple of (modified_payload, audit_logs)
    """
    keywords = config.get("keywords", [])
    if not keywords:
        return payload, []

    match_mode = config.get("match_mode", "any")  # "any" or "all"
    case_sensitive = config.get("case_sensitive", False)
    action = config.get("action", "block")  # "block", "sanitize", "log"
    replacement = config.get("replacement", "[REDACTED]")

    modified_payload = copy.deepcopy(payload)
    messages = modified_payload.get("messages", [])
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
    
    for message in messages:
        content = message.get("content", "")
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
        return payload, []
        
    # Execute Action
    if action == "block":
        if match_mode == "any":
            # Just report the first one we found to match format in requirements
            # "Request blocked: found keyword '{word}'"
            # Since present_keywords is a set, order isn't guaranteed, but any is fine.
            # We already set trigger_reason accordingly for 'any', but let's be precise.
            first_kw = list(present_keywords)[0]
            raise GuardBlockedError(f"Request blocked: found keyword '{first_kw}'")
        else:
             raise GuardBlockedError(f"Request blocked: {trigger_reason}")

    elif action == "log":
        audit_logs.append(f"Keyword filter triggered: {trigger_reason}")
        return payload, audit_logs
        
    elif action == "sanitize":
        audit_logs.append(f"Sanitized keywords: {', '.join(present_keywords)}")
        
        def replace_in_text(text_val: str) -> str:
            new_text = text_val
            for kw in keywords:
                # If match_mode is 'any', we replace any found kw.
                # If match_mode is 'all', we replace all kw (since we are triggered, all are present).
                # Wait, if match_mode is 'all', should we replace ONLY if all are present? Yes, `triggered` check ensures that.
                # But should we replace only the keywords in the list? Yes.
                # Should we check if kw is in present_keywords? 
                # For 'all', yes they are all there. 
                # For 'any', yes we should only replace present ones? 
                # Actually, replace logic can just run regex sub on all keywords. 
                # If a keyword is not there, it won't be replaced.
                # But to avoid unnecessary regex ops, we can filter by present_keywords if we want.
                # But just iterating keywords is safer if there are multiple occurrences.
                pattern = re.escape(kw)
                new_text = re.sub(pattern, replacement, new_text, flags=flags)
            return new_text

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                message["content"] = replace_in_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = replace_in_text(part.get("text", ""))
        
        return modified_payload, audit_logs

    return payload, []
