"""LLM-driven input inspection guard."""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src import log
from src.guards import GuardBlockedError

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


logger = log.setup_logger(__name__)

DEFAULT_MAX_CHARS = 8000
MAX_ALLOWED_CHARS = 50000
VIOLATION_ACTIONS = {"block", "log"}
ERROR_ACTIONS = {"allow", "block"}
INSPECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["allow", "block"]},
        "reason": {"type": "string"},
    },
    "required": ["decision"],
    "additionalProperties": False,
}


def apply(chat: "Chat", llm: "LLM", config: Dict[str, Any]) -> List[str]:
    """Inspect input text with an LLM and decide whether to allow or block."""
    prompt = str(config.get("prompt") or "").strip()
    if not prompt:
        logger.warning("llm_input_inspection: missing prompt, skipping")
        return []

    on_violation = _normalize_choice(config.get("on_violation"), VIOLATION_ACTIONS, "block")
    on_error = _normalize_choice(config.get("on_error"), ERROR_ACTIONS, "allow")
    max_chars = _normalize_max_chars(config.get("max_chars", DEFAULT_MAX_CHARS))
    inspector_model = config.get("inspector_model")
    inspect_roles = _normalize_inspect_roles(config.get("inspect_roles"))

    inspected_text = _collect_inspected_text(chat, max_chars=max_chars, inspect_roles=inspect_roles)
    if not inspected_text:
        return []

    try:
        decision, reason = _inspect_with_llm(
            llm=llm,
            instructions=prompt,
            inspected_text=inspected_text,
            inspector_model=inspector_model,
        )
    except Exception as exc:
        error_msg = f"llm_input_inspection: inspection unavailable/error ({exc})"
        logger.warning(error_msg)
        if on_error == "block":
            raise GuardBlockedError("Request blocked: llm_input_inspection failed")
        return [f"{error_msg}; allowed by on_error=allow"]

    if decision == "block":
        reason_text = reason or "policy violation detected"
        if on_violation == "block":
            raise GuardBlockedError(f"Request blocked by llm_input_inspection: {reason_text}")
        return [f"llm_input_inspection: violation detected; on_violation=log; reason={reason_text}"]

    return []


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value).strip().lower() if value is not None else default
    return normalized if normalized in allowed else default


def _normalize_max_chars(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_CHARS
    return max(1, min(parsed, MAX_ALLOWED_CHARS))


def _normalize_inspect_roles(value: Any) -> frozenset[str]:
    default = frozenset({"user"})
    if not isinstance(value, list):
        return default
    normalized = [str(r).strip().lower() for r in value if isinstance(r, str) and str(r).strip()]
    roles = frozenset(normalized)
    return roles or default


def _collect_inspected_text(
    chat: "Chat", max_chars: int, inspect_roles: frozenset[str] = frozenset({"user"})
) -> str:
    parts: List[str] = []

    for node in chat.plain():
        role = (getattr(node, "role", None) or "").strip().lower()
        if role not in inspect_roles:
            continue

        text = _extract_text(node.content)
        if text:
            parts.append(text)

    if not parts:
        return ""

    combined = "\n\n".join(parts).strip()
    if len(combined) <= max_chars:
        return combined

    # Keep tail — recent messages are more likely to contain attacks
    return combined[-max_chars:]


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") == "text" and isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
        return "\n".join(text_parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    if isinstance(content, (int, float, bool)):
        return str(content)

    return ""


def _inspect_with_llm(
    llm: "LLM", instructions: str, inspected_text: str, inspector_model: Any
) -> Tuple[str, str]:
    model = inspector_model or getattr(llm, "model", None)

    inspect_completion = getattr(llm, "inspect_completion", None)
    if not callable(inspect_completion) or not model:
        raise RuntimeError("missing llm inspection capability in current context")

    system_prompt = (
        "You are an input safety inspector. "
        "Return a strict JSON object: "
        '{"decision":"allow|block","reason":"short reason"}. '
        "No extra text."
    )

    user_prompt = (
        f"Inspection policy:\n{instructions}\n\n"
        f"Candidate input:\n{inspected_text}\n\n"
        "Decide if this input should be blocked."
    )

    try:
        output_text = inspect_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "InputInspectionDecision",
                    "schema": INSPECTION_SCHEMA,
                },
            },
        )
    except Exception:
        output_text = inspect_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            response_format=None,
        )

    return _parse_decision(output_text)


def _parse_decision(raw_text: str) -> Tuple[str, str]:
    if isinstance(raw_text, dict):
        decision, reason = _decision_from_object(raw_text)
        if decision is not None:
            return decision, reason
        raw_text = json.dumps(raw_text)

    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty inspector output")

    parsed = _try_parse_json(text)
    if parsed is not None:
        decision, reason = _decision_from_object(parsed)
        if decision is not None:
            return decision, reason

    direct = _decision_from_text(text)
    if direct:
        return direct

    return ("block", "ambiguous inspector output — defaulting to block")


def _try_parse_json(text: str) -> Any:
    candidate = text
    if "```" in candidate:
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    return None


def _decision_from_object(data: Any) -> Tuple[Optional[str], str]:
    if isinstance(data, str):
        decision = _normalize_decision_word(data)
        if decision:
            return decision, ""
        return None, ""

    if isinstance(data, bool):
        return ("block", "policy violation") if data else ("allow", "")

    if not isinstance(data, dict):
        return None, ""

    decision_fields = ["decision", "verdict", "action", "result"]
    for field in decision_fields:
        value = data.get(field)
        decision = _normalize_decision_word(value)
        if decision:
            reason = str(data.get("reason") or data.get("explanation") or "").strip()
            return decision, reason

    if isinstance(data.get("block"), bool):
        return ("block", "policy violation") if data["block"] else ("allow", "")

    if isinstance(data.get("allow"), bool):
        return ("allow", "") if data["allow"] else ("block", "policy violation")

    return None, ""


def _normalize_decision_word(value: Any) -> str | None:
    if value is None:
        return None

    token = str(value).strip().lower()

    if token in {
        "allow",
        "allowed",
        "pass",
        "safe",
        "ok",
        "approve",
        "approved",
        "permit",
        "permitted",
        "benign",
        "harmless",
        "clean",
        "legitimate",
        "acceptable",
        "accepted",
        "false",
    }:
        return "allow"

    if token in {
        "block",
        "blocked",
        "deny",
        "denied",
        "reject",
        "rejected",
        "unsafe",
        "violation",
        "forbidden",
        "harmful",
        "malicious",
        "dangerous",
        "suspicious",
        "flagged",
        "detected",
        "threat",
        "disallow",
        "disallowed",
        "refuse",
        "refused",
        "true",
    }:
        return "block"

    return None


def _decision_from_text(text: str) -> Tuple[str, str] | None:
    normalized = text.strip().lower()

    # Try direct synonym recognition for single-word responses
    word_decision = _normalize_decision_word(normalized)
    if word_decision:
        return word_decision, ""

    decision_match = re.search(
        r"\b(decision|verdict|action|result)\b\s*[:=\-]\s*(allow|block)",
        normalized,
    )
    if decision_match:
        return decision_match.group(2), ""

    if re.fullmatch(r"\s*(allow|block)\s*[.!]?\s*", normalized):
        return normalized.strip(" .!"), ""

    has_block = bool(re.search(r"\b(block|deny|reject|unsafe|violation)\b", normalized))
    has_allow = bool(re.search(r"\b(allow|safe|ok|pass|approved?)\b", normalized))

    if has_block and not has_allow:
        return "block", "policy violation"
    if has_allow and not has_block:
        return "allow", ""

    # Fail-safe: ambiguous response defaults to block
    if has_block and has_allow:
        return "block", "ambiguous inspector response"

    return None
