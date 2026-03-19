"""PII detection guard — structured PII redaction via regex and libphonenumber.

Phone numbers: phonenumbers.PhoneNumberMatcher (Google libphonenumber port) —
covers 250+ country formats.
Structured identity/financial PII: validated regex (Luhn-10 for credit cards,
ISO 13616 mod-97 for IBAN, octet-range check for IPv4).
Context awareness: high-FP types (IPv4, IBAN, DoB, EIN) are only activated
when a context keyword appears within 120 characters of the match.
Default action: redact (replace with <protected:TYPE>).

For credential detection (API keys, tokens, private keys), use credential_filter.
"""

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import phonenumbers
import phonenumbers.phonenumberutil
from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


# ─── Text normalisation ───────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """NFKC-normalise and strip invisible/bidirectional Unicode."""
    text = unicodedata.normalize("NFKC", text)
    # zero-width, soft-hyphen, BOM, bidirectional overrides
    text = re.sub(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", text)
    return text


_SPACED_EMAIL = re.compile(
    r"(?<!\S)(?:[A-Za-z0-9._%+\-] ){1,40}@ (?:[A-Za-z0-9.\-] ){1,40}[A-Za-z]{2,}(?=\s|$)",
    re.IGNORECASE,
)


def _despace_emails(text: str) -> str:
    """Collapse 'j o h n @ e x a m p l e . c o m' obfuscation before matching."""
    return _SPACED_EMAIL.sub(lambda m: m.group(0).replace(" ", ""), text)


# ─── Replacement helpers ──────────────────────────────────────────────────────


def _make_replacement(pii_type: str, action: str, raw_value: str) -> str:
    if action == "mask":
        return "****"
    if action == "hash":
        digest = hashlib.sha256(raw_value.encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"<protected:{pii_type}:{digest}>"
    return f"<protected:{pii_type}>"


# ─── Algorithmic validators ───────────────────────────────────────────────────


def _luhn_valid(raw: str) -> bool:
    """Luhn-10 checksum — rejects random digit sequences from CC matches."""
    digits = [int(c) for c in raw if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_valid(raw: str) -> bool:
    """ISO 13616 mod-97 — rejects random alphanumeric strings."""
    s = raw.replace(" ", "").replace("-", "").upper()
    if not (15 <= len(s) <= 34) or not s[:2].isalpha() or not s[2:4].isdigit():
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _ipv4_valid(raw: str) -> bool:
    """Octet-range validation — rejects version strings and other dotted numbers."""
    parts = raw.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    return all(0 <= o <= 255 for o in octets) and octets not in ([0, 0, 0, 0], [255, 255, 255, 255])


# ─── Tier 2: PII detection (regex + phonenumbers) ────────────────────────────


_SENSITIVITY_RANK: Dict[str, int] = {"relaxed": 0, "balanced": 1, "strict": 2}


@dataclass
class _PiiDetector:
    pii_type: str
    pattern: re.Pattern  # type: ignore[type-arg]
    min_sensitivity: str
    validator: Optional[Callable[[str], bool]] = None
    context_keywords: Optional[List[str]] = None
    context_window: int = 120


def _context_near(text: str, start: int, end: int, keywords: List[str], window: int) -> bool:
    region = text[max(0, start - window) : min(len(text), end + window)].lower()
    return any(kw in region for kw in keywords)


# ── Pattern definitions ────────────────────────────────────────────────────

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# SSN — rejects invalid area/group/serial ranges at pattern level
_SSN = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# Credit cards — 4×4 with separator, Amex 4-6-5, raw 15/16 digits; Luhn-validated
_CREDITCARD = re.compile(
    r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"
    r"|\b\d{4}[-\s]\d{6}[-\s]\d{5}\b"
    r"|(?<!\d)\b\d{15,16}\b(?!\d)"
)

# IBAN — structure + mod-97 checksum
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

# IPv4 — octet-range validated
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Date of birth (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD)
_DOB = re.compile(
    r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
    r"|\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])\b"
)

# Salutation-prefixed names (the salutation is the context signal itself)
_NAME = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Sir|Lord|Lady|Mx)\.?\s+"
    r"[A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,3}\b"
)

# US Employer Identification Number
_EIN = re.compile(r"\b(?!00)\d{2}-(?!0000000)\d{7}\b")

# US Passport — high FP risk, only at strict sensitivity
_PASSPORT_US = re.compile(r"\b[A-Z]\d{8}\b")

# Generic hex secret — statistical fallback for unknown secret formats
_HEX_SECRET = re.compile(r"(?<![A-Za-z0-9])[0-9a-f]{32,64}(?![A-Za-z0-9])", re.IGNORECASE)

# Conservative phone regex for llm.stream_patterns (phonenumbers can't provide
# a regex, so we register this as a best-effort output filter)
_STREAM_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b")

# ── PII detector registry ─────────────────────────────────────────────────────
# "phone" is handled by phonenumbers, not by _PiiDetector — see _replace_phones.

_PII_DETECTORS: List[_PiiDetector] = [
    # ── relaxed+ ──────────────────────────────────────────────────────────────
    _PiiDetector("email", _EMAIL, "relaxed"),
    _PiiDetector("ssn", _SSN, "relaxed"),
    _PiiDetector("creditcard", _CREDITCARD, "relaxed", validator=_luhn_valid),
    # ── balanced+ ─────────────────────────────────────────────────────────────
    _PiiDetector(
        "ipv4",
        _IPV4,
        "balanced",
        validator=_ipv4_valid,
        context_keywords=[
            "ip",
            "address",
            "host",
            "server",
            "client",
            "remote",
            "origin",
            "access from",
            "login from",
            "ip:",
        ],
    ),
    _PiiDetector(
        "iban",
        _IBAN,
        "balanced",
        validator=_iban_valid,
        context_keywords=["iban", "bank account", "account number", "transfer", "wire", "bic"],
    ),
    _PiiDetector(
        "dob",
        _DOB,
        "balanced",
        context_keywords=[
            "born",
            "birthday",
            "dob",
            "date of birth",
            "birth date",
            "birth year",
            "born on",
        ],
    ),
    _PiiDetector("name", _NAME, "balanced"),
    _PiiDetector(
        "ein",
        _EIN,
        "balanced",
        context_keywords=[
            "ein",
            "fein",
            "employer id",
            "employer identification",
            "tax id",
            "federal tax",
            "federal id",
        ],
    ),
    # ── strict+ ───────────────────────────────────────────────────────────────
    _PiiDetector(
        "passport",
        _PASSPORT_US,
        "strict",
        context_keywords=["passport", "travel document", "visa application", "document number"],
    ),
    _PiiDetector(
        "secret_hex",
        _HEX_SECRET,
        "strict",
        context_keywords=[
            "secret",
            "token",
            "key",
            "api",
            "password",
            "credential",
            "auth",
            "bearer",
        ],
    ),
]

_PII_BY_TYPE: Dict[str, _PiiDetector] = {d.pii_type: d for d in _PII_DETECTORS}
_PHONE_TYPE = "phone"


def _active_pii_setup(
    pii_types: Optional[List[str]],
    custom_patterns: List[Any],
    sensitivity: str,
) -> Tuple[List[_PiiDetector], bool]:
    """Returns (active_pii_detectors, phone_active)."""
    rank = _SENSITIVITY_RANK[sensitivity]
    if pii_types is not None:
        detectors = [_PII_BY_TYPE[t] for t in pii_types if t in _PII_BY_TYPE]
        phone_active = _PHONE_TYPE in pii_types
    else:
        detectors = [d for d in _PII_DETECTORS if _SENSITIVITY_RANK[d.min_sensitivity] <= rank]
        # Phone is active at balanced+ (same tier as ipv4/iban)
        phone_active = rank >= _SENSITIVITY_RANK["balanced"]

    for cp in custom_patterns:
        try:
            compiled = re.compile(cp.pattern)
        except re.error:
            continue
        detectors.append(_PiiDetector(cp.name, compiled, "relaxed"))

    return detectors, phone_active


def _replace_phones(
    text: str,
    pii_action: str,
    allowlist_lower: Set[str],
) -> Tuple[str, int]:
    """Span-accurate phone redaction using Google's libphonenumber."""
    matches = list(
        phonenumbers.PhoneNumberMatcher(text, "US", leniency=phonenumbers.Leniency.VALID)
    )
    if not matches:
        return text, 0

    count = 0
    parts: List[str] = []
    prev_end = 0
    for m in matches:
        raw = text[m.start : m.end]
        if raw.lower() in allowlist_lower:
            parts.append(text[prev_end : m.end])
        else:
            parts.append(text[prev_end : m.start])
            parts.append(_make_replacement(_PHONE_TYPE, pii_action, raw))
            count += 1
        prev_end = m.end
    parts.append(text[prev_end:])
    return "".join(parts), count


def _scan_pii(
    text: str,
    detectors: List[_PiiDetector],
    phone_active: bool,
    pii_action: str,
    allowlist_lower: Set[str],
    sensitivity_rank: int,
    label: str,
) -> Tuple[str, List[str], bool]:
    """Scan *text* for PII. Returns (redacted_text, audit_logs, pii_found)."""
    working = _normalize(text)
    working = _despace_emails(working)
    counts: Dict[str, int] = defaultdict(int)

    # Phone via Google libphonenumber
    if phone_active:
        working, phone_count = _replace_phones(working, pii_action, allowlist_lower)
        if phone_count:
            counts[_PHONE_TYPE] += phone_count

    # Regex-based PII
    for det in detectors:

        def _make_repl(d: _PiiDetector) -> Callable[[re.Match], str]:  # type: ignore[type-arg]
            def _repl(m: re.Match) -> str:  # type: ignore[type-arg]
                raw = m.group(0)
                if raw.lower() in allowlist_lower:
                    return raw
                if d.validator is not None and not d.validator(raw):
                    return raw
                ctx_kws = d.context_keywords
                below_strict = sensitivity_rank < _SENSITIVITY_RANK["strict"]
                if ctx_kws is not None and below_strict:
                    if not _context_near(working, m.start(), m.end(), ctx_kws, d.context_window):
                        return raw
                counts[d.pii_type] += 1
                return _make_replacement(d.pii_type, pii_action, raw)

            return _repl

        working = det.pattern.sub(_make_repl(det), working)

    audit_logs = [f"pii_filter: found {t} in {label}: {n} match(es)" for t, n in counts.items()]
    return working, audit_logs, bool(counts)


# ─── Config ───────────────────────────────────────────────────────────────────


class _CustomPattern(BaseModel):
    name: str = Field(description="Identifier used in audit logs and replacement tokens.")
    pattern: str = Field(description="Python regex string to match against message content.")


class Config(BaseModel):
    # ── PII tier ───────────────────────────────────────────────────────────────
    pii_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "PII types to scan for. When omitted, all detectors at the configured "
            "sensitivity level run. Available: email, phone, ssn, creditcard, "
            "ipv4, iban, dob, name, ein, passport, secret_hex. "
            "Include 'phone' to activate the libphonenumber scanner explicitly."
        ),
    )
    sensitivity: Literal["relaxed", "balanced", "strict"] = Field(
        default="balanced",
        description=(
            "'relaxed' = email, SSN, credit card only (no phone). "
            "'balanced' (default) = + phone (libphonenumber), IPv4, IBAN, DoB, names, EIN. "
            "'strict' = all types, no context requirement."
        ),
    )
    action: Literal["redact", "mask", "hash"] = Field(
        default="redact",
        description=(
            "Replacement format for detected PII (not credentials). "
            "'redact' → `[protected:TYPE]` (default). "
            "'mask' → ****. "
            "'hash' → `[protected:TYPE:HASH8]` (referential integrity)."
        ),
    )
    allowlist: List[str] = Field(
        default_factory=list,
        description="Values left untouched regardless of detection (e.g. known-safe emails).",
    )
    custom_patterns: List[_CustomPattern] = Field(
        default_factory=list,
        description="Additional regex patterns added to the PII tier as relaxed-level detectors.",
    )


# ─── apply() ─────────────────────────────────────────────────────────────────


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    PII firewall: redacts structured personal data before the request reaches the LLM.
    Also registers patterns on llm.stream_patterns to filter PII from streamed output.
    """
    cfg = Config.model_validate(config)
    allowlist_lower: Set[str] = {v.lower() for v in cfg.allowlist}
    sensitivity_rank = _SENSITIVITY_RANK[cfg.sensitivity]

    pii_detectors, phone_active = _active_pii_setup(
        cfg.pii_types, cfg.custom_patterns, cfg.sensitivity
    )

    # Stream-side filters (best-effort; no validators or context gates available)
    if hasattr(llm, "stream_patterns"):
        stream_pii_action = cfg.action if cfg.action != "hash" else "redact"
        for pii_det in pii_detectors:
            llm.stream_patterns.append(
                (pii_det.pattern, _make_replacement(pii_det.pii_type, stream_pii_action, ""))
            )
        if phone_active:
            llm.stream_patterns.append(
                (_STREAM_PHONE, _make_replacement(_PHONE_TYPE, stream_pii_action, ""))
            )

    audit_logs: List[str] = []
    pii_found = False

    for idx, node in enumerate(chat.plain()):
        content = node.content
        if content is None:
            continue

        label = f"message {idx} ({node.role})"

        if isinstance(content, str):
            redacted, pii_logs, found = _scan_pii(
                content,
                pii_detectors,
                phone_active,
                cfg.action,
                allowlist_lower,
                sensitivity_rank,
                label,
            )
            audit_logs.extend(pii_logs)
            if found:
                node.content = redacted
                pii_found = True

        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if not (isinstance(part, dict) and "text" in part):
                    continue
                part_label = f"{label} part {part_idx}"
                redacted, pii_logs, found = _scan_pii(
                    part["text"],
                    pii_detectors,
                    phone_active,
                    cfg.action,
                    allowlist_lower,
                    sensitivity_rank,
                    part_label,
                )
                audit_logs.extend(pii_logs)
                if found:
                    part["text"] = redacted
                    pii_found = True

    if pii_found:
        chat.system(
            "Note: PII has been filtered from the following conversation. "
            "Protected items appear as <protected:type>."
        )

    return audit_logs


# ─── Guard metadata ───────────────────────────────────────────────────────────

META = GuardMeta(
    name="pii_filter",
    description=(
        "PII detection guard: redacts structured personal data (email, phone, SSN, "
        "credit card, IBAN, IPv4, DoB, names, EIN) from messages before they reach "
        "the LLM.  Uses Google libphonenumber for phone numbers and validated regex "
        "for all other types.  For credential detection, use credential_filter."
    ),
    config_schema=Config,
    docs=(
        "**Sensitivity levels**\n"
        "- *relaxed*: email, SSN, credit card (Luhn-validated)\n"
        "- *balanced* (default): + phone (Google libphonenumber, 250+ countries),\n"
        "  IPv4 (octet-validated), IBAN (mod-97), DoB, salutation-prefixed names, EIN\n"
        "- *strict*: + US passport numbers, generic hex secrets\n\n"
        "**Options**: `pii_types`, `allowlist`, `custom_patterns`,\n"
        "`action` (redact/mask/hash), `sensitivity`."
    ),
    examples=[
        {"sensitivity": "balanced"},
        {"pii_types": ["email", "phone", "ssn", "creditcard"]},
        {"action": "hash", "sensitivity": "strict"},
        {
            "custom_patterns": [{"name": "employee_id", "pattern": r"\bEMP-\d{6}\b"}],
        },
    ],
)
