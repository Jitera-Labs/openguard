"""Credential detection guard — blocks or redacts secrets via detect-secrets.

Uses the battle-tested detect-secrets pattern corpus to identify 12+ credential
families in message content before they reach the LLM.  Default action is
**block** (raises GuardBlockedError), mirroring the behaviour of KeywordFilter.
Set ``action: redact`` to sanitize in-place instead.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from detect_secrets.plugins.aws import AWSKeyDetector
from detect_secrets.plugins.azure_storage_key import AzureStorageKeyDetector
from detect_secrets.plugins.discord import DiscordBotTokenDetector
from detect_secrets.plugins.github_token import GitHubTokenDetector
from detect_secrets.plugins.gitlab_token import GitLabTokenDetector
from detect_secrets.plugins.jwt import JwtTokenDetector
from detect_secrets.plugins.npm import NpmDetector
from detect_secrets.plugins.openai import OpenAIDetector
from detect_secrets.plugins.private_key import PrivateKeyDetector
from detect_secrets.plugins.slack import SlackDetector
from detect_secrets.plugins.stripe import StripeDetector
from detect_secrets.plugins.twilio import TwilioKeyDetector
from pydantic import BaseModel, Field

from src.guard_meta import GuardMeta
from src.guards import GuardBlockedError

if TYPE_CHECKING:
    from src.chat import Chat
    from src.llm import LLM


@dataclass
class _CredDetector:
    """Wraps a detect-secrets plugin and exposes a uniform scan/redact API."""

    cred_type: str
    # detect_secrets plugin instance (typed as Any — library ships no stubs)
    plugin: Any
    # PrivateKeyDetector's denylist patterns only match PEM headers, not the
    # full block.  Use this override for span-accurate redaction.
    redact_override: Optional[re.Pattern] = None  # type: ignore[type-arg]


_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY[^-]*-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |PGP )?PRIVATE KEY[^-]*-----"
    r"|PuTTY-User-Key-File-2[\s\S]*?Private-MAC:[ \t]*[0-9a-f]+",
    re.MULTILINE,
)

_CRED_DETECTORS: List[_CredDetector] = [
    _CredDetector("aws_key", AWSKeyDetector()),
    _CredDetector("github_token", GitHubTokenDetector()),
    _CredDetector("gitlab_token", GitLabTokenDetector()),
    _CredDetector("jwt", JwtTokenDetector()),
    _CredDetector("slack_token", SlackDetector()),
    _CredDetector("private_key", PrivateKeyDetector(), redact_override=_PRIVATE_KEY_BLOCK),
    _CredDetector("openai_key", OpenAIDetector()),
    _CredDetector("stripe_key", StripeDetector()),
    _CredDetector("twilio_key", TwilioKeyDetector()),
    _CredDetector("discord_token", DiscordBotTokenDetector()),
    _CredDetector("npm_token", NpmDetector()),
    _CredDetector("azure_storage_key", AzureStorageKeyDetector()),
]

_CRED_BY_TYPE: Dict[str, _CredDetector] = {d.cred_type: d for d in _CRED_DETECTORS}


def _active_cred_detectors(credential_types: Optional[List[str]]) -> List[_CredDetector]:
    if credential_types is None:
        return list(_CRED_DETECTORS)
    return [_CRED_BY_TYPE[t] for t in credential_types if t in _CRED_BY_TYPE]


def _redact_credential(text: str, det: _CredDetector, repl: str) -> str:
    if det.redact_override is not None:
        return det.redact_override.sub(repl, text)
    for pattern in det.plugin.denylist:
        text = pattern.sub(repl, text)
    return text


def _scan_credentials(
    text: str,
    active: List[_CredDetector],
    action: str,
    label: str,
) -> Tuple[str, List[str], List[str]]:
    """
    Scan *text* for credentials.

    Returns (possibly_redacted_text, detected_types, audit_logs).
    """
    detected: List[str] = []
    working = text

    for det in active:
        if not list(det.plugin.analyze_string(working)):
            continue
        detected.append(det.cred_type)
        if action in ("redact", "block"):
            working = _redact_credential(working, det, f"<protected:{det.cred_type}>")

    logs = [f"credential_filter: credential '{t}' detected in {label}" for t in detected]
    return working, detected, logs


class Config(BaseModel):
    action: Literal["block", "redact", "log"] = Field(
        default="block",
        description=(
            "Action when credentials are detected. "
            "'block' rejects the request with a 403 (default — mirrors KeywordFilter). "
            "'redact' replaces credentials with `[protected:TYPE]` and lets the request through. "
            "'log' only records the finding without modifying or blocking."
        ),
    )
    credential_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Credential types to scan for. When omitted, all 12 detectors run. "
            "Available: aws_key, github_token, gitlab_token, jwt, slack_token, "
            "private_key, openai_key, stripe_key, twilio_key, discord_token, "
            "npm_token, azure_storage_key."
        ),
    )


def apply(chat: "Chat", llm: "LLM", config: dict) -> List[str]:
    """
    Scan all messages for credentials (API keys, tokens, private keys, etc.).

    Uses the detect-secrets library (12 credential families).
    Default action: block (raises GuardBlockedError with credential types listed).
    Set action=redact to replace in-place, or action=log to audit-only.
    """
    cfg = Config.model_validate(config)
    active_creds = _active_cred_detectors(cfg.credential_types)

    # Register stream-side redaction patterns when action=redact
    if hasattr(llm, "stream_patterns") and cfg.action == "redact":
        for det in active_creds:
            repl = f"<protected:{det.cred_type}>"
            patterns = (
                [det.redact_override]
                if det.redact_override is not None
                else list(det.plugin.denylist)
            )
            for p in patterns:
                llm.stream_patterns.append((p, repl))

    audit_logs: List[str] = []
    all_cred_types: List[str] = []

    for idx, node in enumerate(chat.plain()):
        content = node.content
        if content is None:
            continue

        label = f"message {idx} ({node.role})"

        if isinstance(content, str):
            new_text, cred_types, cred_logs = _scan_credentials(
                content, active_creds, cfg.action, label
            )
            audit_logs.extend(cred_logs)
            all_cred_types.extend(cred_types)
            if cred_types and cfg.action == "redact":
                node.content = new_text

        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if not (isinstance(part, dict) and "text" in part):
                    continue
                part_label = f"{label} part {part_idx}"
                new_raw, cred_types, cred_logs = _scan_credentials(
                    part["text"], active_creds, cfg.action, part_label
                )
                audit_logs.extend(cred_logs)
                all_cred_types.extend(cred_types)
                if cred_types and cfg.action == "redact":
                    part["text"] = new_raw

    # Raise after all messages are scanned so audit_logs are complete
    if all_cred_types and cfg.action == "block":
        unique = sorted(set(all_cred_types))
        raise GuardBlockedError(
            f"Request blocked: credential type(s) detected: {', '.join(unique)}"
        )

    return audit_logs


META = GuardMeta(
    name="credential_filter",
    description=(
        "Credential detection guard: scans messages for API keys, tokens, and secrets "
        "using the detect-secrets library (12 credential families). "
        "Default action: block (raises 403). Set action=redact to sanitize in-place."
    ),
    config_schema=Config,
    docs=(
        "Covers AWS IAM keys, GitHub/GitLab tokens, JWTs, Slack tokens, PEM private\n"
        "keys, OpenAI keys, Stripe keys, Twilio, Discord, npm, Azure Storage keys.\n\n"
        "**Options**\n"
        "- `action`: block (default) / redact / log\n"
        "- `credential_types`: list of specific types to check; omit for all 12.\n\n"
        "**Available types**: aws_key, github_token, gitlab_token, jwt, slack_token,\n"
        "private_key, openai_key, stripe_key, twilio_key, discord_token, npm_token,\n"
        "azure_storage_key."
    ),
    examples=[
        {"action": "block"},
        {"action": "block", "credential_types": ["aws_key", "github_token", "jwt", "private_key"]},
        {"action": "redact"},
        {"action": "log"},
    ],
)
