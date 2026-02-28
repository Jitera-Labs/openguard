from typing import Dict

from src.config import OPENGUARD_PORT

from .setup import setup_claude, setup_codex, setup_opencode
from .types import Integration, LaunchStrategy, StrategyConfig

# Standard OpenGuard port from configuration
OPENGUARD_URL = f"http://127.0.0.1:{OPENGUARD_PORT.value}"

INTEGRATIONS: Dict[str, Integration] = {
    "opencode": Integration(
        name="opencode",
        default_command="opencode",
        setup_callback=setup_opencode,
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.FILE,
                params={
                    "file_path": "~/.config/opencode/opencode.json",
                    "data": {
                        "provider": {
                            "openguard": {
                                "npm": "@ai-sdk/openai-compatible",
                                "name": "OpenGuard",
                                "options": {
                                    "baseURL": f"{OPENGUARD_URL}/v1",
                                    "apiKey": "sk-openguard-placeholder",
                                },
                                "models": {},
                            }
                        },
                        "model": "openguard:gpt-4o",
                    },
                    "format": "json",
                },
            )
        ],
    ),
    "claude": Integration(
        name="claude",
        default_command="claude",
        setup_callback=setup_claude,
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.ENV,
                params={
                    "env_vars": {
                        # Only redirect the base URL.  Claude Code manages its own
                        # credentials (OAuth token or ANTHROPIC_API_KEY from the user's
                        # shell) and attaches them to every request.  OpenGuard forwards
                        # those headers verbatim to api.anthropic.com.
                        "ANTHROPIC_BASE_URL": OPENGUARD_URL,
                    }
                },
            )
        ],
    ),
    "codex": Integration(
        name="codex",
        default_command="codex",
        setup_callback=setup_codex,
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.ENV,
                params={
                    "env_vars": {
                        "OPENAI_BASE_URL": f"{OPENGUARD_URL}/v1",
                        "OPENAI_API_KEY": "sk-openguard-placeholder",
                    }
                },
            )
        ],
    ),
}
