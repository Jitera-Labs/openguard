from typing import Dict

from src.config import OPENGUARD_PORT

from .setup import setup_opencode
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
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.ENV,
                params={
                    "env_vars": {
                        "ANTHROPIC_BASE_URL": OPENGUARD_URL,
                        "ANTHROPIC_API_KEY": "sk-openguard-placeholder",
                    }
                },
            )
        ],
    ),
    "codex": Integration(
        name="codex",
        default_command="codex",
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.ENV,
                params={
                    "env_vars": {
                        "OPENAI_BASE_URL": OPENGUARD_URL,
                        "OPENAI_API_KEY": "sk-openguard-placeholder",
                    }
                },
            )
        ],
    ),
}
