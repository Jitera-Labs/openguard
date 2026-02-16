from typing import Dict

from .types import Integration, LaunchStrategy, StrategyConfig

# Standard OpenGuard port from docker-compose.yml
OPENGUARD_PORT = 23294
OPENGUARD_URL = f"http://127.0.0.1:{OPENGUARD_PORT}"

INTEGRATIONS: Dict[str, Integration] = {
    "opencode": Integration(
        name="opencode",
        default_command="opencode",
        strategies=[
            StrategyConfig(
                type=LaunchStrategy.FILE,
                params={
                    "file_path": "~/.config/opencode/opencode.json",
                    "data": {
                        "provider": {
                            "type": "openai",
                            "base_url": OPENGUARD_URL,
                            "api_key": "sk-openguard-placeholder",
                        }
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
                type=LaunchStrategy.ARG,
                params={
                    "args": [
                        "--api-base",
                        OPENGUARD_URL,
                        "--api-key",
                        "sk-openguard-placeholder",
                    ]
                },
            )
        ],
    ),
}
