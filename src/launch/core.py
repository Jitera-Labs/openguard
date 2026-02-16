import os
import subprocess
import sys
from typing import List, Type

from .strategies import CliArgStrategy, ConfigFileStrategy, EnvVarStrategy, Strategy
from .types import LaunchStrategy


def get_strategy_implementation(strategy_type: LaunchStrategy) -> Type[Strategy]:
    if strategy_type == LaunchStrategy.ENV:
        return EnvVarStrategy
    elif strategy_type == LaunchStrategy.FILE:
        return ConfigFileStrategy
    elif strategy_type == LaunchStrategy.ARG:
        return CliArgStrategy
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def launch_integration(name: str, args: List[str]) -> int:
    """
    Launch an integration with the given name and arguments.
    """
    try:
        from .integrations import INTEGRATIONS
    except ImportError:
        # Fallback if integrations.py has issues
        INTEGRATIONS = {}
    except Exception:
        # Avoid crashing if integrations file is broken for now
        INTEGRATIONS = {}

    integration = INTEGRATIONS.get(name)
    if not integration:
        print(f"Error: Integration '{name}' not found.", file=sys.stderr)
        if INTEGRATIONS:
            print(f"Available integrations: {', '.join(INTEGRATIONS.keys())}", file=sys.stderr)
        return 1

    # Prepare command and environment
    # Start with the default command from integration
    command = [integration.default_command]

    env = os.environ.copy()

    # Apply strategies
    # Strategies might modify command (ARG) or env (ENV) or files (FILE)
    for strategy_config in integration.strategies:
        try:
            strategy_cls = get_strategy_implementation(strategy_config.type)
            strategy = strategy_cls()
            strategy.apply(command, env, strategy_config.params)
        except Exception as e:
            print(f"Error applying strategy {strategy_config.type}: {e}", file=sys.stderr)
            return 1

    # Append user arguments
    if args:
        command.extend(args)

    # Execute
    try:
        # Use execvp to replace current process if possible
        # but subprocess.run is safer for python wrapper
        result = subprocess.run(command, env=env)
        return result.returncode
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is it installed?", file=sys.stderr)
        return 127
    except Exception as e:
        print(f"Error launching integration: {e}", file=sys.stderr)
        return 1
