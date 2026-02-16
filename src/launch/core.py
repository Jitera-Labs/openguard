import os
import socket
import subprocess
import sys
import time
from typing import List, Optional, Type

from src.config import OPENGUARD_HOST, OPENGUARD_PORT

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


def ensure_server_running(host: str, port: int) -> Optional[subprocess.Popen]:
    """
    Ensure the OpenGuard server is running on the given host and port.
    Returns the process handle if a new server was started, or None if it was already running.
    """
    # Check if port is in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            print(f"OpenGuard server already running at http://{host}:{port}", file=sys.stderr)
            return None

    print(f"Starting OpenGuard server at http://{host}:{port}...", file=sys.stderr)

    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # Capture stderr to show if startup fails
    )

    # Wait for port to be active
    start_time = time.time()
    while time.time() - start_time < 10:  # 10 seconds timeout
        if server_process.poll() is not None:
            # Process exited prematurely
            stdout, stderr = server_process.communicate()
            print(f"Server failed to start:\n{stderr.decode()}", file=sys.stderr)
            return None

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                print("Server started successfully.", file=sys.stderr)
                return server_process
        time.sleep(0.5)

    print("Timeout waiting for server to start.", file=sys.stderr)
    server_process.terminate()
    return None


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

    # Ensure server is running
    host = OPENGUARD_HOST.value
    port = OPENGUARD_PORT.value
    server_process = ensure_server_running(host, port)

    try:
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
            # Use subprocess.run to execute the integration command
            result = subprocess.run(command, env=env)
            return result.returncode
        except FileNotFoundError:
            print(f"Error: Command '{command[0]}' not found. Is it installed?", file=sys.stderr)
            return 127
        except Exception as e:
            print(f"Error launching integration: {e}", file=sys.stderr)
            return 1
    finally:
        # Clean up server process if we started it
        if server_process:
            print("Stopping OpenGuard server...", file=sys.stderr)
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
