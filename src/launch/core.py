import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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


def fetch_openguard_models(host: str, port: int) -> List[str]:
    """
    Fetch available models from the OpenGuard server.
    Returns a list of model IDs.
    """
    url = f"http://{host}:{port}/v1/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                content = response.read().decode()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    return []

                # Parse OpenAI-compatible model list
                # { "object": "list", "data": [ { "id": "model-id", ... }, ... ] }
                if data.get("object") == "list" and isinstance(data.get("data"), list):
                    return [
                        model["id"]
                        for model in data["data"]
                        if isinstance(model, dict) and "id" in model
                    ]
    except Exception as e:
        print(f"Warning: Failed to fetch models from OpenGuard: {e}", file=sys.stderr)

    return []


def kill_server_on_port(host: str, port: int) -> None:
    """Kill any process listening on the given port."""
    try:
        # Check if port is in use using lsof which is standard on Linux/macOS
        # Windows support would consistently require netstat parsing or psutil
        check_proc = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pids = check_proc.stdout.strip().split()

        if pids:
            print(
                f"Stopping existing OpenGuard instance (PIDs: {', '.join(pids)})...",
                file=sys.stderr,
            )
            try:
                subprocess.run(
                    ["kill"] + pids, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except (ProcessLookupError, OSError):
                pass

            # Wait for port to be free
            start_time = time.time()
            while time.time() - start_time < 5:
                # Check if port is still bound
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, port)) != 0:
                        return  # Port is free
                time.sleep(0.5)

            # Force kill if still running
            print("Force killing process...", file=sys.stderr)
            try:
                subprocess.run(
                    ["kill", "-9"] + pids, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except (ProcessLookupError, OSError):
                pass
            time.sleep(1)  # Give it a second to release the port completely

    except FileNotFoundError:
        # lsof not found, fallback to trying connection and warning?
        print("Warning: lsof not found, cannot kill existing server.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to stop existing server: {e}", file=sys.stderr)


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
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ],
        stdin=subprocess.DEVNULL,
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
                # Drain stderr in background to prevent pipe buffer deadlock
                threading.Thread(
                    target=lambda p: p.stderr.read() if p.stderr else None,
                    args=(server_process,),
                    daemon=True,
                ).start()
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

    # Run setup callback if provided
    if integration.setup_callback:
        print(f"Running setup for {name}...", file=sys.stderr)
        try:
            integration.setup_callback()
        except Exception as e:
            print(f"Error running setup for {name}: {e}", file=sys.stderr)
            # Decide if we should continue or exit.
            # Usually setup failure is critical for first run, but maybe not subsequent?
            # Let's verify with user intent, but for now continue with warning.

    # Ensure server is running
    host = OPENGUARD_HOST.value
    port = OPENGUARD_PORT.value
    server_process = ensure_server_running(host, port)

    # dynamically update configuration if needed
    if name == "opencode":
        try:
            # Wait a bit for server to be ready if we just started it?
            # ensure_server_running waits for port open, so it should be fine.
            models = fetch_openguard_models(host, port)
            if models:
                for s_config in integration.strategies:
                    if s_config.type == LaunchStrategy.FILE:
                        # We need to be careful not to persist this change
                        # if we were running in a long-lived process
                        # But since it's a CLI tool, it's fine.
                        data = s_config.params.get("data", {})
                        if "provider" in data and "openguard" in data["provider"]:
                            openguard_config = data["provider"]["openguard"]
                            # Format: "gpt-4o": {"name": "GPT-4o (Guarded)"}
                            model_map = {m: {"name": f"{m} (Guarded)"} for m in models}
                            openguard_config["models"] = model_map
        except Exception as e:
            print(
                f"Warning: Failed to update opencode configuration with dynamic models: {e}",
                file=sys.stderr,
            )

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
