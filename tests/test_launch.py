import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to sys.path to ensure we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from launch.core import ensure_server_running, fetch_openguard_models, launch_integration

# Try to import INTEGRATIONS, but handle if it fails (as seen in launch code)
try:
    from launch.integrations import INTEGRATIONS
except ImportError:
    INTEGRATIONS = {}

# Constants used in the tests
OPENGUARD_URL = "http://127.0.0.1:23294"
OPENGUARD_HOST = "127.0.0.1"
OPENGUARD_PORT = 23294


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


@pytest.fixture
def mock_popen():
    with patch("subprocess.Popen") as mock_popen:
        process_mock = MagicMock()
        process_mock.poll.return_value = None
        process_mock.terminate.return_value = None
        process_mock.kill.return_value = None
        process_mock.wait.return_value = None
        mock_popen.return_value = process_mock
        yield mock_popen


@pytest.fixture
def mock_socket():
    with patch("socket.socket") as mock_sock:
        yield mock_sock


@pytest.fixture
def mock_launch_deps():
    """Mock dependencies used by launch_integration."""
    with (
        patch("launch.core.kill_server_on_port") as mock_kill,
        patch("launch.core.fetch_openguard_models") as mock_fetch,
    ):
        # Mock setup_callback on opencode integration to avoid file side effects
        mock_setup = None
        orig_setup = None
        if "opencode" in INTEGRATIONS:
            orig_setup = INTEGRATIONS["opencode"].setup_callback
            mock_setup = MagicMock()
            INTEGRATIONS["opencode"].setup_callback = mock_setup

        yield mock_kill, mock_fetch, mock_setup

        # Restore setup_callback
        if "opencode" in INTEGRATIONS and orig_setup:
            INTEGRATIONS["opencode"].setup_callback = orig_setup


def test_ensure_server_already_running(mock_socket):
    # Setup socket to indicate port is open (connect_ex returns 0)
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    server_process = ensure_server_running(OPENGUARD_HOST, OPENGUARD_PORT)

    # Should perform check and return None (no new process started)
    assert server_process is None
    # Verify connect_ex called with correct host/port
    connect_ex = mock_socket.return_value.__enter__.return_value.connect_ex
    connect_ex.assert_called_with((OPENGUARD_HOST, OPENGUARD_PORT))


def test_ensure_server_starts_successfully(mock_socket, mock_popen):
    # Setup socket: first call (check if running) returns 1 (not running)
    # Subsequent calls (wait for start) return 1 then 0 (success)
    mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = [1, 1, 0]

    server_process = ensure_server_running(OPENGUARD_HOST, OPENGUARD_PORT)

    # Should start process and return it
    assert server_process is not None
    assert mock_popen.called
    # Should have checked at least once for startup
    assert mock_socket.return_value.__enter__.return_value.connect_ex.call_count >= 2


def test_ensure_server_fails_to_start(mock_socket, mock_popen):
    # Setup socket: always returns 1 (not running)
    # Setup process: poll returns 1 (exited with error)
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 1

    process_mock = mock_popen.return_value
    process_mock.poll.return_value = 1  # Process exited
    process_mock.communicate.return_value = (b"", b"Error starting")

    server_process = ensure_server_running(OPENGUARD_HOST, OPENGUARD_PORT)

    # Should return None as it failed
    assert server_process is None


def test_launch_integration_server_already_running(
    mock_launch_deps, mock_subprocess_run, mock_socket
):
    # Test that launch_integration proceeds when server is running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    exit_code = launch_integration("opencode", [])

    assert exit_code == 0
    mock_subprocess_run.assert_called_once()

    # Also verify dependencies call
    mock_kill, _, mock_setup = mock_launch_deps
    if mock_setup:
        mock_setup.assert_called_once()


def test_launch_integration_starts_and_stops_server(
    mock_launch_deps, mock_subprocess_run, mock_socket, mock_popen
):
    # Server not running initially, then starts
    mock_socket.return_value.__enter__.return_value.connect_ex.side_effect = [1, 0]

    # We need to verify terminate is called on the process

    process_mock = mock_popen.return_value

    exit_code = launch_integration("opencode", [])

    assert exit_code == 0
    mock_subprocess_run.assert_called_once()

    # Verify server was terminated
    process_mock.terminate.assert_called_once()
    process_mock.wait.assert_called_once()


def test_launch_opencode_config_strategy(
    mock_launch_deps, mock_subprocess_run, mock_socket, tmp_path
):
    """
    Test ConfigFileStrategy with 'opencode' integration.
    Verifies that the config file is created/updated with the correct settings.
    """
    # Ensure server is seen as running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    # Create a dummy home directory in tmp_path
    dummy_home = tmp_path / "home"
    dummy_home.mkdir()

    # Define where the config file should be
    config_dir = dummy_home / ".config" / "opencode"
    config_file = config_dir / "opencode.json"

    # Mock os.path.expanduser to return our dummy home path instead of real home
    # We need to match the specific call in ConfigFileStrategy
    with patch("os.path.expanduser", side_effect=lambda p: str(p).replace("~", str(dummy_home))):
        # Call the function under test
        exit_code = launch_integration("opencode", [])

        # Assertions
        assert exit_code == 0

        # Check if subprocess.run was called with correct command
        # opencode default command is "opencode"
        mock_subprocess_run.assert_called_once()
        args, kwargs = mock_subprocess_run.call_args
        assert args[0][0] == "opencode"

        # Verify file creation
        assert config_file.exists()

        # Verify file content
        with open(config_file, "r") as f:
            content = json.load(f)

        assert "provider" in content
        assert "openguard" in content["provider"]
        openguard_config = content["provider"]["openguard"]
        assert openguard_config["npm"] == "@ai-sdk/openai-compatible"
        assert openguard_config["options"]["baseURL"] == f"{OPENGUARD_URL}/v1"
        assert openguard_config["options"]["apiKey"] == "sk-openguard-placeholder"
        assert content["model"] == "openguard:gpt-4o"


def test_launch_claude_env_strategy(mock_launch_deps, mock_subprocess_run, mock_socket):
    """
    Test EnvVarStrategy with 'claude' integration.
    Verifies that environment variables are injected.
    """
    # Ensure server is seen as running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    # Call the function under test
    exit_code = launch_integration("claude", [])

    # Assertions
    assert exit_code == 0

    # Check if subprocess.run was called
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    command = args[0]
    env = kwargs.get("env")

    assert command[0] == "claude"

    # Verify environment variables
    assert env is not None
    assert env["ANTHROPIC_BASE_URL"] == OPENGUARD_URL
    # Claude Code manages its own credentials; OpenGuard only redirects the base URL


def test_launch_codex_env_strategy(mock_launch_deps, mock_subprocess_run, mock_socket):
    """
    Test EnvVarStrategy with 'codex' integration.
    Verifies that environment variables are injected (not CLI args).
    """
    # Ensure server is seen as running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    # Call the function under test
    exit_code = launch_integration("codex", [])

    # Assertions
    assert exit_code == 0

    # Check if subprocess.run was called
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    command = args[0]
    env = kwargs.get("env")

    # Command is just ["codex"] — no injected CLI args
    assert command[0] == "codex"
    assert "--api-base" not in command
    assert "--api-key" not in command

    # Verify environment variables
    assert env is not None
    # OpenAI-compatible clients expect the /v1 suffix on the base URL
    assert env["OPENAI_BASE_URL"] == f"{OPENGUARD_URL}/v1"
    assert env["OPENAI_API_KEY"] == "sk-openguard-placeholder"


def test_launch_integration_not_found(mock_launch_deps, mock_subprocess_run):
    """
    Test behavior when an unknown integration is requested.
    """
    exit_code = launch_integration("unknown_tool", [])
    assert exit_code == 1
    mock_subprocess_run.assert_not_called()


def test_launch_with_extra_args(mock_launch_deps, mock_subprocess_run, mock_socket):
    """
    Test that extra arguments passed to launch_integration are appended to the command.
    """
    # Ensure server is seen as running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    extra_args = ["--verbose", "--dry-run"]
    exit_code = launch_integration("codex", extra_args)

    assert exit_code == 0

    args, _ = mock_subprocess_run.call_args
    command = args[0]

    # "codex" + extra args (no injected CLI args with ENV strategy)
    assert command[-2:] == extra_args


def test_fetch_openguard_models_parsing():
    """
    Unit test for fetch_openguard_models parsing logic.
    """
    import urllib.error

    # 1. Success case
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(
        {
            "object": "list",
            "data": [{"id": "model-a", "object": "model"}, {"id": "model-b", "object": "model"}],
        }
    ).encode()
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        models = fetch_openguard_models("localhost", 1234)
        assert models == ["model-a", "model-b"]

    # 2. Invalid JSON
    mock_response.read.return_value = b"invalid json"
    with patch("urllib.request.urlopen", return_value=mock_response):
        models = fetch_openguard_models("localhost", 1234)
        assert models == []

    # 3. HTTP Error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network error")):
        models = fetch_openguard_models("localhost", 1234)
        assert models == []


def test_launch_integration_opencode_updates_config(
    mock_launch_deps, mock_subprocess_run, mock_socket
):
    """
    Test that opencode integration fetches models and updates configuration.
    """
    # Mock server running
    mock_socket.return_value.__enter__.return_value.connect_ex.return_value = 0

    # Mock fetch_openguard_models
    with patch(
        "launch.core.fetch_openguard_models", return_value=["model-x", "model-y"]
    ) as mock_fetch:
        # Mock ConfigFileStrategy.apply to capture the configuration passed to it
        # Note: referencing the class in launch.core namespace as that's where it is used
        with patch("launch.core.ConfigFileStrategy.apply") as mock_apply:
            launch_integration("opencode", [])

            # Verify fetch called
            mock_fetch.assert_called_once()

            # Verify apply called
            mock_apply.assert_called_once()

            # Inspect arguments passed to apply(command, env, params)
            args, _ = mock_apply.call_args
            params = args[-1]

            # Verify the structure
            assert "data" in params
            assert "provider" in params["data"]
            assert "openguard" in params["data"]["provider"]

            openguard_config = params["data"]["provider"]["openguard"]
            assert "models" in openguard_config

            models = openguard_config["models"]
            assert "model-x" in models
            assert "model-y" in models
            assert models["model-x"]["name"] == "model-x (Guarded)"
