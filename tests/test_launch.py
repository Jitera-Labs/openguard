import json
import os
import sys
from unittest.mock import patch

import pytest

# Add src to sys.path to ensure we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from launch.core import launch_integration

# Constants used in the tests
OPENGUARD_URL = "http://127.0.0.1:23294"


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


def test_launch_opencode_config_strategy(mock_subprocess_run, tmp_path):
    """
    Test ConfigFileStrategy with 'opencode' integration.
    Verifies that the config file is created/updated with the correct settings.
    """
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
        assert content["provider"]["type"] == "openai"
        assert content["provider"]["base_url"] == OPENGUARD_URL
        assert content["provider"]["api_key"] == "sk-openguard-placeholder"


def test_launch_claude_env_strategy(mock_subprocess_run):
    """
    Test EnvVarStrategy with 'claude' integration.
    Verifies that environment variables are injected.
    """
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
    assert env["ANTHROPIC_API_KEY"] == "sk-openguard-placeholder"


def test_launch_codex_arg_strategy(mock_subprocess_run):
    """
    Test CliArgStrategy with 'codex' integration.
    Verifies that CLI arguments are appended.
    """
    # Call the function under test
    exit_code = launch_integration("codex", [])

    # Assertions
    assert exit_code == 0

    # Check if subprocess.run was called
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    command = args[0]

    # Verify command structure
    assert command[0] == "codex"

    # Verify injected arguments
    assert "--api-base" in command
    assert OPENGUARD_URL in command
    assert "--api-key" in command
    assert "sk-openguard-placeholder" in command

    # Verify order (basic check)
    base_idx = command.index("--api-base")
    assert command[base_idx + 1] == OPENGUARD_URL


def test_launch_integration_not_found(mock_subprocess_run):
    """
    Test behavior when an unknown integration is requested.
    """
    exit_code = launch_integration("unknown_tool", [])
    assert exit_code == 1
    mock_subprocess_run.assert_not_called()


def test_launch_with_extra_args(mock_subprocess_run):
    """
    Test that extra arguments passed to launch_integration are appended to the command.
    """
    extra_args = ["--verbose", "--dry-run"]
    exit_code = launch_integration("codex", extra_args)

    assert exit_code == 0

    args, _ = mock_subprocess_run.call_args
    command = args[0]

    # "codex" + injected args + extra args
    assert command[-2:] == extra_args
