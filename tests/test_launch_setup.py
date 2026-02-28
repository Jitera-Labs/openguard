import contextlib
import io
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from launch.setup import _discover_models, setup_opencode  # noqa: E402, I001


def test_discover_models_from_global_config():
    global_cfg = {
        "provider": {
            "openai": {"models": {"gpt-4o": {"name": "GPT-4o"}, "o1": {"name": "o1"}}},
            "anthropic": {"models": {"claude-3-5-sonnet-20241022": {"name": "Claude 3.5"}}},
            "openguard": {"models": {"should-be-skipped": {}}},
        }
    }
    models = _discover_models(global_cfg)
    assert "gpt-4o" in models
    assert models["gpt-4o"]["name"] == "GPT-4o (Guarded)"
    assert "o1" in models
    assert "claude-3-5-sonnet-20241022" in models
    assert "should-be-skipped" not in models


def test_discover_models_empty_when_no_global_config():
    assert _discover_models() == {}
    assert _discover_models({}) == {}
    assert _discover_models({"provider": {}}) == {}


# Mocking the file system interactions for setup_opencode
@patch("launch.setup._seed_opencode_dirs")
@patch("launch.setup.Path")
@patch("launch.setup.shutil.copy2")
@patch("launch.setup._secure_open")
@patch("launch.setup.open", new_callable=mock_open)
@patch("launch.setup.json.load")
@patch("launch.setup.json.dump")
@patch("launch.setup.yaml.safe_load")
@patch("launch.setup.yaml.safe_dump")
def test_setup_opencode_flow(
    mock_yaml_dump,
    mock_yaml_load,
    mock_json_dump,
    mock_json_load,
    mock_file,
    mock_secure_open,
    mock_copy2,
    mock_path_cls,
    mock_seed_dirs,
):
    # --- Setup Mocks ---

    # Make _secure_open return a context manager that yields a StringIO file handle
    def make_ctx():
        buf = io.StringIO()
        return contextlib.contextmanager(lambda: (yield buf))()

    mock_secure_open.side_effect = lambda _path: make_ctx()

    # 1. Mock Path.home() and Path.cwd()
    mock_home = MagicMock()
    mock_cwd = MagicMock()
    mock_path_cls.home.return_value = mock_home
    mock_path_cls.cwd.return_value = mock_cwd

    # 2. Mock specific paths
    auth_path = mock_home / ".local/share/opencode/auth.json"
    auth_path.exists.return_value = True

    opencode_path = mock_cwd / "opencode.json"
    opencode_path.exists.return_value = True

    og_config_dir = mock_home / ".config" / "openguard"

    # Ensure intermediate dirs are created (mock only so no real effect)
    og_config_dir.mkdir.return_value = None

    # 3. Mock file reads (json.load and yaml.safe_load)
    # The setup_opencode function reads:
    #   1. auth.json -> json.load
    #   2. opencode.json -> json.load
    #   3. config.yaml -> yaml.safe_load (if exists)

    # Let's say auth.json has an openai key
    auth_data = {"openai": {"type": "api", "key": "sk-test-key"}}
    # opencode.json has just a model
    opencode_data = {"model": "openai:gpt-4"}

    # config.yaml (OpenGuard) might carry existing data or fail (return None/empty)
    # We will simulate empty or existing to verify append

    # Three json.load calls: auth.json, project opencode.json, global opencode.json
    mock_json_load.side_effect = [
        auth_data,
        opencode_data,
        # Global opencode.json with real model definitions so discovery succeeds
        {
            "provider": {
                "openai": {"models": {"gpt-4o": {"name": "GPT-4o"}, "gpt-4": {"name": "GPT-4"}}}
            }
        },
    ]
    mock_yaml_load.return_value = {}  # Empty config.yaml initially

    # --- Run Function ---
    setup_opencode()

    # --- Verifications ---

    # 1. Verify backup was created for auth.json
    # args of copy2: src, dst
    # We expect `auth_path` to be copied to `auth_path.with_suffix(".json.bak")`
    assert mock_copy2.call_count >= 1
    # Check if first call was for auth.json
    # src should be auth_path (the mock object)
    # assert call_args[0][0] == auth_path

    # 2. Verify OpenGuard config migration
    # We expect yaml.safe_dump to be called with the new provider config
    assert mock_yaml_dump.called
    dumped_data = mock_yaml_dump.call_args[0][0]
    assert "providers" in dumped_data
    # It should have picked up "openai" from auth_data
    providers = dumped_data["providers"]
    assert len(providers) == 1
    assert providers[0]["type"] == "openai"
    assert providers[0]["key"] == "sk-test-key"

    # 3. Verify auth.json update
    # We expect json.dump to be called to add "openguard" to auth.json
    # We can inspect calls to json.dump.
    # One call is updating auth.json, one is updating opencode.json

    auth_dump_call = None
    opencode_dump_call = None

    for call in mock_json_dump.call_args_list:
        data, _ = call[0]
        if "openguard" in data and "key" in data["openguard"]:
            auth_dump_call = data
        if "provider" in data and "openguard" in data["provider"]:
            opencode_dump_call = data

    assert auth_dump_call is not None, "auth.json was not updated with openguard credentials"
    assert auth_dump_call["openguard"]["type"] == "api"

    assert opencode_dump_call is not None, "opencode.json was not updated with openguard provider"
    # Verify models in opencode.json
    og_provider = opencode_dump_call["provider"]["openguard"]
    assert "models" in og_provider
    assert "gpt-4o" in og_provider["models"]

    # 4. Verify opencode.json model selection
    # It should have switched "model" to "openguard:..."
    assert opencode_dump_call["model"].startswith("openguard:")


@patch("launch.setup._seed_opencode_dirs")
@patch("launch.setup.Path")
@patch("builtins.print")
def test_setup_opencode_missing_auth(mock_print, mock_path_cls, mock_seed_dirs):
    # If auth.json is missing, it should just return early (logging warning)
    mock_home = MagicMock()
    mock_path_cls.home.return_value = mock_home
    auth_path = mock_home / ".local/share/opencode/auth.json"
    auth_path.exists.return_value = False

    setup_opencode()

    # Should check for auth path existence
    assert auth_path.exists.called
