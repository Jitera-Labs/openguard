import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from src import config
from src import log as log_module

logger = log_module.setup_logger(__name__)


def setup_claude() -> None:
    """Seed /root/.claude and /root/.claude.json from read-only host mounts.

    The wrapper mounts ~/.claude as /claude-seed:ro and ~/.claude.json as
    /claude-seed.json:ro so the container can never corrupt host ownership.
    This function copies both into the writable /root tmpfs before claude
    starts, giving it a complete, writable config identical to the host.
    """
    log_path = "/tmp/claude-setup.log"

    def _log(msg: str) -> None:
        print(f"[setup_claude] {msg}", file=sys.stderr, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[setup_claude] {msg}\n")

    seed = Path("/claude-seed")
    dest = Path("/root/.claude")

    _log(f"seed exists={seed.exists()} dest exists={dest.exists()}")

    if not seed.exists():
        _log("ERROR: seed not found at /claude-seed — skipping copy")
        logger.warning("Claude seed not found at /claude-seed — skipping copy")
        return
    if dest.exists():
        shutil.rmtree(dest)
    # symlinks=True: ~/.claude often contains symlinks (debug/latest, skills/*).
    # Copying them as symlinks avoids ENOENT when targets don't exist in the
    # container.  ignore_dangling_symlinks silences any remaining edge cases.
    shutil.copytree(seed, dest, symlinks=True, ignore_dangling_symlinks=True)
    _log("copied /claude-seed -> /root/.claude")

    # The seed files were chmod a+rX (world-readable but not writable) by the
    # wrapper so the container could read them without CAP_DAC_OVERRIDE.
    # copytree preserves those permissions, leaving the copies read-only.
    # Claude Code needs to write to its config dir (update tokens, state, etc.),
    # so make the writable tmpfs copies owner-writable.
    for dirpath, dirs, files in os.walk(dest):
        for d in dirs:
            p = os.path.join(dirpath, d)
            if not os.path.islink(p):
                os.chmod(p, 0o755)
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                os.chmod(p, 0o644)
    _log("fixed permissions on /root/.claude (owner-writable)")
    _log(f"  .credentials.json exists={Path('/root/.claude/.credentials.json').exists()}")

    seed_json = Path("/claude-seed.json")
    _log(f"seed_json exists={seed_json.exists()}")
    if seed_json.exists():
        shutil.copy2(seed_json, Path("/root/.claude.json"))
        os.chmod("/root/.claude.json", 0o644)
        _log("copied /claude-seed.json -> /root/.claude.json")
    else:
        _log("WARNING: /claude-seed.json not found — skipping ~/.claude.json copy")

    _log("done. inspect: cat /tmp/claude-setup.log")


def setup_codex() -> None:
    """Seed /root/.codex from read-only host mount and patch config.toml for OpenGuard.

    The wrapper mounts ~/.codex as /codex-seed:ro (auth.json, config.toml, themes/).
    This function copies everything into the writable /root tmpfs, fixes permissions,
    and optionally patches config.toml to add/override the openguard model provider
    with the correct base_url pointing at the local OpenGuard proxy.
    """
    log_path = "/tmp/codex-setup.log"

    def _log(msg: str) -> None:
        print(f"[setup_codex] {msg}", file=sys.stderr, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[setup_codex] {msg}\n")

    seed = Path("/codex-seed")
    dest = Path.home() / ".codex"

    _log(f"seed exists={seed.exists()} dest exists={dest.exists()}")

    if not seed.exists():
        _log("WARNING: seed not found at /codex-seed — skipping copy")
        logger.warning("Codex seed not found at /codex-seed — skipping copy")
        # Still create dest so codex doesn't complain about missing config dir
        dest.mkdir(parents=True, exist_ok=True)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(seed, dest, symlinks=True, ignore_dangling_symlinks=True)
        _log("copied /codex-seed -> /root/.codex")

        # Fix permissions: seed was chmod a+rX (read-only), make writable
        for dirpath, dirs, files in os.walk(dest):
            for d in dirs:
                p = os.path.join(dirpath, d)
                if not os.path.islink(p):
                    os.chmod(p, 0o755)
            for f in files:
                p = os.path.join(dirpath, f)
                if not os.path.islink(p):
                    os.chmod(p, 0o644)
        _log("fixed permissions on /root/.codex (owner-writable)")

    # Patch config.toml to add/override the openguard model provider
    config_path = dest / "config.toml"
    base_url = f"http://{config.OPENGUARD_HOST.value}:{config.OPENGUARD_PORT.value}"
    _patch_codex_config(config_path, base_url, _log)

    _log("done. inspect: cat /tmp/codex-setup.log")


def _patch_codex_config(config_path: Path, base_url: str, _log: Any) -> None:
    """Add or override [model_providers.openguard] in codex config.toml.

    Sets model_provider = "openguard" so codex routes through the local proxy.
    Uses simple text manipulation since no TOML writer is available in stdlib.
    """
    import re
    import tomllib

    content = ""
    parsed: Dict[str, Any] = {}
    if config_path.exists():
        content = config_path.read_text()
        try:
            parsed = tomllib.loads(content)
        except Exception as e:
            _log(f"WARNING: could not parse config.toml: {e} — will append provider block")

    provider_block = (
        '\n[model_providers.openguard]\n'
        'name = "OpenGuard Proxy"\n'
        f'base_url = "{base_url}"\n'
        'env_key = "OPENAI_API_KEY"\n'
    )

    # Check if [model_providers.openguard] already exists
    existing_providers = parsed.get("model_providers", {})
    if "openguard" in existing_providers:
        # Replace the existing section using regex
        # Match [model_providers.openguard] through to the next section or EOF
        pattern = r'\[model_providers\.openguard\][^\[]*'
        content = re.sub(pattern, provider_block.lstrip() + '\n', content)
        _log("replaced existing [model_providers.openguard] section")
    else:
        content = content.rstrip() + '\n' + provider_block
        _log("appended [model_providers.openguard] section")

    # Set model_provider = "openguard" at top level
    if re.search(r'^model_provider\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^model_provider\s*=\s*.*$',
            'model_provider = "openguard"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
        _log("updated model_provider = \"openguard\"")
    else:
        # Insert after model line if present, otherwise prepend
        if re.search(r'^model\s*=', content, re.MULTILINE):
            content = re.sub(
                r'^(model\s*=\s*.*?)$',
                r'\1\nmodel_provider = "openguard"',
                content,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            content = 'model_provider = "openguard"\n' + content
        _log("added model_provider = \"openguard\"")

    config_path.write_text(content)
    os.chmod(str(config_path), 0o644)
    _log(f"wrote patched config.toml to {config_path}")


@contextlib.contextmanager
def _secure_open(path: str) -> Iterator[Any]:
    """Open a file for writing with 0o600 permissions (owner read/write only).

    Uses os.open to atomically set permissions at creation time, avoiding the
    TOCTOU window that exists when using open() followed by os.chmod().
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    f = os.fdopen(fd, "w")
    try:
        yield f
    finally:
        f.close()


KNOWN_PROVIDER_URLS: Dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
}


def _discover_models(
    global_opencode_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Discover models from the user's global OpenCode config
    (~/.config/opencode/opencode.json), skipping the 'openguard' provider.
    Returns an empty dict if no models are found.
    """
    discovered: Dict[str, Dict[str, str]] = {}
    for provider_name, provider_conf in (global_opencode_config or {}).get("provider", {}).items():
        if provider_name == "openguard":
            continue
        if not isinstance(provider_conf, dict):
            continue
        for model_id, model_conf in provider_conf.get("models", {}).items():
            model_name = (
                model_conf.get("name", model_id) if isinstance(model_conf, dict) else model_id
            )
            discovered[model_id] = {"name": f"{model_name} (Guarded)"}
    return discovered


def _seed_opencode_dirs() -> None:
    """Copy read-only seed mounts into the writable /root tmpfs.

    The wrapper mounts host ~/.config/opencode and ~/.local/share/opencode as
    /opencode-seed/config:ro and /opencode-seed/share:ro.  This function copies
    them into the writable tmpfs paths so setup_opencode() and opencode itself
    can write freely.
    """
    seeds = [
        (Path("/opencode-seed/config"), Path.home() / ".config" / "opencode"),
        (Path("/opencode-seed/share"), Path.home() / ".local" / "share" / "opencode"),
    ]
    for seed, dest in seeds:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if seed.exists():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(seed, dest, symlinks=True, ignore_dangling_symlinks=True)
            # Fix permissions: seed was chmod a+rX (read-only), make writable
            for dirpath, dirs, files in os.walk(dest):
                for d in dirs:
                    p = os.path.join(dirpath, d)
                    if not os.path.islink(p):
                        os.chmod(p, 0o755)
                for f in files:
                    p = os.path.join(dirpath, f)
                    if not os.path.islink(p):
                        os.chmod(p, 0o644)
            logger.info(f"Copied seed {seed} -> {dest}")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            logger.info(f"No seed at {seed}, created empty {dest}")


def setup_opencode() -> None:
    """Configure OpenCode to use the local OpenGuard instance."""
    logger.info("Installing OpenGuard configuration for OpenCode...")
    print("Installing OpenGuard configuration for OpenCode...", file=sys.stderr)

    # 0. Copy read-only seed mounts into writable tmpfs
    _seed_opencode_dirs()

    # 1. Read ~/.local/share/opencode/auth.json
    auth_path = Path.home() / ".local/share/opencode/auth.json"

    if not auth_path.exists():
        msg = (
            f"Auth file not found at {auth_path}. "
            "Please ensure OpenCode is installed and has run at least once."
        )

        logger.warning(msg)
        print(f"Warning: {msg}", file=sys.stderr)
        return

    try:
        # Create backup
        backup_path = auth_path.with_suffix(".json.bak")
        shutil.copy2(auth_path, backup_path)
        logger.info(f"Backed up auth.json to {backup_path}")

        with open(auth_path, "r") as f:
            auth_data = json.load(f)

    except Exception as e:
        logger.error(f"Failed to read auth.json: {e}")
        print(f"Error reading auth.json: {e}", file=sys.stderr)
        return

    # 2. Read opencode.json in current directory (the OUTPUT config written by this tool)
    config_path = Path.home() / ".config/opencode/opencode.json"
    config_data: Dict[str, Any] = {}

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Could not parse existing {config_path} (likely JSONC/comments). Backup and reset."
            )
            print(
                f"Warning: Could not parse {config_path} (might be JSONC). Creating fresh config.",
                file=sys.stderr,
            )

            shutil.copy2(config_path, config_path.with_suffix(".json.bak"))
            config_data = {}

    # 2b. Read the user's global OpenCode config (INPUT — contains all their configured models)
    global_opencode_config_path = Path.home() / ".config/opencode/opencode.json"
    global_opencode_config: Dict[str, Any] = {}
    if global_opencode_config_path.exists():
        try:
            with open(global_opencode_config_path, "r") as f:
                global_opencode_config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read global OpenCode config: {e}")

    # 3. Migrate credentials to OpenGuard config
    try:
        providers_list: List[Dict[str, Any]] = []

        # Get provider configurations from both local and global OpenCode configs
        opencode_providers_conf = config_data.get("provider", {})
        global_providers_conf = global_opencode_config.get("provider", {})

        for provider_name, creds in auth_data.items():
            if provider_name == "openguard":
                continue

            if creds.get("type") == "oauth":
                logger.warning(f"Provider '{provider_name}' uses OAuth. Skipping migration.")
                continue

            if creds.get("type") == "api" and "key" in creds:
                provider_key = creds["key"]
                provider_url = ""

                # Check URL: local config → global config → known defaults
                for conf in (
                    opencode_providers_conf.get(provider_name),
                    global_providers_conf.get(provider_name),
                ):
                    if isinstance(conf, dict):
                        provider_url = (
                            conf.get("baseURL") or conf.get("options", {}).get("baseURL") or ""
                        )
                        if provider_url:
                            break

                if not provider_url:
                    provider_url = KNOWN_PROVIDER_URLS.get(provider_name.lower(), "")

                # Determine type
                ptype = "openai"
                if "anthropic" in provider_name.lower():
                    ptype = "anthropic"
                elif "google" in provider_name.lower():
                    ptype = "google"
                elif "groq" in provider_name.lower():
                    ptype = "groq"
                elif "xai" in provider_name.lower() or "grok" in provider_name.lower():
                    ptype = "xai"

                providers_list.append({"type": ptype, "key": provider_key, "url": provider_url})

        if providers_list:
            # Write to OpenGuard config
            og_config_dir = Path.home() / ".config" / "openguard"
            og_config_dir.mkdir(parents=True, exist_ok=True)
            og_config_file = og_config_dir / "config.yaml"

            og_data: Dict[str, Any] = {}

            if og_config_file.exists():
                try:
                    with open(og_config_file, "r") as f:
                        og_data = yaml.safe_load(f) or {}
                except Exception:
                    pass

            # Initialize providers list if not present
            if "providers" not in og_data:
                og_data["providers"] = []

            # Merge providers: update existing, append new
            existing_by_key = {p.get("key"): p for p in og_data["providers"] if isinstance(p, dict)}
            for p in providers_list:
                existing = existing_by_key.get(p["key"])
                if existing is not None:
                    existing.update(p)
                else:
                    og_data["providers"].append(p)

            with _secure_open(str(og_config_file)) as f:
                yaml.safe_dump(og_data, f)
            logger.info(f"Migrated credentials to {og_config_file}")

        # Add openguard credential to auth.json if missing
        if "openguard" not in auth_data:
            auth_data["openguard"] = {"type": "api", "key": "sk-openguard-placeholder"}
            with _secure_open(str(auth_path)) as f:
                json.dump(auth_data, f, indent=2)
            logger.info("Added 'openguard' credentials to auth.json")

    except Exception as e:
        logger.error(f"Failed during migration: {e}")
        print(f"Error during migration: {e}", file=sys.stderr)
        # Continue with model discovery even if migration failed partly

    # 4. Discover models
    discovered_models = _discover_models(global_opencode_config)

    if not discovered_models:
        print(
            "Error: No models found in your OpenCode global config "
            f"({global_opencode_config_path}).\n"
            "Please configure at least one provider with models in OpenCode before running "
            "'openguard launch opencode'.",
            file=sys.stderr,
        )
        return

    # 5. Configure provider in opencode.json
    if "provider" not in config_data:
        config_data["provider"] = {}

    config_data["provider"]["openguard"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "OpenGuard",
        "options": {
            "baseURL": f"http://{config.OPENGUARD_HOST.value}:{config.OPENGUARD_PORT.value}/v1"
        },
        "models": discovered_models,
    }

    # Set default model if needed
    preferred_models = [
        "gpt-4o",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "gpt-4-turbo",
        "gpt-4",
    ]

    selected_model = None

    # Check if current model is in our discovered list
    current_model = config_data.get("model")
    if isinstance(current_model, str) and ":" in current_model:
        if current_model.startswith("openguard:"):
            # If already using openguard, keep it if it's still available
            _, m_id = current_model.split(":", 1)
            if m_id in discovered_models:
                selected_model = current_model
        else:
            # If using another provider, see if we have the equiv model
            _, m_id = current_model.split(":", 1)
            if m_id in discovered_models:
                selected_model = f"openguard:{m_id}"

    if not selected_model:
        for m in preferred_models:
            if m in discovered_models:
                selected_model = f"openguard:{m}"
                break

    if not selected_model and discovered_models:
        selected_model = f"openguard:{next(iter(discovered_models.keys()))}"

    if selected_model:
        config_data["model"] = selected_model

    # 6. Write back opencode.json
    try:
        with _secure_open(str(config_path)) as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Updated {config_path} with OpenGuard provider")
    except Exception as e:
        logger.error(f"Failed to update opencode.json: {e}")
        print(f"Error updating opencode.json: {e}", file=sys.stderr)
        return

    print("✅ OpenGuard installed for OpenCode successfully!", file=sys.stderr)
