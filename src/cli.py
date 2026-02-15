import json
import shutil
from pathlib import Path

import typer
import uvicorn
import yaml

from src import config
from src import log as log_module
from src.guards import get_guards

# Setup logging
logger = log_module.setup_logger(__name__)

app = typer.Typer(no_args_is_help=False)


def _run_server():
    host = config.OPENGUARD_HOST.value
    port = config.OPENGUARD_PORT.value
    log_level = config.OPENGUARD_LOG_LEVEL.value.lower()

    logger.info(f"Starting OpenGuard on {host}:{port}")
    logger.info(f"Config file: {config.OPENGUARD_CONFIG.value}")

    # Load guards at startup
    guards = get_guards()
    logger.info(f"Loaded {len(guards)} guard rules")

    uvicorn.run("src.main:app", host=host, port=port, log_level=log_level, reload=False)


def install_opencode():
    """Configure OpenCode to use the local OpenGuard instance."""
    logger.info("Installing OpenGuard configuration for OpenCode...")

    # 1. Update ~/.local/share/opencode/auth.json
    auth_path = Path.home() / ".local/share/opencode/auth.json"

    if not auth_path.exists():
        logger.error(f"Auth file not found at {auth_path}")
        typer.echo(f"Error: Could not find OpenCode auth file at {auth_path}")
        typer.echo("Please ensure OpenCode is installed and has run at least once.")
        return

    try:
        # Create backup
        backup_path = auth_path.with_suffix(".json.bak")
        shutil.copy2(auth_path, backup_path)
        logger.info(f"Backed up auth.json to {backup_path}")

        with open(auth_path, "r") as f:
            auth_data = json.load(f)

        # Migrating existing credentials to OpenGuard config
        providers_list = []

        # Check for Base URLs in opencode.json
        local_opencode_path = Path.cwd() / "opencode.json"

        opencode_providers_conf = {}
        if local_opencode_path.exists():
            try:
                with open(local_opencode_path, "r") as f:
                     local_data = json.load(f)
                     opencode_providers_conf = local_data.get("provider", {})
            except Exception:
                pass

        for provider_name, creds in auth_data.items():
            if provider_name == "openguard":
                continue

            if creds.get("type") == "oauth":
                logger.warning(f"Provider '{provider_name}' uses OAuth. Skipping migration.")
                continue

            if creds.get("type") == "api" and "key" in creds:
                provider_key = creds["key"]
                provider_url = ""

                # Check known URL for provider in opencode.json
                p_conf = opencode_providers_conf.get(provider_name)
                if p_conf:
                    provider_url = (
                        p_conf.get("baseURL")
                        or p_conf.get("options", {}).get("baseURL")
                        or ""
                    )

                # Determine type
                ptype = "openai"
                if "anthropic" in provider_name.lower():
                    ptype = "anthropic"

                providers_list.append({
                    "type": ptype,
                    "key": provider_key,
                    "url": provider_url
                })

        if providers_list:
            # Write to OpenGuard config
            og_config_dir = Path.home() / ".config" / "openguard"
            og_config_dir.mkdir(parents=True, exist_ok=True)
            og_config_file = og_config_dir / "config.yaml"

            og_data = {}
            if og_config_file.exists():
                try:
                    with open(og_config_file, "r") as f:
                        og_data = yaml.safe_load(f) or {}
                except Exception:
                    pass

            og_data["providers"] = providers_list

            with open(og_config_file, "w") as f:
                yaml.safe_dump(og_data, f)

            logger.info(f"Migrated credentials to {og_config_file}")

        # Add openguard credential
        auth_data["openguard"] = {
            "type": "api",
            "key": "sk-openguard"
        }

        with open(auth_path, "w") as f:
            json.dump(auth_data, f, indent=2)

        logger.info("Added 'openguard' credentials to auth.json")

    except Exception as e:
        logger.error(f"Failed to update auth.json: {e}")
        typer.echo(f"Error updating auth.json: {e}")
        return

    # 2. Update opencode.json in current directory
    config_path = Path.cwd() / "opencode.json"
    config_data = {}

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Could not parse existing {config_path} "
                "(likely JSONC/comments). Backup and reset."
            )
            typer.echo(
                f"Warning: Could not parse {config_path} "
                "(might be JSONC). Creating fresh config."
            )
            shutil.copy2(config_path, config_path.with_suffix(".json.bak"))
            config_data = {}

    # Configure provider
    if "provider" not in config_data:
        config_data["provider"] = {}

    config_data["provider"]["openguard"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "OpenGuard",
        "options": {
            "baseURL": f"http://{config.OPENGUARD_HOST.value}:{config.OPENGUARD_PORT.value}/v1"
        },
        "models": {
            "gpt-4o": { "name": "GPT-4o (Guarded)" },
            "claude-3-5-sonnet": { "name": "Claude 3.5 Sonnet (Guarded)" }
        }
    }

    # Set default model to verify installation
    # Overwrite model if it's set, or set if missing. Prompt suggests overwrite is fine.
    config_data["model"] = "openguard:gpt-4o"

    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Updated {config_path} with OpenGuard provider")
    except Exception as e:
        logger.error(f"Failed to update opencode.json: {e}")
        typer.echo(f"Error updating opencode.json: {e}")
        return

    typer.echo("✅ OpenGuard installed for OpenCode successfully!")
    typer.echo("Run 'openguard serve' (or 'make dev') to start the proxy.")


@app.command()
def serve():
    """Start the OpenGuard server."""
    _run_server()


@app.command()
def install(service: str):
    """Install OpenGuard for a specific service."""
    if service.lower() == "opencode":
        install_opencode()
    else:
        typer.echo(f"Service '{service}' not supported yet.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    OpenGuard CLI - Guarding proxy for AI.
    """
    if ctx.invoked_subcommand is None:
        _run_server()


if __name__ == "__main__":
    app()
