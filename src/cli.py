from pathlib import Path
from typing import List, Optional

import typer
import uvicorn

from src import config
from src import guards as guards_module
from src import log as log_module
from src.guards import get_guards
from src.launch.setup import setup_opencode as launch_setup_opencode

# Setup logging
logger = log_module.setup_logger(__name__)

# Read version from pyproject.toml at import time (works with mounted source in Docker)
_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _get_version() -> str:
    try:
        for line in _PYPROJECT.read_text().splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "unknown"


app = typer.Typer(no_args_is_help=False)


def _apply_config_paths(paths: List[str]) -> None:
    """Override the resolved config value and reset the guards cache."""
    config.OPENGUARD_CONFIG.__value__ = ",".join(paths)
    guards_module._guards_cache = None


def _run_server():
    host = config.OPENGUARD_HOST.value
    port = config.OPENGUARD_PORT.value
    log_level = config.OPENGUARD_LOG_LEVEL.value.lower()

    logger.info(f"Starting Louder on {host}:{port}")
    logger.info(f"Config file: {config.OPENGUARD_CONFIG.value}")

    # Load guards at startup
    guards = get_guards()
    logger.info(f"Loaded {len(guards)} guard rules")

    uvicorn.run("src.main:app", host=host, port=port, log_level=log_level, reload=False)


def install_opencode():
    """Configure OpenCode to use the local Louder instance."""
    launch_setup_opencode()
    typer.echo("Run 'openguard serve' (or 'make dev') to start the proxy.")


@app.command()
def serve(
    config_paths: Optional[List[str]] = typer.Option(
        None, "--config", help="Guard config file path (can be repeated)."
    ),
):
    """Start the Louder server."""
    if config_paths:
        _apply_config_paths(config_paths)
    _run_server()


@app.command()
def install(service: str):
    """Install Louder for a specific service."""
    if service.lower() == "opencode":
        install_opencode()
    else:
        typer.echo(f"Service '{service}' not supported yet.")


def _version_callback(value: bool):
    if value:
        typer.echo(_get_version())
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_paths: Optional[List[str]] = typer.Option(
        None, "--config", help="Guard config file path (can be repeated)."
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
):
    """
    Louder CLI - Guarding proxy for AI.
    """
    if ctx.invoked_subcommand is None:
        if config_paths:
            _apply_config_paths(config_paths)
        _run_server()


@app.command()
def version():
    """Show the Louder version."""
    typer.echo(_get_version())


@app.command(
    name="launch",
    help="Launch an integration/tool configured to use Louder.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the integration (e.g. opencode)."),
):
    from src.launch.core import launch_integration

    args = ctx.args
    exit_code = launch_integration(name, args)
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
