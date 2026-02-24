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

app = typer.Typer(no_args_is_help=False)


def _apply_config_paths(paths: List[str]) -> None:
    """Override the resolved config value and reset the guards cache."""
    config.OPENGUARD_CONFIG.__value__ = ",".join(paths)
    guards_module._guards_cache = None


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
    launch_setup_opencode()
    typer.echo("Run 'openguard serve' (or 'make dev') to start the proxy.")


@app.command()
def serve(
    config_paths: Optional[List[str]] = typer.Option(
        None, "--config", help="Guard config file path (can be repeated)."
    ),
):
    """Start the OpenGuard server."""
    if config_paths:
        _apply_config_paths(config_paths)
    _run_server()


@app.command()
def install(service: str):
    """Install OpenGuard for a specific service."""
    if service.lower() == "opencode":
        install_opencode()
    else:
        typer.echo(f"Service '{service}' not supported yet.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config_paths: Optional[List[str]] = typer.Option(
        None, "--config", help="Guard config file path (can be repeated)."
    ),
):
    """
    OpenGuard CLI - Guarding proxy for AI.
    """
    if ctx.invoked_subcommand is None:
        if config_paths:
            _apply_config_paths(config_paths)
        _run_server()


@app.command(
    name="launch",
    help="Launch an integration/tool configured to use OpenGuard.",
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
