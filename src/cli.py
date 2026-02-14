import typer
import uvicorn
from src import config, log as log_module
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


@app.command()
def serve():
    """Start the OpenGuard server."""
    _run_server()


@app.command()
def install(service: str):
    """Install OpenGuard for a specific service."""
    typer.echo(f"Installing openguard into {service}...")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    OpenGuard CLI - Guarding proxy for AI.
    """
    if ctx.invoked_subcommand is None:
        _run_server()


if __name__ == "__main__":
    app()
