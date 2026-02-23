"""Guard configuration loading and validation."""

import os
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field

from src import log
from src.config import OPENGUARD_CONFIG

logger = log.setup_logger(__name__)


class GuardBlockedError(Exception):
    """Exception raised when a guard blocks a request."""

    pass


class GuardAction(BaseModel):
    """A single guard action to apply."""

    type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class GuardRule(BaseModel):
    """A guard rule with match criteria and actions."""

    match: Dict[str, Any]
    apply: List[GuardAction]


class GuardsConfig(BaseModel):
    """Top-level guards configuration."""

    guards: List[GuardRule] = Field(default_factory=list)


# Module-level cache for loaded guards
_guards_cache: List[GuardRule] | None = None


def load_guards_config(config_path: str) -> GuardsConfig:
    """
    Load guards configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        GuardsConfig object with parsed guards

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValidationError: If config doesn't match schema
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Guards configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raw_config = {}

        config = GuardsConfig(**raw_config)
        logger.info(f"Loaded {len(config.guards)} guard(s) from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse guards config: {e}")
        raise


def get_guards() -> List[GuardRule]:
    """
    Get all loaded guard rules (cached).

    Returns:
        List of GuardRule objects
    """
    global _guards_cache

    if _guards_cache is None:
        config_path = OPENGUARD_CONFIG.value
        try:
            config = load_guards_config(config_path)
            _guards_cache = config.guards
        except FileNotFoundError:
            logger.warning(f"No guards config found at {config_path}, using empty guards list")
            _guards_cache = []

    return _guards_cache
