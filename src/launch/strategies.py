import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import yaml


class Strategy(ABC):
    @abstractmethod
    def apply(self, command: List[str], env: Dict[str, str], params: Dict[str, Any]) -> None:
        """
        Apply the strategy to modify command or environment variables.
        params: The parameters specific to this strategy instance.
        """
        pass


class EnvVarStrategy(Strategy):
    def apply(self, command: List[str], env: Dict[str, str], params: Dict[str, Any]) -> None:
        """
        Injects environment variables.
        Expected params:
            env_vars: Dict[str, str] - Map of environment variables to set
        """
        env_vars = params.get("env_vars", {})
        if not isinstance(env_vars, dict):
            return

        for key, value in env_vars.items():
            env[str(key)] = str(value)


class ConfigFileStrategy(Strategy):
    def apply(self, command: List[str], env: Dict[str, str], params: Dict[str, Any]) -> None:
        """
        Reads/modifies/writes JSON/YAML config files.
        Expected params:
            file_path: str - Path to the config file (can be absolute or relative to home ~)
            data: Dict[str, Any] - Data to merge/set in the config file
            format: str - "json" or "yaml" (optional, inferred from extension if missing)
        """
        file_path_str = params.get("file_path")
        if not file_path_str:
            return

        file_path = Path(os.path.expanduser(file_path_str))
        data_to_merge = params.get("data", {})

        # Determine format
        fmt = params.get("format")
        if not fmt:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                fmt = "yaml"
            else:
                fmt = "json"

        existing_data = {}
        if file_path.exists():
            try:
                if fmt == "json":
                    with open(file_path, "r") as f:
                        # Handle empty files
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                elif fmt == "yaml":
                    with open(file_path, "r") as f:
                        existing_data = yaml.safe_load(f) or {}
            except Exception:
                # Log warning in real app, but here we just proceed with empty dict or partial read
                pass

        # Deep merge data_to_merge into existing_data
        # Simple recursive merge for this implementation
        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for k, v in source.items():
                if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                    deep_merge(target[k], v)
                else:
                    target[k] = v

        deep_merge(existing_data, data_to_merge)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write back
        try:
            if fmt == "json":
                with open(file_path, "w") as f:
                    json.dump(existing_data, f, indent=2)
            elif fmt == "yaml":
                with open(file_path, "w") as f:
                    yaml.dump(existing_data, f)
        except Exception:
            # Handle write errors gracefully
            pass


class CliArgStrategy(Strategy):
    def apply(self, command: List[str], env: Dict[str, str], params: Dict[str, Any]) -> None:
        """
        Appends CLI args.
        Expected params:
            args: List[str] - List of arguments to append
        """
        args = params.get("args", [])
        if not isinstance(args, list):
            return

        command.extend([str(arg) for arg in args])
