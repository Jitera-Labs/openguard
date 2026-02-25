import json
import os
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

import yaml

# ----------------- Persistent Configuration -----------------

PERSISTENT_CONFIG: Dict[str, Any] = {}


def _load_persistent_config():
    global PERSISTENT_CONFIG
    config_dir = Path.home() / ".config" / "openguard"
    config_file = config_dir / "config.yaml"
    json_config_file = config_dir / "config.json"

    data = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            pass
    elif json_config_file.exists():
        try:
            with open(json_config_file, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    if data:
        PERSISTENT_CONFIG.update(data)


_load_persistent_config()

T = TypeVar("T")


class ConfigDict(Dict[str, Union[str, int, float, bool]]):
    @classmethod
    def from_string(cls, value: str) -> "ConfigDict":
        result = cls()
        if not value:
            return result

        # Handle if value is somehow passed as a dict (defensive)
        if isinstance(value, dict):
            return cls(value)

        pairs = value.split(",")
        for pair in pairs:
            if "=" not in pair:
                continue
            key, val = pair.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Try to parse the value as int, float, or bool
            if val.lower() == "true":
                result[key] = True
            elif val.lower() == "false":
                result[key] = False
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
        return result


class StrList(List[str]):
    @classmethod
    def from_string(cls, value: str) -> "StrList":
        return (
            cls(item.strip() for item in value.split(";") if item.strip())
            if value.strip()
            else cls()
        )


class FloatList(List[float]):
    @classmethod
    def from_string(cls, value: str) -> "FloatList":
        return (
            cls(float(item.strip()) for item in value.split(";") if item.strip())
            if value.strip()
            else cls()
        )


class BoolList(List[bool]):
    @classmethod
    def from_string(cls, value: str) -> "BoolList":
        return (
            cls(item.strip().lower() == "true" for item in value.split(";") if item.strip())
            if value.strip()
            else cls()
        )


class IntList(List[int]):
    @classmethod
    def from_string(cls, value: str) -> "IntList":
        return (
            cls(int(item.strip()) for item in value.split(";") if item.strip())
            if value.strip()
            else cls()
        )


class Config(Generic[T]):
    name: str
    type: Type[T]
    default: str
    description: Optional[str]
    __value__: T

    def __init__(self, name: str, type: Type[T], default: str, description: Optional[str] = None):
        self.name = name
        self.type = type
        self.default = default
        self.description = description
        self.__value__ = self.resolve_value()  # type: ignore

    @property
    def value(self) -> T:
        return self.__value__

    def resolve_value(self) -> Union[T, List[T]]:
        if "*" in self.name:
            return self._resolve_wildcard()
        else:
            return self._resolve_single()

    def _resolve_single(self) -> T:
        # 1. Check persistent config (mapping OPENGUARD_XYZ -> xyz)
        config_key = self.name.replace("OPENGUARD_", "").lower()
        if config_key in PERSISTENT_CONFIG:
            val = PERSISTENT_CONFIG[config_key]
            # For complex types like StrList, ConfigDict, handle direct assignment from YAML types
            if issubclass(self.type, (ConfigDict, dict)) and isinstance(val, dict):
                return val  # type: ignore
            if issubclass(self.type, list) and isinstance(val, list):
                # Ensure elements are scalar strings/ints/floats/bools depending on subtype.
                # Assuming YAML structure matches.
                return val  # type: ignore

            return self._convert_value(str(val))

        # 2. Check environment variable
        raw_value = os.getenv(self.name, self.default)
        if isinstance(raw_value, list):
            raw_value = raw_value[0] if raw_value else ""
        return self._convert_value(raw_value)

    def _resolve_wildcard(self) -> List[T]:
        # Gather values from both Persistent Config 'providers' and Env Vars
        values: List[T] = []

        # 1. Extract from PERSISTENT_CONFIG['providers']
        providers = PERSISTENT_CONFIG.get("providers", [])
        if isinstance(providers, list):
            target_type = "openai"
            target_field = "key"

            if "ANTHROPIC" in self.name:
                target_type = "anthropic"

            if "URL" in self.name:
                target_field = "url"

            for p in providers:
                if not isinstance(p, dict):
                    continue
                # Default type is openai
                ptype = p.get("type", "openai").lower()

                if ptype == target_type:
                    val = p.get(target_field)
                    if val:
                        values.append(self._convert_value(str(val)))

        # 2. Extract from Environment Variables
        prefix = self.name.replace("*", "")
        matching_vars = [
            (key, value) for key, value in os.environ.items() if key.startswith(prefix)
        ]

        env_values = [self._convert_value(value) for _, value in sorted(matching_vars)]

        # Combine: Config file providers first, then Env vars.
        all_values = values + env_values

        if not all_values:
            if isinstance(self.default, str) and self.default:
                return [self._convert_value(self.default)]
            elif isinstance(self.default, list):
                return self.default  # type: ignore
            else:
                return []

        return all_values

    def _convert_value(self, value: str) -> T:
        if issubclass(self.type, (StrList, IntList, FloatList, BoolList, ConfigDict)):
            return self.type.from_string(value)  # type: ignore
        elif self.type is str:
            return value  # type: ignore
        elif self.type is int:
            return int(value)  # type: ignore
        elif self.type is float:
            return float(value)  # type: ignore
        elif self.type is bool:
            return value.lower() in ("true", "1", "yes", "on")  # type: ignore
        else:
            return self.type(value)  # type: ignore


# ----------------- OpenGuard Configuration -----------------
# NOTE: Config instances below resolve their values at import time by reading
# environment variables and the persistent config file. This means env vars set
# after this module is first imported will NOT be reflected. In tests, set env
# vars before importing this module, or patch Config.value / Config.__value__.

OPENGUARD_CONFIG = Config[str](
    name="OPENGUARD_CONFIG",
    type=str,
    default="./guards.yaml",
    description="Comma-separated list of guard config file paths, merged in order.",
)


def get_config_paths() -> List[str]:
    """Return the ordered list of config file paths from all config sources.

    Sources (in order):
    1. OPENGUARD_CONFIG (comma-separated, or single path)
    2. OPENGUARD_CONFIG_<NAME> env vars, sorted by NAME alphabetically
    """
    paths: List[str] = [p.strip() for p in OPENGUARD_CONFIG.value.split(",") if p.strip()]

    prefix = "OPENGUARD_CONFIG_"
    extra_keys = sorted(key for key in os.environ if key.startswith(prefix))
    for key in extra_keys:
        value = os.environ[key].strip()
        if value:
            paths.append(value)

    # Deduplicate while preserving order
    seen: Dict[str, None] = {}
    for path in paths:
        seen[path] = None
    return list(seen)


OPENGUARD_OPENAI_URLS = Config[str](
    name="OPENGUARD_OPENAI_URL_*",
    type=str,
    default="http://localhost:11434/v1",
    description=(
        "Downstream OpenAI-compatible API URLs. "
        "Use wildcard naming like OPENGUARD_OPENAI_URL_OLLAMA."
    ),
)

OPENGUARD_OPENAI_KEYS = Config[str](
    name="OPENGUARD_OPENAI_KEY_*",
    type=str,
    default="",
    description=(
        "API keys for downstream APIs. Use wildcard naming like OPENGUARD_OPENAI_KEY_OLLAMA."
    ),
)

OPENGUARD_ANTHROPIC_URLS = Config[str](
    name="OPENGUARD_ANTHROPIC_URL_*",
    type=str,
    default="",
    description=(
        "Downstream Anthropic Chat API URLs. Use wildcard naming like OPENGUARD_ANTHROPIC_URL_1."
    ),
)

OPENGUARD_ANTHROPIC_KEYS = Config[str](
    name="OPENGUARD_ANTHROPIC_KEY_*",
    type=str,
    default="",
    description=(
        "API keys for downstream Anthropic APIs. "
        "Use wildcard naming like OPENGUARD_ANTHROPIC_KEY_1."
    ),
)

OPENGUARD_API_KEY = Config[str](
    name="OPENGUARD_API_KEY",
    type=str,
    default="",
    description="API key required to access this OpenGuard proxy. Leave empty to disable auth.",
)

OPENGUARD_API_KEYS = Config[StrList](
    name="OPENGUARD_API_KEYS",
    type=StrList,
    default="",
    description="Additional API keys (semicolon-separated) for proxy access.",
)

OPENGUARD_PUBLIC_URL = Config[str](
    name="OPENGUARD_PUBLIC_URL",
    type=str,
    default="http://localhost:23294",
    description="Public URL where this OpenGuard instance is accessible.",
)

OPENGUARD_PORT = Config[int](
    name="OPENGUARD_PORT",
    type=int,
    default="23294",
    description="Port to run the OpenGuard server on.",
)

OPENGUARD_HOST = Config[str](
    name="OPENGUARD_HOST",
    type=str,
    default="0.0.0.0",
    description="Host to bind the OpenGuard server to.",
)

OPENGUARD_LOG_LEVEL = Config[str](
    name="OPENGUARD_LOG_LEVEL",
    type=str,
    default="INFO",
    description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
)

OPENGUARD_CORS_ORIGINS = Config[StrList](
    name="OPENGUARD_CORS_ORIGINS",
    type=StrList,
    default="",
    description="Semicolon-separated list of allowed CORS origins.",
)

MODEL_FILTER = Config[ConfigDict](
    name="OPENGUARD_MODEL_FILTER",
    type=ConfigDict,
    default="",
    description="Filter to apply to downstream models list (Hasura-style filter).",
)

INTERMEDIATE_OUTPUT = Config[bool](
    name="BOOST_INTERMEDIATE_OUTPUT",  # Legacy "Boost" name — kept for env var compatibility
    type=bool,
    default="false",
    description="Stream intermediate reasoning steps to the client.",
)

EXTRA_LLM_PARAMS = Config[ConfigDict](
    name="BOOST_EXTRA_LLM_PARAMS",  # Legacy "Boost" product name — kept for env var compatibility
    type=ConfigDict,
    default="",
    description="Extra parameters to inject into every LLM request.",
)

BOOST_PUBLIC_URL = Config[str](
    name="BOOST_PUBLIC_URL",  # Legacy "Boost" product name — kept for env var compatibility
    type=str,
    default="http://localhost:23294",
    description="Public URL for Boost artifacts.",
)

STATUS_STYLE = Config[str](
    name="STATUS_STYLE",
    type=str,
    default="md:codeblock",
    description="Style for status messages (md:codeblock, md:h1, md:h2, md:h3, plain, none)",
)

if __name__ == "__main__":
    # Render documentation
    configs = [item for item in globals().values() if isinstance(item, Config)]

    # Reload config to ensure clean state
    # PERSISTENT_CONFIG = {}
    _load_persistent_config()

    docs = """
# OpenGuard Configuration

OpenGuard is configured using environment variables or a configuration file at
`~/.config/openguard/config.yaml`.

## Usage
Create `~/.config/openguard/config.yaml` with the following structure:
```yaml
port: 23294
providers:
  - type: openai
    key: sk-proj-...
    url: ...
  - type: anthropic
    key: sk-ant-...
    url: ...

# Other options (match variable names without OPENGUARD_ prefix, lowercase)
model_filter:
  ...
```

Following options are available:
  """

    for config in configs:
        docs += f"\n\n## {config.name}\n"
        docs += f"> **Type**: `{config.type.__name__}`<br/>\n"
        docs += f"> **Default**: `{config.default}`<br/>\n"

        if config.description:
            docs += f"\n{config.description}\n"

    print(docs)
