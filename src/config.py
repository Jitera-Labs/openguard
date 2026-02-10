import os
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union

T = TypeVar("T")


class ConfigDict(Dict[str, Union[str, int, float, bool]]):
    @classmethod
    def from_string(cls, value: str) -> "ConfigDict":
        result = cls()
        if not value:
            return result
        pairs = value.split(",")
        for pair in pairs:
            key, val = pair.split("=")
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


class IntList(List[int]):
    @classmethod
    def from_string(cls, value: str) -> "IntList":
        return (
            cls(int(item.strip()) for item in value.split(";") if item.strip())
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
        raw_value = os.getenv(self.name, self.default)
        if isinstance(raw_value, list):
            raw_value = raw_value[0] if raw_value else ""
        return self._convert_value(raw_value)

    def _resolve_wildcard(self) -> List[T]:
        prefix = self.name.replace("*", "")
        matching_vars = [
            (key, value) for key, value in os.environ.items() if key.startswith(prefix)
        ]

        if not matching_vars:
            if isinstance(self.default, str):
                return [self._convert_value(self.default)] if self.default else []
            return self.default

        return [self._convert_value(value) for _, value in sorted(matching_vars)]

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

OPENGUARD_CONFIG = Config[str](
    name="OPENGUARD_CONFIG",
    type=str,
    default="./guards.yaml",
    description="Path to guards configuration file.",
)

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
    default="http://localhost:8000",
    description="Public URL where this OpenGuard instance is accessible.",
)

OPENGUARD_PORT = Config[int](
    name="OPENGUARD_PORT",
    type=int,
    default="8000",
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
    default="*",
    description="Semicolon-separated list of allowed CORS origins.",
)

MODEL_FILTER = Config[ConfigDict](
    name="OPENGUARD_MODEL_FILTER",
    type=ConfigDict,
    default="",
    description="Filter to apply to downstream models list (Hasura-style filter).",
)


if __name__ == "__main__":
    # Render documentation
    configs = [item for item in globals().values() if isinstance(item, Config)]

    docs = """
# OpenGuard Configuration

OpenGuard is configured using environment variables. Following options are available:
  """

    for config in configs:
        docs += f"\n\n## {config.name}\n"
        docs += f"> **Type**: `{config.type.__name__}`<br/>\n"
        docs += f"> **Default**: `{config.default}`<br/>\n"

        if config.description:
            docs += f"\n{config.description}\n"

    print(docs)
