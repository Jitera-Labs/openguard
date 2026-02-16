from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class LaunchStrategy(Enum):
    ENV = "env"
    FILE = "file"
    ARG = "arg"


@dataclass
class StrategyConfig:
    type: LaunchStrategy
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Integration:
    name: str
    default_command: str
    strategies: List[StrategyConfig] = field(default_factory=list)
