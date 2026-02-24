from typing import Any, Dict, List, Type

from pydantic import BaseModel


class GuardMeta(BaseModel):
    name: str
    description: str
    config_schema: Type[BaseModel]
    docs: str
    examples: List[Dict[str, Any]]
