from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src import llm


class LLMRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, instance: "llm.LLM"):
        self._registry[instance.id] = instance

    def get(self, llm_id) -> "Optional[llm.LLM]":
        return self._registry.get(llm_id)

    def unregister(self, instance: "llm.LLM"):
        self._registry.pop(instance.id, None)

    def list_all(self):
        return list(self._registry.values())


llm_registry = LLMRegistry()
