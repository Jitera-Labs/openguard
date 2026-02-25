from copy import deepcopy
from typing import Any, Dict, Optional

from . import format, llm, log, selection
from .chat_node import ChatNode

logger = log.setup_logger(__name__)


class Chat:
    tail: ChatNode
    llm: Optional["llm.LLM"]

    @staticmethod
    def from_conversation(messages):
        tail = ChatNode.from_conversation(messages)
        return Chat(tail=tail)

    @staticmethod
    def from_payload(payload: dict, provider: str = "openai"):
        if provider == "anthropic":
            return Chat.from_anthropic_payload(payload)

        messages = payload.get("messages", []) if isinstance(payload, dict) else payload
        return Chat.from_conversation(messages)

    @staticmethod
    def from_anthropic_payload(payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("Anthropic payload must be a dictionary")

        source_payload = deepcopy(payload)
        node_specs = []

        system_content = payload.get("system")
        if isinstance(system_content, str):
            node_specs.append(
                {
                    "role": "system",
                    "content": system_content,
                    "meta": {
                        "anthropic_ref": {
                            "section": "system",
                            "kind": "string",
                        }
                    },
                }
            )
        elif isinstance(system_content, list):
            for block_index, block in enumerate(system_content):
                node_specs.append(
                    Chat._anthropic_block_to_node_spec(block, "system", None, block_index)
                )

        for message_index, message in enumerate(payload.get("messages", [])):
            role = message.get("role", "")
            content = message.get("content", "")

            if isinstance(content, list):
                for block_index, block in enumerate(content):
                    node_specs.append(
                        Chat._anthropic_block_to_node_spec(block, role, message_index, block_index)
                    )
            else:
                node_specs.append(
                    {
                        "role": role,
                        "content": content,
                        "meta": {
                            "anthropic_ref": {
                                "section": "messages",
                                "message_index": message_index,
                                "kind": "message_string",
                            }
                        },
                    }
                )

        tail = ChatNode.from_conversation(
            [
                {
                    "role": spec["role"],
                    "content": spec.get("content", ""),
                    "meta": spec.get("meta", {}),
                }
                for spec in node_specs
            ]
        )

        if tail is not None:
            for node, spec in zip(tail.parents(), node_specs):
                node.meta = spec.get("meta", {})

        return Chat(
            tail=tail,
            provider="anthropic",
            source_payload=source_payload,
        )

    @staticmethod
    def from_tail(chat: "Chat"):
        new_tail = ChatNode(role=chat.tail.role, content=chat.tail.content)
        return Chat(tail=new_tail)

    def __init__(self, **kwargs):
        self.tail = kwargs.get("tail")
        self.llm = kwargs.get("llm")
        self.provider = kwargs.get("provider", "openai")
        self.source_payload = kwargs.get("source_payload", None)
        self.chat_node_type = ChatNode

        self.Chat = Chat
        self.ChatNode = self.chat_node_type

    def clone(self):
        if self.provider == "anthropic":
            return Chat.from_payload(self.serialize("anthropic"), provider="anthropic")
        return Chat.from_conversation(self.history())

    def has_substring(self, substring):
        return any(
            isinstance(msg.content, str) and substring in msg.content for msg in self.plain()
        )

    def match(self, **kwargs):
        return selection.match(self, **kwargs)

    def match_one(self, **kwargs):
        candidates = self.match(**kwargs)
        if not candidates:
            return None
        return candidates[0]

    def add_message(self, role, content):
        logger.debug(
            f"Chat message: {role}: "
            f"{content[:50] if isinstance(content, str) else str(content)[:50]}"
        )

        child = self.__create_node(role=role, content=content)
        if self.tail:
            self.tail.add_child(child)
        self.tail = child

        return self.tail

    def tool_call(self, tool_call):
        self.add_message("assistant", "")
        self.tail.tool_calls = [tool_call]
        return self.tail

    def tool(self, id, content):
        self.add_message("tool", content)
        self.tail.tool_call_id = id
        return self.tail

    def user(self, content):
        return self.add_message("user", content)

    def assistant(self, content):
        return self.add_message("assistant", content)

    def system(self, content):
        self.tail.ancestor().add_parent(self.__create_node(role="system", content=content))
        return self.tail

    def insert(self, after: ChatNode, role, content):
        new_node = self.__create_node(role=role, content=content)
        after.insert_child(new_node)

        if self.tail == after:
            self.tail = new_node

        return self.tail

    def plain(self):
        if not self.tail:
            return []
        return self.tail.parents()

    def history(self):
        if not self.tail:
            return []
        return self.tail.history()

    def serialize(self, provider: Optional[str] = None):
        active_provider = provider or self.provider or "openai"

        if active_provider == "anthropic":
            return self._serialize_anthropic()

        return self.history()

    def serialized(self, provider: Optional[str] = None):
        return self.serialize(provider=provider)

    def root(self):
        if not self.tail:
            return None
        return self.tail.ancestor()

    def text(self):
        # __str__ already does exactly this
        return f"{self}"

    def __create_node(self, **kwargs):
        NodeType = self.chat_node_type
        return NodeType(**kwargs)

    async def advance(self):
        """
        Advance the chat completion

        Will not be streamed back to the client
        """

        if not self.llm:
            raise ValueError("Chat: unable to advance without an LLM")

        response = await self.llm.chat_completion(chat=self)
        params = await self.llm.resolve_request_params()
        self.assistant(self.llm.get_response_content(params, response))
        return response

    async def emit_advance(self, **kwargs):
        """
        Emit the next step in the chat completion

        Will be streamed back to the client
        """

        if not self.llm:
            raise ValueError("Chat: unable to advance without an LLM")

        response = await self.llm.stream_chat_completion(chat=self, **kwargs)
        self.assistant(response)
        return response

    async def emit_status(self, status):
        """
        Emit a status message

        Will be streamed back to the client
        """

        if not self.llm:
            raise ValueError("Chat: unable to emit status without an LLM")

        await self.llm.emit_status(status)

    async def sanitise_artifacts(self):
        tail = self.tail

        while tail:
            tail_content = tail.content
            tail.content = format.remove_html_code_blocks(tail_content)
            tail = tail.parent

    def __str__(self):
        return "\n".join([str(msg) for msg in self.plain()])

    @staticmethod
    def _anthropic_block_to_node_spec(
        block: Any,
        message_role: str,
        message_index: Optional[int],
        block_index: int,
    ) -> Dict[str, Any]:
        block_type = block.get("type") if isinstance(block, dict) else None
        ref = {
            "section": "messages" if message_index is not None else "system",
            "message_index": message_index,
            "block_index": block_index,
            "kind": "block",
        }

        if block_type == "tool_use":
            return {
                "role": "tool_use",
                "content": deepcopy(block.get("input", {})),
                "meta": {
                    "anthropic_ref": ref,
                    "anthropic_block_type": "tool_use",
                    "anthropic_parent_role": message_role,
                    "anthropic_tool_use_id": block.get("id"),
                    "anthropic_tool_name": block.get("name"),
                },
            }

        if block_type == "tool_result":
            return {
                "role": "tool_result",
                "content": deepcopy(block.get("content", "")),
                "meta": {
                    "anthropic_ref": ref,
                    "anthropic_block_type": "tool_result",
                    "anthropic_parent_role": message_role,
                    "anthropic_tool_use_id": block.get("tool_use_id"),
                    "anthropic_is_error": block.get("is_error"),
                },
            }

        if block_type == "text":
            node_content = block.get("text", "")
        else:
            node_content = deepcopy(block)

        return {
            "role": message_role,
            "content": node_content,
            "meta": {
                "anthropic_ref": ref,
                "anthropic_block_type": block_type,
                "anthropic_parent_role": message_role,
            },
        }

    @staticmethod
    def _node_to_anthropic_block(node: ChatNode, existing: Optional[dict] = None) -> dict:
        block = deepcopy(existing) if isinstance(existing, dict) else {}
        block_type = node.meta.get("anthropic_block_type")

        if node.role == "tool_use" or block_type == "tool_use":
            block["type"] = "tool_use"
            if node.meta.get("anthropic_tool_use_id") is not None:
                block["id"] = node.meta.get("anthropic_tool_use_id")
            if node.meta.get("anthropic_tool_name") is not None:
                block["name"] = node.meta.get("anthropic_tool_name")
            block["input"] = deepcopy(node.content)
            return block

        if node.role == "tool_result" or block_type == "tool_result":
            block["type"] = "tool_result"
            if node.meta.get("anthropic_tool_use_id") is not None:
                block["tool_use_id"] = node.meta.get("anthropic_tool_use_id")
            if node.meta.get("anthropic_is_error") is not None:
                block["is_error"] = node.meta.get("anthropic_is_error")
            block["content"] = deepcopy(node.content)
            return block

        if block_type == "text" or isinstance(node.content, str):
            block["type"] = "text"
            block["text"] = node.content if isinstance(node.content, str) else str(node.content)
            return block

        if isinstance(node.content, dict):
            return deepcopy(node.content)

        block["type"] = block_type or "text"
        if block["type"] == "text":
            block["text"] = node.content if isinstance(node.content, str) else str(node.content)
        else:
            block["content"] = deepcopy(node.content)
        return block

    def _serialize_anthropic(self) -> dict:
        payload = deepcopy(self.source_payload) if isinstance(self.source_payload, dict) else {}
        payload.setdefault("messages", [])

        unmapped_nodes = []

        for node in self.plain():
            ref = node.meta.get("anthropic_ref", {})
            section = ref.get("section")
            kind = ref.get("kind")

            if not section:
                unmapped_nodes.append(node)
                continue

            if section == "system":
                if kind == "string":
                    payload["system"] = node.content
                    continue

                block_index = ref.get("block_index")
                system_value = payload.get("system", [])
                if not isinstance(system_value, list):
                    system_value = []
                while len(system_value) <= block_index:
                    system_value.append({"type": "text", "text": ""})
                existing_block = system_value[block_index]
                system_value[block_index] = self._node_to_anthropic_block(node, existing_block)
                payload["system"] = system_value
                continue

            if section == "messages":
                message_index = ref.get("message_index")
                while len(payload["messages"]) <= message_index:
                    payload["messages"].append({"role": "user", "content": []})

                message = payload["messages"][message_index]
                if kind == "message_string":
                    message["content"] = node.content
                    if node.role:
                        message["role"] = node.role
                    continue

                block_index = ref.get("block_index")
                message_content = message.get("content", [])
                if not isinstance(message_content, list):
                    message_content = []
                while len(message_content) <= block_index:
                    message_content.append({"type": "text", "text": ""})

                existing_block = message_content[block_index]
                message_content[block_index] = self._node_to_anthropic_block(node, existing_block)
                if node.meta.get("anthropic_parent_role"):
                    message["role"] = node.meta.get("anthropic_parent_role")
                elif node.role in ("user", "assistant"):
                    message["role"] = node.role
                message["content"] = message_content
                continue

            unmapped_nodes.append(node)

        for node in unmapped_nodes:
            if node.role == "system":
                system_value = payload.get("system")
                if system_value is None:
                    payload["system"] = node.content
                elif isinstance(system_value, str):
                    payload["system"] = [
                        {"type": "text", "text": system_value},
                        {
                            "type": "text",
                            "text": (
                                node.content if isinstance(node.content, str) else str(node.content)
                            ),
                        },
                    ]
                elif isinstance(system_value, list):
                    system_value.append(self._node_to_anthropic_block(node))
                continue

            if node.role in ("tool_use", "tool_result"):
                role = node.meta.get("anthropic_parent_role")
                if role is None:
                    role = "assistant" if node.role == "tool_use" else "user"
                payload["messages"].append(
                    {
                        "role": role,
                        "content": [self._node_to_anthropic_block(node)],
                    }
                )
                continue

            if isinstance(node.content, list):
                content_value = deepcopy(node.content)
            elif isinstance(node.content, str):
                content_value = [{"type": "text", "text": node.content}]
            else:
                content_value = [self._node_to_anthropic_block(node)]

            payload["messages"].append(
                {
                    "role": node.role if node.role in ("user", "assistant") else "user",
                    "content": content_value,
                }
            )

        return payload
