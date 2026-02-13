import copy

from src.chat import Chat


def test_anthropic_deserialize_creates_tool_nodes():
    payload = {
        "model": "claude-3-5-sonnet",
        "system": "You are concise.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Find weather for SF"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "San Francisco"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": [{"type": "text", "text": "63F and sunny"}],
                    }
                ],
            },
        ],
    }

    chat = Chat.from_payload(payload, provider="anthropic")
    nodes = chat.plain()

    assert [node.role for node in nodes] == ["system", "user", "tool_use", "tool_result"]
    assert nodes[2].meta["anthropic_tool_use_id"] == "toolu_123"
    assert nodes[3].meta["anthropic_tool_use_id"] == "toolu_123"


def test_anthropic_round_trip_preserves_unknown_fields_verbatim():
    payload = {
        "model": "claude-3-5-sonnet",
        "custom_top": {"a": 1, "b": [1, 2, 3]},
        "system": [
            {"type": "text", "text": "S1", "cache_control": {"type": "ephemeral"}},
            {"type": "redacted_thinking", "data": {"opaque": True}},
        ],
        "messages": [
            {
                "role": "user",
                "name": "alice",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "citations": [{"kind": "x", "value": 3}],
                    },
                    {"type": "custom_block", "z": 42, "nested": {"k": "v"}},
                ],
                "unknown_message_field": {"preserve": True},
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_9",
                        "name": "lookup",
                        "input": {"q": "abc"},
                        "extra": "keep",
                    }
                ],
            },
        ],
    }

    original = copy.deepcopy(payload)
    chat = Chat.from_payload(payload, provider="anthropic")
    round_trip = chat.serialize("anthropic")

    assert round_trip == original


def test_anthropic_serialization_updates_known_mapped_nodes_only():
    payload = {
        "system": "Original system",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello", "extra": 1}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search",
                        "input": {"q": "old"},
                        "extra": {"keep": True},
                    }
                ],
            },
        ],
        "top_unknown": {"keep": [1, 2, 3]},
    }

    chat = Chat.from_payload(payload, provider="anthropic")
    nodes = chat.plain()

    nodes[0].content = "Updated system"
    nodes[1].content = "hello updated"
    nodes[2].content = {"q": "new"}

    out = chat.serialize("anthropic")

    assert out["system"] == "Updated system"
    assert out["messages"][0]["content"][0]["text"] == "hello updated"
    assert out["messages"][1]["content"][0]["input"] == {"q": "new"}
    assert out["messages"][0]["content"][0]["extra"] == 1
    assert out["messages"][1]["content"][0]["extra"] == {"keep": True}
    assert out["top_unknown"] == {"keep": [1, 2, 3]}


def test_openai_history_keeps_passthrough_message_fields():
    messages = [
        {
            "role": "user",
            "content": "hi",
            "name": "alice",
            "custom_unknown": {"x": 1},
        }
    ]

    chat = Chat.from_conversation(messages)
    history = chat.history()

    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hi"
    assert history[0]["name"] == "alice"
    assert history[0]["custom_unknown"] == {"x": 1}
