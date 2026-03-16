"""Unit tests for src/guard_types/llm_tool_inspection.py."""

from typing import cast

import pytest

from src.chat import Chat
from src.guard_types import llm_tool_inspection
from src.guards import GuardBlockedError
from src.llm import LLM


class MockLLM:
    def __init__(self, tools=None):
        self.model = "test-model"
        self.params = {"tools": tools} if tools is not None else {}
        self.provider = "openai"

    async def inspect_completion(self, **kwargs):
        return '{"decision":"allow"}'


def _openai_tool(name, description=""):
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {}},
    }


@pytest.mark.asyncio
async def test_allow_decision(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("read_file")])

    async def fake_inspect(**kwargs):
        return ("allow", "")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", fake_inspect)

    logs = await llm_tool_inspection.apply(chat, cast(LLM, llm), {"prompt": "check tools"})
    assert logs == []


@pytest.mark.asyncio
async def test_block_decision(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("evil_tool")])

    async def fake_inspect(**kwargs):
        return ("block", "prompt injection in tool description")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", fake_inspect)

    with pytest.raises(GuardBlockedError, match="Request blocked by llm_tool_inspection"):
        await llm_tool_inspection.apply(chat, cast(LLM, llm), {"prompt": "check tools"})


@pytest.mark.asyncio
async def test_on_violation_log(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("evil_tool")])

    async def fake_inspect(**kwargs):
        return ("block", "suspicious description")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", fake_inspect)

    logs = await llm_tool_inspection.apply(
        chat, cast(LLM, llm), {"prompt": "check tools", "on_violation": "log"}
    )
    assert len(logs) == 1
    assert "on_violation=log" in logs[0]
    assert "suspicious description" in logs[0]


@pytest.mark.asyncio
async def test_on_error_allow(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("read_file")])

    async def raise_error(**kwargs):
        raise RuntimeError("inspector down")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", raise_error)

    logs = await llm_tool_inspection.apply(
        chat, cast(LLM, llm), {"prompt": "check tools", "on_error": "allow"}
    )
    assert len(logs) == 1
    assert "on_error=allow" in logs[0]


@pytest.mark.asyncio
async def test_on_error_block(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("read_file")])

    async def raise_error(**kwargs):
        raise RuntimeError("inspector down")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", raise_error)

    with pytest.raises(GuardBlockedError, match="llm_tool_inspection failed"):
        await llm_tool_inspection.apply(
            chat, cast(LLM, llm), {"prompt": "check tools", "on_error": "block"}
        )


@pytest.mark.asyncio
async def test_no_prompt_skips():
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=[_openai_tool("read_file")])

    logs = await llm_tool_inspection.apply(chat, cast(LLM, llm), {"prompt": ""})
    assert logs == []


@pytest.mark.asyncio
async def test_no_tools_skips(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hi"}])
    llm = MockLLM(tools=None)

    # Even with a prompt, no tools means nothing to inspect
    inspected = False

    async def should_not_be_called(**kwargs):
        nonlocal inspected
        inspected = True
        return ("allow", "")

    monkeypatch.setattr(llm_tool_inspection, "_inspect_tools", should_not_be_called)

    logs = await llm_tool_inspection.apply(chat, cast(LLM, llm), {"prompt": "check tools"})
    assert logs == []
    assert not inspected


@pytest.mark.asyncio
async def test_tool_text_includes_tool_calls():
    """Tool call arguments and results from chat appear in inspection text."""
    chat = Chat.from_conversation(
        [
            {"role": "user", "content": "run the command"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": '{"cmd": "ls -la"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt\nfile2.txt"},
        ]
    )

    tools = [_openai_tool("execute", "Execute a command")]
    llm = MockLLM(tools=tools)

    from src.guard_types.llm_tool_inspection import Config, _collect_tool_text

    cfg = Config(prompt="test", include_tool_calls=True)
    text = _collect_tool_text(cast(LLM, llm), chat, cfg)

    assert "execute" in text
    assert "ls -la" in text
    assert "file1.txt" in text
