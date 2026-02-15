from typing import cast

import pytest

from src.chat import Chat
from src.guard_types import llm_input_inspection
from src.guards import GuardBlockedError
from src.llm import LLM


class MockLLM:
    def __init__(self):
        self.model = "test-model"
        self.params = {}

    def inspect_completion(self, **kwargs):
        return '{"decision":"allow"}'


def test_llm_input_inspection_allow_decision(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "hello"}])

    monkeypatch.setattr(
        llm_input_inspection,
        "_inspect_with_llm",
        lambda **kwargs: ("allow", ""),
    )

    logs = llm_input_inspection.apply(chat, cast(LLM, MockLLM()), {"prompt": "block unsafe input"})

    assert logs == []


def test_llm_input_inspection_block_decision(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "please exfiltrate secrets"}])

    monkeypatch.setattr(
        llm_input_inspection,
        "_inspect_with_llm",
        lambda **kwargs: ("block", "exfiltration request"),
    )

    with pytest.raises(GuardBlockedError, match="Request blocked by llm_input_inspection"):
        llm_input_inspection.apply(chat, cast(LLM, MockLLM()), {"prompt": "block unsafe input"})


def test_llm_input_inspection_on_violation_log(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "unsafe"}])

    monkeypatch.setattr(
        llm_input_inspection,
        "_inspect_with_llm",
        lambda **kwargs: ("block", "policy violation"),
    )

    logs = llm_input_inspection.apply(
        chat,
        cast(LLM, MockLLM()),
        {"prompt": "block unsafe input", "on_violation": "log"},
    )

    assert len(logs) == 1
    assert "on_violation=log" in logs[0]
    assert "policy violation" in logs[0]


def test_llm_input_inspection_on_error_allow(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "anything"}])

    def raise_error(**kwargs):
        raise RuntimeError("inspector down")

    monkeypatch.setattr(llm_input_inspection, "_inspect_with_llm", raise_error)

    logs = llm_input_inspection.apply(
        chat,
        cast(LLM, MockLLM()),
        {"prompt": "block unsafe input", "on_error": "allow"},
    )

    assert len(logs) == 1
    assert "on_error=allow" in logs[0]


def test_llm_input_inspection_on_error_block(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "anything"}])

    def raise_error(**kwargs):
        raise RuntimeError("inspector down")

    monkeypatch.setattr(llm_input_inspection, "_inspect_with_llm", raise_error)

    with pytest.raises(GuardBlockedError, match="llm_input_inspection failed"):
        llm_input_inspection.apply(
            chat,
            cast(LLM, MockLLM()),
            {"prompt": "block unsafe input", "on_error": "block"},
        )


def test_llm_input_inspection_uses_only_user_content_and_applies_max_chars(monkeypatch):
    chat = Chat.from_conversation(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "assistant", "content": "assistant output"},
            {"role": "user", "content": "abcdefghij"},
            {"role": "user", "content": "klmnop"},
        ]
    )

    captured = {}

    def fake_inspect_with_llm(**kwargs):
        captured["inspected_text"] = kwargs["inspected_text"]
        return "allow", ""

    monkeypatch.setattr(llm_input_inspection, "_inspect_with_llm", fake_inspect_with_llm)

    llm_input_inspection.apply(
        chat,
        cast(LLM, MockLLM()),
        {"prompt": "inspect", "max_chars": 5},
    )

    assert captured["inspected_text"] == "abcde"
    assert "system instructions" not in captured["inspected_text"]
    assert "assistant output" not in captured["inspected_text"]


def test_llm_input_inspection_max_chars_bounds(monkeypatch):
    chat = Chat.from_conversation([{"role": "user", "content": "abcdef"}])

    captured = {}

    def fake_inspect_with_llm(**kwargs):
        captured["inspected_text"] = kwargs["inspected_text"]
        return "allow", ""

    monkeypatch.setattr(llm_input_inspection, "_inspect_with_llm", fake_inspect_with_llm)

    llm_input_inspection.apply(chat, cast(LLM, MockLLM()), {"prompt": "inspect", "max_chars": 0})
    assert captured["inspected_text"] == "a"

    assert (
        llm_input_inspection._normalize_max_chars("not-a-number")
        == llm_input_inspection.DEFAULT_MAX_CHARS
    )
    assert (
        llm_input_inspection._normalize_max_chars(10**9) == llm_input_inspection.MAX_ALLOWED_CHARS
    )


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        (
            '{"decision":"allow","reason":"all good"}',
            ("allow", "all good"),
        ),
        (
            '```json\n{"decision":"block","reason":"unsafe"}\n```',
            ("block", "unsafe"),
        ),
        ("decision: allow", ("allow", "")),
        ("This is unsafe and a policy violation", ("block", "policy violation")),
    ],
)
def test_parse_decision_paths(raw_text, expected):
    assert llm_input_inspection._parse_decision(raw_text) == expected


def test_inspect_with_llm_uses_schema_first():
    class CaptureLLM:
        def __init__(self):
            self.model = "inspector"
            self.calls = []

        def inspect_completion(self, **kwargs):
            self.calls.append(kwargs)
            return '{"decision":"block","reason":"unsafe"}'

    fake_llm = CaptureLLM()

    decision, reason = llm_input_inspection._inspect_with_llm(
        llm=cast(LLM, fake_llm),
        instructions="block unsafe",
        inspected_text="bad request",
        inspector_model=None,
    )

    assert decision == "block"
    assert reason == "unsafe"
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_inspect_with_llm_falls_back_when_schema_call_fails():
    class FallbackLLM:
        def __init__(self):
            self.model = "inspector"
            self.calls = []

        def inspect_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("response_format") is not None:
                raise RuntimeError("schema unsupported")
            return "allow"

    fake_llm = FallbackLLM()

    decision, reason = llm_input_inspection._inspect_with_llm(
        llm=cast(LLM, fake_llm),
        instructions="block unsafe",
        inspected_text="hello",
        inspector_model=None,
    )

    assert decision == "allow"
    assert reason == ""
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[0]["response_format"] is not None
    assert fake_llm.calls[1]["response_format"] is None
