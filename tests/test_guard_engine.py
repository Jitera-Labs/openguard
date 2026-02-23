"""Unit tests for src/guard_engine.py error paths."""

from unittest.mock import MagicMock

import pytest

from src.chat import Chat
from src.guards import GuardAction, GuardBlockedError, GuardRule


def make_mock_llm(model="test-model"):
    llm = MagicMock()
    llm.model = model
    llm.params = {}
    llm.provider = "openai"
    llm.raw_payload = None
    return llm


def make_chat(content="Hello world"):
    return Chat.from_conversation([{"role": "user", "content": content}])


@pytest.mark.asyncio
async def test_apply_guards_no_match():
    """Guard that doesn't match leaves chat unchanged."""
    from src.guard_engine import apply_guards

    chat = make_chat("Hello world")
    llm = make_mock_llm(model="some-other-model")
    guard = GuardRule(
        match={"model": {"_eq": "specific-model"}},
        apply=[GuardAction(type="content_filter", config={"blocked_words": ["hello"]})],
    )

    result_chat, logs = await apply_guards(chat, llm, [guard])
    # Guard didn't match, so content is unchanged
    assert result_chat.plain()[0].content == "Hello world"
    assert logs == []


@pytest.mark.asyncio
async def test_apply_guards_match_and_apply():
    """Matched guard applies action and returns audit logs."""
    from src.guard_engine import apply_guards

    chat = make_chat("This has badword in it")
    llm = make_mock_llm(model="test-model")
    guard = GuardRule(
        match={"model": {"_eq": "test-model"}},
        apply=[GuardAction(type="content_filter", config={"blocked_words": ["badword"]})],
    )

    result_chat, logs = await apply_guards(chat, llm, [guard])
    assert "[FILTERED]" in result_chat.plain()[0].content
    assert len(logs) == 1
    assert "badword" in logs[0]


@pytest.mark.asyncio
async def test_apply_guards_raises_guard_blocked_error():
    """GuardBlockedError propagates out of apply_guards."""
    from src.guard_engine import apply_guards

    chat = make_chat("forbidden content")
    llm = make_mock_llm(model="test-model")
    guard = GuardRule(
        match={"model": {"_eq": "test-model"}},
        apply=[
            GuardAction(
                type="keyword_filter", config={"keywords": ["forbidden"], "action": "block"}
            )
        ],
    )

    with pytest.raises(GuardBlockedError):
        await apply_guards(chat, llm, [guard])


@pytest.mark.asyncio
async def test_apply_guards_module_not_found_logs_error():
    """Missing guard module logs error and continues (no crash)."""
    from src.guard_engine import apply_guards

    chat = make_chat("Hello")
    llm = make_mock_llm(model="test-model")
    guard = GuardRule(
        match={"model": {"_eq": "test-model"}},
        apply=[GuardAction(type="nonexistent_guard_type", config={})],
    )

    # Should not raise — logs error and continues
    result_chat, logs = await apply_guards(chat, llm, [guard])
    assert result_chat is chat
    assert logs == []


@pytest.mark.asyncio
async def test_apply_guards_empty_guards_list():
    """Empty guards list returns chat unchanged with empty logs."""
    from src.guard_engine import apply_guards

    chat = make_chat("Hello")
    llm = make_mock_llm()

    result_chat, logs = await apply_guards(chat, llm, [])
    assert result_chat is chat
    assert logs == []


@pytest.mark.asyncio
async def test_apply_guards_match_error_skips_guard():
    """Exception during match evaluation is caught; guard is skipped."""
    from src.guard_engine import apply_guards

    chat = make_chat("Hello")
    llm = make_mock_llm(model="test-model")

    # Bad filter that will raise during match_filter
    guard = GuardRule(
        match={"_bad_op": "invalid"},
        apply=[GuardAction(type="content_filter", config={"blocked_words": ["hello"]})],
    )

    # Should not raise — logs error and skips guard
    result_chat, logs = await apply_guards(chat, llm, [guard])
    # Content is unchanged since guard was skipped
    assert result_chat.plain()[0].content == "Hello"


@pytest.mark.asyncio
async def test_apply_guards_multiple_guards_all_match():
    """Multiple matching guards are all applied in order."""
    from src.guard_engine import apply_guards

    chat = make_chat("Remove badword and spam please")
    llm = make_mock_llm(model="test-model")
    guards = [
        GuardRule(
            match={"model": {"_eq": "test-model"}},
            apply=[GuardAction(type="content_filter", config={"blocked_words": ["badword"]})],
        ),
        GuardRule(
            match={"model": {"_eq": "test-model"}},
            apply=[GuardAction(type="content_filter", config={"blocked_words": ["spam"]})],
        ),
    ]

    result_chat, logs = await apply_guards(chat, llm, guards)
    content = result_chat.plain()[0].content
    assert "badword" not in content
    assert "spam" not in content
    assert content.count("[FILTERED]") == 2
    assert len(logs) == 2
