"""Tests for guard system."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chat import Chat
from src.guard_engine import apply_guards
from src.guard_types import content_filter, max_tokens, pii_filter
from src.guards import GuardAction, GuardRule


class MockLLM:
    def __init__(self, model="test-model", params=None):
        self.model = model
        self.params = params or {}


def test_content_filter_basic():
    """Test content filter replaces blocked words."""
    messages = [
        {"role": "user", "content": "This contains a badword in it"},
        {"role": "assistant", "content": "Normal response"},
    ]
    chat = Chat.from_conversation(messages)

    config = {"blocked_words": ["badword"]}
    logs = content_filter.apply(chat, MockLLM(), config)

    # Convert back to dict-like structure or query nodes
    nodes = chat.plain()

    assert "[FILTERED]" in nodes[0].content
    assert "badword" not in nodes[0].content.lower()
    assert len(logs) == 1
    assert "badword" in logs[0]


def test_content_filter_case_insensitive():
    """Test content filter is case-insensitive."""
    messages = [{"role": "user", "content": "This has BadWord and BADWORD and badword"}]
    chat = Chat.from_conversation(messages)

    config = {"blocked_words": ["badword"]}
    logs = content_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    # All variations should be filtered
    assert nodes[0].content.count("[FILTERED]") == 3
    assert "badword" not in nodes[0].content.lower()


def test_content_filter_multimodal():
    """Test content filter handles array content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This has badword"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
            ],
        }
    ]
    chat = Chat.from_conversation(messages)

    config = {"blocked_words": ["badword"]}
    logs = content_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    assert "[FILTERED]" in nodes[0].content[0]["text"]
    assert len(logs) == 1


def test_pii_filter_email():
    """Test PII filter detects and replaces emails."""
    messages = [{"role": "user", "content": "Contact me at john.doe@example.com for details"}]
    chat = Chat.from_conversation(messages)

    config = {"enabled": True}
    logs = pii_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    # Check email was replaced (now at index 1 after system message potentially)
    # Note: pii_filter adds a system message at the root.
    # Chat.plain() traverses parents.

    # If system message is added as root, it should be first.
    # The original message should be second.

    # Check if system message was added
    assert nodes[0].role == "system"
    assert "PII has been filtered" in nodes[0].content

    assert "<protected:email>" in nodes[1].content
    assert "john.doe@example.com" not in nodes[1].content

    # Check audit log
    assert len(logs) == 1
    assert "email" in logs[0]
    assert "john.doe@example.com" in logs[0]


def test_pii_filter_phone():
    """Test PII filter detects phone numbers."""
    messages = [{"role": "user", "content": "Call me at 555-123-4567"}]
    chat = Chat.from_conversation(messages)

    config = {}
    logs = pii_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    assert "<protected:phone>" in nodes[1].content
    assert "555-123-4567" not in nodes[1].content


def test_pii_filter_ssn():
    """Test PII filter detects SSN."""
    messages = [{"role": "user", "content": "My SSN is 123-45-6789"}]
    chat = Chat.from_conversation(messages)

    config = {}
    logs = pii_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    assert "<protected:ssn>" in nodes[1].content
    assert "123-45-6789" not in nodes[1].content


def test_pii_filter_creditcard():
    """Test PII filter detects credit card numbers."""
    messages = [{"role": "user", "content": "My card is 1234 5678 9012 3456"}]
    chat = Chat.from_conversation(messages)

    config = {}
    logs = pii_filter.apply(chat, MockLLM(), config)

    nodes = chat.plain()
    assert "<protected:creditcard>" in nodes[1].content
    assert "1234" not in nodes[1].content


def test_pii_filter_multimodal():
    """Test PII filter handles array content."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Email me at test@example.com"}]}
    ]
    chat = Chat.from_conversation(messages)

    config = {}
    logs = pii_filter.apply(chat, MockLLM(), config)

    # System message is at index 0, original message at index 1
    nodes = chat.plain()
    assert "<protected:email>" in nodes[1].content[0]["text"]


def test_max_tokens_adds_limit():
    """Test max_tokens adds limit when not present."""
    messages = [{"role": "user", "content": "Hello"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM()

    config = {"max_tokens": 1000}
    logs = max_tokens.apply(chat, llm, config)

    assert llm.params["max_tokens"] == 1000
    assert len(logs) == 1
    assert "1000" in logs[0]


def test_max_tokens_enforces_limit():
    """Test max_tokens enforces limit when exceeded."""
    messages = [{"role": "user", "content": "Hello"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM(params={"max_tokens": 5000})

    config = {"max_tokens": 1000}
    logs = max_tokens.apply(chat, llm, config)

    assert llm.params["max_tokens"] == 1000
    assert len(logs) == 1
    assert "5000" in logs[0]  # Should mention old value


def test_max_tokens_preserves_lower():
    """Test max_tokens doesn't change lower values."""
    messages = [{"role": "user", "content": "Hello"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM(params={"max_tokens": 500})

    config = {"max_tokens": 1000}
    logs = max_tokens.apply(chat, llm, config)

    assert llm.params["max_tokens"] == 500
    assert len(logs) == 0  # No change needed


def test_guard_engine_applies_matching_guards():
    """Test guard engine applies guards that match."""
    messages = [{"role": "user", "content": "Test message"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM(model="openrouter/model")

    guards = [
        GuardRule(
            match={"model": {"_ilike": "%openrouter%"}},
            apply=[GuardAction(type="max_tokens", config={"max_tokens": 2000})],
        )
    ]

    modified_chat, logs = apply_guards(chat, llm, guards)

    assert llm.params["max_tokens"] == 2000
    assert len(logs) > 0


def test_guard_engine_skips_non_matching():
    """Test guard engine skips guards that don't match."""
    messages = [{"role": "user", "content": "Test message"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM(model="gpt-4")

    guards = [
        GuardRule(
            match={"model": {"_ilike": "%openrouter%"}},
            apply=[GuardAction(type="max_tokens", config={"max_tokens": 2000})],
        )
    ]

    modified_chat, logs = apply_guards(chat, llm, guards)

    assert "max_tokens" not in llm.params
    assert len(logs) == 0


def test_guard_engine_applies_multiple_actions():
    """Test guard engine applies multiple actions in order."""
    messages = [{"role": "user", "content": "Email me at test@example.com with badword"}]
    chat = Chat.from_conversation(messages)
    llm = MockLLM(model="test-model")

    guards = [
        GuardRule(
            match={"model": "test-model"},
            apply=[
                GuardAction(type="content_filter", config={"blocked_words": ["badword"]}),
                GuardAction(type="pii_filter", config={}),
                GuardAction(type="max_tokens", config={"max_tokens": 1000}),
            ],
        )
    ]

    modified_chat, logs = apply_guards(chat, llm, guards)

    # All guards should have been applied
    nodes = chat.plain()
    # PII filter adds system message at index 0
    assert "[FILTERED]" in nodes[1].content  # content_filter
    assert "<protected:email>" in nodes[1].content  # pii_filter
    assert llm.params["max_tokens"] == 1000  # max_tokens
    assert len(logs) >= 3  # At least 3 audit logs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
