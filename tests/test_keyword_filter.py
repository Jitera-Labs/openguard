import pytest

from src.chat import Chat
from src.guard_types import keyword_filter
from src.guards import GuardBlockedError


class MockLLM:
    pass


class TestKeywordFilter:
    def test_block_match_any_single_keyword(self):
        """Test blocking when match_mode is 'any' and one keyword is found."""
        config = {"keywords": ["forbidden"], "action": "block", "match_mode": "any"}
        messages = [{"role": "user", "content": "This contains the forbidden word."}]
        chat = Chat.from_conversation(messages)

        with pytest.raises(GuardBlockedError) as excinfo:
            keyword_filter.apply(chat, MockLLM(), config)

        assert "found keyword 'forbidden'" in str(excinfo.value)

    def test_block_match_all_avoid_partial_match(self):
        """
        Test that action is NOT taken when match_mode is 'all'
        and only some keywords are present.
        """
        config = {"keywords": ["secret", "confidential"], "action": "block", "match_mode": "all"}
        messages = [{"role": "user", "content": "This is just secret but not the other one."}]
        chat = Chat.from_conversation(messages)

        # Should not raise
        logs = keyword_filter.apply(chat, MockLLM(), config)
        assert logs == []

    def test_block_match_all_hit_full_match(self):
        """Test blocking when match_mode is 'all' and ALL keywords are found."""
        config = {"keywords": ["secret", "confidential"], "action": "block", "match_mode": "all"}
        messages = [{"role": "user", "content": "This is both secret and confidential info."}]
        chat = Chat.from_conversation(messages)

        with pytest.raises(GuardBlockedError) as excinfo:
            keyword_filter.apply(chat, MockLLM(), config)

        assert "found all required keywords" in str(excinfo.value)

    def test_sanitization(self):
        """Test keyword sanitization/replacement."""
        config = {"keywords": ["badwords"], "action": "sanitize", "replacement": "[REDACTED]"}
        messages = [{"role": "user", "content": "This has badwords in it."}]
        chat = Chat.from_conversation(messages)

        logs = keyword_filter.apply(chat, MockLLM(), config)

        # Check logs
        assert len(logs) > 0
        assert "Sanitized keywords" in logs[0]

        # Check replacement
        content = chat.plain()[0].content
        assert "This has [REDACTED] in it." == content

    def test_logging_only(self):
        """Test 'log' action which should not block or modify, just audit."""
        config = {"keywords": ["monitor"], "action": "log"}
        messages = [{"role": "user", "content": "Please monitor this conversation."}]
        chat = Chat.from_conversation(messages)

        logs = keyword_filter.apply(chat, MockLLM(), config)

        # Should populate logs
        assert len(logs) > 0
        assert "Keyword filter triggered" in logs[0]

        # Payload should be unchanged
        content = chat.plain()[0].content
        assert content == "Please monitor this conversation."

    def test_case_sensitivity_ignore(self):
        """Test that case_sensitive: true causes case mismatches to be ignored."""
        config = {"keywords": ["STRICT"], "action": "block", "case_sensitive": True}
        messages = [{"role": "user", "content": "this is strict lowercase, so it should pass."}]
        chat = Chat.from_conversation(messages)

        # Should not raise exception because case mismatch
        keyword_filter.apply(chat, MockLLM(), config)

    def test_case_sensitivity_match(self):
        """Test that case_sensitive: true blocks exact matches."""
        config = {"keywords": ["STRICT"], "action": "block", "case_sensitive": True}
        messages = [{"role": "user", "content": "This is STRICT uppercase."}]
        chat = Chat.from_conversation(messages)

        with pytest.raises(GuardBlockedError):
            keyword_filter.apply(chat, MockLLM(), config)

    def test_multiple_messages_match_all(self):
        """
        Test 'match_mode: all' where keywords are spread across different messages.
        The implementation seems to scan all messages, assuming current
        implementation aggregates finding. Let's verify behavior. Based on
        reading, it iterates messages and adds to present_keywords.
        So split keywords should trigger it.
        """
        config = {"keywords": ["part1", "part2"], "action": "block", "match_mode": "all"}
        messages = [
            {"role": "system", "content": "System says part1"},
            {"role": "user", "content": "User says part2"},
        ]
        chat = Chat.from_conversation(messages)

        with pytest.raises(GuardBlockedError):
            keyword_filter.apply(chat, MockLLM(), config)
