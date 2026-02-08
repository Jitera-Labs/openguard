"""Tests for guard system."""
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.guard_types import content_filter, pii_filter, max_tokens
from src.guard_engine import apply_guards
from src.guards import GuardRule, GuardAction


def test_content_filter_basic():
    """Test content filter replaces blocked words."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'This contains a badword in it'},
            {'role': 'assistant', 'content': 'Normal response'}
        ]
    }

    config = {'blocked_words': ['badword']}
    modified, logs = content_filter.apply(payload, config)

    assert '[FILTERED]' in modified['messages'][0]['content']
    assert 'badword' not in modified['messages'][0]['content'].lower()
    assert len(logs) == 1
    assert 'badword' in logs[0]


def test_content_filter_case_insensitive():
    """Test content filter is case-insensitive."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'This has BadWord and BADWORD and badword'}
        ]
    }

    config = {'blocked_words': ['badword']}
    modified, logs = content_filter.apply(payload, config)

    # All variations should be filtered
    assert modified['messages'][0]['content'].count('[FILTERED]') == 3
    assert 'badword' not in modified['messages'][0]['content'].lower()


def test_content_filter_multimodal():
    """Test content filter handles array content."""
    payload = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'This has badword'},
                    {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.jpg'}}
                ]
            }
        ]
    }

    config = {'blocked_words': ['badword']}
    modified, logs = content_filter.apply(payload, config)

    assert '[FILTERED]' in modified['messages'][0]['content'][0]['text']
    assert len(logs) == 1


def test_pii_filter_email():
    """Test PII filter detects and replaces emails."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Contact me at john.doe@example.com for details'}
        ]
    }

    config = {'enabled': True}
    modified, logs = pii_filter.apply(payload, config)

    # Check email was replaced (now at index 1 after system message)
    assert '<protected:email>' in modified['messages'][1]['content']
    assert 'john.doe@example.com' not in modified['messages'][1]['content']

    # Check system message was added
    assert modified['messages'][0]['role'] == 'system'
    assert 'PII has been filtered' in modified['messages'][0]['content']

    # Check audit log
    assert len(logs) == 1
    assert 'email' in logs[0]
    assert 'john.doe@example.com' in logs[0]


def test_pii_filter_phone():
    """Test PII filter detects phone numbers."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Call me at 555-123-4567'}
        ]
    }

    config = {}
    modified, logs = pii_filter.apply(payload, config)

    assert '<protected:phone>' in modified['messages'][1]['content']
    assert '555-123-4567' not in modified['messages'][1]['content']


def test_pii_filter_ssn():
    """Test PII filter detects SSN."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'My SSN is 123-45-6789'}
        ]
    }

    config = {}
    modified, logs = pii_filter.apply(payload, config)

    assert '<protected:ssn>' in modified['messages'][1]['content']
    assert '123-45-6789' not in modified['messages'][1]['content']


def test_pii_filter_creditcard():
    """Test PII filter detects credit card numbers."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'My card is 1234 5678 9012 3456'}
        ]
    }

    config = {}
    modified, logs = pii_filter.apply(payload, config)

    assert '<protected:creditcard>' in modified['messages'][1]['content']
    assert '1234' not in modified['messages'][1]['content']


def test_pii_filter_multimodal():
    """Test PII filter handles array content."""
    payload = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Email me at test@example.com'}
                ]
            }
        ]
    }

    config = {}
    modified, logs = pii_filter.apply(payload, config)

    # System message is at index 0, original message at index 1
    assert '<protected:email>' in modified['messages'][1]['content'][0]['text']


def test_max_tokens_adds_limit():
    """Test max_tokens adds limit when not present."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Hello'}
        ]
    }

    config = {'max_tokens': 1000}
    modified, logs = max_tokens.apply(payload, config)

    assert modified['max_tokens'] == 1000
    assert len(logs) == 1
    assert '1000' in logs[0]


def test_max_tokens_enforces_limit():
    """Test max_tokens enforces limit when exceeded."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Hello'}
        ],
        'max_tokens': 5000
    }

    config = {'max_tokens': 1000}
    modified, logs = max_tokens.apply(payload, config)

    assert modified['max_tokens'] == 1000
    assert len(logs) == 1
    assert '5000' in logs[0]  # Should mention old value


def test_max_tokens_preserves_lower():
    """Test max_tokens doesn't change lower values."""
    payload = {
        'messages': [
            {'role': 'user', 'content': 'Hello'}
        ],
        'max_tokens': 500
    }

    config = {'max_tokens': 1000}
    modified, logs = max_tokens.apply(payload, config)

    assert modified['max_tokens'] == 500
    assert len(logs) == 0  # No change needed


def test_guard_engine_applies_matching_guards():
    """Test guard engine applies guards that match."""
    payload = {
        'model': 'openrouter/model',
        'messages': [
            {'role': 'user', 'content': 'Test message'}
        ]
    }

    guards = [
        GuardRule(
            match={'model': {'_ilike': '%openrouter%'}},
            apply=[
                GuardAction(type='max_tokens', config={'max_tokens': 2000})
            ]
        )
    ]

    modified, logs = apply_guards(payload, guards)

    assert modified['max_tokens'] == 2000
    assert len(logs) > 0


def test_guard_engine_skips_non_matching():
    """Test guard engine skips guards that don't match."""
    payload = {
        'model': 'gpt-4',
        'messages': [
            {'role': 'user', 'content': 'Test message'}
        ]
    }

    guards = [
        GuardRule(
            match={'model': {'_ilike': '%openrouter%'}},
            apply=[
                GuardAction(type='max_tokens', config={'max_tokens': 2000})
            ]
        )
    ]

    modified, logs = apply_guards(payload, guards)

    assert 'max_tokens' not in modified
    assert len(logs) == 0


def test_guard_engine_applies_multiple_actions():
    """Test guard engine applies multiple actions in order."""
    payload = {
        'model': 'test-model',
        'messages': [
            {'role': 'user', 'content': 'Email me at test@example.com with badword'}
        ]
    }

    guards = [
        GuardRule(
            match={'model': 'test-model'},
            apply=[
                GuardAction(type='content_filter', config={'blocked_words': ['badword']}),
                GuardAction(type='pii_filter', config={}),
                GuardAction(type='max_tokens', config={'max_tokens': 1000})
            ]
        )
    ]

    modified, logs = apply_guards(payload, guards)

    # All guards should have been applied
    assert '[FILTERED]' in modified['messages'][1]['content']  # content_filter
    assert '<protected:email>' in modified['messages'][1]['content']  # pii_filter
    assert modified['max_tokens'] == 1000  # max_tokens
    assert len(logs) >= 3  # At least 3 audit logs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
