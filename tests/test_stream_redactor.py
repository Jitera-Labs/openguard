import re

import pytest

import src.guards
from src.llm import StreamRedactor


def test_stream_redactor_basic_passthrough():
    redactor = StreamRedactor(patterns=[], blocks=[], window_size=5)

    assert redactor.push("hel") == ""
    assert redactor.push("lo ") == "h"
    assert redactor.push("world") == "ello "
    assert redactor.flush() == "world"


def test_stream_redactor_pattern_replacement():
    # Replace email pattern with <email>
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    patterns = [(email_pattern, "<email>")]

    redactor = StreamRedactor(patterns=patterns, blocks=[], window_size=10)

    assert redactor.push("My email ") == ""
    assert redactor.push("is john") == "My ema"
    assert redactor.push(".doe@exam") == "il is joh"
    # n.doe@example.com gets replaced by <email>
    assert redactor.push("ple.com, ") == ""
    assert redactor.push("bye!") == "<em"
    assert redactor.flush() == "ail>, bye!"


def test_stream_redactor_blocking():
    # Block on "secret_key"
    secret_pattern = re.compile(r"secret_key")
    blocks = [(secret_pattern, "secret_key")]

    redactor = StreamRedactor(patterns=[], blocks=blocks, window_size=5)

    # "Here is " (8) emits "Her", leaves "e is "
    assert redactor.push("Here is ") == "Her"
    # "e is " + "the se" = "e is the se" (11) emits "e is t", leaves "he se"
    assert redactor.push("the se") == "e is t"

    with pytest.raises(
        src.guards.GuardBlockedError,
        match="Request blocked: found keyword 'secret_key' in streaming response",
    ):
        redactor.push("cret_key: 123")


def test_stream_redactor_flush_blocking():
    secret_pattern = re.compile(r"badword")
    blocks = [(secret_pattern, "badword")]

    # window size is large so it stays in buffer
    redactor = StreamRedactor(patterns=[], blocks=blocks, window_size=20)

    assert redactor.push("This contains badw") == ""

    with pytest.raises(
        src.guards.GuardBlockedError,
        match="Request blocked: found keyword 'badword' in streaming response",
    ):
        redactor.push("ord")


def test_stream_redactor_large_window():
    # If the window is larger than the input, nothing should be emitted until flush
    redactor = StreamRedactor(patterns=[], blocks=[], window_size=100)
    assert redactor.push("chunk 1") == ""
    assert redactor.push("chunk 2") == ""
    assert redactor.flush() == "chunk 1chunk 2"


def test_stream_redactor_overlapping_replacements():
    # A generic pattern that might match early and replace
    pattern = re.compile(r"apple")
    patterns = [(pattern, "orange")]
    redactor = StreamRedactor(patterns=patterns, blocks=[], window_size=3)

    assert redactor.push("I like a") == "I lik"
    assert redactor.push("pples") == "e oran"
    assert redactor.flush() == "ges"
