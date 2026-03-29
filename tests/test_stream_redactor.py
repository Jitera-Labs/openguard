import re

import pytest

from src.llm import StreamRedactor


def test_stream_redactor_basic_passthrough():
    redactor = StreamRedactor(patterns=[], window_size=5)

    assert redactor.push("hel") == ""
    assert redactor.push("lo ") == "h"
    assert redactor.push("world") == "ello "
    assert redactor.flush() == "world"


def test_stream_redactor_pattern_replacement():
    # Replace email pattern with <email>
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    patterns = [(email_pattern, "<email>")]

    redactor = StreamRedactor(patterns=patterns, window_size=10)

    assert redactor.push("My email ") == ""
    assert redactor.push("is john") == "My ema"
    assert redactor.push(".doe@exam") == "il is joh"
    # n.doe@example.com gets replaced by <email>
    assert redactor.push("ple.com, ") == ""
    assert redactor.push("bye!") == "<em"
    assert redactor.flush() == "ail>, bye!"


def test_stream_redactor_large_window():
    # If the window is larger than the input, nothing should be emitted until flush
    redactor = StreamRedactor(patterns=[], window_size=100)
    assert redactor.push("chunk 1") == ""
    assert redactor.push("chunk 2") == ""
    assert redactor.flush() == "chunk 1chunk 2"


def test_stream_redactor_overlapping_replacements():
    # A generic pattern that might match early and replace
    pattern = re.compile(r"apple")
    patterns = [(pattern, "orange")]
    redactor = StreamRedactor(patterns=patterns, window_size=3)

    assert redactor.push("I like a") == "I lik"
    assert redactor.push("pples") == "e oran"
    assert redactor.flush() == "ges"
