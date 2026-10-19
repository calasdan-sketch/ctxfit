"""Tests for the canonical message model."""

import pytest

from ctxfit import PairingBroken, fit
from ctxfit.messages import (
    KIND_TEXT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Message,
    assign_section,
    detect_flavor,
    from_dicts,
    normalize,
    turn_indexes,
    validate_pairing,
)


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.kind == KIND_TEXT
    assert m.section == "history"
    assert not m.pinned
    assert m.to_dict() == {"role": "user", "content": "hi"}


def test_message_rejects_bad_section():
    with pytest.raises(ValueError, match="unknown section"):
        Message(role="user", content="x", section="nope")


def test_message_none_content_becomes_empty():
    assert Message(role="user", content=None).content == ""


def test_assign_section_rules():
    assert assign_section("system", KIND_TEXT) == "system"
    assert assign_section("tool", KIND_TEXT) == "retrieved"
    assert assign_section("user", KIND_TOOL_RESULT) == "retrieved"
    assert assign_section("user", KIND_TEXT) == "history"
    assert assign_section("developer", KIND_TEXT) == "scratchpad"
    assert assign_section("user", KIND_TEXT, explicit="scratchpad") == "scratchpad"
    # explicit garbage is ignored (never inferred from content length)
    assert assign_section("user", KIND_TEXT, explicit="nope") == "history"


def test_from_dicts_basic():
    msgs = from_dicts([{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}])
    assert msgs[0].section == "system" and msgs[0].pinned
    assert msgs[1].section == "history" and not msgs[1].pinned


def test_from_dicts_tool_result_section():
    msgs = from_dicts([{"role": "tool", "content": "out", "tool_call_id": "c1"}])
    assert msgs[0].kind == KIND_TEXT  # generic dicts don't set kind
    assert msgs[0].section == "retrieved"
    assert msgs[0].call_id == "c1"


def test_from_dicts_content_parts_joined():
    msgs = from_dicts(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
    )
    assert "look" in msgs[0].content
    assert "non-text part omitted" in msgs[0].content


def test_from_dicts_rejects_non_dict():
    with pytest.raises(TypeError):
        from_dicts(["nope"])


def test_from_dicts_passes_through_messages():
    m = Message(role="user", content="x")
    assert from_dicts([m]) == [m]


def test_detect_flavor_openai():
    assert detect_flavor([{"role": "assistant", "tool_calls": []}]) == "openai"
    assert detect_flavor([{"role": "tool", "content": "x"}]) == "openai"


def test_detect_flavor_anthropic():
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
        }
    ]
    assert detect_flavor(msgs) == "anthropic"


def test_detect_flavor_generic():
    assert detect_flavor([{"role": "user", "content": "hi"}]) == "generic"


def test_detect_flavor_langchain_ducktyped():
    class FakeLC:
        content = "hi"
        type = "human"

    assert detect_flavor([FakeLC()]) == "langchain"


def test_normalize_message_passthrough():
    m = Message(role="user", content="x")
    assert normalize([m]) == [m]


def test_normalize_explicit_flavor():
    msgs = normalize([{"role": "user", "content": "hi"}], flavor="generic")
    assert msgs[0].role == "user"


def test_validate_pairing_ok():
    msgs = [
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="tool", content="ok", kind=KIND_TOOL_RESULT, call_id="c1"),
    ]
    assert validate_pairing(msgs) == []


def test_validate_pairing_orphaned_call():
    msgs = [Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1")]
    problems = validate_pairing(msgs)
    assert len(problems) == 1 and "orphaned tool_call" in problems[0]


def test_validate_pairing_orphaned_result():
    msgs = [Message(role="tool", content="ok", kind=KIND_TOOL_RESULT, call_id="c9")]
    problems = validate_pairing(msgs)
    assert len(problems) == 1 and "orphaned tool_result" in problems[0]


def test_validate_pairing_duplicate_call_ids():
    msgs = [
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="tool", content="ok", kind=KIND_TOOL_RESULT, call_id="c1"),
    ]
    problems = validate_pairing(msgs)
    assert any("duplicate tool_call id" in p for p in problems)


def test_validate_pairing_duplicate_result_ids():
    msgs = [
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="tool", content="ok", kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content="ok again", kind=KIND_TOOL_RESULT, call_id="c1"),
    ]
    problems = validate_pairing(msgs)
    assert any("duplicate tool_result id" in p for p in problems)


def test_fit_rejects_duplicate_call_ids():
    raw = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
        {"role": "tool", "content": "ok again", "tool_call_id": "c1"},
    ]
    with pytest.raises(PairingBroken, match="duplicate tool_result id"):
        fit(raw, total_budget=100_000)


def test_turn_indexes():
    msgs = [
        Message(role="system", content="s"),
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
        Message(role="tool", content="r", kind=KIND_TOOL_RESULT),
    ]
    assert turn_indexes(msgs) == [-1, 0, 0, 1, 1, 1]


def test_turn_indexes_no_user():
    msgs = [Message(role="assistant", content="a")]
    assert turn_indexes(msgs) == [0]
