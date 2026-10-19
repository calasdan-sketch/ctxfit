"""Tests for framework adapters."""

import json

from ctxfit.adapters.anthropic import from_anthropic, to_anthropic
from ctxfit.adapters.langchain import from_lc_messages, to_lc_messages
from ctxfit.adapters.openai import from_openai, to_openai
from ctxfit.messages import KIND_TOOL_CALL, KIND_TOOL_RESULT, validate_pairing


def openai_conversation():
    return [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What time is it?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": json.dumps({"tz": "UTC"})},
                }
            ],
        },
        {"role": "tool", "content": "12:00 UTC", "tool_call_id": "call_1", "name": "get_time"},
        {"role": "assistant", "content": "It is noon."},
    ]


def test_openai_round_trip():
    raw = openai_conversation()
    msgs = from_openai(raw)
    # assistant split into text + tool_call
    kinds = [m.kind for m in msgs]
    assert kinds.count(KIND_TOOL_CALL) == 1
    assert kinds.count(KIND_TOOL_RESULT) == 1
    tc = next(m for m in msgs if m.kind == KIND_TOOL_CALL)
    tr = next(m for m in msgs if m.kind == KIND_TOOL_RESULT)
    assert tc.call_id == tr.call_id == "call_1"
    assert tc.name == "get_time"
    assert json.loads(tc.content) == {"tz": "UTC"}  # args intact
    assert tr.section == "retrieved"
    assert validate_pairing(msgs) == []

    back = to_openai(msgs)
    assert len(back) == len(raw)
    assert back[0] == {"role": "system", "content": "Be helpful."}
    asst = back[2]
    assert asst["role"] == "assistant"
    assert asst["content"] == "Let me check."
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert asst["tool_calls"][0]["function"]["name"] == "get_time"
    tool = back[3]
    assert tool == {
        "role": "tool",
        "content": "12:00 UTC",
        "tool_call_id": "call_1",
        "name": "get_time",
    }


def test_openai_tool_call_args_never_parsed_away():
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
    ]
    msgs = from_openai(raw)
    assert msgs[0].content == '{"a": 1}'


def test_openai_content_parts_and_legacy_function_role():
    raw = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {}}],
        },
        {"role": "function", "content": "old-style", "name": "f"},
    ]
    msgs = from_openai(raw)
    assert "see" in msgs[0].content and "image" in msgs[0].content
    assert msgs[1].role == "tool" and msgs[1].section == "retrieved"


def test_openai_system_pinned():
    msgs = from_openai([{"role": "system", "content": "s"}])
    assert msgs[0].pinned and msgs[0].section == "system"


def anthropic_conversation():
    return [
        {"role": "user", "content": [{"type": "text", "text": "What time is it?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {"tz": "UTC"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "12:00 UTC"}],
        },
    ]


def test_anthropic_round_trip():
    raw = anthropic_conversation()
    msgs = from_anthropic(raw)
    tc = next(m for m in msgs if m.kind == KIND_TOOL_CALL)
    tr = next(m for m in msgs if m.kind == KIND_TOOL_RESULT)
    assert tc.call_id == tr.call_id == "tu_1"
    assert tc.name == "get_time"
    assert json.loads(tc.content) == {"tz": "UTC"}
    assert validate_pairing(msgs) == []

    back = to_anthropic(msgs)
    assert len(back) == 3
    asst_blocks = back[1]["content"]
    assert {"type": "text", "text": "Checking."} in asst_blocks
    tu = next(b for b in asst_blocks if b["type"] == "tool_use")
    assert tu["id"] == "tu_1" and tu["input"] == {"tz": "UTC"}
    tr_block = back[2]["content"][0]
    assert tr_block["type"] == "tool_result"
    assert tr_block["tool_use_id"] == "tu_1"
    assert tr_block["content"] == "12:00 UTC"


def test_anthropic_tool_result_list_content():
    raw = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t9",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ],
        }
    ]
    msgs = from_anthropic(raw)
    assert msgs[0].content == "a\nb"


def test_anthropic_string_content():
    msgs = from_anthropic([{"role": "user", "content": "plain"}])
    assert msgs[0].content == "plain" and msgs[0].kind == "text"


class FakeHuman:
    type = "human"

    def __init__(self, content):
        self.content = content


class FakeAI:
    type = "ai"

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = {}


class FakeTool:
    type = "tool"

    def __init__(self, content, tool_call_id):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = "get_time"


def test_langchain_ducktyped():
    raw = [
        FakeHuman("What time?"),
        FakeAI(
            "",
            tool_calls=[
                {"name": "get_time", "args": {"tz": "UTC"}, "id": "lc1", "type": "tool_call"}
            ],
        ),
        FakeTool("12:00", "lc1"),
    ]
    # adapt: tool_calls shaped with function/arguments like OpenAI for this test
    raw[1].tool_calls = [
        {
            "id": "lc1",
            "type": "function",
            "function": {"name": "get_time", "arguments": json.dumps({"tz": "UTC"})},
        }
    ]
    msgs = from_lc_messages(raw)
    assert msgs[0].role == "user"
    tc = next(m for m in msgs if m.kind == KIND_TOOL_CALL)
    tr = next(m for m in msgs if m.kind == KIND_TOOL_RESULT)
    assert tc.call_id == tr.call_id == "lc1"
    assert validate_pairing(msgs) == []

    back = to_lc_messages(msgs)
    assert back[0]["type"] == "human"
    assert back[2]["type"] == "tool" and back[2]["tool_call_id"] == "lc1"


def test_langchain_classname_fallback():
    class SystemMessage:
        def __init__(self, content):
            self.content = content

    msgs = from_lc_messages([SystemMessage("s")])
    assert msgs[0].role == "system" and msgs[0].pinned


def test_langchain_rejects_dicts():
    import pytest

    with pytest.raises(TypeError, match="use flavor='generic'"):
        from_lc_messages([{"role": "user", "content": "hi"}])
