"""LangChain-*shaped* adapter — duck-typed, no langchain dependency.

Accepts any objects with a ``.content`` attribute (LangChain message
objects satisfy this). The message type is read from ``.type`` when
present, else from the class name (HumanMessage/AIMessage/SystemMessage/
ToolMessage/...). Tool calls come from ``additional_kwargs["tool_calls"]``
or a ``.tool_calls`` attribute; tool results from ``.tool_call_id``.

``to_lc_messages`` returns plain dicts shaped like LangChain messages
(``{"type": ..., "content": ...}``) rather than importing langchain —
ctxfit stays dependency-free.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..messages import (
    KIND_TEXT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Message,
    assign_section,
)

_CLASS_TO_ROLE = {
    "humanmessage": "user",
    "human": "user",
    "aimessage": "assistant",
    "ai": "assistant",
    "systemmessage": "system",
    "system": "system",
    "toolmessage": "tool",
    "tool": "tool",
    "functionmessage": "tool",
    "function": "tool",
    "chatmessage": "user",
}


def _lc_type(obj: Any) -> str:
    t = getattr(obj, "type", None)
    if isinstance(t, str) and t:
        return t.lower()
    return type(obj).__name__.lower()


def _lc_role(obj: Any) -> str:
    t = _lc_type(obj)
    if t in _CLASS_TO_ROLE:
        return _CLASS_TO_ROLE[t]
    role = getattr(obj, "role", None)
    if isinstance(role, str) and role:
        return role.lower()
    return "user"


def _lc_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append("[non-text part omitted]")
        return "\n".join(parts)
    return str(content)


def from_lc_messages(raw: Sequence[Any]) -> list[Message]:
    out: list[Message] = []
    for i, obj in enumerate(raw):
        if isinstance(obj, dict):
            raise TypeError(
                f"message #{i} is a dict; use flavor='generic' or 'openai' for "
                "dict inputs, 'langchain' is for LangChain message objects"
            )
        role = _lc_role(obj)
        text = _lc_content_text(getattr(obj, "content", ""))
        name = getattr(obj, "name", None)
        extra = getattr(obj, "additional_kwargs", {}) or {}
        tool_calls = list(getattr(obj, "tool_calls", None) or extra.get("tool_calls") or [])
        tool_call_id = getattr(obj, "tool_call_id", None)
        section = assign_section(
            role, KIND_TOOL_RESULT if tool_call_id else KIND_TEXT, getattr(obj, "section", None)
        )
        pinned = bool(getattr(obj, "pinned", False)) or role == "system"
        meta = {"src_index": i, "src": "langchain"}

        if tool_calls and role == "assistant":
            if text:
                out.append(
                    Message(
                        role=role,
                        content=text,
                        kind=KIND_TEXT,
                        name=name,
                        section=section,
                        pinned=pinned,
                        meta={**meta, "part": "text"},
                    )
                )
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                out.append(
                    Message(
                        role="assistant",
                        content=args,
                        kind=KIND_TOOL_CALL,
                        name=fn.get("name") or getattr(tc, "name", None),
                        call_id=tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                        section=section,
                        pinned=pinned,
                        meta={**meta, "part": "tool_call"},
                    )
                )
        elif tool_call_id:
            out.append(
                Message(
                    role="tool",
                    content=text,
                    kind=KIND_TOOL_RESULT,
                    name=name,
                    call_id=tool_call_id,
                    section=section,
                    pinned=pinned,
                    meta=meta,
                )
            )
        else:
            out.append(
                Message(
                    role=role,
                    content=text,
                    kind=KIND_TEXT,
                    name=name,
                    section=section,
                    pinned=pinned,
                    meta=meta,
                )
            )
    return out


def to_lc_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Plain dicts shaped like LangChain messages (no langchain import)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.kind == KIND_TOOL_CALL:
            out.append(
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": m.name or "",
                            "args": m.content,
                            "id": m.call_id,
                            "type": "tool_call",
                        }
                    ],
                }
            )
        elif m.kind == KIND_TOOL_RESULT:
            out.append(
                {"type": "tool", "content": m.content, "tool_call_id": m.call_id, "name": m.name}
            )
        else:
            t = {"system": "system", "user": "human", "assistant": "ai", "tool": "tool"}.get(
                m.role, "human"
            )
            out.append({"type": t, "content": m.content, "name": m.name})
    return out
