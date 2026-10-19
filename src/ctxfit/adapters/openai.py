"""OpenAI chat-completions shaped adapter.

Handles ``{"role", "content", "tool_calls", "tool_call_id", "name"}`` dicts.
Assistant messages carrying ``tool_calls`` are split into one text Message
(optional) plus one ``kind="tool_call"`` Message per call, linked by
``call_id`` to the ``role="tool"`` result messages. ``to_openai`` rejoins
them via the shared ``src_index`` recorded at split time.

Also covers OpenAI-compatible endpoints (freellmapi, Ollama, etc.).
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


def _content_to_text(content: Any) -> tuple[str, dict[str, Any]]:
    """Flatten string or content-part lists. Returns (text, notes)."""
    notes: dict[str, Any] = {}
    if content is None:
        return "", notes
    if isinstance(content, str):
        return content, notes
    if isinstance(content, list):
        parts: list[str] = []
        images = 0
        for p in content:
            if isinstance(p, dict):
                t = p.get("type")
                if t in ("text", "input_text"):
                    parts.append(str(p.get("text", "")))
                elif t in ("image_url", "image"):
                    images += 1
                else:
                    parts.append("[non-text part omitted]")
            else:
                parts.append(str(p))
        if images:
            notes["omitted_images"] = images
            parts.append(f"[{images} image part(s) omitted]")
        return "\n".join(parts), notes
    return str(content), notes


def from_openai(raw: Sequence[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for i, m in enumerate(raw):
        role = str(m.get("role", "user"))
        if role == "function":
            role = "tool"  # legacy function-calling shape
        text, notes = _content_to_text(m.get("content"))
        section = assign_section(role, KIND_TEXT, m.get("section"))
        pinned = bool(m.get("pinned", False)) or role == "system"
        base_meta = {"src_index": i, "src": "openai", **notes}

        tool_calls = m.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            if text:
                out.append(
                    Message(
                        role=role,
                        content=text,
                        kind=KIND_TEXT,
                        section=section,
                        pinned=pinned,
                        meta={**base_meta, "part": "text"},
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
                        content=args,  # NEVER modified by strategies
                        kind=KIND_TOOL_CALL,
                        name=fn.get("name"),
                        call_id=tc.get("id") if isinstance(tc, dict) else None,
                        section=section,
                        pinned=pinned,
                        meta={
                            **base_meta,
                            "part": "tool_call",
                            "tool_call_type": tc.get("type", "function")
                            if isinstance(tc, dict)
                            else "function",
                        },
                    )
                )
        elif role in ("tool",):
            out.append(
                Message(
                    role="tool",
                    content=text,
                    kind=KIND_TOOL_RESULT,
                    name=m.get("name"),
                    call_id=m.get("tool_call_id"),
                    section=assign_section(role, KIND_TOOL_RESULT, m.get("section")),
                    pinned=pinned,
                    meta=base_meta,
                )
            )
        else:
            out.append(
                Message(
                    role=role,
                    content=text,
                    kind=KIND_TEXT,
                    name=m.get("name"),
                    section=section,
                    pinned=pinned,
                    meta=base_meta,
                )
            )
    return out


def to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Rejoin split assistant tool_calls; preserve original ordering."""
    grouped: dict[int, list[Message]] = {}
    order: list[int] = []
    ungrouped: list[Message] = []
    for m in messages:
        src = m.meta.get("src_index")
        if src is None:
            ungrouped.append(m)
            continue
        if src not in grouped:
            grouped[src] = []
            order.append(src)
        grouped[src].append(m)

    out: list[dict[str, Any]] = []
    for src in order:
        parts = grouped[src]
        first = parts[0]
        if first.role == "assistant" and any(p.kind == KIND_TOOL_CALL for p in parts):
            text_parts = [p.content for p in parts if p.kind == KIND_TEXT]
            tool_calls = [
                {
                    "id": p.call_id,
                    "type": p.meta.get("tool_call_type", "function"),
                    "function": {"name": p.name or "", "arguments": p.content},
                }
                for p in parts
                if p.kind == KIND_TOOL_CALL
            ]
            d: dict[str, Any] = {"role": "assistant"}
            d["content"] = "\n".join(text_parts) if text_parts else None
            d["tool_calls"] = tool_calls
            out.append(d)
        elif len(parts) == 1:
            out.append(_single_to_openai(parts[0]))
        else:
            # Same source produced several non-tool messages (e.g. dedup
            # markers); emit each in order.
            for p in parts:
                out.append(_single_to_openai(p))
    for m in ungrouped:
        out.append(_single_to_openai(m))
    return out


def _single_to_openai(m: Message) -> dict[str, Any]:
    if m.kind == KIND_TOOL_RESULT:
        d: dict[str, Any] = {"role": "tool", "content": m.content}
        if m.call_id:
            d["tool_call_id"] = m.call_id
        if m.name:
            d["name"] = m.name
        return d
    if m.kind == KIND_TOOL_CALL:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": m.call_id,
                    "type": "function",
                    "function": {"name": m.name or "", "arguments": m.content},
                }
            ],
        }
    return {"role": m.role, "content": m.content}
