"""Anthropic messages-API shaped adapter (block-based content).

``tool_use`` blocks become ``kind="tool_call"`` messages and
``tool_result`` blocks become ``kind="tool_result"`` messages, paired via
the block ``id`` / ``tool_use_id``. ``to_anthropic`` rebuilds blocks,
grouped by the shared ``src_index`` recorded at split time.
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


def _blocks_to_parts(
    blocks: Sequence[dict[str, Any]], src_index: int, role: str, section: str, pinned: bool
) -> list[Message]:
    out: list[Message] = []
    for b in blocks:
        btype = b.get("type")
        meta = {"src_index": src_index, "src": "anthropic"}
        if btype == "tool_use":
            out.append(
                Message(
                    role="assistant",
                    content=json.dumps(b.get("input", {}), ensure_ascii=False),
                    kind=KIND_TOOL_CALL,
                    name=b.get("name"),
                    call_id=b.get("id"),
                    section=section,
                    pinned=pinned,
                    meta={**meta, "part": "tool_use"},
                )
            )
        elif btype == "tool_result":
            content = b.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            out.append(
                Message(
                    role="user",  # tool results live in user messages in Anthropic's API
                    content=str(content),
                    kind=KIND_TOOL_RESULT,
                    call_id=b.get("tool_use_id"),
                    section=assign_section("user", KIND_TOOL_RESULT, None),
                    pinned=pinned,
                    meta={
                        **meta,
                        "part": "tool_result",
                        "is_error": bool(b.get("is_error", False)),
                    },
                )
            )
        elif btype == "text":
            out.append(
                Message(
                    role=role,
                    content=str(b.get("text", "")),
                    kind=KIND_TEXT,
                    section=section,
                    pinned=pinned,
                    meta={**meta, "part": "text"},
                )
            )
        # other block types (image, document...) are dropped with a note in
        # the surviving text part's meta — strategies only handle text.
    return out


def from_anthropic(raw: Sequence[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for i, m in enumerate(raw):
        role = str(m.get("role", "user"))
        section = assign_section(role, KIND_TEXT, m.get("section"))
        pinned = bool(m.get("pinned", False))
        content = m.get("content", "")
        if isinstance(content, list):
            out.extend(_blocks_to_parts(content, i, role, section, pinned))
        else:
            out.append(
                Message(
                    role=role,
                    content=str(content),
                    kind=KIND_TEXT,
                    section=section,
                    pinned=pinned,
                    meta={"src_index": i, "src": "anthropic"},
                )
            )
    return out


def to_anthropic(messages: Sequence[Message]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Message]] = {}
    order: list[int] = []
    for m in messages:
        src = m.meta.get("src_index")
        if src is None:
            src = -1 - len(order)  # unique negative key, keeps order
        if src not in grouped:
            grouped[src] = []
            order.append(src)
        grouped[src].append(m)

    out: list[dict[str, Any]] = []
    for src in order:
        parts = grouped[src]
        # Anthropic role: assistant if any assistant/tool_call part, else user.
        # (System messages are top-level in the real API; we keep them as
        # role "system" dicts and let the caller hoist them.)
        roles = {p.role for p in parts}
        role = (
            "assistant"
            if ("assistant" in roles or any(p.kind == KIND_TOOL_CALL for p in parts))
            else "user"
        )
        if roles == {"system"}:
            role = "system"
        blocks: list[dict[str, Any]] = []
        for p in parts:
            if p.kind == KIND_TOOL_CALL:
                try:
                    input_data = json.loads(p.content) if p.content else {}
                except (json.JSONDecodeError, ValueError):
                    input_data = {}
                blocks.append(
                    {"type": "tool_use", "id": p.call_id, "name": p.name or "", "input": input_data}
                )
            elif p.kind == KIND_TOOL_RESULT:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": p.call_id,
                    "content": p.content,
                }
                if p.meta.get("is_error"):
                    block["is_error"] = True
                blocks.append(block)
            else:
                blocks.append({"type": "text", "text": p.content})
        if role == "system" and all(b.get("type") == "text" for b in blocks):
            out.append({"role": "system", "content": "\n".join(b["text"] for b in blocks)})
        else:
            out.append({"role": role, "content": blocks})
    return out
