"""Canonical message model for ctxfit.

The core pipeline never touches framework shapes directly. Every input is
normalized into a list of :class:`Message` objects, strategies operate on
those, and adapters convert back at the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Roles ctxfit understands. Unknown roles are passed through untouched.
KNOWN_ROLES = ("system", "developer", "user", "assistant", "tool", "function")

# Sections a message can belong to. Budgets are declared per section.
SECTIONS = ("system", "retrieved", "history", "scratchpad")

# Kinds of message content. tool_call / tool_result pairs are atomic:
# strategies must never orphan one side of a pair.
KIND_TEXT = "text"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """One normalized message.

    Attributes:
        role: API role (system/user/assistant/tool/...).
        content: Normalized plain-text content.
        kind: "text" | "tool_call" | "tool_result".
        name: Tool/function name when applicable.
        call_id: Pairing key linking a tool_call to its tool_result.
        section: Budget section: system|retrieved|history|scratchpad.
        pinned: Exempt from dropping; truncated only with explicit consent.
        meta: Adapter round-trip bookkeeping. Never user-visible content.
    """

    role: str
    content: str = ""
    kind: str = KIND_TEXT
    name: str | None = None
    call_id: str | None = None
    section: str | None = None
    pinned: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-assign section from role/kind unless explicitly given.
        # (Assignment is deterministic — never inferred from content length,
        # so budgets can't be gamed by mislabeling.)
        if self.section is None:
            self.section = assign_section(self.role, self.kind, None)
        if self.section not in SECTIONS:
            raise ValueError(f"unknown section {self.section!r}; expected one of {SECTIONS}")
        if self.content is None:
            self.content = ""
        # System messages are pinned by default: the task definition must
        # never be silently dropped by a sliding window.
        if self.role == "system":
            self.pinned = True

    def char_len(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Generic dict form (the first-class raw message-list shape)."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.kind != KIND_TEXT:
            d["kind"] = self.kind
        if self.name:
            d["name"] = self.name
        if self.call_id:
            d["call_id"] = self.call_id
        if self.section != "history":
            d["section"] = self.section
        if self.pinned:
            d["pinned"] = True
        return d


def assign_section(role: str, kind: str, explicit: str | None = None) -> str:
    """Deterministic section assignment.

    Never inferred from content length (that would let callers game
    budgets by mislabeling). Explicit per-message overrides win, then
    role/kind rules.
    """
    if explicit in SECTIONS:
        return explicit  # type: ignore[return-value]
    if role == "system":
        return "system"
    if role in ("tool", "function") or kind == KIND_TOOL_RESULT:
        return "retrieved"
    if role == "developer":
        return "scratchpad"
    return "history"


def from_dicts(raw: Sequence[dict[str, Any]]) -> list[Message]:
    """Normalize generic {"role","content"} dicts into Messages."""
    out: list[Message] = []
    for i, m in enumerate(raw):
        if isinstance(m, Message):
            out.append(m)
            continue
        if not isinstance(m, dict):
            raise TypeError(f"message #{i} must be a dict or Message, got {type(m).__name__}")
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, list):
            # Content-part lists (multimodal leftovers): join text parts.
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append("[non-text part omitted]")
            content = "\n".join(parts)
        content = "" if content is None else str(content)
        kind = str(m.get("kind", KIND_TEXT))
        section = assign_section(role, kind, m.get("section"))
        pinned = bool(m.get("pinned", False)) or role == "system"
        out.append(
            Message(
                role=role,
                content=content,
                kind=kind,
                name=m.get("name"),
                call_id=m.get("call_id") or m.get("tool_call_id"),
                section=section,
                pinned=pinned,
                meta={"src_index": i, "src": "generic"},
            )
        )
    return out


def detect_flavor(raw: Sequence[Any]) -> str:
    """Best-effort flavor detection for dict-shaped inputs.

    Returns one of "openai", "anthropic", "langchain", "generic".
    """
    for m in raw:
        if isinstance(m, Message):
            continue
        if not isinstance(m, dict):
            if hasattr(m, "content"):
                return "langchain"  # duck-typed LC-style message object
            continue
        if "tool_calls" in m or m.get("role") == "tool" or m.get("tool_call_id"):
            return "openai"
        content = m.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") in ("tool_use", "tool_result"):
                    return "anthropic"
    return "generic"


def normalize(raw: Sequence[Any], flavor: str = "auto") -> list[Message]:
    """Normalize any supported input into canonical Messages.

    flavor: "auto" | "openai" | "anthropic" | "langchain" | "generic".
    Message objects pass through (re-pinned/re-sectioned only if unset).
    """
    if raw and all(isinstance(m, Message) for m in raw):
        return list(raw)  # type: ignore[arg-type]
    if flavor == "auto":
        flavor = detect_flavor(raw)
    if flavor == "openai":
        from .adapters.openai import from_openai

        return from_openai(raw)  # type: ignore[arg-type]
    if flavor == "anthropic":
        from .adapters.anthropic import from_anthropic

        return from_anthropic(raw)  # type: ignore[arg-type]
    if flavor == "langchain":
        from .adapters.langchain import from_lc_messages

        return from_lc_messages(raw)
    return from_dicts(raw)  # type: ignore[arg-type]


def validate_pairing(messages: Sequence[Message]) -> list[str]:
    """Check the pair-atomicity invariant.

    Every tool_call id must have exactly one matching tool_result and
    vice versa — orphans and duplicate ids are both violations.
    Returns a list of human-readable violations (empty = OK).
    """
    calls: dict[str, int] = {}
    results: dict[str, int] = {}
    problems: list[str] = []
    for i, m in enumerate(messages):
        if m.kind == KIND_TOOL_CALL and m.call_id:
            if m.call_id in calls:
                problems.append(
                    f"duplicate tool_call id {m.call_id!r} at #{i} "
                    f"(first seen at #{calls[m.call_id]})"
                )
            else:
                calls[m.call_id] = i
        elif m.kind == KIND_TOOL_RESULT and m.call_id:
            if m.call_id in results:
                problems.append(
                    f"duplicate tool_result id {m.call_id!r} at #{i} "
                    f"(first seen at #{results[m.call_id]})"
                )
            else:
                results[m.call_id] = i
    for cid, i in calls.items():
        if cid not in results:
            problems.append(f"orphaned tool_call #{i} (call_id={cid!r}): no matching tool_result")
    for cid, i in results.items():
        if cid not in calls:
            problems.append(f"orphaned tool_result #{i} (call_id={cid!r}): no matching tool_call")
    # A tool_call must never carry truncated/invalid args silently: content
    # must be non-empty JSON-ish. (Adapters guarantee this; strategies must
    # never touch tool_call content at all.)
    return problems


def turn_indexes(messages: Sequence[Message]) -> list[int]:
    """Assign each message a turn index.

    A new turn starts at every "user" role message. System messages get
    turn -1 (they are pinned anyway). Tool pairs stay inside the turn of
    the assistant message that produced them because they always follow it.
    """
    turns: list[int] = []
    turn = -1
    seen_user = False
    for m in messages:
        if m.role == "system":
            turns.append(-1)
        elif m.role == "user":
            if seen_user:
                turn += 1
            else:
                seen_user = True
                turn = 0
            turns.append(turn)
        else:
            if not seen_user:
                turn = 0
                seen_user = True
            turns.append(turn)
    return turns
