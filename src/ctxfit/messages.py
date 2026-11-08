"""Canonical message model for ctxfit.

The core pipeline never touches framework shapes directly. Every input is
normalized into a list of :class:`Message` objects, strategies operate on
those, and adapters convert back at the boundary.

The opt-in running-notes layer (:mod:`ctxfit.notes`) is the exception: it
runs on the raw provider JSON *before* (instead of) :func:`normalize`, so
the blocks it does not rewrite pass through byte for byte, and it has its own
turn and pin rules (shared with the TypeScript port; see that module). Given
``Message`` objects it processes their :meth:`Message.to_dict` form and hands
back ``Message`` objects with ``meta`` and every other field kept, only
``content`` changed.
"""

from __future__ import annotations

import re
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
        # Only what the caller could not get back by re-reading the dict: the section when it is
        # not the one the role/kind implies, and ``pinned`` when the role does not pin by itself.
        # (A system message used to come back with ``section``/``pinned`` keys it never had,
        # which a provider API may reject as unknown fields.)
        if self.section != assign_section(self.role, self.kind):
            d["section"] = self.section
        if self.pinned and self.role != "system":
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


_ZERO_WIDTH = "\u200b-\u200f\u2028-\u202e\u2060-\u2064\u2066-\u2069\ufeff"
_UNSAFE_KEEP_WS_RE = re.compile(f"[\x00-\x08\x0b-\x1f\x7f-\x9f{_ZERO_WIDTH}]")
_UNSAFE_ALL_RE = re.compile(f"[\x00-\x1f\x7f-\x9f{_ZERO_WIDTH}]")


def printable(text: str, limit: int | None = None) -> str:
    """Untrusted text made safe to print: control, bidi and zero-width characters become \\uXXXX.

    With ``limit`` only the first ``limit`` characters are used and the result is one line
    (newlines and tabs are escaped too); without it newlines and tabs are kept.
    """
    if limit is None:
        return _UNSAFE_KEEP_WS_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", text)
    return _UNSAFE_ALL_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", text[:limit])


_KNOWN_KEYS = frozenset(
    {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "pinned",
        "section",
        "kind",
        "call_id",
    }
)
_PLAIN_TEXT = frozenset({"type", "text"})
_CTX_KEYS = ("pinned", "section", "kind", "call_id")  # ctxfit's own directives; never handed back


def _is_opaque(m: Any) -> bool:
    """True when a raw message carries parts the canonical text form cannot hold.

    Images, documents, thinking blocks (with signatures), cache_control marks, extra fields on
    content blocks or tool calls (for example a provider's thought signature). Rebuilding such a
    message from text would silently delete them, so ctxfit keeps the original: strategies do not
    rewrite it (it can still be dropped whole) and it is handed back unchanged.
    """
    if not isinstance(m, dict):
        return False
    content = m.get("content")
    if content is not None and not isinstance(content, (str, list)):
        return True  # numbers, objects: not text, so not ours to restate
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                return True
            t = p.get("type")
            if t in ("text", "input_text"):
                if set(p) - _PLAIN_TEXT:
                    return True
            elif t == "tool_use":
                if set(p) - {"type", "id", "name", "input"}:
                    return True
            elif t == "tool_result":
                if set(p) - {"type", "tool_use_id", "content", "is_error"}:
                    return True
                inner = p.get("content")
                if isinstance(inner, list) and any(
                    not (
                        isinstance(q, dict) and q.get("type") == "text" and not set(q) - _PLAIN_TEXT
                    )
                    for q in inner
                ):
                    return True
            else:
                return True
    calls = m.get("tool_calls")
    if isinstance(calls, list):
        for tc in calls:
            if not isinstance(tc, dict) or set(tc) - {"id", "type", "function"}:
                return True
            fn = tc.get("function")
            if isinstance(fn, dict) and set(fn) - {"name", "arguments"}:
                return True
    return False


def _mark_opaque(raw: Sequence[Any], out: list[Message]) -> list[Message]:
    """Record, per source message, what a rebuild would lose (see :func:`raw_passthrough`)."""
    counts: dict[int, int] = {}
    for msg in out:
        i = msg.meta.get("src_index")
        if isinstance(i, int):
            counts[i] = counts.get(i, 0) + 1
    for msg in out:
        i = msg.meta.get("src_index")
        if not (isinstance(i, int) and 0 <= i < len(raw) and isinstance(raw[i], dict)):
            continue
        src = raw[i]
        if _is_opaque(src):
            msg.meta["ctxfit_raw"] = src
            msg.meta["ctxfit_orig"] = msg.content
            msg.meta["ctxfit_nparts"] = counts[i]
        extras = {k: v for k, v in src.items() if k not in _KNOWN_KEYS or k == "name"}
        if extras:
            msg.meta["ctxfit_extras"] = extras
    return out


def raw_passthrough(parts: Sequence[Message]) -> dict[str, Any] | None:
    """The original message dict for an opaque source nobody changed, else None."""
    first = parts[0]
    raw = first.meta.get("ctxfit_raw")
    if raw is None or len(parts) != first.meta.get("ctxfit_nparts"):
        return None
    if any(
        p.meta.get("ctxfit_raw") is not raw or p.content != p.meta.get("ctxfit_orig") for p in parts
    ):
        return None
    import copy

    d = copy.deepcopy(raw)
    for k in _CTX_KEYS:
        d.pop(k, None)
    return d


def with_extras(d: dict[str, Any], parts: Sequence[Message]) -> dict[str, Any]:
    """Add back the provider fields ctxfit does not model (``refusal``, ``reasoning_content``, ...)."""
    extras = parts[0].meta.get("ctxfit_extras")
    if extras:
        import copy

        for k, v in extras.items():
            d.setdefault(k, copy.deepcopy(v))
    return d


def is_opaque_message(m: Message) -> bool:
    """True for a message ctxfit must not rewrite (it carries non-text parts)."""
    return "ctxfit_raw" in m.meta


def count_opaque(messages: Sequence[Message]) -> int:
    """How many source messages are being passed through unchanged (see :func:`_is_opaque`)."""
    return len({id(m.meta["ctxfit_raw"]) for m in messages if "ctxfit_raw" in m.meta})


def normalize(raw: Sequence[Any], flavor: str = "auto") -> list[Message]:
    """Normalize any supported input into canonical Messages.

    flavor: "auto" | "openai" | "anthropic" | "langchain" | "generic".
    Message objects pass through (re-pinned/re-sectioned only if unset).
    """
    if raw and all(isinstance(m, Message) for m in raw):
        return list(raw)  # type: ignore[arg-type]
    if flavor == "auto":
        flavor = detect_flavor(raw)
    if flavor not in ("openai", "anthropic", "langchain", "generic"):
        raise ValueError(
            f"unknown flavor {flavor!r}; expected 'auto', 'openai', 'anthropic', 'langchain' or 'generic'"
        )
    if flavor == "openai":
        from .adapters.openai import from_openai

        return _mark_opaque(raw, from_openai(raw))  # type: ignore[arg-type]
    if flavor == "anthropic":
        from .adapters.anthropic import from_anthropic

        return _mark_opaque(raw, from_anthropic(raw))  # type: ignore[arg-type]
    if flavor == "langchain":
        from .adapters.langchain import from_lc_messages

        return from_lc_messages(raw)
    return _mark_opaque(raw, from_dicts(raw))  # type: ignore[arg-type]


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


def repair_pairing(messages: Sequence[Message]) -> tuple[list[Message], list[str]]:
    """Drop what breaks the pair-atomicity invariant; return ``(kept, what_was_dropped)``.

    A conversation that was cut in the middle (a tool result whose call was sliced off, a call
    whose result has not arrived yet) or that repeats a tool id cannot be fitted safely. This
    removes the unmatched tool_call / tool_result messages and the second and later users of a
    repeated id. Nothing else is touched. A message that also holds images or other non-text
    parts is never dropped here: the caller gets a :class:`ValueError` instead (fail closed).
    """
    first_call: dict[str, int] = {}
    first_result: dict[str, int] = {}
    for i, m in enumerate(messages):
        if m.call_id and m.kind == KIND_TOOL_CALL:
            first_call.setdefault(m.call_id, i)
        elif m.call_id and m.kind == KIND_TOOL_RESULT:
            first_result.setdefault(m.call_id, i)
    kept: list[Message] = []
    dropped: list[str] = []
    for i, m in enumerate(messages):
        why = ""
        if m.call_id and m.kind == KIND_TOOL_CALL:
            if first_call[m.call_id] != i:
                why = "repeated tool_call id"
            elif m.call_id not in first_result:
                why = "tool_call with no tool_result"
        elif m.call_id and m.kind == KIND_TOOL_RESULT:
            if first_result[m.call_id] != i:
                why = "repeated tool_result id"
            elif m.call_id not in first_call:
                why = "tool_result with no tool_call"
        if not why:
            kept.append(m)
            continue
        if "ctxfit_raw" in m.meta:
            raise ValueError(
                f"message #{i} ({why}, id {printable(str(m.call_id), 60)!r}) also holds images or "
                "other non-text parts, so ctxfit will not drop it; fix the conversation first"
            )
        dropped.append(f"#{i}: {why}, id {printable(str(m.call_id), 60)!r}")
    return kept, dropped


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
