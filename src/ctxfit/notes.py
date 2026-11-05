"""Running notes: observation masking + incremental conversation notes.

Two layers, applied in order, both opt-in:

- **Layer 1 — observation masking** (:func:`mask_observations`): old tool
  outputs are replaced with a short, deterministic marker. No model call.
  The masking boundary moves in chunks of ``mask_chunk_turns`` turns, so the
  rewritten prefix stays byte-identical between moves and provider prompt
  caching keeps working.
- **Layer 2 — running notes** (:class:`NotesKeeper`): once the masked
  conversation passes ``high_water_tokens``, the oldest unpinned turns are
  replaced by ONE notes message written by a cheap summariser model. Notes
  are updated incrementally (one summariser call per cut move), reused with
  no model call while the history prefix is unchanged, and discarded if the
  history forks.

This module works on provider JSON directly (Anthropic Messages API
``messages`` arrays, OpenAI chat.completions ``messages`` arrays, or
ctxfit's generic dicts) rather than round-tripping through
:class:`~ctxfit.messages.Message`, so every block it does not rewrite —
images, ``cache_control``, unknown keys — passes through untouched, and the
output matches the TypeScript port (calas-gateway) byte for byte on the
shared fixtures.

Safety invariants (same as the rest of ctxfit):

- inputs are never mutated (the output is a deep copy);
- tool calls are never modified, and a tool call is never separated from its
  result (cuts only land on pair-safe turn boundaries; asserted at the end);
- pinned content (system/developer messages, ``pinned: true``, the first
  user message, any message whose text contains ``[pin]``) is never masked
  and never summarised;
- a summariser failure never fails the request: the previous notes and cut
  are kept and the failure is reported.

The spec both implementations follow is ``SPEC-running-notes.md``; the
"Resolved ambiguities" section of the README in the shared ``fixtures/``
directory records every interpretation made here.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .messages import Message, detect_flavor, from_dicts

# ---------------------------------------------------------------------------
# Constants (identical in the TypeScript port)
# ---------------------------------------------------------------------------

STATE_VERSION = 1

SUMMARISER_SYSTEM_PROMPT = (
    "You maintain compact running notes for an AI agent's long conversation. "
    "Update the notes using ONLY the new events. Keep: the user's goal, decisions and why, "
    "facts learned (names, numbers, paths, ids), files touched, errors and fixes, open tasks "
    "and next step. Drop chit-chat and anything superseded. Never invent. Never include "
    "secrets (keys, tokens, passwords) - write [redacted]. Plain text with short headed "
    "sections: Goal, Decisions, Facts, Files, Open items, Next step. Stay under "
    "{notes_max_tokens} tokens."
)
"""Summariser system prompt. ``{notes_max_tokens}`` is replaced literally."""

SUMMARISER_USER_PROMPT = "CURRENT NOTES:\n{previous_notes}\n\nNEW EVENTS:\n{rendered}"
"""Summariser user prompt. ``{previous_notes}`` is ``(none yet)`` on the first call."""

NO_NOTES_YET = "(none yet)"

MASK_MARKER = "[ctxfit: tool output hidden - {tool} - {n} chars - id {id8}]"
NOTES_HEADER = (
    "[ctxfit running notes - replaces {k} earlier messages; pinned rules above still apply]"
)

NON_TEXT = "[non-text]"
RENDER_ARGS_MAX_CHARS = 500
RENDER_RESULT_MAX_CHARS = 1500
REDACTED = "[redacted]"

_PIN_RE = re.compile(r"\[pin\]", re.IGNORECASE)

# Secret scrubber. Applied in this order. ASCII semantics on purpose so the
# JavaScript port (no `u` flag) matches exactly.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.ASCII,
        ),
    ),
    ("anthropic_or_openai_key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}", re.ASCII)),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b", re.ASCII)),
    (
        "github_token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})", re.ASCII),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*", re.ASCII),
    ),
    ("password", re.compile(r"password[\"']?\s*[:=]\s*\S+", re.ASCII | re.IGNORECASE)),
)


def scrub_secrets(text: str) -> str:
    """Replace secret-looking substrings with ``[redacted]``.

    Covers ``sk-``/``sk-ant-`` API keys, AWS ``AKIA`` access keys, GitHub
    ``ghp_``/``github_pat_`` tokens, JWTs, PEM private-key blocks and
    ``password: ...`` / ``password=...`` assignments. Best effort — a regex
    scrubber is a safety net, not a guarantee.
    """
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


# ---------------------------------------------------------------------------
# Deterministic JSON + estimates (shared definitions with the TS port)
# ---------------------------------------------------------------------------


def _js_number(f: float) -> str:
    """Format a float exactly like JavaScript's ``Number.prototype.toString``."""
    if not math.isfinite(f):
        return "null"
    if f == 0:
        return "0"
    sign = "-" if f < 0 else ""
    _s, digits_t, exp = Decimal(repr(abs(f))).normalize().as_tuple()
    digits = "".join(str(d) for d in digits_t)
    k = len(digits)
    n = k + int(exp)  # value = 0.DIGITS * 10**n
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits[0] + ("." + digits[1:] if k > 1 else "")
        out = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + out


def _json(obj: Any, sort_keys: bool = False) -> str:
    """Compact JSON matching ``JSON.stringify`` (optionally with sorted keys)."""
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _js_number(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, dict):
        keys = sorted(obj, key=str) if sort_keys else list(obj)
        return (
            "{"
            + ",".join(
                json.dumps(str(k), ensure_ascii=False) + ":" + _json(obj[k], sort_keys)
                for k in keys
            )
            + "}"
        )
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_json(v, sort_keys) for v in obj) + "]"
    return json.dumps(str(obj), ensure_ascii=False)


def canonical_json(obj: Any) -> str:
    """Sorted-key compact JSON used for prefix and cache hashes."""
    return _json(obj, sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def estimate_tokens(chars: int) -> int:
    """The spec's shared token estimate: ``ceil(chars / 4)``.

    ``chars`` is a count of Unicode code points. Both implementations use
    this exact rule for triggers and reports so their numbers agree.
    """
    return (max(0, chars) + 3) // 4


# ---------------------------------------------------------------------------
# Provider-shape views
# ---------------------------------------------------------------------------


@dataclass
class _View:
    """Read-only facts about one raw message (never rendered as output)."""

    index: int
    role: str
    starts_turn: bool
    text: str  # non-tool text (for [pin] detection and rendering)
    calls: list[tuple[str | None, str, str]]  # (call_id, tool_name, args_json)
    results: list[tuple[str | None, str]]  # (call_id, original_text)
    pinned: bool = False
    turn: int = 0


def _parts_text(parts: Sequence[Any], text_types: tuple[str, ...]) -> str:
    out: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") in text_types:
            out.append(str(p.get("text", "")))
        elif isinstance(p, str):
            out.append(p)
        else:
            out.append(NON_TEXT)
    return "\n".join(out)


def _content_text(content: Any, text_types: tuple[str, ...] = ("text",)) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _parts_text(content, text_types)
    return str(content)


def _anthropic_result_text(block: dict[str, Any]) -> str:
    return _content_text(block.get("content", ""))


def _openai_args(args: Any) -> str:
    return args if isinstance(args, str) else _json(args)


_OPENAI_TEXT = ("text", "input_text")


def _view_anthropic(i: int, m: dict[str, Any]) -> _View:
    role = str(m.get("role", "user"))
    content = m.get("content", "")
    text_parts: list[str] = []
    calls: list[tuple[str | None, str, str]] = []
    results: list[tuple[str | None, str]] = []
    if isinstance(content, list):
        for b in content:
            btype = b.get("type") if isinstance(b, dict) else None
            if btype == "tool_use":
                calls.append((b.get("id"), str(b.get("name") or "tool"), _json(b.get("input", {}))))
            elif btype == "tool_result":
                results.append((b.get("tool_use_id"), _anthropic_result_text(b)))
            elif btype == "text":
                text_parts.append(str(b.get("text", "")))
            else:
                text_parts.append(NON_TEXT)
        only_results = bool(content) and len(results) == len(content)
    else:
        text_parts.append(_content_text(content))
        only_results = False
    return _View(
        index=i,
        role=role,
        starts_turn=role == "user" and not only_results,
        text="\n".join(text_parts),
        calls=calls,
        results=results,
    )


def _view_openai(i: int, m: dict[str, Any]) -> _View:
    role = str(m.get("role", "user"))
    if role in ("tool", "function"):
        return _View(
            index=i,
            role=role,
            starts_turn=False,
            text="",
            calls=[],
            results=[(m.get("tool_call_id"), _content_text(m.get("content"), _OPENAI_TEXT))],
        )
    calls: list[tuple[str | None, str, str]] = []
    for tc in m.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        calls.append(
            (tc.get("id"), str(fn.get("name") or "tool"), _openai_args(fn.get("arguments", "")))
        )
    return _View(
        index=i,
        role=role,
        starts_turn=role == "user",
        text=_content_text(m.get("content"), _OPENAI_TEXT),
        calls=calls,
        results=[],
    )


def _view_generic(i: int, m: dict[str, Any]) -> _View:
    role = str(m.get("role", "user"))
    kind = m.get("kind", "text")
    cid = m.get("call_id") or m.get("tool_call_id")
    content = _content_text(m.get("content"), _OPENAI_TEXT)
    if kind == "tool_result" or role in ("tool", "function"):
        return _View(i, role, False, "", [], [(cid, content)])
    if kind == "tool_call":
        return _View(i, role, False, "", [(cid, str(m.get("name") or "tool"), content)], [])
    return _View(i, role, role == "user", content, [], [])


_VIEWERS = {"anthropic": _view_anthropic, "openai": _view_openai, "generic": _view_generic}


class _Conversation:
    """Turn numbering, pin flags and tool-name lookup over raw messages."""

    def __init__(self, flavor: str, raw: list[dict[str, Any]]) -> None:
        self.flavor = flavor
        self.raw = raw
        viewer = _VIEWERS[flavor]
        self.views = [viewer(i, m) for i, m in enumerate(raw)]
        # Turns: a turn starts at each user message that is not purely tool
        # results. Anything before the first turn start belongs to turn 0.
        turn = -1
        first_user: int | None = None
        for v in self.views:
            if v.starts_turn:
                turn += 1
                if first_user is None:
                    first_user = v.index
            v.turn = max(turn, 0)
        self.turn_count = (turn + 1) if turn >= 0 else (1 if raw else 0)
        self.turn_start: dict[int, int] = {}
        for v in self.views:
            if v.starts_turn:
                self.turn_start.setdefault(v.turn, v.index)
        for v, m in zip(self.views, raw):
            v.pinned = (
                v.role in ("system", "developer")
                or (isinstance(m, dict) and m.get("pinned") is True)
                or v.index == first_user
                or bool(_PIN_RE.search(v.text))
            )
        # Tool name per call id (results inherit the name of their call).
        self.tool_names: dict[str, str] = {}
        self.call_turn: dict[str, int] = {}
        self.result_turn: dict[str, int] = {}
        self.call_msg: dict[str, int] = {}
        self.result_msg: dict[str, int] = {}
        for v in self.views:
            for cid, name, _args in v.calls:
                if cid:
                    self.tool_names.setdefault(cid, name)
                    self.call_turn.setdefault(cid, v.turn)
                    self.call_msg.setdefault(cid, v.index)
            for cid, _text in v.results:
                if cid:
                    self.result_turn.setdefault(cid, v.turn)
                    self.result_msg.setdefault(cid, v.index)

    def tool_name(self, call_id: str | None, raw_msg: dict[str, Any]) -> str:
        if call_id and call_id in self.tool_names:
            return self.tool_names[call_id]
        name = raw_msg.get("name") if isinstance(raw_msg, dict) else None
        return str(name) if name else "tool"

    def pair_safe_boundary(self, c: int) -> bool:
        """True when no tool call/result pair straddles turn boundary ``c``."""
        for cid, ct in self.call_turn.items():
            rt = self.result_turn.get(cid)
            if rt is None:
                continue
            if min(ct, rt) < c <= max(ct, rt):
                return False
        return True

    def kept_before(self, c: int) -> list[int]:
        """Indexes kept ahead of cut ``c``: pinned messages, closed over pairs.

        A pinned message's tool-call/result partners are kept with it so a
        cut never orphans half of a pair.
        """
        keep = {v.index for v in self.views if v.turn < c and v.pinned}
        changed = True
        while changed:
            changed = False
            for idx in list(keep):
                v = self.views[idx]
                partners = [self.result_msg.get(cid) for cid, _n, _a in v.calls if cid]
                partners += [self.call_msg.get(cid) for cid, _t in v.results if cid]
                for p in partners:
                    if p is not None and self.views[p].turn < c and p not in keep:
                        keep.add(p)
                        changed = True
        return sorted(keep)


def _resolve_flavor(raw: Sequence[Any], flavor: str) -> tuple[str, list[dict[str, Any]], bool]:
    """Return (flavor, list of dict messages, input_was_Message_objects)."""
    items = list(raw)
    was_messages = bool(items) and all(isinstance(m, Message) for m in items)
    if was_messages:
        items = [m.to_dict() for m in items]
        if flavor == "auto":
            flavor = "generic"
    if flavor == "auto":
        flavor = detect_flavor(items)
    if flavor not in _VIEWERS:
        raise ValueError(
            f"running notes support flavors 'anthropic', 'openai' and 'generic' (got {flavor!r})"
        )
    for i, m in enumerate(items):
        if not isinstance(m, dict):
            raise TypeError(f"message #{i} must be a dict, got {type(m).__name__}")
    return flavor, items, was_messages


def _message_chars(flavor: str, m: dict[str, Any]) -> int:
    """Code-point length of a raw message's content (the estimate basis)."""
    content = m.get("content")
    total = 0
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for b in content:
            btype = b.get("type") if isinstance(b, dict) else None
            if flavor == "anthropic" and btype == "tool_use":
                total += len(_json(b.get("input", {})))
            elif flavor == "anthropic" and btype == "tool_result":
                total += len(_anthropic_result_text(b))
            elif btype in ("text", "input_text"):
                total += len(str(b.get("text", "")))
            elif isinstance(b, str):
                total += len(b)
            else:
                total += len(NON_TEXT)
    if flavor == "openai":
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                total += len(_openai_args((tc.get("function") or {}).get("arguments", "")))
    return total


def _system_chars(system: Any) -> int:
    if system is None:
        return 0
    return len(_content_text(system))


# ---------------------------------------------------------------------------
# Layer 1: observation masking
# ---------------------------------------------------------------------------


def mask_boundary(turn_count: int, keep_recent_turns: int, mask_chunk_turns: int) -> int:
    """Chunked masking boundary B (outputs in turns < B are masked)."""
    if turn_count - keep_recent_turns <= 0:
        return 0
    return ((turn_count - keep_recent_turns) // mask_chunk_turns) * mask_chunk_turns


def _marker(tool: str, text: str) -> tuple[str, str]:
    id8 = _sha256(text)[:8]
    return MASK_MARKER.format(tool=tool, n=len(text), id8=id8), id8


@dataclass
class _Masked:
    messages: list[dict[str, Any]]
    boundary: int
    # raw index -> list of (id8, original_text) masked in that message
    masked_at: dict[int, list[tuple[str, str]]]


def _apply_mask(
    conv: _Conversation, keep_recent_turns: int, mask_chunk_turns: int, min_mask_chars: int
) -> _Masked:
    b = mask_boundary(conv.turn_count, keep_recent_turns, mask_chunk_turns)
    out = copy.deepcopy(conv.raw)
    masked_at: dict[int, list[tuple[str, str]]] = {}
    if b <= 0:
        return _Masked(out, b, masked_at)
    for v in conv.views:
        if v.turn >= b or v.pinned or not v.results:
            continue
        m = out[v.index]
        if conv.flavor == "anthropic":
            content = m.get("content")
            if not isinstance(content, list):
                continue
            new_blocks = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    text = _anthropic_result_text(blk)
                    if len(text) >= min_mask_chars:
                        tool = conv.tool_name(blk.get("tool_use_id"), {})
                        marker, id8 = _marker(tool, text)
                        blk = dict(blk)
                        if isinstance(blk.get("content"), list):
                            blk["content"] = [{"type": "text", "text": marker}]
                        else:
                            blk["content"] = marker
                        masked_at.setdefault(v.index, []).append((id8, text))
                new_blocks.append(blk)
            m["content"] = new_blocks
        else:
            cid, text = v.results[0]
            if len(text) >= min_mask_chars:
                marker, id8 = _marker(conv.tool_name(cid, m), text)
                m["content"] = marker
                masked_at.setdefault(v.index, []).append((id8, text))
    return _Masked(out, b, masked_at)


@dataclass
class MaskReport:
    """What :func:`mask_observations` did. All token figures are chars/4 estimates."""

    turns: int
    mask_boundary: int
    masked_outputs: int
    masked_chars: int
    tokens_before_est: int
    tokens_after_est: int
    offload: dict[str, str] = field(default_factory=dict)

    @property
    def saved_tokens_est(self) -> int:
        return self.tokens_before_est - self.tokens_after_est

    def summary(self) -> str:
        pct = (
            100.0 * self.saved_tokens_est / self.tokens_before_est if self.tokens_before_est else 0
        )
        lines = [
            f"est. tokens: {self.tokens_before_est:,} → {self.tokens_after_est:,} "
            f"(−{pct:.1f}%) [chars/4]",
            f"  turns {self.turns}, mask boundary at turn {self.mask_boundary}",
            f"  masked {self.masked_outputs} tool output(s), {self.masked_chars:,} chars",
        ]
        return "\n".join(lines)

    def to_dict(self, include_offload: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "turns": self.turns,
            "mask_boundary": self.mask_boundary,
            "masked_outputs": self.masked_outputs,
            "masked_chars": self.masked_chars,
            "tokens_before_est": self.tokens_before_est,
            "tokens_after_est": self.tokens_after_est,
            "saved_tokens_est": self.saved_tokens_est,
            "offload_ids": sorted(self.offload),
        }
        if include_offload:
            d["offload"] = dict(self.offload)
        return d


def _check_mask_config(keep_recent_turns: int, mask_chunk_turns: int, min_mask_chars: int) -> None:
    if keep_recent_turns < 0:
        raise ValueError("keep_recent_turns must be >= 0")
    if mask_chunk_turns < 1:
        raise ValueError("mask_chunk_turns must be >= 1")
    if min_mask_chars < 0:
        raise ValueError("min_mask_chars must be >= 0")


def _restore_shape(out: list[dict[str, Any]], was_messages: bool) -> list[Any]:
    return from_dicts(out) if was_messages else out


def mask_observations(
    messages: Sequence[Any],
    keep_recent_turns: int = 4,
    mask_chunk_turns: int = 4,
    min_mask_chars: int = 400,
    flavor: str = "auto",
) -> tuple[list[Any], MaskReport]:
    """Layer 1: hide old tool outputs behind a deterministic marker.

    Number the turns 0..T-1; the boundary is
    ``B = floor((T - keep_recent_turns) / mask_chunk_turns) * mask_chunk_turns``
    (minimum 0). Every unpinned tool output in a turn before ``B`` that is at
    least ``min_mask_chars`` long becomes exactly
    ``[ctxfit: tool output hidden - {tool} - {n} chars - id {id8}]``.
    Tool calls, ``tool_use_id``/``tool_call_id``, ``is_error`` and every
    other key are preserved; only the output content changes.

    Returns ``(messages_in_original_shape, MaskReport)``; ``report.offload``
    maps each ``id8`` to the original text so callers can store it for recall.
    Input is never mutated.
    """
    _check_mask_config(keep_recent_turns, mask_chunk_turns, min_mask_chars)
    fl, raw, was_messages = _resolve_flavor(messages, flavor)
    conv = _Conversation(fl, raw)
    masked = _apply_mask(conv, keep_recent_turns, mask_chunk_turns, min_mask_chars)
    offload: dict[str, str] = {}
    chars = 0
    count = 0
    for entries in masked.masked_at.values():
        for id8, text in entries:
            offload[id8] = text
            chars += len(text)
            count += 1
    report = MaskReport(
        turns=conv.turn_count,
        mask_boundary=masked.boundary,
        masked_outputs=count,
        masked_chars=chars,
        tokens_before_est=estimate_tokens(sum(_message_chars(fl, m) for m in raw)),
        tokens_after_est=estimate_tokens(sum(_message_chars(fl, m) for m in masked.messages)),
        offload=offload,
    )
    return _restore_shape(masked.messages, was_messages), report


# ---------------------------------------------------------------------------
# Layer 2: running notes
# ---------------------------------------------------------------------------


def render_events(conv: _Conversation, indexes: Sequence[int]) -> str:
    """Render messages as plain text for the summariser (unscrubbed).

    Per message: ``[{role}] {text}`` for its text, then
    ``[assistant -> {tool}] {args}`` per tool call (args capped at 500
    chars), then ``[{tool} result] {text}`` per tool result (capped at
    1,500 chars). Original (unmasked) content is used.
    """
    lines: list[str] = []
    for idx in indexes:
        v = conv.views[idx]
        raw = conv.raw[idx]
        if v.text:
            lines.append(f"[{v.role}] {v.text}")
        for _cid, name, args in v.calls:
            lines.append(f"[assistant -> {name}] {args[:RENDER_ARGS_MAX_CHARS]}")
        for cid, text in v.results:
            lines.append(f"[{conv.tool_name(cid, raw)} result] {text[:RENDER_RESULT_MAX_CHARS]}")
    return "\n".join(lines)


def build_summariser_prompts(
    previous_notes: str | None, rendered: str, notes_max_tokens: int
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` exactly as sent to the summariser."""
    system = SUMMARISER_SYSTEM_PROMPT.replace("{notes_max_tokens}", str(notes_max_tokens))
    user = SUMMARISER_USER_PROMPT.replace(
        "{previous_notes}", previous_notes if previous_notes else NO_NOTES_YET
    ).replace("{rendered}", rendered)
    return system, user


SummarizerResult = Any  # str | tuple[str, int | None, int | None]
Summarizer = Callable[[str, str], SummarizerResult]


@dataclass
class NotesReport:
    """Honest accounting for one :meth:`NotesKeeper.fit` call.

    Token figures are ``ceil(chars / 4)`` estimates of input tokens not sent.
    Summariser spend is real and always subtracted in
    ``net_saved_tokens_est``; it uses provider-reported counts when the
    summariser returns them (``summariser_tokens_estimated`` is then False).
    """

    tokens_before_est: int = 0
    tokens_after_est: int = 0
    masked_outputs: int = 0
    masked_chars: int = 0
    notes_applied: bool = False
    evicted_messages: int = 0
    summariser_calls: int = 0
    summariser_input_tokens: int = 0
    summariser_output_tokens: int = 0
    summariser_tokens_estimated: bool = False
    summariser_failed: bool = False
    failure_reason: str | None = None
    cache_stable: bool | None = None
    turns: int = 0
    mask_boundary: int = 0
    cut_turn: int = 0
    previous_cut_turn: int = 0
    notes_tokens: int = 0
    state_reused: bool = False
    reasons: list[str] = field(default_factory=list)
    offload: dict[str, str] = field(default_factory=dict)

    @property
    def net_saved_tokens_est(self) -> int:
        return (
            self.tokens_before_est
            - self.tokens_after_est
            - self.summariser_input_tokens
            - self.summariser_output_tokens
        )

    def summary(self) -> str:
        b, a = self.tokens_before_est, self.tokens_after_est
        pct = 100.0 * (b - a) / b if b else 0.0
        lines = [f"est. tokens: {b:,} → {a:,} (−{pct:.1f}%) [chars/4]"]
        lines.append(
            f"  masked {self.masked_outputs} tool output(s), {self.masked_chars:,} chars "
            f"(boundary turn {self.mask_boundary} of {self.turns})"
        )
        if self.notes_applied:
            lines.append(
                f"  running notes replace {self.evicted_messages} earlier message(s) "
                f"(cut at turn {self.cut_turn}, notes ~{self.notes_tokens:,} est. tokens)"
            )
        if self.summariser_calls:
            flag = " (estimated)" if self.summariser_tokens_estimated else ""
            lines.append(
                f"  summariser: {self.summariser_calls} call, "
                f"{self.summariser_input_tokens:,} in + {self.summariser_output_tokens:,} out "
                f"tokens{flag}"
            )
        if self.summariser_failed:
            lines.append(f"  ! summariser failed ({self.failure_reason}); kept previous notes")
        lines.append(f"  net saved: {self.net_saved_tokens_est:,} est. tokens")
        cache = {True: "stable", False: "changed", None: "unknown (first request)"}
        lines.append(f"  prompt-cache prefix: {cache[self.cache_stable]}")
        if self.reasons:
            lines.append(f"  reasons: {', '.join(self.reasons)}")
        return "\n".join(lines)

    def to_dict(self, include_offload: bool = False) -> dict[str, Any]:
        d = asdict(self)
        d.pop("offload")
        d["net_saved_tokens_est"] = self.net_saved_tokens_est
        d["offload_ids"] = sorted(self.offload)
        if include_offload:
            d["offload"] = dict(self.offload)
        return d


@dataclass
class _Built:
    messages: list[dict[str, Any]]
    notes_applied: bool
    evicted: int
    kept_indexes: list[int]  # raw indexes present in the output


def _notes_text(k: int, notes: str) -> str:
    return NOTES_HEADER.format(k=k) + "\n" + notes


def _assemble(conv: _Conversation, masked: _Masked, cut: int, notes: str | None) -> _Built:
    """Output = pinned-before-cut, ONE notes message, turns >= cut (masked)."""
    if cut <= 0 or notes is None or cut not in conv.turn_start:
        return _Built(copy.deepcopy(masked.messages), False, 0, list(range(len(conv.raw))))
    before = conv.kept_before(cut)
    before_set = set(before)
    evicted = sum(1 for v in conv.views if v.turn < cut and v.index not in before_set)
    start = conv.turn_start[cut]
    after = list(range(start, len(conv.raw)))
    text = _notes_text(evicted, notes)
    out: list[dict[str, Any]] = [copy.deepcopy(masked.messages[i]) for i in before]
    first = copy.deepcopy(masked.messages[start])
    rest = [copy.deepcopy(masked.messages[i]) for i in after[1:]]
    if conv.flavor == "anthropic":
        if first.get("role") == "user":
            content = first.get("content")
            if isinstance(content, list):
                blocks = list(content)
            elif content is None or content == "":
                blocks = []
            else:
                blocks = [{"type": "text", "text": str(content)}]
            # "First text block": after any leading tool_result blocks, which
            # the Messages API requires to come first in a user message.
            pos = 0
            while (
                pos < len(blocks)
                and isinstance(blocks[pos], dict)
                and blocks[pos].get("type") == "tool_result"
            ):
                pos += 1
            first["content"] = blocks[:pos] + [{"type": "text", "text": text}] + blocks[pos:]
            out.append(first)
        else:
            out.append({"role": "user", "content": [{"type": "text", "text": text}]})
            out.append(first)
    else:
        out.append({"role": "user", "content": text})
        out.append(first)
    out.extend(rest)
    return _Built(out, True, evicted, before + after)


def _pairing_intact(conv: _Conversation, kept: Sequence[int]) -> bool:
    keep = set(kept)
    for cid, ci in conv.call_msg.items():
        ri = conv.result_msg.get(cid)
        if ri is not None and ((ci in keep) != (ri in keep)):
            return False
    return True


def _coerce_summary(result: Any) -> tuple[Any, int | None, int | None]:
    if isinstance(result, tuple):
        if len(result) != 3:
            raise TypeError("summarizer tuple must be (text, input_tokens, output_tokens)")
        return result[0], result[1], result[2]
    return result, None, None


@dataclass
class _State:
    cut_turn: int = 0
    prefix_hash: str | None = None
    notes: str | None = None
    notes_tokens: int = 0
    evicted_messages: int = 0
    summariser_calls: int = 0
    summariser_input_tokens: int = 0
    summariser_output_tokens: int = 0
    updated_at: str | None = None
    last_output_hash: str | None = None
    last_output_len: int = 0


class NotesKeeper:
    """Stateful running-notes manager (spec layer 1 + layer 2).

    Args:
        summarizer: ``(system_prompt, user_prompt) -> notes_text``, or a
            richer callable returning ``(notes_text, input_tokens,
            output_tokens)`` with provider-reported counts (``None`` for
            unknown). See :func:`anthropic_summarizer` /
            :func:`openai_summarizer`. Use a cheap model.
        keep_recent_turns / mask_chunk_turns / min_mask_chars: layer 1.
        high_water_tokens: notes engage when the prompt estimate exceeds this.
        low_water_tokens: a new cut aims to bring the estimate under this.
        notes_max_tokens: notes length target; output over twice this is
            rejected as a failure.
        min_evict_turns: a cut advances by at least this many turns.
        summarizer_timeout_s: optional wall-clock limit per summariser call
            (runs it in a worker thread; a late result is discarded).
        offload: optional ``(id8, original_text) -> None`` callback for
            every tool output masked in a request's output. Masked originals
            are also kept in :attr:`offload` (in memory) for :meth:`recall`.
        clock: optional ``() -> datetime`` for ``updated_at`` (tests).

    Call :meth:`fit` before every model request with the full, original
    history. Persist :meth:`state_dict` between processes if needed.
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        keep_recent_turns: int = 4,
        mask_chunk_turns: int = 4,
        min_mask_chars: int = 400,
        high_water_tokens: int = 60_000,
        low_water_tokens: int = 30_000,
        notes_max_tokens: int = 1_500,
        min_evict_turns: int = 4,
        summarizer_timeout_s: float | None = None,
        offload: Callable[[str, str], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(summarizer):
            raise TypeError("summarizer must be callable")
        _check_mask_config(keep_recent_turns, mask_chunk_turns, min_mask_chars)
        if low_water_tokens <= 0 or high_water_tokens <= low_water_tokens:
            raise ValueError("need 0 < low_water_tokens < high_water_tokens")
        if notes_max_tokens < 1:
            raise ValueError("notes_max_tokens must be >= 1")
        if min_evict_turns < 1:
            raise ValueError("min_evict_turns must be >= 1")
        if summarizer_timeout_s is not None and summarizer_timeout_s <= 0:
            raise ValueError("summarizer_timeout_s must be > 0")
        self.summarizer = summarizer
        self.keep_recent_turns = keep_recent_turns
        self.mask_chunk_turns = mask_chunk_turns
        self.min_mask_chars = min_mask_chars
        self.high_water_tokens = high_water_tokens
        self.low_water_tokens = low_water_tokens
        self.notes_max_tokens = notes_max_tokens
        self.min_evict_turns = min_evict_turns
        self.summarizer_timeout_s = summarizer_timeout_s
        self.offload_callback = offload
        self.offload: dict[str, str] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state = _State()

    # -- state -------------------------------------------------------------

    @property
    def notes(self) -> str | None:
        """The current notes text (``None`` before the first successful cut)."""
        return self._state.notes

    @property
    def cut_turn(self) -> int:
        return self._state.cut_turn

    def reset(self) -> None:
        """Forget notes, cut and cache bookkeeping (offload store is kept)."""
        self._state = _State()

    def recall(self, id8: str) -> str | None:
        """Original text of a masked tool output, if it was offloaded here."""
        return self.offload.get(id8)

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe state for persistence (spec field names, ``version: 1``)."""
        s = self._state
        return {
            "version": STATE_VERSION,
            "cut_turn": s.cut_turn,
            "prefix_hash": s.prefix_hash,
            "notes": s.notes,
            "notes_tokens": s.notes_tokens,
            "evicted_messages": s.evicted_messages,
            "summariser_calls": s.summariser_calls,
            "summariser_input_tokens": s.summariser_input_tokens,
            "summariser_output_tokens": s.summariser_output_tokens,
            "updated_at": s.updated_at,
            "last_output_hash": s.last_output_hash,
            "last_output_len": s.last_output_len,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output. Raises ValueError on bad input."""
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported state version {state.get('version')!r}")

        def _int(key: str) -> int:
            v = state.get(key, 0)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"state field {key!r} must be a non-negative int")
            return v

        def _opt_str(key: str) -> str | None:
            v = state.get(key)
            if v is not None and not isinstance(v, str):
                raise ValueError(f"state field {key!r} must be a string or null")
            return v

        new = _State(
            cut_turn=_int("cut_turn"),
            prefix_hash=_opt_str("prefix_hash"),
            notes=_opt_str("notes"),
            notes_tokens=_int("notes_tokens"),
            evicted_messages=_int("evicted_messages"),
            summariser_calls=_int("summariser_calls"),
            summariser_input_tokens=_int("summariser_input_tokens"),
            summariser_output_tokens=_int("summariser_output_tokens"),
            updated_at=_opt_str("updated_at"),
            last_output_hash=_opt_str("last_output_hash"),
            last_output_len=_int("last_output_len"),
        )
        if new.cut_turn > 0 and (new.prefix_hash is None or new.notes is None):
            raise ValueError("state with cut_turn > 0 needs prefix_hash and notes")
        self._state = new

    # -- core --------------------------------------------------------------

    def _prefix_hash(self, conv: _Conversation, cut: int) -> str:
        return _sha256(canonical_json([conv.raw[v.index] for v in conv.views if v.turn < cut]))

    def _est(self, flavor: str, msgs: Sequence[dict[str, Any]], system_chars: int) -> int:
        return estimate_tokens(system_chars + sum(_message_chars(flavor, m) for m in msgs))

    def _choose_cut(
        self, conv: _Conversation, masked: _Masked, prev_cut: int, system_chars: int
    ) -> tuple[int | None, bool]:
        """Return (cut, reached_low_water) or (None, False) if no boundary is allowed."""
        latest = conv.turn_count - self.keep_recent_turns
        allowed = [
            c
            for c in range(prev_cut + self.min_evict_turns, latest + 1)
            if c > prev_cut and c in conv.turn_start and conv.pair_safe_boundary(c)
        ]
        if not allowed:
            return None, False
        fl = conv.flavor
        chars = [_message_chars(fl, m) for m in masked.messages]
        for c in allowed:
            kept = conv.kept_before(c) + list(range(conv.turn_start[c], len(conv.raw)))
            est = estimate_tokens(system_chars + sum(chars[i] for i in kept))
            if est + self.notes_max_tokens <= self.low_water_tokens:
                return c, True
        return allowed[-1], False

    def _call_summarizer(self, system: str, user: str) -> tuple[Any, int | None, int | None]:
        if self.summarizer_timeout_s is None:
            return _coerce_summary(self.summarizer(system, user))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(self.summarizer, system, user)
            return _coerce_summary(fut.result(timeout=self.summarizer_timeout_s))
        finally:
            pool.shutdown(wait=False)

    def fit(
        self, raw_messages: Sequence[Any], flavor: str = "auto", system: Any = None
    ) -> tuple[list[Any], NotesReport]:
        """Apply masking + running notes to the full original history.

        Args:
            raw_messages: the ``messages`` array (Anthropic or OpenAI shape,
                generic dicts, or ctxfit ``Message`` objects).
            flavor: "auto" | "anthropic" | "openai" | "generic".
            system: Anthropic top-level ``system`` (string or blocks). Counted
                in the estimates as pinned content; never altered or returned.

        Returns ``(messages_in_original_shape, NotesReport)``. Never raises
        because of the summariser; input is never mutated.
        """
        fl, raw, was_messages = _resolve_flavor(raw_messages, flavor)
        raw = copy.deepcopy(raw)
        conv = _Conversation(fl, raw)
        masked = _apply_mask(
            conv, self.keep_recent_turns, self.mask_chunk_turns, self.min_mask_chars
        )
        sys_chars = _system_chars(system)
        report = NotesReport(
            turns=conv.turn_count,
            mask_boundary=masked.boundary,
            tokens_before_est=self._est(fl, raw, sys_chars),
        )

        # 1. Does the stored state still apply to this history?
        st = self._state
        prev_cut, notes = 0, None
        if st.cut_turn > 0:
            if (
                st.cut_turn < conv.turn_count
                and st.cut_turn in conv.turn_start
                and self._prefix_hash(conv, st.cut_turn) == st.prefix_hash
            ):
                prev_cut, notes = st.cut_turn, st.notes
                report.state_reused = True
                report.reasons.append("state_reused")
            else:
                report.reasons.append("history_forked")
                last_hash, last_len = st.last_output_hash, st.last_output_len
                self._state = st = _State(last_output_hash=last_hash, last_output_len=last_len)
        report.previous_cut_turn = prev_cut

        built = _assemble(conv, masked, prev_cut, notes)
        current_est = self._est(fl, built.messages, sys_chars)

        # 2. Over the high-water mark: move the cut (one summariser call).
        if current_est <= self.high_water_tokens:
            report.reasons.append("below_high_water")
        else:
            cut, reached = self._choose_cut(conv, masked, prev_cut, sys_chars)
            if cut is None:
                report.reasons.append("no_cut_allowed")
            else:
                new_notes = self._summarise(conv, prev_cut, cut, notes, report)
                if new_notes is not None:
                    candidate = _assemble(conv, masked, cut, new_notes)
                    if _pairing_intact(conv, candidate.kept_indexes):
                        built, notes = candidate, new_notes
                        report.reasons.append("cut_advanced")
                        if not reached:
                            report.reasons.append("low_water_not_reached")
                        st.cut_turn = cut
                        st.prefix_hash = self._prefix_hash(conv, cut)
                        st.notes = new_notes
                        st.notes_tokens = estimate_tokens(len(new_notes))
                        st.evicted_messages = candidate.evicted
                        st.updated_at = self._clock().isoformat()
                    else:  # pragma: no cover - guarded by pair_safe_boundary
                        report.reasons.append("pairing_guard")

        # 3. Final safety assertion: never ship a broken pair.
        if not _pairing_intact(conv, built.kept_indexes):  # pragma: no cover
            built = _assemble(conv, masked, 0, None)
            report.reasons.append("pairing_guard")

        out = built.messages
        report.notes_applied = built.notes_applied
        report.evicted_messages = built.evicted
        report.cut_turn = self._state.cut_turn if built.notes_applied else 0
        report.notes_tokens = estimate_tokens(len(notes)) if built.notes_applied and notes else 0
        report.tokens_after_est = self._est(fl, out, sys_chars)
        kept = set(built.kept_indexes)
        for idx, entries in masked.masked_at.items():
            if idx not in kept:
                continue
            for id8, text in entries:
                report.masked_outputs += 1
                report.masked_chars += len(text)
                report.offload[id8] = text
                if id8 not in self.offload:
                    self.offload[id8] = text
                    if self.offload_callback is not None:
                        try:
                            self.offload_callback(id8, text)
                        except Exception as exc:  # never fail the request
                            report.reasons.append(f"offload_callback_failed:{type(exc).__name__}")
        if report.masked_outputs:
            report.reasons.insert(0, "masked")

        # 4. Cache stability: is the previous output a prefix of this one?
        st = self._state
        if st.last_output_hash is not None and st.last_output_len > 0:
            n = st.last_output_len
            report.cache_stable = (
                n <= len(out) and _sha256(canonical_json(out[:n])) == st.last_output_hash
            )
        st.last_output_hash = _sha256(canonical_json(out))
        st.last_output_len = len(out)
        return _restore_shape(out, was_messages), report

    def _summarise(
        self,
        conv: _Conversation,
        prev_cut: int,
        cut: int,
        previous_notes: str | None,
        report: NotesReport,
    ) -> str | None:
        """One summariser call for turns [prev_cut, cut). None on failure."""
        kept_new = set(conv.kept_before(cut))
        evicted = [
            v.index for v in conv.views if prev_cut <= v.turn < cut and v.index not in kept_new
        ]
        rendered = scrub_secrets(render_events(conv, evicted))
        prev = scrub_secrets(previous_notes) if previous_notes else None
        system, user = build_summariser_prompts(prev, rendered, self.notes_max_tokens)
        in_est = estimate_tokens(len(system) + len(user))
        report.summariser_calls = 1
        st = self._state
        st.summariser_calls += 1

        def spend(inp: int, outp: int, estimated: bool) -> None:
            report.summariser_input_tokens += inp
            report.summariser_output_tokens += outp
            report.summariser_tokens_estimated = report.summariser_tokens_estimated or estimated
            st.summariser_input_tokens += inp
            st.summariser_output_tokens += outp

        def fail(reason: str) -> None:
            report.summariser_failed = True
            report.failure_reason = reason
            report.reasons.append("summariser_failed")

        try:
            text, tin, tout = self._call_summarizer(system, user)
        except (TimeoutError, concurrent.futures.TimeoutError):
            # The call most likely ran and was billed: count the input.
            spend(in_est, 0, True)
            fail("timeout")
            return None
        except Exception as exc:  # never fail the user's request
            fail(f"exception: {type(exc).__name__}: {exc}"[:300])
            return None
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        out_est = estimate_tokens(len(text))
        spend(
            tin if isinstance(tin, int) and tin >= 0 else in_est,
            tout if isinstance(tout, int) and tout >= 0 else out_est,
            not (isinstance(tin, int) and isinstance(tout, int)),
        )
        if not text.strip():
            fail("empty")
            return None
        if out_est > 2 * self.notes_max_tokens:
            fail("oversized")
            return None
        return text


# ---------------------------------------------------------------------------
# Summariser factories (no SDK imports; use the client objects passed in)
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def anthropic_summarizer(
    client: Any, model: str = "claude-haiku-5-5", max_tokens: int = 3_000
) -> Callable[[str, str], tuple[str, int | None, int | None]]:
    """Summariser backed by an ``anthropic.Anthropic`` client you pass in.

    Returns provider-reported token usage so the report's net savings are
    exact. ``max_tokens`` defaults to 2x the default ``notes_max_tokens``.
    """

    def summarise(system: str, user: str) -> tuple[str, int | None, int | None]:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            str(_get(b, "text", ""))
            for b in (_get(resp, "content") or [])
            if _get(b, "type") == "text"
        )
        usage = _get(resp, "usage")
        return text, _get(usage, "input_tokens"), _get(usage, "output_tokens")

    return summarise


def openai_summarizer(
    client: Any, model: str, max_completion_tokens: int | None = None
) -> Callable[[str, str], tuple[str, int | None, int | None]]:
    """Summariser backed by an ``openai.OpenAI``-compatible client you pass in.

    ``model`` is required (pick your provider's cheapest capable model).
    Works with OpenAI-compatible endpoints that implement chat.completions.
    """

    def summarise(system: str, user: str) -> tuple[str, int | None, int | None]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        resp = client.chat.completions.create(**kwargs)
        choices = _get(resp, "choices") or []
        msg = _get(choices[0], "message") if choices else None
        text = _get(msg, "content") or ""
        usage = _get(resp, "usage")
        return str(text), _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")

    return summarise


__all__ = [
    "MASK_MARKER",
    "NOTES_HEADER",
    "SUMMARISER_SYSTEM_PROMPT",
    "SUMMARISER_USER_PROMPT",
    "MaskReport",
    "NotesKeeper",
    "NotesReport",
    "anthropic_summarizer",
    "build_summariser_prompts",
    "canonical_json",
    "estimate_tokens",
    "mask_boundary",
    "mask_observations",
    "openai_summarizer",
    "scrub_secrets",
]
