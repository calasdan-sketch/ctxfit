"""Pluggable, pure-function compression strategies.

Each strategy is ``(messages, ctx) -> (messages, StrategyReport)``.
Strategies never mutate their input list in place; they return a new list.

Ordering (cheapest/safest first) is the pipeline's job; each strategy
declares which budget sections it relieves and only engages when one of
them overflows — unless constructed with ``when="always"``.

Hard safety rules enforced here, not left to callers:
- tool_call content is NEVER modified (API-validity invariant).
- tool_call/tool_result pairs are atomic: kept, truncated, or dropped together.
- pinned messages are never dropped; truncated only with explicit consent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .budgets import Budget, BudgetLedger
from .estimate import TokenEstimator
from .messages import (
    KIND_TEXT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Message,
    is_opaque_message,
    turn_indexes,
)

_SEVERITY_RE = re.compile(
    r"\b(fatal|error|errors|exception|traceback|panic|critical|fail|failed|failure|denied|"
    r"refused|timed out|timeout|segfault|aborted|cannot|unable|invalid|unhandled)\b",
    re.IGNORECASE,
)
MARKER_PREFIX = "[ctxfit"


def _marker(text: str) -> str:
    return f"{MARKER_PREFIX}: {text}]"


@dataclass
class StrategyContext:
    estimator: TokenEstimator
    budget: Budget
    ledger: BudgetLedger
    aggressive: bool = False
    # Optional mapping that receives the full text of every tool output a strategy cuts,
    # keyed by output_id(text); the cut marker then names that id (see ``fit(offload=...)``).
    offload: MutableMapping[str, str] | None = None


def output_id(text: str) -> str:
    """The 8-hex id of a tool output (first 8 hex of its sha256), the same id running notes use."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:8]


@dataclass
class StrategyReport:
    strategy: str
    saved_tokens_est: int = 0
    actions: list[str] = field(default_factory=list)
    messages_affected: int = 0

    def log(self, action: str, saved: int = 0) -> None:
        self.actions.append(action)
        self.saved_tokens_est += saved
        self.messages_affected += 1


class Strategy:
    """Base class. Subclass and override :meth:`_apply` for custom strategies.

    Contract for custom strategies:
    - Pure function: never mutate the input list or the Message objects in
      it — build and return a new list (copy-and-modify Messages you change).
    - Preserve relative message order (adapters rejoin split messages by
      position; reordering can break framework shapes).
    - Never modify ``kind="tool_call"`` content and never orphan one half
      of a tool_call/tool_result pair — the pipeline raises PairingBroken
      if you do.
    - Never drop or truncate pinned messages without explicit opt-in.
    """

    name = "base"
    # Sections whose overflow this strategy relieves.
    target_sections: tuple[str, ...] = ("history",)
    # "over_budget": engage only when a target section overflows.
    # "always": engage regardless (hygiene mode).
    when: str = "over_budget"

    def __init__(self, when: str = "over_budget") -> None:
        if when not in ("over_budget", "always"):
            raise ValueError("when must be 'over_budget' or 'always'")
        self.when = when

    def engages(self, ctx: StrategyContext) -> bool:
        if self.when == "always":
            return True
        over = {w.section for w in ctx.ledger.overflows()}
        return bool(over & set(self.target_sections))

    def __call__(
        self, messages: list[Message], ctx: StrategyContext
    ) -> tuple[list[Message], StrategyReport]:
        report = StrategyReport(strategy=self.name)
        if not self.engages(ctx):
            return list(messages), report
        return self._apply(list(messages), ctx, report)

    def _apply(
        self, messages: list[Message], ctx: StrategyContext, report: StrategyReport
    ) -> tuple[list[Message], StrategyReport]:
        raise NotImplementedError


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _call_index(messages: list[Message]) -> dict[str, list[int]]:
    by_call: dict[str, list[int]] = {}
    for j, m in enumerate(messages):
        if m.call_id and m.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
            by_call.setdefault(m.call_id, []).append(j)
    return by_call


def _pair_closure(
    messages: list[Message], drop: set, by_call: dict[str, list[int]] | None = None
) -> set:
    """Expand a drop-set so tool_call/tool_result pairs drop atomically.

    Follows call_id links to a fixpoint: if any message in the drop set is
    half of a pair, its partner joins the set no matter which section or
    turn it lives in. Callers must check the expanded set for pinned
    messages *before* dropping — a pinned partner vetoes the whole drop.
    """
    if by_call is None:
        by_call = _call_index(messages)
    expanded = set(drop)
    changed = True
    while changed:
        changed = False
        for j in list(expanded):
            m = messages[j]
            if m.call_id and m.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
                for p in by_call.get(m.call_id, []):
                    if p not in expanded:
                        expanded.add(p)
                        changed = True
    return expanded


class Deduplicate(Strategy):
    """Replace exact-duplicate message bodies with a pointer to the first.

    Only ``kind in (text, tool_result)``, only above ``min_chars`` (short
    repeats cost more as markers than they save), never pinned, never
    tool_call args.
    """

    name = "deduplicate"
    target_sections = ("retrieved", "history")

    def __init__(self, min_chars: int = 120, **kw: Any) -> None:
        super().__init__(**kw)
        self.min_chars = min_chars

    def _apply(self, messages, ctx, report):
        seen: dict[str, int] = {}
        out: list[Message] = []
        for i, m in enumerate(messages):
            if (
                m.kind in (KIND_TEXT, KIND_TOOL_RESULT)
                and not m.pinned
                and not is_opaque_message(m)
                and len(m.content) >= self.min_chars
            ):
                key = f"{m.kind}:{_norm_text(m.content)}"
                if key in seen:
                    marker = _marker(
                        f"duplicate of an earlier identical message — "
                        f"{len(m.content):,} chars omitted"
                    )
                    saved = ctx.estimator.count(m.content) - ctx.estimator.count(marker)
                    out.append(
                        Message(
                            role=m.role,
                            content=marker,
                            kind=m.kind,
                            name=m.name,
                            call_id=m.call_id,
                            section=m.section,
                            meta={**m.meta, "ctxfit_dedup_of": seen[key]},
                        )
                    )
                    report.log(f"deduped message #{i} (identical to #{seen[key]})", saved)
                    continue
                seen[key] = i
            out.append(m)
        return out, report


class _Lit(float):
    """A JSON float that remembers how it was written (``0.10`` stays ``0.10``, ``1e400`` stays)."""

    text: str

    def __new__(cls, text: str) -> _Lit:
        obj = float.__new__(cls, text)
        obj.text = text
        return obj


class _DuplicateKey(ValueError):
    pass


def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise _DuplicateKey(k)
        d[k] = v
    return d


def _loads_exact(text: str) -> Any:
    """json.loads that keeps number spelling and refuses duplicate keys (they would lose data)."""
    return json.loads(text, parse_float=_Lit, object_pairs_hook=_no_dup_pairs)


def _jd(v: Any) -> str:
    """Compact JSON text of a parsed value, with numbers as originally written."""
    if isinstance(v, _Lit):
        return v.text
    if isinstance(v, dict):
        return (
            "{"
            + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_jd(x)}" for k, x in v.items())
            + "}"
        )
    if isinstance(v, list):
        return "[" + ", ".join(_jd(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def _one_line(s: str) -> str:
    """No line break survives in a table cell or flat value (a value cannot forge a new row)."""
    return re.sub(r"[\r\n\u2028\u2029\x0b\x0c\x85]+", " ", s)


def _outlier_rows(data: list[dict[str, Any]], keys: list[str], skip: int, limit: int) -> list[int]:
    """Indexes (>= ``skip``) of rows that differ from an otherwise uniform column.

    A column counts when one value fills at least 60% of the rows and the
    column has few distinct values (<= max(12, n/10)); a row whose value
    there occurs at most twice is an outlier. Rows with a severity word
    (error, failed, ...) in any string cell also count. Deterministic.
    """
    n = len(data)
    if limit <= 0 or n <= skip:
        return []
    flags: dict[int, int] = {}
    for k in keys:
        col = [_jd(r[k]) for r in data]
        freq: dict[str, int] = {}
        for v in col:
            freq[v] = freq.get(v, 0) + 1
        if len(freq) > max(12, n // 10) or max(freq.values()) < 0.6 * n:
            continue
        for i in range(skip, n):
            if freq[col[i]] <= 2:
                flags[i] = flags.get(i, 0) + 2
    for i in range(skip, n):
        for k in keys:
            v = data[i][k]
            if isinstance(v, str) and _SEVERITY_RE.search(v):
                flags[i] = flags.get(i, 0) + 1
                break
    ranked = sorted(flags, key=lambda i: (-flags[i], i))[:limit]
    return sorted(ranked)


def _json_table(data: list[dict[str, Any]], max_rows: int, max_outliers: int = 10) -> str | None:
    if not data or not all(isinstance(r, dict) for r in data):
        return None
    keys: list[str] = list(data[0].keys())
    if any(set(r.keys()) != set(keys) for r in data):
        return None  # not uniform — leave it alone
    if not keys:
        return None

    def cell(v: Any) -> str:
        s = v if isinstance(v, str) else _jd(v)
        s = _one_line(str(s).replace("|", "\\|"))
        return s if len(s) <= 120 else s[:117] + "…"

    def head(k: str) -> str:
        return _one_line(k.replace("|", "\\|"))

    lines = [
        "| " + " | ".join(head(k) for k in keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    rows = data[:max_rows]
    for r in rows:
        lines.append("| " + " | ".join(cell(r[k]) for k in keys) + " |")
    omitted = len(data) - len(rows)
    if omitted:
        lines.append(f"| … {omitted} more rows omitted |")
        picked = _outlier_rows(data, keys, len(rows), max_outliers)
        if picked:
            lines.append(
                f"[ctxfit: {len(picked)} outlier row(s) from the omitted rows; (row n) is the 0-based index]"
            )
            for i in picked:
                lines.append("| " + " | ".join(cell(data[i][k]) for k in keys) + f" | (row {i})")
    return "\n".join(lines)


def _scalar(v: Any) -> str:
    """A leaf as JSON spells it (``true``/``null``, not Python's ``True``/``None``); strings bare."""
    return v if isinstance(v, str) else _jd(v)


def _json_flat(data: dict[str, Any], max_keys: int, max_depth: int) -> str | None:
    lines: list[str] = []

    def walk(obj: Any, path: str, depth: int) -> None:
        if len(lines) >= max_keys or depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(obj, list) and obj and all(not isinstance(x, (dict, list)) for x in obj):
            lines.append(
                f"- {_one_line(path)}: [{', '.join(_one_line(_scalar(x)) for x in obj[:10])}]"
            )
        elif isinstance(obj, (dict, list)):
            lines.append(f"- {_one_line(path)}: {_jd(obj)[:200]}")
        else:
            lines.append(f"- {_one_line(path)}: {_one_line(_scalar(obj))}")

    walk(data, "", 0)
    if not lines:
        return None
    out = lines[:max_keys]
    if len(lines) > max_keys:
        out.append(f"- … {len(lines) - max_keys} more keys omitted")
    return "\n".join(out)


class CompactJson(Strategy):
    """JSON tool outputs / text → compact markdown tables or flat key lists.

    Savings-gated: if the compacted form isn't smaller, the JSON is left
    untouched. Never touches tool_call content (must stay valid JSON).
    Only whole-message JSON in v1 (no substring extraction).
    """

    name = "compact_json"
    target_sections = ("retrieved", "history")

    def __init__(
        self,
        max_rows: int = 50,
        max_keys: int = 120,
        max_depth: int = 6,
        max_outliers: int = 10,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.max_outliers = max_outliers
        self.max_rows = max_rows
        self.max_keys = max_keys
        self.max_depth = max_depth

    def _apply(self, messages, ctx, report):
        out: list[Message] = []
        for i, m in enumerate(messages):
            if m.kind not in (KIND_TEXT, KIND_TOOL_RESULT) or m.pinned or is_opaque_message(m):
                out.append(m)
                continue
            compacted: str | None = None
            try:
                data = _loads_exact(m.content.strip())
                if isinstance(data, list):
                    compacted = _json_table(data, self.max_rows, self.max_outliers)
                elif isinstance(data, dict):
                    compacted = _json_flat(data, self.max_keys, self.max_depth)
            except (ValueError, RecursionError):  # not JSON, duplicate keys, or nested too deep
                out.append(m)
                continue
            if not compacted:
                out.append(m)
                continue
            # No content marker: the markdown table is self-evident and the
            # action is logged in the strategy report. (A marker line would
            # eat into the savings on small payloads.)
            new_content = compacted
            before = ctx.estimator.count(m.content)
            after = ctx.estimator.count(new_content)
            if after >= before:
                out.append(m)  # no savings — leave it alone
                continue
            out.append(
                Message(
                    role=m.role,
                    content=new_content,
                    kind=m.kind,
                    name=m.name,
                    call_id=m.call_id,
                    section=m.section,
                    meta={**m.meta, "ctxfit_compacted": True},
                )
            )
            report.log(
                f"compacted JSON in message #{i}: ~{before:,} → ~{after:,} est. tokens",
                before - after,
            )
        return out, report


_TEMPLATE_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|[0-9a-fA-F]{8,}|\d+")
_WS_RE = re.compile(r"\s+")
_LINE_CAP = 300


def _line_template(line: str) -> str:
    """The line with numbers/hex ids blanked, so 'same kind of line' compares equal."""
    return _WS_RE.sub(" ", _TEMPLATE_NUM_RE.sub("#", line.strip()))


def salient_lines(middle: str, budget_chars: int, max_lines: int = 20) -> list[tuple[int, str]]:
    """Pick the lines of ``middle`` a reader would regret losing.

    Deterministic, no model. Two signals: severity words (error, fatal,
    traceback, ...), and rarity: when most lines are repeats of a few
    templates (logs, tables, generated code), the lines whose template
    occurs once or twice are the odd ones out. Repeated severity lines are
    collapsed to the first one plus a count. The first and last "line" of
    ``middle`` are skipped (they are cut mid-line). Returns
    ``[(index_in_middle, text)]`` in original order within ``budget_chars``.
    """
    lines = middle.split("\n")
    if len(lines) < 3:
        return []
    seen_tpl: dict[str, str] = {}  # repeated lines (logs) are templated once
    templates = []
    for ln in lines:
        t = seen_tpl.get(ln)
        if t is None:
            t = seen_tpl[ln] = _line_template(ln)
        templates.append(t)
    counts: dict[str, int] = {}
    for t in templates:
        if t:
            counts[t] = counts.get(t, 0) + 1
    nonblank = sum(counts.values())
    boiler = sum(c for c in counts.values() if c >= 3)
    rarity_informative = nonblank > 0 and boiler / nonblank >= 0.3
    first_seen: dict[str, int] = {}
    scored: list[tuple[int, int, str]] = []  # (-score, index, text)
    for i in range(1, len(lines) - 1):
        t = templates[i]
        if not t:
            continue
        severe = bool(_SEVERITY_RE.search(lines[i]))
        rare = rarity_informative and counts[t] <= 2
        if not (severe or rare):
            continue
        if severe and counts[t] >= 3:
            if t in first_seen:
                continue  # repeated severity line: only the first is kept
            first_seen[t] = i
        text = lines[i].strip()
        if len(text) > _LINE_CAP:
            text = text[:_LINE_CAP] + "…"
        if severe and counts[t] >= 3:
            text += f"  [x{counts[t]} similar]"
        scored.append((-(2 * severe + rare), i, text))
    scored.sort()
    chosen: list[tuple[int, str]] = []
    used = 0
    for _neg, i, text in scored:
        if len(chosen) >= max_lines or used + len(text) + 8 > budget_chars:
            continue
        chosen.append((i, text))
        used += len(text) + 8
    chosen.sort()
    return chosen


class TruncateToolOutput(Strategy):
    """Head+tail truncation for oversized tool results.

    Heads carry schema/headers, tails carry the errors and final results —
    both are kept, the middle is replaced with one explicit marker line.
    Pair-preserving (truncation never orphans a pair). Pinned messages are
    skipped unless ``truncate_pinned=True``.

    With ``salient=True`` (default, unreleased 0.8.0 candidate) the omitted middle is not
    thrown away blindly: up to ``salient_chars`` of its error lines and
    odd-one-out lines are kept, with their original line numbers (see
    :func:`salient_lines`). ``salient=False`` gives the old head+tail only.
    """

    name = "truncate_tool_output"
    target_sections = ("retrieved",)

    def __init__(
        self,
        max_chars: int = 4000,
        head_chars: int = 1200,
        tail_chars: int = 1200,
        truncate_pinned: bool = False,
        include_retrieved_text: bool = False,
        salient: bool = True,
        salient_chars: int = 1200,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        if head_chars + tail_chars >= max_chars:
            raise ValueError("head_chars + tail_chars must be < max_chars")
        if salient_chars < 0:
            raise ValueError("salient_chars must be >= 0")
        self.salient = salient
        self.salient_chars = salient_chars
        self.max_chars = max_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.truncate_pinned = truncate_pinned
        self.include_retrieved_text = include_retrieved_text

    def _eligible(self, m: Message) -> bool:
        if m.kind == KIND_TOOL_RESULT:
            return True
        return self.include_retrieved_text and m.kind == KIND_TEXT and m.section == "retrieved"

    def _apply(self, messages, ctx, report):
        out: list[Message] = []
        for i, m in enumerate(messages):
            if (
                not self._eligible(m)
                or (m.pinned and not self.truncate_pinned)
                or is_opaque_message(m)
                or len(m.content) <= self.max_chars
            ):
                out.append(m)
                continue
            head = m.content[: self.head_chars]
            tail = m.content[-self.tail_chars :]
            omitted = len(m.content) - self.head_chars - self.tail_chars
            omitted_tokens = ctx.estimator.count(
                m.content[self.head_chars : len(m.content) - self.tail_chars]
            )
            marker = _marker(
                f"truncated — kept {self.head_chars:,} head + {self.tail_chars:,} tail chars, "
                f"omitted {omitted:,} chars (~{omitted_tokens:,} est. tokens)"
            )
            oid = ""
            if ctx.offload is not None:
                oid = output_id(m.content)
                ctx.offload[oid] = m.content
                marker = marker[:-1] + f" — full text id {oid}]"
            new_content = head + "\n" + marker + "\n" + tail
            if self.salient and self.salient_chars:
                middle = m.content[self.head_chars : len(m.content) - self.tail_chars]
                picked = salient_lines(middle, self.salient_chars)
                if picked:
                    base = m.content[: self.head_chars].count("\n") + 1
                    body = "\n".join(f"L{base + i}: {text}" for i, text in picked)
                    note = _marker(
                        f"kept {len(picked)} salient line(s) from the omitted middle "
                        "(errors / odd-one-out lines); L<n> = original line number"
                    )
                    candidate = head + "\n" + marker + "\n" + note + "\n" + body + "\n" + tail
                    if len(candidate) < len(m.content):
                        new_content = candidate
            saved = ctx.estimator.count(m.content) - ctx.estimator.count(new_content)
            out.append(
                Message(
                    role=m.role,
                    content=new_content,
                    kind=m.kind,
                    name=m.name,
                    call_id=m.call_id,
                    section=m.section,
                    meta={**m.meta, "ctxfit_truncated": True},
                )
            )
            report.log(
                f"truncated message #{i}: {len(m.content):,} → {len(new_content):,} chars", saved
            )
        return out, report


class SlidingWindow(Strategy):
    """Drop oldest turns while the history section overflows.

    Whole-turn granularity keeps tool pairs intact. Never drops pinned
    messages (a turn containing any pinned message is skipped, not split),
    never drops the latest turn, and keeps at least ``min_turns`` turns.
    With ``when="always"`` it additionally caps history at ``keep_turns``.
    """

    name = "sliding_window"
    target_sections = ("history",)

    def __init__(self, keep_turns: int = 10, min_turns: int = 3, **kw: Any) -> None:
        super().__init__(**kw)
        if min_turns < 1:
            raise ValueError("min_turns must be >= 1")
        self.keep_turns = keep_turns
        self.min_turns = min_turns

    def _apply(self, messages, ctx, report):
        turns = turn_indexes(messages)
        if not turns:
            return messages, report
        latest = max(turns)
        # Group message indexes by turn (history-section, non-system only).
        by_turn: dict[int, list[int]] = {}
        for idx, (m, t) in enumerate(zip(messages, turns)):
            if t >= 0 and m.section == "history":
                by_turn.setdefault(t, []).append(idx)
        ordered = sorted(by_turn)
        if len(ordered) <= self.min_turns:
            return messages, report

        allowance = ctx.budget.allowance("history")

        # Count each message once; keep a running history total instead of
        # re-counting the whole conversation per candidate turn (was O(n^2)).
        call_index = _call_index(messages)
        counts = [ctx.estimator.count_message(m) for m in messages]
        history_total = sum(c for m, c in zip(messages, counts) if m.section == "history")

        drop: set[int] = set()
        dropped_turns = 0
        for t in ordered:
            if t == latest:
                continue  # never drop the latest turn
            if len(ordered) - dropped_turns <= self.min_turns:
                break
            idxs = by_turn[t]
            if any(messages[j].pinned for j in idxs):
                report.actions.append(f"sliding_window: kept turn {t} (contains pinned message)")
                continue
            # Pair closure: a turn's tool_call may pair with a tool_result
            # living in another section (retrieved). They drop together or
            # not at all — a pinned partner vetoes the whole turn.
            candidate = _pair_closure(messages, set(idxs), call_index)
            if any(messages[j].pinned for j in candidate):
                report.actions.append(f"sliding_window: kept turn {t} (pair partner pinned)")
                continue
            need = (history_total > allowance) or (
                self.when == "always" and len(ordered) - dropped_turns > self.keep_turns
            )
            if not need:
                break
            for j in candidate - drop:
                if messages[j].section == "history":
                    history_total -= counts[j]
            drop.update(candidate)
            dropped_turns += 1

        if drop:
            saved = sum(counts[j] for j in drop)
            kept = [m for j, m in enumerate(messages) if j not in drop]
            report.saved_tokens_est = saved
            report.messages_affected = len(drop)
            report.actions.append(
                f"sliding_window: dropped {len(drop)} messages ({dropped_turns} oldest turns)"
            )
            return kept, report
        return messages, report


class DropOldestUnpinned(Strategy):
    """Last-resort strategy (aggressive mode): drop oldest unpinned history
    messages one at a time until the total budget fits.

    Pair-safe: a tool_call/tool_result is only dropped together with its
    partner, and never if the partner is pinned. Never touches the latest
    turn or system messages.
    """

    name = "drop_oldest_unpinned"
    target_sections = ("history", "retrieved", "system", "scratchpad")

    def _apply(self, messages, ctx, report):
        # Single left-to-right scan; indexes are rebuilt after every
        # mutation so pair bookkeeping can never go stale.
        return self._scan(messages, ctx, report)

    def _scan(self, messages, ctx, report):
        turns = turn_indexes(messages)
        latest = max(turns) if turns else -1
        by_call: dict[str, list[int]] = {}
        for j, m in enumerate(messages):
            if m.call_id and m.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
                by_call.setdefault(m.call_id, []).append(j)

        # Running total: a message's count never depends on its position.
        running = sum(ctx.estimator.count_message(m) for m in messages)

        def total_over() -> bool:
            return running > ctx.budget.total

        idx = 0
        while idx < len(messages) and total_over():
            m, t = messages[idx], turns[idx]
            if m.pinned or m.section == "system" or t == latest:
                idx += 1
                continue
            if m.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT) and m.call_id:
                partners = [p for p in by_call.get(m.call_id, []) if p != idx]
                if any(messages[p].pinned for p in partners):
                    idx += 1
                    continue
                doomed = [idx] + partners
                saved = sum(ctx.estimator.count_message(messages[p]) for p in doomed)
                report.log(f"drop_oldest_unpinned: dropped pair at #{idx}", saved)
            else:
                doomed = [idx]
                report.log(
                    f"drop_oldest_unpinned: dropped message #{idx} ({m.role}/{m.kind})",
                    ctx.estimator.count_message(m),
                )
            doomed_set = set(doomed)
            running -= sum(ctx.estimator.count_message(messages[k]) for k in doomed_set)
            messages = [mm for k, mm in enumerate(messages) if k not in doomed_set]
            # rebuild bookkeeping after mutation
            turns = turn_indexes(messages)
            latest = max(turns) if turns else -1
            by_call = {}
            for k, mm in enumerate(messages):
                if mm.call_id and mm.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
                    by_call.setdefault(mm.call_id, []).append(k)
            # do not increment idx: next oldest slid into this position
        return messages, report


DEFAULT_STRATEGIES: Sequence[type] = (
    Deduplicate,
    CompactJson,
    TruncateToolOutput,
    SlidingWindow,
)

AGGRESSIVE_STRATEGIES: Sequence[type] = DEFAULT_STRATEGIES + (DropOldestUnpinned,)
