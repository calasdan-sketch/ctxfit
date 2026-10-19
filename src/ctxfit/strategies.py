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

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .budgets import Budget, BudgetLedger
from .estimate import TokenEstimator, count_by_section
from .messages import (
    KIND_TEXT,
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Message,
    turn_indexes,
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


def _pair_closure(messages: list[Message], drop: set) -> set:
    """Expand a drop-set so tool_call/tool_result pairs drop atomically.

    Follows call_id links to a fixpoint: if any message in the drop set is
    half of a pair, its partner joins the set no matter which section or
    turn it lives in. Callers must check the expanded set for pinned
    messages *before* dropping — a pinned partner vetoes the whole drop.
    """
    by_call: dict[str, list[int]] = {}
    for j, m in enumerate(messages):
        if m.call_id and m.kind in (KIND_TOOL_CALL, KIND_TOOL_RESULT):
            by_call.setdefault(m.call_id, []).append(j)
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


def _json_table(data: list[dict[str, Any]], max_rows: int) -> str | None:
    if not data or not all(isinstance(r, dict) for r in data):
        return None
    keys: list[str] = list(data[0].keys())
    if any(set(r.keys()) != set(keys) for r in data):
        return None  # not uniform — leave it alone
    if not keys:
        return None

    def cell(v: Any) -> str:
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        s = str(s).replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= 120 else s[:117] + "…"

    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    rows = data[:max_rows]
    for r in rows:
        lines.append("| " + " | ".join(cell(r[k]) for k in keys) + " |")
    omitted = len(data) - len(rows)
    if omitted:
        lines.append(f"| … {omitted} more rows omitted |")
    return "\n".join(lines)


def _json_flat(data: dict[str, Any], max_keys: int, max_depth: int) -> str | None:
    lines: list[str] = []

    def walk(obj: Any, path: str, depth: int) -> None:
        if len(lines) >= max_keys or depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(obj, list) and obj and all(not isinstance(x, (dict, list)) for x in obj):
            lines.append(f"- {path}: [{', '.join(str(x) for x in obj[:10])}]")
        elif isinstance(obj, (dict, list)):
            lines.append(f"- {path}: {json.dumps(obj, ensure_ascii=False)[:200]}")
        else:
            lines.append(f"- {path}: {obj}")

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
        self, max_rows: int = 50, max_keys: int = 120, max_depth: int = 6, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.max_rows = max_rows
        self.max_keys = max_keys
        self.max_depth = max_depth

    def _apply(self, messages, ctx, report):
        out: list[Message] = []
        for i, m in enumerate(messages):
            if m.kind not in (KIND_TEXT, KIND_TOOL_RESULT) or m.pinned:
                out.append(m)
                continue
            compacted: str | None = None
            try:
                data = json.loads(m.content.strip())
            except (json.JSONDecodeError, ValueError):
                out.append(m)
                continue
            if isinstance(data, list):
                compacted = _json_table(data, self.max_rows)
            elif isinstance(data, dict):
                compacted = _json_flat(data, self.max_keys, self.max_depth)
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


class TruncateToolOutput(Strategy):
    """Head+tail truncation for oversized tool results.

    Heads carry schema/headers, tails carry the errors and final results —
    both are kept, the middle is replaced with one explicit marker line.
    Pair-preserving (truncation never orphans a pair). Pinned messages are
    skipped unless ``truncate_pinned=True``.
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
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        if head_chars + tail_chars >= max_chars:
            raise ValueError("head_chars + tail_chars must be < max_chars")
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
            new_content = head + "\n" + marker + "\n" + tail
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

        def history_over(excluded: set[int]) -> bool:
            usage = count_by_section(
                [m for j, m in enumerate(messages) if j not in excluded],
                ctx.estimator,
            )
            return usage.get("history", 0) > allowance

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
            candidate = _pair_closure(messages, set(idxs))
            if any(messages[j].pinned for j in candidate):
                report.actions.append(f"sliding_window: kept turn {t} (pair partner pinned)")
                continue
            need = history_over(drop) or (
                self.when == "always" and len(ordered) - dropped_turns > self.keep_turns
            )
            if not need:
                break
            drop.update(candidate)
            dropped_turns += 1

        if drop:
            saved = sum(ctx.estimator.count_message(messages[j]) for j in drop)
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

        def total_over() -> bool:
            usage = count_by_section(messages, ctx.estimator)
            return sum(usage.values()) > ctx.budget.total

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
