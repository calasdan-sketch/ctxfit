"""Tests for compression strategies."""

import json

import pytest

from ctxfit.budgets import Budget, BudgetLedger
from ctxfit.estimate import HeuristicEstimator, count_by_section
from ctxfit.messages import (
    KIND_TOOL_CALL,
    KIND_TOOL_RESULT,
    Message,
    validate_pairing,
)
from ctxfit.strategies import (
    CompactJson,
    Deduplicate,
    DropOldestUnpinned,
    SlidingWindow,
    StrategyContext,
    TruncateToolOutput,
)


def make_ctx(messages, total=128_000, **kw):
    est = HeuristicEstimator()
    budget = Budget(total=total)
    ledger = BudgetLedger(budget=budget, usage_est=count_by_section(messages, est))
    return StrategyContext(estimator=est, budget=budget, ledger=ledger, **kw)


def over_budget_ctx(messages, section):
    """Force a context where `section` overflows its budget."""
    est = HeuristicEstimator()
    budget = Budget(total=100)  # tiny budget -> everything overflows
    ledger = BudgetLedger(budget=budget, usage_est=count_by_section(messages, est))
    assert any(w.section == section for w in ledger.overflows())
    return StrategyContext(estimator=est, budget=budget, ledger=ledger)


# --- Deduplicate ---


def test_deduplicate_replaces_repeat():
    body = "x" * 500
    msgs = [
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c2"),
    ]
    out, report = Deduplicate()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert len(out) == 2
    assert "duplicate of an earlier identical message" in out[1].content
    assert out[0].content == body  # first kept intact
    assert report.messages_affected == 1
    assert report.saved_tokens_est > 0


def test_deduplicate_ignores_short_messages():
    msgs = [Message(role="user", content="ok"), Message(role="user", content="ok")]
    ctx = make_ctx(msgs)
    out, report = Deduplicate()(msgs, ctx)
    assert out[1].content == "ok" and report.messages_affected == 0


def test_deduplicate_skips_pinned_and_tool_calls():
    body = "y" * 500
    msgs = [
        Message(role="system", content=body, pinned=True),
        Message(role="system", content=body, pinned=True),
        Message(role="assistant", content=body, kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="assistant", content=body, kind=KIND_TOOL_CALL, call_id="c2"),
    ]
    out, report = Deduplicate()(msgs, over_budget_ctx(msgs, "system"))
    assert all(m.content == body for m in out)
    assert report.messages_affected == 0


def test_deduplicate_noop_when_under_budget():
    body = "z" * 500
    msgs = [
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c2"),
    ]
    out, report = Deduplicate()(msgs, make_ctx(msgs))  # huge budget, no overflow
    assert out[1].content == body and report.messages_affected == 0


def test_deduplicate_always_mode():
    body = "w" * 500
    msgs = [
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content=body, kind=KIND_TOOL_RESULT, call_id="c2"),
    ]
    out, report = Deduplicate(when="always")(msgs, make_ctx(msgs))
    assert "duplicate of an earlier identical message" in out[1].content


# --- CompactJson ---


def test_compact_json_table():
    rows = [{"name": f"svc-{i}", "status": "ok", "latency_ms": i} for i in range(20)]
    content = json.dumps(rows)
    msgs = [Message(role="tool", content=content, kind=KIND_TOOL_RESULT, call_id="c1")]
    out, report = CompactJson()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "|" in out[0].content and "name" in out[0].content
    assert out[0].meta.get("ctxfit_compacted") is True
    assert report.saved_tokens_est > 0


def test_compact_json_row_cap():
    rows = [{"a": i} for i in range(200)]
    msgs = [Message(role="tool", content=json.dumps(rows), kind=KIND_TOOL_RESULT, call_id="c1")]
    out, _ = CompactJson(max_rows=10)(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "more rows omitted" in out[0].content


def test_compact_json_dict_flattened():
    data = {"a": {"b": {"c": 1}}, "x": [1, 2, 3]}
    msgs = [Message(role="tool", content=json.dumps(data), kind=KIND_TOOL_RESULT, call_id="c1")]
    out, _ = CompactJson(when="always")(msgs, make_ctx(msgs))
    assert "a.b.c: 1" in out[0].content


def test_compact_json_never_touches_tool_call():
    args = json.dumps([{"k": "v"}] * 30)
    msgs = [Message(role="assistant", content=args, kind=KIND_TOOL_CALL, call_id="c1")]
    out, report = CompactJson()(msgs, over_budget_ctx(msgs, "history"))
    assert out[0].content == args and report.messages_affected == 0


def test_compact_json_skips_non_uniform_and_invalid():
    msgs = [
        Message(role="tool", content='[{"a": 1}, {"b": 2}]', kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content="not json at all", kind=KIND_TOOL_RESULT, call_id="c2"),
        Message(role="tool", content="just a short string", kind=KIND_TOOL_RESULT, call_id="c3"),
    ]
    out, report = CompactJson(when="always")(msgs, make_ctx(msgs))
    assert [m.content for m in out] == [m.content for m in msgs]
    assert report.messages_affected == 0


def test_compact_json_savings_gated():
    # Tiny JSON where the table is bigger -> untouched
    content = json.dumps([{"a": 1}])
    msgs = [Message(role="tool", content=content, kind=KIND_TOOL_RESULT, call_id="c1")]
    out, report = CompactJson(when="always")(msgs, make_ctx(msgs))
    assert out[0].content == content


# --- TruncateToolOutput ---


def test_truncate_keeps_head_and_tail():
    head = "HEADER " * 200
    mid = "MIDDLE " * 2000
    tail = "TRACEBACK: boom" * 100
    content = head + mid + tail
    msgs = [
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="tool", content=content, kind=KIND_TOOL_RESULT, call_id="c1"),
    ]
    out, report = TruncateToolOutput(max_chars=1000, head_chars=300, tail_chars=300)(
        msgs, over_budget_ctx(msgs, "retrieved")
    )
    assert out[0].content == "{}"  # tool_call untouched
    new = out[1].content
    assert new.startswith(head[:300])
    assert new.endswith(tail[-300:])
    assert "TRACEBACK: boom" in new
    assert "truncated" in new and "omitted" in new
    assert len(new) < len(content)
    assert report.saved_tokens_est > 0
    # pairing intact
    assert validate_pairing(out) == []


def test_truncate_skips_short_and_pinned():
    msgs = [
        Message(role="tool", content="short", kind=KIND_TOOL_RESULT, call_id="c1"),
        Message(role="tool", content="p" * 9000, kind=KIND_TOOL_RESULT, call_id="c2", pinned=True),
    ]
    out, report = TruncateToolOutput()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert out[0].content == "short"
    assert out[1].content == "p" * 9000
    assert report.messages_affected == 0


def test_truncate_pinned_with_consent():
    msgs = [
        Message(role="tool", content="p" * 9000, kind=KIND_TOOL_RESULT, call_id="c1", pinned=True)
    ]
    out, report = TruncateToolOutput(truncate_pinned=True)(msgs, over_budget_ctx(msgs, "retrieved"))
    assert len(out[0].content) < 9000 and report.messages_affected == 1


def test_truncate_rejects_bad_geometry():
    with pytest.raises(ValueError, match="head_chars \\+ tail_chars"):
        TruncateToolOutput(max_chars=100, head_chars=60, tail_chars=60)


def test_truncate_never_touches_tool_call_args():
    args = "x" * 9000
    msgs = [
        Message(role="assistant", content=args, kind=KIND_TOOL_CALL, call_id="c1"),
        Message(role="tool", content="ok " * 3000, kind=KIND_TOOL_RESULT, call_id="c1"),
    ]
    out, _ = TruncateToolOutput()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert out[0].content == args  # tool_call args byte-identical
    assert validate_pairing(out) == []


# --- SlidingWindow ---


def _conversation(n_turns, chars_per_turn=2000):
    msgs = [Message(role="system", content="sys", section="system", pinned=True)]
    for t in range(n_turns):
        msgs.append(Message(role="user", content=f"q{t} " + "u" * chars_per_turn))
        msgs.append(Message(role="assistant", content=f"a{t} " + "a" * chars_per_turn))
    return msgs


def test_sliding_window_drops_oldest_when_over():
    msgs = _conversation(10)
    out, report = SlidingWindow(keep_turns=10, min_turns=3)(msgs, over_budget_ctx(msgs, "history"))
    # tiny budget (100 tokens): drops down to min_turns + latest handling
    turns_left = {m.content.split()[0] for m in out if m.role == "user"}
    assert "q0" not in turns_left  # oldest dropped
    assert any(m.role == "user" and m.content.startswith("q9") for m in out)  # latest kept
    assert any(m.role == "system" for m in out)  # system kept
    assert validate_pairing(out) == []


def test_sliding_window_never_drops_pinned_turn():
    msgs = _conversation(6)
    msgs[1] = Message(role="user", content=msgs[1].content, pinned=True)  # pin q0
    out, _ = SlidingWindow(keep_turns=10, min_turns=2)(msgs, over_budget_ctx(msgs, "history"))
    assert any(m.content.startswith("q0") for m in out if m.role == "user")


def test_sliding_window_noop_under_budget():
    msgs = _conversation(4, chars_per_turn=10)
    out, report = SlidingWindow()(msgs, make_ctx(msgs))
    assert len(out) == len(msgs) and report.messages_affected == 0


def test_sliding_window_keeps_pairs_together():
    msgs = [Message(role="system", content="s", section="system", pinned=True)]
    for t in range(8):
        msgs.append(Message(role="user", content=f"q{t} " + "u" * 3000))
        msgs.append(Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id=f"c{t}"))
        msgs.append(
            Message(
                role="tool", content=f"r{t} " + "r" * 3000, kind=KIND_TOOL_RESULT, call_id=f"c{t}"
            )
        )
    out, _ = SlidingWindow(min_turns=2)(msgs, over_budget_ctx(msgs, "history"))
    assert validate_pairing(out) == []


# --- DropOldestUnpinned ---


def test_drop_oldest_unpinned_pair_atomic():
    msgs = [Message(role="system", content="s", section="system", pinned=True)]
    for t in range(6):
        msgs.append(Message(role="user", content=f"q{t} " + "u" * 2000))
        msgs.append(Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id=f"c{t}"))
        msgs.append(
            Message(
                role="tool", content=f"r{t} " + "r" * 2000, kind=KIND_TOOL_RESULT, call_id=f"c{t}"
            )
        )
    msgs.append(Message(role="user", content="latest q"))
    out, report = DropOldestUnpinned()(msgs, over_budget_ctx(msgs, "history"))
    assert validate_pairing(out) == []
    # every surviving call_id appears exactly twice (call + result)
    surviving = [m.call_id for m in out if m.call_id]
    assert all(surviving.count(cid) == 2 for cid in set(surviving))
    assert any(m.content == "latest q" for m in out)  # latest turn kept
    assert any(m.role == "system" for m in out)
    assert report.messages_affected > 0


def test_drop_oldest_unpinned_respects_pinned_pair():
    msgs = [
        Message(role="assistant", content="{}", kind=KIND_TOOL_CALL, call_id="c1"),
        Message(
            role="tool", content="pinned result", kind=KIND_TOOL_RESULT, call_id="c1", pinned=True
        ),
        Message(role="user", content="u" * 5000),
    ]
    out, _ = DropOldestUnpinned()(msgs, over_budget_ctx(msgs, "history"))
    # pair with pinned partner must survive together
    assert any(m.call_id == "c1" and m.kind == KIND_TOOL_CALL for m in out)
    assert validate_pairing(out) == []
