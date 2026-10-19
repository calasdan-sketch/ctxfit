"""Tests for the pipeline, FitReport, and the top-level fit() API."""

import pytest

import ctxfit
from ctxfit import Budget, BudgetOverflow, FitPipeline, fit
from ctxfit.messages import Message, validate_pairing
from ctxfit.pipeline import PairingBroken
from ctxfit.strategies import Deduplicate


def big_tool_result(cid, n=6000):
    return {"role": "tool", "content": "LOG " * n, "tool_call_id": cid}


def test_fit_end_to_end_openai_shape():
    raw = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "read the logs"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_logs", "arguments": "{}"},
                }
            ],
        },
        big_tool_result("c1"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "read_logs", "arguments": "{}"},
                }
            ],
        },
        big_tool_result("c2"),  # duplicate body -> dedup candidate
        {"role": "user", "content": "summarize"},
    ]
    fitted, report = fit(raw, total_budget=4000)
    assert isinstance(fitted, list) and all(isinstance(m, dict) for m in fitted)
    assert report.total_after < report.total_before
    assert report.savings_pct > 0
    # OpenAI shape preserved: tool_calls reattached, tool messages intact
    asst = [m for m in fitted if m.get("tool_calls")]
    assert len(asst) == 2
    assert all(m["tool_calls"][0]["function"]["arguments"] == "{}" for m in asst)
    tools = [m for m in fitted if m["role"] == "tool"]
    assert len(tools) == 2 and all(m["tool_call_id"] in ("c1", "c2") for m in tools)
    # pairing still valid in canonical form
    assert validate_pairing(ctxfit.normalize(fitted, flavor="openai")) == []


def test_fit_noop_when_under_budget():
    raw = [{"role": "user", "content": "hi"}]
    fitted, report = fit(raw)
    assert fitted == raw
    assert report.saved_est == 0
    assert not report.over_budget
    assert "within budget" in report.summary()


def test_fit_report_dict_shape():
    raw = [{"role": "user", "content": "hi"}]
    _, report = fit(raw, total_budget=1000)
    d = report.to_dict()
    assert d["total_before"] == d["total_after"]
    assert d["over_budget"] is False
    assert isinstance(d["strategies"], list) and len(d["strategies"]) == 4


def test_strict_raises_on_surviving_overflow():
    raw = [{"role": "user", "content": "x" * 100_000}]
    with pytest.raises(BudgetOverflow):
        fit(raw, total_budget=100, strict=True)


def test_non_strict_warns_instead():
    raw = [{"role": "user", "content": "x" * 100_000}]
    _, report = fit(raw, total_budget=100)
    assert report.over_budget
    assert any("history" in w for w in [str(x) for x in report.warnings])
    assert "warnings:" in report.summary()


def test_aggressive_mode_drops_more():
    raw = [{"role": "system", "content": "s"}]
    for t in range(8):
        raw.append({"role": "user", "content": f"q{t} " + "u" * 5000})
        raw.append({"role": "assistant", "content": f"a{t}"})
    _, default_report = fit(raw, total_budget=2000)
    _, aggro_report = fit(raw, total_budget=2000, aggressive=True)
    assert aggro_report.total_after <= default_report.total_after


def test_custom_strategy_list():
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "LOG " * 3000, "tool_call_id": "c1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "LOG " * 3000, "tool_call_id": "c2"},
    ]
    # only dedup enabled, tiny budget forces overflow so it engages
    fitted, report = fit(raw, total_budget=500, strategies=[Deduplicate()])
    assert len(report.strategy_reports) == 1
    assert report.strategy_reports[0].strategy == "deduplicate"
    assert report.strategy_reports[0].messages_affected == 1


def test_unknown_strategy_preset_rejected():
    with pytest.raises(ValueError, match="unknown strategies preset"):
        fit([{"role": "user", "content": "hi"}], strategies="turbo")


def test_custom_strategy_subclass_hook():
    from ctxfit import Strategy

    class Shout(Strategy):
        name = "shout"
        target_sections = ("history",)

        def _apply(self, messages, ctx, report):
            out = []
            for m in messages:
                m2 = Message(
                    role=m.role,
                    content=m.content.upper(),
                    kind=m.kind,
                    name=m.name,
                    call_id=m.call_id,
                    section=m.section,
                    pinned=m.pinned,
                    meta=m.meta,
                )
                out.append(m2)
            report.log("shouted", 0)
            return out, report

    fitted, report = fit(
        [{"role": "user", "content": "hi"}], total_budget=10, strategies=[Shout(when="always")]
    )
    assert fitted[0]["content"] == "HI"
    assert report.strategy_reports[0].strategy == "shout"


def test_pairing_broken_raises():
    from ctxfit import Strategy

    class Evil(Strategy):
        name = "evil"
        target_sections = ("retrieved",)

        def _apply(self, messages, ctx, report):
            # drop every tool_result -> orphans tool_calls
            return [m for m in messages if m.kind != "tool_result"], report

    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]
    with pytest.raises(PairingBroken, match="orphaned tool_call"):
        fit(raw, total_budget=100_000, strategies=[Evil(when="always")])


def test_analyze_does_not_modify():
    raw = [{"role": "user", "content": "hello"}]
    pipeline = FitPipeline(budget=Budget(total=1000))
    info = pipeline.analyze(raw)
    assert info["message_count"] == 1
    assert info["usage_est"]["history"] > 0
    assert info["total_used_est"] == sum(info["usage_est"].values())
    assert len(info["top_consumers"]) == 1
    assert info["over_budget"] is False
    assert raw == [{"role": "user", "content": "hello"}]  # input untouched


def test_budget_overrides_plumbed():
    raw = [{"role": "user", "content": "x" * 4000}]
    _, report = fit(
        raw,
        total_budget=1000,
        budgets={"history": 0.9, "system": 0.05, "retrieved": 0.025, "scratchpad": 0.025},
    )
    assert report.total_budget == 1000


def test_fit_accepts_message_objects():
    msgs = [Message(role="user", content="hi")]
    fitted, report = fit(msgs)
    assert fitted[0]["role"] == "user"  # generic dict shape out
    assert not report.over_budget


def test_version_exported():
    assert isinstance(ctxfit.__version__, str)


def test_fit_empty_conversation():
    fitted, report = fit([])
    assert fitted == []
    assert report.total_before == 0
    assert report.savings_pct == 0.0
    assert not report.over_budget


def test_summary_hints_aggressive_on_surviving_overflow():
    raw = [{"role": "user", "content": "x" * 100_000}]
    _, report = fit(raw, total_budget=100)
    assert "aggressive=True" in report.summary()


def test_summary_no_hint_when_aggressive_already_on():
    raw = [{"role": "user", "content": "x" * 100_000}]
    _, report = fit(raw, total_budget=100, aggressive=True)
    assert "aggressive=True" not in report.summary()
