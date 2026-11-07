"""Expert round 2: salient truncation, outlier rows, piece estimator, fact audit, scaling."""

from __future__ import annotations

import json
import math
import random
import time

import ctxfit
from ctxfit.budgets import Budget, BudgetLedger
from ctxfit.estimate import HeuristicEstimator, PieceEstimator, count_by_section
from ctxfit.messages import KIND_TOOL_RESULT, Message
from ctxfit.strategies import CompactJson, StrategyContext, TruncateToolOutput, salient_lines


def over_budget_ctx(messages, section):
    est = HeuristicEstimator()
    budget = Budget(total=100)
    ledger = BudgetLedger(budget=budget, usage_est=count_by_section(messages, est))
    assert any(w.section == section for w in ledger.overflows())
    return StrategyContext(estimator=est, budget=budget, ledger=ledger)


def _tool(content, cid="c1"):
    return Message(role="tool", content=content, kind=KIND_TOOL_RESULT, call_id=cid)


def _log(n=600, at=300):
    lines = [
        f"2026-11-01 INFO worker-{i % 4} processed batch {1000 + i} status=ok" for i in range(n)
    ]
    lines.insert(at, "ERROR migration 0042_add_index failed E4242: relation already exists")
    return "\n".join(lines)


def test_salient_keeps_error_line_from_omitted_middle():
    msgs = [_tool(_log())]
    out, _ = TruncateToolOutput()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "0042_add_index" in out[0].content
    assert "salient" in out[0].content and len(out[0].content) < len(msgs[0].content)


def test_salient_off_restores_head_tail_only():
    msgs = [_tool(_log())]
    out, _ = TruncateToolOutput(salient=False)(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "0042_add_index" not in out[0].content


def test_salient_line_numbers_are_original():
    text = _log(at=300)
    msgs = [_tool(text)]
    out, _ = TruncateToolOutput()(msgs, over_budget_ctx(msgs, "retrieved"))
    n = next(ln for ln in out[0].content.split("\n") if "0042_add_index" in ln)
    num = int(n.split(":")[0][1:])
    assert text.split("\n")[num - 1].startswith("ERROR migration 0042")


def test_salient_repeated_errors_collapse_with_count():
    lines = [f"ok line {i}" for i in range(200)] + ["ERROR timeout talking to db 7"] * 30
    middle = "\n".join(["x"] + lines + ["y"])
    picked = salient_lines(middle, 1000)
    assert len(picked) == 1 and "[x30 similar]" in picked[0][1]


def test_salient_not_informative_when_all_lines_unique():
    middle = "\n".join(f"unique statement number {chr(65 + i % 26)}{'q' * i}" for i in range(80))
    assert salient_lines(middle, 1000) == []  # no rarity signal -> nothing invented


def test_salient_respects_budget_and_cap():
    middle = "\n".join(["boiler 1"] * 50 + [f"fatal error code {i} {'z' * 100}" for i in range(50)])
    picked = salient_lines("a\n" + middle + "\nb", 400)
    assert sum(len(t) for _, t in picked) <= 400


def test_salient_never_touches_pinned_or_tool_call():
    big = _log()
    pinned = Message(role="tool", content=big, kind=KIND_TOOL_RESULT, call_id="p", pinned=True)
    out, _ = TruncateToolOutput()([pinned], over_budget_ctx([pinned], "retrieved"))
    assert out[0].content == big


def test_salient_hostile_input_does_not_blow_up():
    rng = random.Random(1)
    junk = "".join(
        chr(rng.choice([0, 7, 27, 0x202E, 0x200B, 65, 10, 0x1F600])) for _ in range(30000)
    )
    msgs = [_tool(junk)]
    out, _ = TruncateToolOutput()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert out[0].content


def test_json_outlier_rows_survive_row_cap():
    rows = [{"id": i, "status": "active"} for i in range(400)]
    rows[222]["status"] = "recalled-batch-7731"
    msgs = [_tool(json.dumps(rows))]
    out, _ = CompactJson()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "recalled-batch-7731" in out[0].content and "(row 222)" in out[0].content


def test_json_outliers_can_be_disabled():
    rows = [{"id": i, "status": "active"} for i in range(400)]
    rows[222]["status"] = "recalled-batch-7731"
    msgs = [_tool(json.dumps(rows))]
    out, _ = CompactJson(max_outliers=0)(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "recalled-batch-7731" not in out[0].content


def test_json_no_outliers_when_column_is_not_uniform():
    rows = [{"id": i, "v": i % 7} for i in range(300)]
    msgs = [_tool(json.dumps(rows))]
    out, _ = CompactJson()(msgs, over_budget_ctx(msgs, "retrieved"))
    assert "outlier" not in out[0].content


def test_piece_estimator_matches_legacy_on_plain_text_scale():
    e = PieceEstimator()
    assert e.count("") == 0 and e.count("hello world") >= 2
    assert e.count("x" * 4000) < 700  # one long run: not 4000


def test_piece_estimator_counts_structured_text_higher_than_legacy():
    log = _log(100)
    assert PieceEstimator().count(log) > HeuristicEstimator().count(log) * 1.15


def test_piece_estimator_cjk_and_emoji():
    e = PieceEstimator()
    assert e.count("日本語" * 100) == math.ceil(300 * 1.5)
    assert e.count("😀" * 50) >= 100


def test_fact_audit_reports_lost_and_kept():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "r", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": _log(600, 300)},
        {"role": "user", "content": "ok"},
    ]
    _, kept = ctxfit.fit(msgs, total_budget=4000, audit_facts=True)
    assert kept.facts_seen and "facts:" in kept.summary()
    _, plain = ctxfit.fit(msgs, total_budget=4000)
    assert plain.facts_seen is None and "facts:" not in plain.summary()
    # head+tail only loses specifics the salient pass would have kept
    from ctxfit import FitPipeline

    pipe = FitPipeline(
        strategies=[TruncateToolOutput(salient=False, when="always")], audit_facts=True
    )
    _, r = pipe.fit(msgs)
    assert r.facts_lost_count > 0 and "E4242" in r.facts_lost
    assert r.to_dict()["facts_lost_count"] == r.facts_lost_count


def test_fit_scales_roughly_linearly():
    def mk(n):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        for i in range(n):
            cid = f"c{i}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": cid, "content": f"out {i} " * 200})
            msgs.append({"role": "user", "content": f"next {i}"})
        return msgs

    small, big = mk(150), mk(600)
    t = time.perf_counter()
    ctxfit.fit(small, total_budget=8000, estimator="heuristic")
    a = time.perf_counter() - t
    t = time.perf_counter()
    ctxfit.fit(big, total_budget=8000, estimator="heuristic")
    b = time.perf_counter() - t
    # 4x the input: the old O(n^2) code took ~16x; allow generous slack for CI noise.
    assert b < max(a, 0.02) * 12 + 0.5


def test_fact_audit_summary_escapes_terminal_control_chars():
    from ctxfit.pipeline import FitReport

    r = FitReport(
        tokens_before={},
        tokens_after={},
        strategy_reports=[],
        warnings=[],
        facts_seen=3,
        facts_lost=["/tmp/a\x1b[31m‮b"],
        facts_lost_count=1,
    )
    out = r.summary()
    assert "\x1b" not in out and "‮" not in out and "\\u001b" in out


def test_salient_lines_huge_inputs_stay_fast():
    one_line = "x" * 5_000_000
    t = time.perf_counter()
    assert salient_lines(one_line, 1000) == []
    many = "\n".join(f"row {i} ok" for i in range(300_000)) + "\nFATAL disk gone\nrow end"
    got = salient_lines(many, 1000)
    assert any("FATAL" in text for _, text in got)
    assert time.perf_counter() - t < 15


def test_cli_global_options_work_after_the_subcommand(tmp_path, capsys):
    from ctxfit.cli import main

    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"role": "user", "content": "hi"}]), encoding="utf-8")
    assert main(["analyze", str(p), "--budget", "5000", "--estimator=heuristic"]) == 0
    assert "5,000" in capsys.readouterr().out
    assert main(["--budget", "5000", "analyze", str(p)]) == 0  # old order still fine
    assert main(["fit", str(p), "--audit-facts", "--strict"]) == 0
