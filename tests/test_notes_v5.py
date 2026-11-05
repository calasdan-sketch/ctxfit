"""Running notes spec v5: lossless arrival compaction, cold-cache moments, HTTP fact window."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from test_notes import FakeSummarizer, anthropic_convo, keeper

from ctxfit.notes import (
    ALGO,
    NotesConfig,
    NotesKeeper,
    arrival_form,
    extract_facts,
    finish_context,
    mask_observations,
    plan_context,
)

NO_MASK = dict(keep_recent_turns=50, cache_ttl_seconds=None)


def _convo(outputs):
    msgs = [{"role": "user", "content": "task"}]
    for t, out in enumerate(outputs):
        if t:
            msgs.append({"role": "user", "content": f"next {t}"})
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{t}", "name": "bash", "input": {"n": t}}]})  # fmt: skip
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{t}", "content": out}]})  # fmt: skip
    return msgs


def _results(msgs):
    return [
        b["content"]
        for m in msgs
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]


def test_algo_and_config():
    assert ALGO == "rn-5"
    with pytest.raises(ValueError):
        NotesConfig(arrival="lossy")
    with pytest.raises(ValueError):
        NotesConfig(cache_ttl_seconds=-1)
    NotesConfig(arrival="lossless", cache_ttl_seconds=None)


def test_run_collapse_and_literal_guard():
    line = "retrying connection to db..."
    form, kind = arrival_form("\n".join([line] * 10 + ["y"]))
    assert kind == "runs" and form == f"[ctxfit: lossless - runs]\n{line}  [x10]\ny"
    assert arrival_form("\n".join([line] * 10 + ["y  [x2] literal"])) is None
    assert arrival_form("\n".join(["x"] * 2 + ["y"])) is None  # no run of 3


def test_arrival_only_if_at_most_85_percent():
    assert arrival_form("\n".join(["a", "a", "a", "b"])) is None  # header makes it longer


def test_tables_on_arrival_are_unredacted():
    rows = json.dumps([{"user": f"u{i}", "password": f"p{i}"} for i in range(6)], indent=2)
    form, kind = arrival_form(rows)
    assert kind == "table" and "| u0 | p0 |" in form


def test_duplicates_reference_only_visible_earlier_copies():
    file = "\n".join(f"line {i} of the file" for i in range(60))
    msgs = _convo([file, "ok", file, file])
    out, r = mask_observations(msgs, arrival="lossless", **{"keep_recent_turns": 50})
    res = _results(out)
    assert res[0] == file and res[2].startswith("[ctxfit: identical to earlier output id ")
    assert res[3] == res[2] and r.arrival_compacted == 2
    # turns 0-1 hidden by masking -> turn 2 is the first full copy, turn 3 references it
    out, _ = mask_observations(msgs, arrival="lossless", keep_recent_turns=2, mask_chunk_turns=1)
    res = _results(out)
    assert res[0].startswith("[ctxfit: tool output hidden") and res[1] == "ok"
    assert res[2] == file and res[3].startswith("[ctxfit: identical to earlier output id ")


def test_pinned_and_image_outputs_untouched():
    file = "\n".join(f"line {i}" for i in range(80))
    msgs = _convo([file, file, [{"type": "text", "text": file}, {"type": "image", "source": {}}]])
    msgs[2]["pinned"] = True
    msgs[5]["pinned"] = True
    out, _ = mask_observations(msgs, arrival="lossless", keep_recent_turns=50)
    res = _results(out)
    assert res[0] == file and res[1] == file  # pinned: never replaced
    assert isinstance(res[2], list) and len(res[2]) == 2


def test_arrival_forms_are_stable_as_history_grows():
    file = "\n".join(f"row {i}" for i in range(50))
    rows = json.dumps([{"a": i, "b": "x"} for i in range(30)], indent=2)
    outs = [file, rows, "\n".join(["warn"] * 9), file, rows, "done"]
    prev = None
    for n in range(1, len(outs) + 1):
        out, _ = mask_observations(_convo(outs[:n]), arrival="lossless", keep_recent_turns=50)
        if prev is not None:
            assert out[: len(prev)] == prev
        prev = out


def test_cold_detection_and_now_parsing():
    cfg = NotesConfig(high_water_tokens=10**7, low_water_tokens=1)
    p = plan_context(anthropic_convo(4), config=cfg, now="2026-11-05T00:00:00Z")
    r = finish_context(p, None)
    assert r.report.cache_cold and r.state.last_request_at == 1793836800
    for now, cold in (("2026-11-05T00:05:00Z", False), ("2026-11-05T00:05:00.001Z", True),
                      ("2026-11-05T01:05:00+01:00", False), (1793836800 + 301, True)):  # fmt: skip
        p = plan_context(anthropic_convo(4), config=cfg, state=r.state, now=now)
        assert finish_context(p, None).report.cache_cold is cold, now
    with pytest.raises(ValueError):
        plan_context([], now="2026-11-05T00:00:00")  # no timezone
    p = plan_context([], config=NotesConfig(cache_ttl_seconds=None))
    assert finish_context(p, None).report.cache_cold is False


def test_cold_cut_from_low_water_at_latest_boundary():
    msgs = anthropic_convo(10)
    base = dict(high_water_tokens=10**6, low_water_tokens=1500, notes_max_tokens=100)
    plan = plan_context(msgs, config=NotesConfig(**base), now=0)
    assert plan.cache_cold and plan.cut == 6 and plan.summary is not None  # T=10, keep 4
    warm = NotesConfig(**base, cache_ttl_seconds=None)
    assert plan_context(msgs, config=warm).summary is None


def test_keeper_uses_clock_and_skip_updates_last_request_at():
    t = [datetime(2026, 11, 5, tzinfo=timezone.utc)]
    k = NotesKeeper(
        FakeSummarizer(), high_water_tokens=10**7, low_water_tokens=1, clock=lambda: t[0]
    )
    _, r = k.fit(anthropic_convo(4))
    assert r.cache_cold
    t[0] += timedelta(seconds=10)
    _, r = k.fit(anthropic_convo(4), context_management={"edits": []})
    assert r.reasons == ["provider_context"] and not r.cache_cold
    assert k.state_dict()["last_request_at"] == 1793836810
    t[0] += timedelta(seconds=400)
    _, r = k.fit(anthropic_convo(5))
    assert r.cache_cold


def test_http_status_window_counts_from_keyword_end():
    assert extract_facts("HTTP" + " " * 8 + "201") == ["201"]
    assert extract_facts("HTTP" + " " * 9 + "202") == []
    assert extract_facts("httpa.b=404") == ["404"]  # http ends 4 before 404


def test_keeper_arrival_reports_savings():
    file = "\n".join(f"line {i} of the file" for i in range(60))
    k = keeper(arrival="lossless", high_water_tokens=10**7, low_water_tokens=1)
    out, r = k.fit(_convo([file, file]))
    assert r.arrival_compacted == 1 and r.arrival_saved_chars > 1000
