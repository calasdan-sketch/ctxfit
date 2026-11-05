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


def test_cold_cut_needs_high_water_and_lands_latest():
    msgs = anthropic_convo(10)
    base = dict(high_water_tokens=3000, low_water_tokens=2700, notes_max_tokens=100)
    cold = plan_context(msgs, config=NotesConfig(**base), now=0)
    warm = plan_context(msgs, config=NotesConfig(**base, cache_ttl_seconds=None))
    assert cold.cache_cold and cold.cut == 6  # T=10, keep 4: the latest allowed boundary
    assert warm.cut is not None and warm.cut < 6  # the earliest one reaching low water
    high = dict(base, high_water_tokens=10**6, low_water_tokens=1500)
    assert plan_context(msgs, config=NotesConfig(**high), now=0).summary is None


def test_parse_now_strict():
    from ctxfit.notes import parse_now

    assert parse_now("2026-11-05T00:00:00.5Z") == 1793836800500
    assert parse_now("2026-11-05T01:00:00+01:00") == 1793836800000
    assert parse_now(1793836800.0005) == 1793836800001
    for bad in ("2026-11-05T00:00:00", "2026-02-30T00:00:00Z", "2026-11-05T00:00:00.1234Z",
                True, float("nan"), "2026-11-05T00:00:00+0100", []):  # fmt: skip
        with pytest.raises(ValueError):
            parse_now(bad)


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


def _split_cells(line):
    assert line.startswith("| ") and line.endswith(" |")
    body, cells, cur, i = line[2:-2], [], "", 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            cur += body[i : i + 2]
            i += 2
        elif body.startswith(" | ", i):
            cells.append(cur)
            cur, i = "", i + 3
        else:
            cur += body[i]
            i += 1
    return cells + [cur]


def _decode(cell):
    import re as _re

    if cell.startswith('"'):
        return ("s", json.loads(cell.replace("\\|", "|")))
    if cell in ("true", "false", "null"):
        return ("c", cell)
    if _re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", cell):
        return ("n", cell)
    out, i = "", 0
    while i < len(cell):
        if cell[i] == "\\":
            out += {"\\": "\\", "|": "|", "n": "\n", "r": "\r"}[cell[i + 1]]
            i += 2
        else:
            out += cell[i]
            i += 1
    return ("s", out)


def _typed(v):
    if isinstance(v, str):
        return ("s", v)
    if v is True or v is False or v is None:
        return ("c", json.dumps(v))
    return ("n", v)


def test_lossless_table_round_trips_values_exactly():
    import random

    rng = random.Random(3)
    nums = ["12345678901234567891", "0.10", "-0.0", "1e400", "1E+2", "2.50e-3", "0", "-7"]
    strs = ["", "a|b", "x\ny", "back\\slash", " lead", "trail ", '"q', "42", "null", "é", "\\n"]
    for _ in range(200):
        items = []
        for _r in range(rng.randint(3, 6)):
            parts = []
            for k in rng.sample(["a", "b", "c", "10"], rng.randint(1, 4)):
                v = rng.choice([rng.choice(nums), json.dumps(rng.choice(strs)), "true", "null"])
                parts.append(f'"{k}": {v}')
            items.append("{" + ", ".join(parts) + "}")
        src = "[" + ",\n".join(items) + "]"
        form = arrival_form(src)
        lines = form[0].split("\n")[1:] if form and form[1] == "table" else None
        if lines is None:
            from ctxfit.notes import _table_lines, _table_source

            lines = _table_lines(_table_source(src), redact=False)
        keys = [_decode(c)[1] for c in _split_cells(lines[0])]
        want = json.loads(src, parse_int=lambda x: ("n", x), parse_float=lambda x: ("n", x))
        for row, item in zip(lines[1:], want):
            got = {k: _decode(c) for k, c in zip(keys, _split_cells(row)) if c != ""}
            assert got == {k: v if isinstance(v, tuple) else _typed(v) for k, v in item.items()}


def test_no_table_with_duplicate_keys():
    from ctxfit.notes import lossless_table

    rows = "[" + ",".join(f'{{"id": {i}, "id": {i}, "pad": "{"p" * 40}"}}' for i in range(6)) + "]"
    assert lossless_table("t", rows) is None and arrival_form(rows) is None


def test_url_facts_drop_query_and_fragment():
    assert extract_facts("see https://a.io/p/q.html?x=1#top and https://b.io/r#s") == [
        "https://a.io/p/q.html",
        "https://b.io/r",
    ]
