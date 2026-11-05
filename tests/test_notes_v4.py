"""Running notes spec v4: facts on markers, recall index, lossless tables, prose, auto threshold."""

from __future__ import annotations

import json
import math
import random
import re

import pytest
from test_notes import FakeSummarizer, anthropic_convo, keeper

from ctxfit.notes import (
    ALGO,
    FACT_PATTERNS,
    NotesConfig,
    auto_clear_threshold,
    extract_facts,
    lossless_table,
    mask_observations,
    scrub_secrets,
    squeeze_digest,
)


def test_algo_and_config_validation():
    assert ALGO == "rn-6"
    assert NotesConfig().keep_facts is True and NotesConfig().max_index == 20
    for bad in (
        dict(clear_at_least_chars="sometimes"),
        dict(clear_at_least_chars=-1),
        dict(cache_read_multiplier=1),
        dict(cache_write_multiplier=0.05),
        dict(expected_reuse=0),
        dict(keep_facts=1),
        dict(squeeze_recent_json=True),  # needs squeeze
        dict(max_facts=-1),
    ):
        with pytest.raises(ValueError):
            NotesConfig(**bad)
    NotesConfig(clear_at_least_chars="auto", mask_style="squeeze", squeeze_recent_json=True)


def test_auto_threshold_formula():
    cfg = NotesConfig(clear_at_least_chars="auto")
    assert auto_clear_threshold(10_000, cfg) == math.ceil(10_000 * (1.25 - 0.1) / (1 - 0.1) / 4)
    assert auto_clear_threshold(0, cfg) == 0
    cfg2 = NotesConfig(cache_write_multiplier=2, cache_read_multiplier=0.05, expected_reuse=1.5)
    assert auto_clear_threshold(1000, cfg2) == math.ceil(1000 * 1.95 / 0.95 / 1.5)


def test_facts_on_hide_marker_and_off_switch():
    msgs = anthropic_convo(8, out_chars=10)
    msgs[2]["content"][0]["content"] = "edited app/main.py; HTTP 500 at v1.2.3 " + "x" * 500
    out, _ = mask_observations(msgs)
    line = out[2]["content"][0]["content"]
    assert line.endswith("] - keeps: app/main.py; 500; v1.2.3")
    out, _ = mask_observations(msgs, keep_facts=False)
    assert out[2]["content"][0]["content"].endswith("]")


def test_facts_never_see_secrets():
    text = "key sk-ant-api03-AAAABBBBCCCCDDDDEEEE at https://u:pw123@x.io/a and ghp_" + "a" * 30
    facts = extract_facts(scrub_secrets(text))
    assert all("AAAA" not in f and "pw123" not in f and "redacted" not in f for f in facts)


def test_fact_limits():
    text = " ".join(f"app/f{i}.py" for i in range(20))
    assert len(extract_facts(text)) == 8 and len(extract_facts(text, 3)) == 3
    long = " ".join(f"/{c * 100}{i}/x.y" for i, c in enumerate("abcdef"))
    facts = extract_facts(long)
    assert all(len(f) == 80 for f in facts) and len("; ".join(facts)) <= 240 and len(facts) == 2


def test_path_search_matches_plain_regex():
    plain = re.compile(r"[\w.-]*/[\w./-]+", re.ASCII)
    fast = FACT_PATTERNS[0][0]
    rng = random.Random(7)
    alphabet = "ab./- _x9:"
    for _ in range(3000):
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        p = rng.randint(0, len(t))
        a, b = plain.search(t, p), fast.search(t, p)
        assert (a and a.span()) == (b and b.span()), (t, p)


def test_lossless_table_round_trips():
    rows = [{"a": i, "b": f"v|{i}\n", "c": None, "d": "1" if i else ""} for i in range(5)]
    table = lossless_table("t", json.dumps(rows))
    lines = table.split("\n")
    assert lines[0].endswith(" - lossless table]") and lines[1] == "| a | b | c | d |"
    assert lines[2] == '| 0 | v\\|0\\n | null | "" |'
    assert lines[3] == '| 1 | v\\|1\\n | null | "1" |'
    assert lossless_table("t", json.dumps(rows[:2])) is None  # < 3 rows
    assert lossless_table("t", json.dumps([{"a": [1]}] * 5)) is None  # nested
    assert lossless_table("t", json.dumps([{f"k{i}": 1 for i in range(31)}] * 3)) is None


def test_squeeze_recent_json_tables_recent_outputs_only_when_on():
    msgs = anthropic_convo(6, out_chars=10)
    rows = json.dumps([{"id": i, "name": f"n{i}"} for i in range(40)], indent=2)
    msgs[-2]["content"][0]["content"] = rows  # last turn, inside the keep window
    _, r = mask_observations(msgs, mask_style="squeeze")
    assert r.masked_outputs == 0
    out, r = mask_observations(msgs, mask_style="squeeze", squeeze_recent_json=True)
    assert r.lossless_tables == 1 and "lossless table]" in out[-2]["content"][0]["content"]


def test_prose_digest_and_fallback():
    sentences = ["Intro sentence here."] + [f"Filler number {'x' * 40} here." for _ in range(3)]
    sentences += ["Value was 42 today.", "See app/x.py now.", "Closing words."]
    text = " ".join(sentences * 4)
    body = squeeze_digest("t", text).split("\n", 1)[1]
    parts = body.split(" ... ")
    assert parts[0] == "Intro sentence here." and parts[-1] == "Closing words."
    assert "Value was 42 today." in parts and "See app/x.py now." in parts
    assert len(parts) <= 8 and len(body) <= 600
    one = squeeze_digest("t", "q" * 900).split("\n", 1)[1]
    assert one == "q" * 300 + " ... " + "q" * 200


def test_evicted_outputs_are_recallable_and_indexed():
    msgs = anthropic_convo(10)
    msgs[2]["content"][0]["content"] = "tiny"  # never masked, still evicted
    s = FakeSummarizer("n")
    k = keeper(s)
    out, r = k.fit(msgs)
    assert r.notes_applied
    import hashlib

    tiny = hashlib.sha256(b"tiny").hexdigest()[:8]
    assert k.recall(tiny) == "tiny" and tiny in r.offload
    notes_text = out[1]["content"][0]["text"]
    assert "[hidden outputs you can recall: " in notes_text and tiny not in notes_text
    assert f"- id {tiny}]" in s.calls[0][1]
    assert "(id 1a2b3c4d)" in s.calls[0][0]


def test_keeper_auto_threshold_reports():
    k = keeper(high_water_tokens=10**7, low_water_tokens=1, clear_at_least_chars="auto")
    _, r = k.fit(anthropic_convo(8))
    assert r.mask_boundary == 4 and r.clear_threshold_chars and r.clear_threshold_chars > 0
    _, r = k.fit(anthropic_convo(9))
    assert r.clear_threshold_chars is None
