"""Round 5: author line, other scripts and base64 in the estimator, property tests, golden files,
per-tool limits, JSON-lines input, salient-line ranking, and the defects found on the way."""

from __future__ import annotations

import base64
import copy
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ctxfit
from ctxfit import cli
from ctxfit.cli import main
from ctxfit.estimate import PieceEstimator
from ctxfit.messages import Message, normalize, validate_pairing
from ctxfit.strategies import (
    CompactJson,
    Deduplicate,
    TruncateToolOutput,
    salient_lines,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
AUTHOR = "Calas Automations Inc."


def run_cli(*args: str, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "ctxfit", *args],
        capture_output=True,
        input=stdin.encode() if stdin is not None else None,
        timeout=120,
    )


def _call(i: int, name: str = "run") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _convo(*outputs: tuple[str, str]) -> list[dict]:
    """OpenAI-shaped conversation; each output is (tool name, text)."""
    msgs: list[dict] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "go"},
    ]
    for i, (name, text) in enumerate(outputs):
        msgs.append(_call(i, name))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": text})
    msgs.append({"role": "user", "content": "and now?"})
    return msgs


# --- 0. the author line ----------------------------------------------------------------------


def test_author_is_the_company_everywhere_it_is_declared():
    assert ctxfit.__author__ == AUTHOR and ctxfit.__license__ == "MIT"
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'authors = [{{ name = "{AUTHOR}" }}]' in toml
    assert f'maintainers = [{{ name = "{AUTHOR}" }}]' in toml
    assert "email" not in toml.lower()  # no address was given for this project: none is invented
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert f"Copyright (c) 2026 {AUTHOR}" in lic
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"MIT — {AUTHOR}" in readme
    out = run_cli("--help").stdout.decode()
    assert AUTHOR in out


def test_installed_metadata_names_the_company():
    md = pytest.importorskip("importlib.metadata")
    try:
        meta = md.metadata("ctxfit")
    except md.PackageNotFoundError:  # run from a bare checkout
        pytest.skip("ctxfit is not installed")
    assert meta["Author"] == AUTHOR or AUTHOR in (meta["Author-email"] or "")


def test_no_other_owner_label_in_the_tree():
    banned = re.compile(r"DC Enterprises|Dan Calas|@decnutz", re.IGNORECASE)
    allowed = {"tests/test_round4.py", "tests/test_round5.py", "docs/ROUND4.md", "docs/ROUND5.md"}
    for path in (
        list(ROOT.glob("*.md")) + list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("*.toml"))
    ):
        rel = path.relative_to(ROOT).as_posix()
        if rel in allowed:
            continue
        assert not banned.search(path.read_text(encoding="utf-8")), rel


# --- 1. estimator: other scripts, accents, base64 -----------------------------------------------

sys.path.insert(0, str(ROOT / "benchmarks"))
try:  # an sdist ships benchmarks/; a bare wheel test run does not
    from held_out_text import HELD_OUT
except ImportError:  # pragma: no cover
    HELD_OUT = []


@pytest.mark.parametrize(("name", "text", "real"), HELD_OUT, ids=[h[0] for h in HELD_OUT])
def test_non_latin_and_accented_text_is_estimated_within_a_quarter(name, text, real):
    est = PieceEstimator().count(text)
    assert abs(est - real) / real <= 0.25, (name, est, real)


def test_greek_and_hindi_are_no_longer_undercounted_by_a_third():
    # Before Round 5 these came out about 35-45% too low (every letter counted as 0.7).
    e = PieceEstimator()
    assert e.count("Η γρήγορη καφέ αλεπού πηδά πάνω από το τεμπέλικο σκυλί " * 10) > 400
    assert e.count("तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है " * 10) > 400


def test_base64_blob_is_counted_by_the_character_not_as_a_few_words():
    blob = base64.b64encode(random.Random(3).randbytes(3000)).decode()
    n = PieceEstimator().count(blob)
    assert 0.6 * len(blob) <= n <= 0.85 * len(blob)  # cl100k_base measured 0.72 per character
    data_url = json.dumps({"image": "data:image/png;base64," + blob})
    assert PieceEstimator().count(data_url) >= 0.6 * len(blob)


def test_paths_hashes_and_words_are_not_mistaken_for_base64():
    e = PieceEstimator()
    path = "/usr/local/lib/python3/site-packages/Some/Pkg/Mod1/file/another/deep/path/x"
    assert e.count(path) < 0.6 * len(path)
    assert e.count("a3f9c2d1e87b4650" * 4) <= 70  # a 64-char hex hash stays about 1 per char
    assert e.count("internationalization" * 3) < 40


def test_estimator_never_fails_on_any_text():
    e = PieceEstimator()
    nasty = ["\x00" * 50, "\ud800" * 10, "‮" * 30, "é" * 5000, "Ж" * 5000, "A1" * 5000, "😀" * 999]
    for t in nasty:
        assert e.count(t) >= 1


# --- 2. property tests (hypothesis) ------------------------------------------------------------

hyp = pytest.importorskip("hypothesis", reason="pip install hypothesis (it is in the dev extra)")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(
    max_examples=int(os.environ.get("CTXFIT_HYPOTHESIS_EXAMPLES", "120")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)

_TEXT = st.one_of(
    st.text(max_size=200),
    st.lists(st.text(alphabet="abc 0123.:", max_size=40), max_size=60).map("\n".join),
    st.text(alphabet=st.sampled_from(list("ERROR boom\n\t 0123é你😀‮\x1b\x00Жα")), max_size=1500),
)


@st.composite
def openai_conversations(draw):
    msgs: list[dict] = []
    if draw(st.booleans()):
        msgs.append({"role": "system", "content": draw(_TEXT)})
    for i in range(draw(st.integers(0, 7))):
        kind = draw(st.sampled_from(["user", "assistant", "tool", "tool"]))
        if kind != "tool":
            msgs.append({"role": kind, "content": draw(_TEXT)})
        else:
            msgs.append(_call(i))
            body = draw(_TEXT) * (1 + 40 * draw(st.integers(0, 3)))
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": body})
    msgs.append({"role": "user", "content": draw(_TEXT)})
    return msgs


@st.composite
def anthropic_conversations(draw):
    msgs: list[dict] = []
    for i in range(draw(st.integers(0, 6))):
        cid = f"toolu_{i}"
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": draw(_TEXT)},
                    {
                        "type": "tool_use",
                        "id": cid,
                        "name": "t",
                        "input": {"n": draw(st.integers())},
                    },
                ],
            }
        )
        body = draw(_TEXT) * draw(st.integers(1, 60))
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": cid, "content": body}],
            }
        )
    msgs.append({"role": "user", "content": draw(_TEXT)})
    return msgs


def _texts(msgs) -> list[str]:
    return [m.content for m in normalize(msgs)]


@_SETTINGS
@given(
    openai_conversations(),
    st.integers(100, 6000),
    st.sampled_from(["pieces", "heuristic"]),
    st.booleans(),
)
def test_property_fit_keeps_every_promise_on_openai_shapes(msgs, budget, est, aggressive):
    before = copy.deepcopy(msgs)
    out, rep = ctxfit.fit(
        msgs, total_budget=budget, estimator=est, aggressive=aggressive, offload={}
    )
    assert msgs == before  # the input is never modified
    assert validate_pairing(normalize(out)) == []  # no orphaned tool call or result
    assert rep.total_after <= rep.total_before  # never makes things bigger
    assert out[-1]["content"] == msgs[-1]["content"]  # the newest message is untouched
    if msgs[0]["role"] == "system":
        assert out[0] == msgs[0]  # and the system prompt comes back exactly as it went in
    again, _ = ctxfit.fit(
        msgs, total_budget=budget, estimator=est, aggressive=aggressive, offload={}
    )
    assert again == out  # deterministic
    json.dumps(out)  # serialisable
    refit, rep2 = ctxfit.fit(out, total_budget=budget, estimator=est, aggressive=aggressive)
    assert rep2.total_after <= rep2.total_before


@_SETTINGS
@given(anthropic_conversations(), st.integers(100, 8000))
def test_property_fit_keeps_every_promise_on_anthropic_shapes(msgs, budget):
    before = copy.deepcopy(msgs)
    out, rep = ctxfit.fit(msgs, total_budget=budget, estimator="pieces")
    assert msgs == before
    assert validate_pairing(normalize(out)) == []
    assert rep.total_after <= rep.total_before
    assert _texts(out)[-1] == _texts(msgs)[-1]
    if not any(sr.actions for sr in rep.strategy_reports):
        assert _texts(out) == _texts(msgs)  # nothing to do means nothing changed
    json.dumps(out)


@_SETTINGS
@given(openai_conversations(), st.integers(100, 200_000))
def test_property_a_fit_with_nothing_to_do_changes_no_text(msgs, budget):
    out, rep = ctxfit.fit(msgs, total_budget=budget, estimator="pieces")
    if not any(sr.actions for sr in rep.strategy_reports):
        assert _texts(out) == _texts(msgs)
        # and does not invent keys on messages that had none (a provider may reject them)
        for a, b in zip(out, msgs, strict=True):
            assert set(a) <= set(b) | {"content", "tool_call_id", "tool_calls"}


@_SETTINGS
@given(st.text(max_size=3000), st.text(max_size=3000))
def test_property_estimator_is_sane(a, b):
    e = PieceEstimator()
    assert e.count(a) >= 0 and (e.count(a) >= 1) == bool(a)
    # joining two texts never costs more than counting them apart plus a little rounding
    assert e.count(a + b) <= e.count(a) + e.count(b) + 8


@_SETTINGS
@given(st.text(max_size=400), st.integers(0, 3000), st.integers(0, 3000))
def test_property_truncation_never_grows_a_message(text, head, tail):
    msgs = _convo(("run", text * 30))
    try:
        st_ = TruncateToolOutput(
            max_chars=8000, head_chars=min(head, 3000), tail_chars=min(tail, 4000)
        )
    except ValueError:
        return
    out, rep = ctxfit.fit(msgs, total_budget=300, strategies=[st_], estimator="pieces")
    assert len(out[3]["content"]) <= len(msgs[3]["content"]) + 400  # at most the marker lines


# --- 3. golden files: every output format -------------------------------------------------------


def _golden_inputs():
    big = "\n".join(f"line {i} ok" for i in range(900)) + "\nERROR E42 boom\n" + "tail\n" * 200
    openai = _convo(("run", big), ("run", big))
    anth = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}],
        },
        {"role": "user", "content": "and?"},
    ]
    generic = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "task", "pinned": True},
        {"role": "assistant", "content": big, "section": "retrieved"},
        {"role": "user", "content": "bye"},
    ]
    return {"openai": openai, "anthropic": anth, "generic": generic}


def _golden_cases() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, msgs in _golden_inputs().items():
        fitted, rep = ctxfit.fit(msgs, total_budget=1500, estimator="pieces")
        out[f"fit-{name}.json"] = json.dumps(fitted, indent=2, ensure_ascii=False) + "\n"
        out[f"report-{name}.json"] = json.dumps(rep.to_dict(), indent=2) + "\n"
        out[f"summary-{name}.txt"] = rep.summary() + "\n"
        info = ctxfit.FitPipeline(estimator="pieces", budget=ctxfit.Budget(total=1500)).analyze(
            msgs
        )
        out[f"analyze-{name}.json"] = json.dumps(info, indent=2) + "\n"
    return out


@pytest.mark.parametrize("name", sorted(_golden_cases()))
def test_golden_output_has_not_changed(name):
    got = _golden_cases()[name]
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(got, encoding="utf-8", newline="\n")
    assert path.exists(), f"missing golden file {name}; run with UPDATE_GOLDEN=1 and review it"
    want = path.read_text(encoding="utf-8")
    assert got == want, (
        f"{name} changed; if on purpose, run UPDATE_GOLDEN=1 pytest and review the diff"
    )


def test_golden_cli_table_and_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_golden_inputs()["openai"]), encoding="utf-8")
    r = run_cli("--estimator", "pieces", "--budget", "1500", "analyze", str(p))
    table = r.stdout.decode().replace(str(p), "<file>")
    path = GOLDEN / "cli-analyze.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(table, encoding="utf-8", newline="\n")
    assert r.returncode == 2 and table == path.read_text(encoding="utf-8")


# --- 4. defects found in this round ---------------------------------------------------------------


def test_a_system_message_comes_back_with_no_invented_keys():
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    out, _ = ctxfit.fit(msgs, estimator="pieces")
    assert out == msgs  # used to gain "section": "system" and "pinned": true


def test_generic_dicts_keep_the_keys_that_carry_information():
    msgs = [
        {"role": "user", "content": "task", "pinned": True},
        {"role": "user", "content": "x", "section": "scratchpad"},
        {"role": "user", "content": "y"},
    ]
    out, _ = ctxfit.fit(msgs, estimator="pieces")
    assert out[0].get("pinned") is True and out[1].get("section") == "scratchpad"
    assert "pinned" not in out[2] and "section" not in out[2]
    assert Message(role="tool", content="t").to_dict() == {"role": "tool", "content": "t"}


def test_tail_chars_zero_does_not_make_the_output_bigger():
    # content[-0:] is the whole string: tail_chars=0 used to append a full copy of the output.
    text = "line\n" * 3000
    msgs = _convo(("run", text))
    out, rep = ctxfit.fit(
        msgs,
        total_budget=500,
        strategies=[TruncateToolOutput(head_chars=500, tail_chars=0, salient=False)],
        estimator="pieces",
    )
    assert len(out[3]["content"]) < 1000 and rep.total_after < rep.total_before


def _flood(n: int = 900) -> tuple[str, str]:
    rng = random.Random(4)
    lines = [
        f"2026-11-01T10:00:{i % 60:02d}Z INFO worker-{rng.randint(1, 4)} batch {rng.randint(1000, 9999)} ok"
        for i in range(n)
    ]
    words = ["timeout", "reset", "busy", "stale", "slow", "retry", "eof", "dns"]
    for k, i in enumerate(sorted(rng.sample(range(50, 500), 18))):
        lines[i] = (
            f"2026-11-01T10:00:{k:02d}Z WARN net transient error {words[k % 8]}-{k}, retrying"
        )
    lines[700] = "2026-11-01T10:30:00Z ERROR db FATAL migration 0042_add_index failed"
    return "\n".join(lines), "0042_add_index"


def test_a_real_fatal_beats_a_pile_of_warn_lines_that_mention_error():
    text, needle = _flood()
    picked = salient_lines(text, 1200)
    assert any(needle in t for _, t in picked)
    msgs = _convo(("run", text))
    out, _ = ctxfit.fit(msgs, total_budget=1500, estimator="pieces")
    assert needle in out[3]["content"]


def test_salient_ranking_order_is_fatal_then_error_then_warn():
    from ctxfit.strategies import _severity_score as sev

    assert sev("ERROR db FATAL x") > sev("ERROR db x") > sev("WARN transient error") > 0
    assert sev("INFO all fine") == 0 and sev("DEBUG error path taken") < sev("error path taken")


# --- 5. per-tool limits -----------------------------------------------------------------------------


def test_tool_limit_zero_leaves_one_tools_output_alone():
    big = "x line\n" * 3000
    msgs = _convo(("read_file", big), ("bash", big))
    out, _ = ctxfit.fit(msgs, total_budget=2000, tool_limits={"read_file": 0}, estimator="pieces")
    assert out[3]["content"] == big  # untouched
    assert len(out[5]["content"]) < len(big)  # the other tool is still shortened


def test_tool_limit_number_sets_that_tools_cut_point():
    big = "".join(f"row {i}\n" for i in range(2000))
    msgs = _convo(("bash", big))
    default, _ = ctxfit.fit(msgs, total_budget=500, estimator="pieces")
    limited, _ = ctxfit.fit(msgs, total_budget=500, tool_limits={"bash": 900}, estimator="pieces")
    assert len(limited[3]["content"]) < len(default[3]["content"])
    small = _convo(("bash", "y" * 2000))  # under the default 4000, over this tool's 900
    out, _ = ctxfit.fit(small, total_budget=100, tool_limits={"bash": 900}, estimator="pieces")
    assert "[ctxfit: truncated" in out[3]["content"]


def test_tool_limit_protects_json_from_compaction_and_duplicates_from_dedup():
    rows = json.dumps([{"id": i, "name": f"n{i}", "ok": True} for i in range(300)])
    msgs = _convo(("db", rows), ("db", rows))
    out, _ = ctxfit.fit(msgs, total_budget=300, tool_limits={"db": 0}, estimator="pieces")
    assert out[3]["content"] == rows and out[5]["content"] == rows
    out2, _ = ctxfit.fit(msgs, total_budget=300, estimator="pieces")
    assert out2[3]["content"] != rows


def test_tool_limit_works_for_anthropic_tool_results():
    big = "x line\n" * 3000
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]},
        {"role": "user", "content": "next"},
    ]
    out, _ = ctxfit.fit(msgs, total_budget=500, tool_limits={"Read": 0}, estimator="pieces")
    assert out[2]["content"][0]["content"] == big


def test_tool_limit_rejects_nonsense():
    for bad in ({"x": -1}, {"x": 50}, {"x": True}, {"x": "9"}):
        with pytest.raises(ValueError):
            ctxfit.fit(_convo(("x", "y")), tool_limits=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default strategies"):
        ctxfit.fit(_convo(("x", "y")), strategies=[Deduplicate()], tool_limits={"x": 0})
    assert CompactJson(protect_tools=["a"]).protect_tools == frozenset({"a"})


def test_cli_tool_limit(tmp_path):
    big = "x line\n" * 3000
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_convo(("read_file", big))), encoding="utf-8")
    o = tmp_path / "o.json"
    r = run_cli("--budget", "800", "fit", str(p), "--out", str(o), "--tool-limit", "read_file=0")
    assert json.loads(o.read_text(encoding="utf-8"))[3]["content"] == big, r.stderr
    bad = run_cli("fit", str(p), "--tool-limit", "oops")
    assert (
        bad.returncode == 1 and b"NAME=CHARACTERS" in bad.stderr and b"Traceback" not in bad.stderr
    )
    low = run_cli("fit", str(p), "--tool-limit", "read_file=10")
    assert low.returncode == 1 and b"300" in low.stderr


# --- 6. JSON-lines and Claude Code transcripts --------------------------------------------------------


def test_json_lines_file_of_messages(tmp_path):
    msgs = _convo(("run", "line\n" * 4000))
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    r = run_cli("--budget", "800", "fit", str(p), "--json")
    assert r.returncode in (0, 2), r.stderr
    assert json.loads(r.stdout)["total_after"] < json.loads(r.stdout)["total_before"]
    assert run_cli("analyze", str(p), "--json").returncode in (0, 2)


def _transcript(path: Path, n_calls: int = 4) -> None:
    rows, parent, t = [], None, 0

    def add(kind: str, message: dict) -> None:
        nonlocal parent, t
        t += 1
        uid = f"u{t}"
        rows.append(
            {
                "type": kind,
                "uuid": uid,
                "parentUuid": parent,
                "sessionId": "s1",
                "timestamp": f"2026-11-01T10:00:{t:02d}.000Z",
                "message": message,
            }
        )
        parent = uid

    add("user", {"role": "user", "content": "fix the tests"})
    for i in range(n_calls):
        add(
            "assistant",
            {
                "id": f"m{i}",
                "role": "assistant",
                "model": "m",
                "content": [
                    {"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"c": i}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
        body = "ok line\n" * 2000 + ("FATAL disk full\n" if i == 1 else "") + "ok line\n" * 600
        add(
            "user",
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": body}],
            },
        )
    add(
        "assistant",
        {
            "id": "mz",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_claude_code_transcript_is_read_and_fitted(tmp_path):
    p = tmp_path / "session one.jsonl"  # a space in the name
    _transcript(p)
    o = tmp_path / "o.json"
    r = run_cli("--budget", "2500", "fit", str(p), "--out", str(o))
    assert r.returncode in (0, 2), r.stderr
    assert b"Claude Code transcript" in r.stderr and b"not counted" in r.stderr
    out = json.loads(o.read_text(encoding="utf-8"))
    assert "FATAL disk full" in json.dumps(out)  # the planted error survived
    assert validate_pairing(normalize(out)) == []


def test_json_lines_with_a_raw_line_separator_inside_a_string(tmp_path):
    # JSON allows U+2028 / U+0085 / form feed unescaped inside strings; they are not line breaks.
    msgs = [{"role": "user", "content": "a\u2028b\x85c\x0cd"}, {"role": "user", "content": "e"}]
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\r\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\r\n", encoding="utf-8"
    )
    out = run_cli("fit", str(p), "--out", "-")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)[0]["content"] == "a\u2028b\x85c\x0cd"


def test_transcript_errors_are_one_plain_sentence(tmp_path):
    p = tmp_path / "t.jsonl"
    _transcript(p)
    r = run_cli("analyze", "-", stdin=p.read_text(encoding="utf-8"))
    assert r.returncode == 1 and b"from a file" in r.stderr and b"Traceback" not in r.stderr
    star = tmp_path / "t*.jsonl"
    star.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    assert b"no * or ?" in run_cli("analyze", str(star)).stderr
    only_user = tmp_path / "u.jsonl"
    only_user.write_text(
        '{"type":"user","uuid":"a","parentUuid":null,"timestamp":"2026-11-01T10:00:00.000Z",'
        '"message":{"role":"user","content":"hi"}}\n'
        '{"type":"user","uuid":"b","parentUuid":"a","timestamp":"2026-11-01T10:00:01.000Z",'
        '"message":{"role":"user","content":"yo"}}\n',
        encoding="utf-8",
    )
    r2 = run_cli("analyze", str(only_user))
    assert r2.returncode == 1 and b"no model request" in r2.stderr


def test_plain_garbage_still_gets_the_old_json_error(tmp_path):
    p = tmp_path / "g.json"
    p.write_text("not json\nsecond line\n", encoding="utf-8")
    r = run_cli("analyze", str(p))
    assert r.returncode == 1 and b"is not valid JSON" in r.stderr
    q = tmp_path / "h.jsonl"
    q.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")  # JSON lines, but not messages
    assert b"is not valid JSON" in run_cli("analyze", str(q)).stderr


# --- 7. upgrade path from the registry release (0.1.1) -------------------------------------------------


def test_report_keys_of_0_1_1_are_still_there():
    old_keys = {
        "estimator", "total_budget", "tokens_before", "tokens_after", "total_before",
        "total_after", "saved_est", "savings_pct", "over_budget", "warnings", "strategies",
    }  # fmt: skip
    _, rep = ctxfit.fit(_convo(("run", "x")), estimator="pieces")
    assert old_keys <= set(rep.to_dict())
    old_analyze = {
        "allowances", "estimator", "message_count", "over_budget", "top_consumers",
        "total_budget", "total_used_est", "usage_est", "warnings",
    }  # fmt: skip
    info = ctxfit.FitPipeline(estimator="pieces").analyze(_convo(("run", "x")))
    assert old_analyze <= set(info)


def test_cli_module_is_importable_without_side_effects():
    assert callable(cli.main) and callable(main)


# --- 8. tests added after a mutation check found them missing -------------------------------------------


def test_salient_lines_never_make_a_message_longer_than_the_original():
    # Every middle line is a different error: the "kept lines" block would be bigger than what it
    # replaces, so the plain head+tail form must be used instead.
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
    lines = [f"ERROR {w} subsystem failed unexpectedly during step {w}" for w in words * 2]
    text = "\n".join(lines)
    msgs = _convo(("run", text))
    st_ = TruncateToolOutput(max_chars=1000, head_chars=300, tail_chars=300)
    out, _ = ctxfit.fit(msgs, total_budget=100, strategies=[st_], estimator="pieces")
    body = out[3]["content"]
    assert len(body) < len(text) and "salient line" not in body


def test_head_plus_tail_must_leave_room_for_a_middle():
    with pytest.raises(ValueError):
        TruncateToolOutput(max_chars=1000, head_chars=500, tail_chars=500)
    TruncateToolOutput(max_chars=1000, head_chars=500, tail_chars=499)


def test_tool_limit_scales_head_and_tail_to_the_limit():
    text = "".join(f"row {i:05d}\n" for i in range(3000))
    msgs = _convo(("bash", text))
    out, _ = ctxfit.fit(msgs, total_budget=100, tool_limits={"bash": 900}, estimator="pieces")
    body = out[3]["content"]
    assert len(body) < 900 + 700  # head and tail shrank to a third of the limit each, plus markers


def test_odd_lines_are_kept_when_a_third_of_the_lines_are_boilerplate():
    boiler = [f"heartbeat {i} ok" for i in range(40)]
    prose = [f"unique note number {i}: {'xyz' * (i + 1)}" for i in range(40)]
    middle = "\n".join(["cut"] + boiler + prose + ["cut"])
    assert salient_lines(middle, 1200)  # 50% boilerplate: rare lines are informative
    assert salient_lines("\n".join(["cut"] + prose + ["cut"]), 1200) == []  # nothing repeats


def test_a_level_word_outranks_the_word_error_in_ordinary_text():
    from ctxfit.strategies import _severity_score as sev

    assert sev("ERROR db x") > sev("db error x") > sev("WARN db error x")
    assert sev("FATAL db x") == sev("db fatal x") + 2  # the level token adds on top of the word


def test_a_system_message_is_pinned_when_built_directly():
    assert Message(role="system", content="x").pinned is True
    assert Message(role="user", content="x").pinned is False


def test_cli_tool_limit_needs_a_number(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_convo(("x", "y"))), encoding="utf-8")
    r = run_cli("fit", str(p), "--tool-limit", "x=abc")
    assert r.returncode == 1 and b"NAME=CHARACTERS" in r.stderr
    r2 = run_cli("fit", str(p), "--tool-limit", "x=-3")
    assert r2.returncode == 1 and b"NAME=CHARACTERS" in r2.stderr


def test_tool_limit_works_for_langchain_shaped_messages():
    class Human:
        type = "human"

        def __init__(self, content):
            self.content = content

    class AI:
        type = "ai"

        def __init__(self, content, tool_calls):
            self.content, self.tool_calls, self.additional_kwargs = content, tool_calls, {}

    class Tool:
        type = "tool"

        def __init__(self, content, tool_call_id):
            self.content, self.tool_call_id = content, tool_call_id

    big = "x line\n" * 3000
    call = {"name": "read_file", "args": {}, "id": "lc1", "type": "tool_call"}
    raw = [Human("go"), AI("", [call]), Tool(big, "lc1"), Human("next")]
    out, _ = ctxfit.fit(raw, total_budget=500, tool_limits={"read_file": 0}, estimator="pieces")
    assert out[2]["content"] == big
    out2, _ = ctxfit.fit(raw, total_budget=500, estimator="pieces")
    assert len(out2[2]["content"]) < len(big)


def test_langchain_native_tool_calls_keep_name_and_arguments():
    # AIMessage.tool_calls entries are {"name", "args" (dict), "id", "type": "tool_call"}; the
    # adapter used to read only the OpenAI shape, so name and arguments came out empty.
    from ctxfit.adapters.langchain import from_lc_messages

    class AI:
        type = "ai"
        additional_kwargs: dict = {}

        def __init__(self, tool_calls):
            self.content, self.tool_calls = "", tool_calls

    args = {"path": "src/a.py", "lines": [1, 2], "note": "é"}
    call = {"name": "read_file", "args": args, "id": "lc9", "type": "tool_call"}
    [m] = from_lc_messages([AI([call])])
    assert m.name == "read_file" and m.call_id == "lc9" and json.loads(m.content) == args
    out, _ = ctxfit.fit(
        [AI([call]), type("T", (), {"type": "tool", "content": "ok", "tool_call_id": "lc9"})()],
        estimator="pieces",
    )
    assert out[0]["tool_calls"][0]["args"] == args  # a dict again, as LangChain expects
    assert out[0]["tool_calls"][0]["name"] == "read_file"


def test_langchain_openai_shaped_tool_calls_still_work():
    from ctxfit.adapters.langchain import from_lc_messages

    class AI:
        type = "ai"
        additional_kwargs: dict = {}

        def __init__(self, tool_calls):
            self.content, self.tool_calls = "", tool_calls

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    [m] = from_lc_messages([AI([call])])
    assert (m.name, m.content, m.call_id) == ("f", '{"a": 1}', "c1")
