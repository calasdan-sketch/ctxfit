"""Spec v6: replay, anchor tools, probe eval, proof kit, Responses API."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctxfit.cli import main
from ctxfit.evaluate import (
    BLANK,
    EvalBudgetError,
    anthropic_backend,
    context_text,
    run_eval,
)
from ctxfit.notes import NotesConfig, finish_context, mask_observations, plan_context
from ctxfit.proof import DEMO_DIR, badge, demo_hash
from ctxfit.replay import dumps, load_corpus, load_prices, replay

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REPLAY = FIXTURES / "replay"
needs_fixtures = pytest.mark.skipif(not REPLAY.is_dir(), reason="shared fixtures not present")
DEMO_SHA256 = "2f8b17d0ab49a576f59da040fb336b522726e465925e5072be0b9dda5e456fde"


def _prices():
    return load_prices(REPLAY / "prices.json")


@needs_fixtures
def test_replay_output_is_the_shared_fixture_byte_for_byte():
    want = (FIXTURES / "replay-01-expected.json").read_text(encoding="utf-8")
    assert dumps(replay([str(REPLAY)], prices=_prices())) == want
    want = (FIXTURES / "replay-02-since-redact.json").read_text(encoding="utf-8")
    got = dumps(replay([str(REPLAY)], prices=_prices(), since="2026-11-04", redact=True))
    assert got == want and "sessions" not in json.loads(got)


@needs_fixtures
def test_replay_cli_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(
        [
            "replay",
            str(REPLAY),
            "--json",
            "--prices",
            str(REPLAY / "prices.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.read_text(encoding="utf-8") == (FIXTURES / "replay-01-expected.json").read_text(
        encoding="utf-8"
    )


@needs_fixtures
def test_replay_corpus_shape():
    corpus = load_corpus([str(REPLAY)])
    ids = [r.message_id for r in corpus.requests]
    assert len(ids) == len(set(ids))  # one request per message.id (split lines merged)
    report = replay([str(REPLAY)], prices=_prices())
    c = report["corpus"]
    assert c["sessions"] == 3 and c["conversations"] == 4
    assert c["sidechain_conversations"] == 1 and c["compaction_segments"] == 1
    assert c["unpriced_requests"] == 1 and c["skipped_lines"] == 1
    resumed = [r for r in corpus.requests if r.session.startswith("cccccccc")]
    main_conv = [r for r in corpus.requests if r.session.startswith("aaaaaaaa") and not r.sidechain]
    assert resumed and {r.conversation for r in resumed} == {main_conv[0].conversation}
    assert all(s["anchor_touched"] == 0 for s in report["strategies"])


@needs_fixtures
def test_replay_calibration(tmp_path):
    report = replay([str(REPLAY)], prices=_prices())
    assert not report["baseline"]["calibration_off"]
    assert abs(report["baseline"]["calibration_bp"] - 10000) <= 500
    assert report["strategies"][0]["cost_vs_baseline_bp"] == 10000
    for f in REPLAY.glob("session-*.jsonl"):  # recorded usage doubled: the model no longer fits
        lines = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            u = (
                (e.get("message") or {}).get("usage")
                if isinstance(e.get("message"), dict)
                else None
            )
            if isinstance(u, dict):
                for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                    if k in u:
                        u[k] *= 2
            lines.append(json.dumps(e))
        (tmp_path / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    off = replay([str(tmp_path)], prices=_prices())
    assert off["baseline"]["calibration_off"] is True


def test_replay_of_nothing(tmp_path):
    report = replay([str(tmp_path)])
    assert report["corpus"]["requests"] == 0 and report["baseline"]["calibration_bp"] == 0


# -- anchor tools ---------------------------------------------------------------


def test_anchor_tools_only_duplicates():
    src = "\n".join(f"line {i} = value {i * 3}" for i in range(80))
    msgs = [{"role": "user", "content": "task"}]
    for t, name in enumerate(["Read", "bash", "Read", "bash", "x", "x"]):
        if t:
            msgs.append({"role": "user", "content": f"next {t}"})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{t}", "name": name, "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{t}",
                        "content": src if name != "x" else "ok",
                    }
                ],
            }
        )
    out, r = mask_observations(msgs, keep_recent_turns=2, mask_chunk_turns=2, arrival="lossless")
    res = [
        b["content"]
        for m in out
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert res[0] == src  # Read below the boundary: never hidden
    assert res[1].startswith("[ctxfit: tool output hidden - bash")  # bash: masked as usual
    assert res[2].startswith("[ctxfit: identical to earlier output id ")  # Read: duplicates only
    plan = plan_context(
        msgs,
        mode="mask",
        config=NotesConfig(keep_recent_turns=2, mask_chunk_turns=2, anchor_tools=()),
    )
    res2 = finish_context(plan, None).messages
    assert res2[2]["content"][0]["content"].startswith("[ctxfit: tool output hidden - Read")
    with pytest.raises(ValueError):
        NotesConfig(anchor_tools="Read")  # type: ignore[arg-type]


# -- Responses API ------------------------------------------------------------------


def _responses(turns):
    items = [{"role": "developer", "content": "careful"}, {"role": "user", "content": "task"}]
    for t in range(turns):
        if t:
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"next {t}"}],
                }
            )
        items.append(
            {
                "type": "reasoning",
                "id": f"rs{t}",
                "summary": [],
                "encrypted_content": "gAAAA" + str(t),
            }
        )
        items.append({"type": "function_call", "call_id": f"c{t}", "name": "sh", "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"c{t}", "output": "x\n" * 300})
    return items


def test_responses_auto_detected_and_encrypted_items_untouched():
    items = _responses(8)
    out, r = mask_observations(items)
    assert r.masked_outputs == 4
    assert [m for m in out if m.get("type") == "reasoning"] == [
        m for m in items if m.get("type") == "reasoning"
    ]
    assert out[4]["output"].startswith("[ctxfit: tool output hidden - sh - 600 chars")
    out, r = mask_observations(items, previous_response_id="resp_1")
    assert r.reasons == ["previous_response_id"] and out == items


def test_responses_notes_message_shape():
    items = _responses(10)
    cfg = NotesConfig(high_water_tokens=400, low_water_tokens=200, keep_recent_turns=2, mask_chunk_turns=2,
                      cache_ttl_seconds=None)  # fmt: skip
    plan = plan_context(items, config=cfg)
    from ctxfit.notes import SummaryOutcome

    res = finish_context(plan, SummaryOutcome.ok("n"))
    notes = [m for m in res.messages if "running notes" in str(m.get("content"))]
    assert notes and notes[0]["type"] == "message" and notes[0]["role"] == "user"
    calls = {m["call_id"] for m in res.messages if m.get("type") == "function_call"}
    outs = {m["call_id"] for m in res.messages if m.get("type") == "function_call_output"}
    assert calls == outs


# -- probe eval -----------------------------------------------------------------------


def fake_backend(log):
    """Answers a cloze from the context when the blanked line is visible; 'n/a' otherwise."""

    def run(calls):
        out = {}
        for c in calls:
            last = c.messages[-1]["content"]
            if isinstance(last, str):
                out[c.key] = ("4", 50, 3)
                log.append(c.key)
                continue
            ctx = (
                context_text(c.messages[:-1])
                + "\n"
                + context_text([{"role": "user", "content": last[:-1]}])
            )
            q = last[-1]["text"]
            log.append(c.key)
            m = re.search(r"by ____:\n(.*)\nWhat exactly", q)
            if m:
                pat = re.escape(m.group(1)).replace(re.escape(BLANK), "(.+?)")
                hit = re.search(pat, ctx)
                ans = hit.group(1) if hit else "UNKNOWN"
            elif "Score the candidate" in str(c.messages[-1]["content"]):
                ans = "4"
            else:
                ans = "the task was fixing tests"
            from ctxfit.notes import _estimate, estimate_tokens

            out[c.key] = (ans, _estimate("anthropic", c.messages, 0), estimate_tokens(len(ans)) + 3)
        return out

    return run


@needs_fixtures
def test_eval_scores_and_budget():
    log = []
    with pytest.raises(EvalBudgetError):
        run_eval([str(REPLAY)], fake_backend(log), prices=_prices())
    assert log == []  # refused before any call
    seen = []
    with pytest.raises(EvalBudgetError):
        run_eval(
            [str(REPLAY)],
            fake_backend(log),
            prices=_prices(),
            max_usd=1e-9,
            on_estimate=seen.append,
        )
    assert log == [] and seen and seen[0] > 0
    res = run_eval([str(REPLAY)], fake_backend(log), prices=_prices(), max_usd=1.0, strategy="mask")
    assert res["probes"] >= 1
    assert res["cloze"]["full_correct"] == res["probes"]  # the full context always has the line
    assert 0 <= res["cloze"]["strategy_correct"] <= res["probes"]
    assert abs(res["spent_usd"] - res["estimated_usd"]) <= 0.10 * res["estimated_usd"]
    j = run_eval([str(REPLAY)], fake_backend(log), prices=_prices(), max_usd=1.0, judge=True)
    assert j["judge"] and set(j["judge"]) == {"recall", "artifact", "continuation", "decision"}
    assert all(v == 4 for v in j["judge"].values())


def test_eval_cli_refuses_without_max_usd(capsys):
    assert main(["eval", str(DEMO_DIR)]) == 2
    assert "pass --max-usd" in capsys.readouterr().err


def test_anthropic_batch_backend():
    created = {}

    def create(requests):
        created["requests"] = requests
        return SimpleNamespace(id="b1", processing_status="in_progress")

    def retrieve(bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    def results(bid):
        for r in created["requests"]:
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="v1.2.3")],
                                  usage=SimpleNamespace(input_tokens=10, output_tokens=2))  # fmt: skip
            yield SimpleNamespace(
                custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)
            )

    batches = SimpleNamespace(create=create, retrieve=retrieve, results=results)
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    from ctxfit.evaluate import Call

    run = anthropic_backend(client, "m", batch=True, sleep=lambda s: None)
    out = run([Call("k1", [{"role": "user", "content": "q"}], 8, 2)])
    assert out == {"k1": ("v1.2.3", 10, 2)} and created["requests"][0]["params"]["model"] == "m"


# -- proof kit ----------------------------------------------------------------------------


def test_demo_corpus_hash_and_fixture_copy():
    assert demo_hash() == DEMO_SHA256
    if REPLAY.is_dir():
        for p in DEMO_DIR.glob("*.jsonl"):
            assert p.read_bytes() == (REPLAY / p.name).read_bytes()


def test_demo_replay_is_committed():
    committed = Path(__file__).resolve().parents[1] / "docs" / "demo-replay.json"
    assert dumps(replay([str(DEMO_DIR)], redact=True)) == committed.read_text(encoding="utf-8")


def test_badge_schema(tmp_path):
    data = badge()
    assert set(data) == {"schemaVersion", "label", "message", "color"}
    assert data["schemaVersion"] == 1 and isinstance(data["message"], str)
    assert re.fullmatch(r"[+-]\d+\.\d% \(.+\)", data["message"])
    assert main(["badge", "--out", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text()) == data


def test_demo_tape_and_ci_job_exist():
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts" / "demo.tape").is_file():
        pytest.skip("repo-only files (not shipped in the sdist)")
    tape = (root / "scripts" / "demo.tape").read_text()
    assert "Output docs/demo.gif" in tape and "ctxfit replay src/ctxfit/demo" in tape
    ci = (root / ".github" / "workflows" / "proof.yml").read_text()
    assert "running_notes_bench.py --check" in ci and "docs/demo-replay.json" in ci


def test_copy_of_demo_still_replays(tmp_path):
    for p in DEMO_DIR.glob("*.jsonl"):
        shutil.copy(p, tmp_path / p.name)
    assert replay([str(tmp_path)], redact=True) == replay([str(DEMO_DIR)], redact=True)


# -- panel review of rn-6 ----------------------------------------------------------


@needs_fixtures
def test_replay_odd_input_fixture():
    want = (FIXTURES / "replay-03-odd.json").read_text(encoding="utf-8")
    assert dumps(replay([str(FIXTURES / "replay-odd")], prices=_prices())) == want
    rep = json.loads(want)
    assert rep["corpus"]["skipped_lines"] == 4 and rep["corpus"]["requests"] == 3


def test_replay_never_follows_directory_symlinks(tmp_path):
    from ctxfit.replay import expand_paths

    (tmp_path / "d").mkdir()
    shutil.copy(next(DEMO_DIR.glob("*.jsonl")), tmp_path / "d" / "a.jsonl")
    (tmp_path / "d" / "loop").symlink_to(tmp_path / "d", target_is_directory=True)
    (tmp_path / "d" / "same.jsonl").symlink_to(tmp_path / "d" / "a.jsonl")
    assert len(expand_paths([str(tmp_path)])) == 1
    assert len(expand_paths([str(tmp_path / "**" / "*.jsonl")])) == 1
    assert expand_paths([str(tmp_path / "d" / "?.jsonl")]) == [str(tmp_path / "d" / "a.jsonl")]
    assert expand_paths([str(tmp_path / "nope" / "*.jsonl")]) == []


def test_replay_line_rules(tmp_path):
    deep = "[" * 300 + "]" * 300
    good = next(DEMO_DIR.glob("session-b.jsonl")).read_text(encoding="utf-8").split("\n")[0]
    (tmp_path / "x.jsonl").write_text(
        "\n".join([deep, '{"a": NaN}', " \t", " ", good, "[1]"]) + "\n", encoding="utf-8"
    )
    c = load_corpus([str(tmp_path)])
    assert c.skipped_lines == 4 and c.entries == 1


def test_replay_usage_numbers():
    from ctxfit.replay import _usage

    u = _usage({"input_tokens": 3.0, "cache_read_input_tokens": 2.5, "output_tokens": True,
                "cache_creation_input_tokens": 2**53 + 3})  # fmt: skip
    assert u == {"input": 3, "read": 0, "write5": 2**53 + 4, "write1h": 0, "output": 0}


def test_badge_refuses_other_corpora_and_rounds(tmp_path):
    from ctxfit.proof import badge_message

    (tmp_path / "x.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        badge(str(tmp_path))
    for p in DEMO_DIR.glob("*.jsonl"):
        shutil.copy(p, tmp_path / p.name)
    (tmp_path / "x.jsonl").unlink()
    assert badge(str(tmp_path)) == badge()
    assert badge_message(8635, "s") == ("-13.6% (s)", "brightgreen")  # -136.5 -> -136
    assert badge_message(8634, "s")[0] == "-13.7% (s)"
    assert badge_message(10000, "s") == ("+0.0% (s)", "orange")
    assert badge_message(9995, "s") == ("+0.0% (s)", "orange")  # -0.5 tenths -> 0
    assert badge_message(9994, "s") == ("-0.1% (s)", "green")


def test_eval_max_usd_must_be_finite_positive():
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(EvalBudgetError):
            run_eval([str(DEMO_DIR)], lambda calls: {}, max_usd=bad)


def test_eval_sends_plain_text_only():
    from ctxfit.evaluate import flatten

    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "sig"},
                {"type": "tool_use", "id": "a", "name": "Read", "input": {"p": 1}},
            ],
        },  # fmt: skip
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": "key sk-" + "ant-api03-AAAABBBBCCCCDDDDEEEE",
                },
                {"type": "image", "source": {}},
            ],
        },  # fmt: skip
        {"role": "user", "content": "more"},
    ]
    out = flatten(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    blob = json.dumps(out)
    assert "thinking" not in blob and "signature" not in blob and "tool_use" not in blob
    assert "[tool call Read]" in blob and "[non-text]" in blob and "AAAABBBB" not in blob
    assert all(b["type"] == "text" for m in out for b in m["content"])


def test_eval_batch_errors_are_errors():
    from ctxfit.evaluate import Call

    def create(requests):
        return SimpleNamespace(id="b", processing_status="ended", requests=requests)

    def results(bid):
        yield SimpleNamespace(custom_id="p0", result=SimpleNamespace(type="errored"))

    client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
        create=create, retrieve=lambda b: None, results=results)))  # fmt: skip
    run = anthropic_backend(client, "m", batch=True)
    with pytest.raises(RuntimeError):
        run([Call("k", [{"role": "user", "content": "q"}], 8, 2)])


def test_eval_cli_warns_about_sending_content(capsys):
    main(["eval", str(DEMO_DIR)])
    assert "sends transcript content" in capsys.readouterr().err


def test_responses_any_call_id_item_pairs():
    items = [
        {"role": "user", "content": "task"},
        {
            "type": "local_shell_call",
            "call_id": "ls1",
            "action": {"type": "exec", "command": ["ls"]},
        },
        {"type": "message", "role": "user", "content": "next"},
        {"type": "local_shell_call_output", "call_id": "ls1", "output": "x" * 500},
        {"type": "message", "role": "user", "content": "next 2"},
    ]
    plan = plan_context(items, mode="mask", config=NotesConfig(cache_ttl_seconds=None))
    assert not plan.conv.pair_safe_boundary(1) and plan.conv.pair_safe_boundary(2)
    assert finish_context(plan, None).messages == items  # passed through untouched
