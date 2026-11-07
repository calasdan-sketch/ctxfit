"""Running notes spec v3: exclude_tools, clear_at_least_chars, squeeze, recall."""

from __future__ import annotations

import json

import pytest
from test_notes import FakeSummarizer, anthropic_convo, keeper, openai_convo

from ctxfit.notes import (
    ALGO,
    RECALL_TOOL_NAME,
    NotesConfig,
    NotesKeeper,
    NotesState,
    finish_context,
    handle_recall,
    mask_observations,
    plan_context,
    recall_tool,
    squeeze_digest,
)


def test_algo_is_rn3_and_old_states_are_spec_changed():
    assert ALGO == "rn-6"
    plan = plan_context(anthropic_convo(4), state={"version": 2, "algo": "rn-2"})
    assert "spec_changed" in plan.reasons


def test_config_validation():
    with pytest.raises(ValueError):
        NotesConfig(mask_style="zip")
    with pytest.raises(ValueError):
        NotesConfig(exclude_tools="bash")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NotesConfig(clear_at_least_chars=-1)
    assert NotesConfig(exclude_tools=["a"]).exclude_tools == ("a",)


def test_exclude_tools_never_masked_or_squeezed():
    msgs = anthropic_convo(8)
    for style in ("hide", "squeeze"):
        out, r = mask_observations(msgs, exclude_tools=["bash"], mask_style=style)
        assert r.masked_outputs == 0 and out == msgs
    o = openai_convo(8)
    out, r = mask_observations(o, exclude_tools=["sql"])
    assert r.masked_outputs == 0


def _mask_step(msgs, state, **cfg):
    cfg = {"cache_ttl_seconds": None, **cfg}
    plan = plan_context(msgs, mode="mask", config=NotesConfig(**cfg), state=state)
    res = finish_context(plan, None)
    return res.report, res.state


def test_clear_at_least_holds_boundary_until_worth_it():
    cfg = dict(keep_recent_turns=4, mask_chunk_turns=4, clear_at_least_chars=10_000)
    r, st = _mask_step(anthropic_convo(8, out_chars=2000), None, **cfg)
    assert r.mask_boundary == 0 and st.mask_boundary == 0  # 4 x 2000 < 10,000
    r, st = _mask_step(anthropic_convo(9, out_chars=2000), st, **cfg)
    assert r.mask_boundary == 0
    cfg["clear_at_least_chars"] = 8000
    r, st = _mask_step(anthropic_convo(9, out_chars=2000), st, **cfg)
    assert r.mask_boundary == 4 and st.mask_boundary == 4
    # never backwards while the history only grows
    r, st = _mask_step(anthropic_convo(12, out_chars=10), st, **cfg)
    assert r.mask_boundary == 4
    # shorter history than the stored boundary needs: reset and recompute
    r, st = _mask_step(anthropic_convo(5, out_chars=2000), NotesState(mask_boundary=8), **cfg)
    assert r.mask_boundary == 0


def test_clear_at_least_zero_is_v2_behaviour():
    a, _ = _mask_step(anthropic_convo(12), None)
    assert a.mask_boundary == 8


def test_keeper_remembers_boundary_and_fork_resets_it():
    k = keeper(high_water_tokens=10**7, low_water_tokens=1, clear_at_least_chars=1)
    k.fit(anthropic_convo(8))
    assert k.state_dict()["mask_boundary"] == 4


def test_squeeze_json_array_object_lines_text():
    rows = json.dumps([{"a": 1, "b": "x" * 10}, {"c": 2}] * 40)
    d = squeeze_digest("api", rows)
    assert d.split("\n")[1].startswith("array of 80 items; first item: {")
    assert d.split("\n")[2] == "keys: a, b, c"
    obj = json.dumps({"z": 1, "10": "ten", "list": [1], "s": "y" * 200})
    lines = squeeze_digest("api", obj).split("\n")
    assert lines[1] == "object with keys: 10, z, list, s"
    assert lines[2:] == ["10 = ten", "z = 1", "s = " + "y" * 80]
    log = "\n".join(
        ["start", "a", "b"]
        + ["same"] * 5
        + ["boom ERROR here"]
        + [f"l{i}" for i in range(9)]
        + ["x", "y", "z"]
    )
    lines = squeeze_digest("bash", log).split("\n")
    assert lines[1:4] == ["start", "a", "b"]
    assert lines[4] == "... 14 lines omitted ..."
    assert lines[5] == "boom ERROR here" and lines[6:] == ["x", "y", "z"]
    dup = "\n".join(["h1", "h2", "h3"] + ["fail!"] * 4 + ["t1", "t2", "t3"])
    assert "fail!  (x4)" in squeeze_digest("bash", dup)
    text = "q" * 400 + "w" * 400
    body = squeeze_digest("t", text).split("\n", 1)[1]
    assert body == "q" * 300 + " ... " + "w" * 200


def test_squeeze_falls_back_to_hide_when_digest_too_big():
    msgs = anthropic_convo(8, out_chars=450)  # one line: digest > 40%
    out, r = mask_observations(msgs, mask_style="squeeze")
    assert r.masked_outputs == 4 and r.squeezed_outputs == 0
    assert out[2]["content"][0]["content"].startswith("[ctxfit: tool output hidden")


def test_squeeze_scrubs_secrets_and_hashes_original():
    import hashlib

    text = "\n".join(
        ["l"] * 3 + [f"row {i}" for i in range(40)] + ["fatal: token=abcdefsecret"] + ["e"] * 3
    )
    d = squeeze_digest("bash", text)
    assert "abcdefsecret" not in d and "token=[redacted]" in d
    assert hashlib.sha256(text.encode()).hexdigest()[:8] in d.split("\n")[0]


def test_squeeze_report_and_keeper_recall():
    msgs = anthropic_convo(8, out_chars=10)
    for i, m in enumerate(msgs):
        if isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result":
            m["content"][0]["content"] = "\n".join(f"line {j} of output {i}" for j in range(60))
    k = keeper(high_water_tokens=10**7, low_water_tokens=1, mask_style="squeeze")
    out, r = k.fit(msgs)
    assert r.squeezed_outputs == 4 == r.masked_outputs
    id8 = sorted(r.offload)[0]
    assert k.recall(id8) == r.offload[id8]


def test_recall_tool_definitions():
    a = recall_tool("anthropic")
    assert a["name"] == RECALL_TOOL_NAME and a["input_schema"]["required"] == ["id"]
    o = recall_tool("openai")
    assert (
        o["type"] == "function"
        and o["function"]["parameters"]["properties"]["id"]["type"] == "string"
    )
    with pytest.raises(ValueError):
        recall_tool("gemini")


def test_handle_recall_both_flavors():
    k = NotesKeeper(FakeSummarizer(), high_water_tokens=10**7, low_water_tokens=1)
    _, r = k.fit(anthropic_convo(8))
    id8 = next(iter(r.offload))
    res = k.handle_recall(
        {"type": "tool_use", "id": "tu_9", "name": RECALL_TOOL_NAME, "input": {"id": id8}}
    )
    assert res == {"type": "tool_result", "tool_use_id": "tu_9", "content": r.offload[id8]}
    res = k.handle_recall(
        {"type": "tool_use", "id": "tu_9", "name": RECALL_TOOL_NAME, "input": {"id": "nope"}}
    )
    assert res["is_error"] is True and "no stored output" in res["content"]
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": RECALL_TOOL_NAME, "arguments": json.dumps({"id": id8.upper()})},
    }
    assert k.handle_recall(call) == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": r.offload[id8],
    }
    bad = {"id": "c2", "function": {"name": RECALL_TOOL_NAME, "arguments": "{oops"}}
    assert "missing" in handle_recall(bad, k.recall)["content"]
    with pytest.raises(ValueError):
        handle_recall({"foo": 1}, k.recall)


# ---------------------------------------------------------------------------
# panel review fixes
# ---------------------------------------------------------------------------

ANT = "sk-" + "ant-api03-" + "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z"


def _filler(n):
    return ("lorem ipsum dolor sit amet " * (n // 27 + 1))[:n]


@pytest.mark.parametrize(
    "text",
    [
        _filler(290) + " " + ANT + " " + _filler(3000),  # head window
        _filler(3000) + " " + ANT + " " + _filler(150),  # tail window
        "\n".join([_filler(185) + " " + ANT] + [f"r{i}" for i in range(20)]),  # line cap
        json.dumps({"note": _filler(70) + " " + ANT, "pad": _filler(3000)}),  # value cap
        json.dumps([{"blob": _filler(280) + " " + ANT}] + [{"i": i} for i in range(200)]),
    ],
)
def test_squeeze_never_leaks_a_secret_prefix(text):
    d = squeeze_digest("t", text)
    assert "sk-ant" not in d and "Ab3dEf" not in d


def test_sensitive_keys_in_digest_and_args():
    from ctxfit.notes import is_sensitive_key

    for k in ("db_pass", "pass", "AWS_SECRET_ACCESS_KEY", "Authorization", "client_secret_x",
              "credentials", "ssh_private_key", "user-passwd", "password", "api-key"):  # fmt: skip
        assert is_sensitive_key(k), k
    for k in ("passenger", "bypass", "tokens", "max_tokens", "keys", "author"):
        assert not is_sensitive_key(k), k
    d = squeeze_digest("t", json.dumps({"db_pass": "x", "n": 1, "pad": "p" * 3000}))
    assert "db_pass = [redacted]" in d


def test_aws_secret_scrubbed():
    from ctxfit.notes import scrub_secrets

    out = scrub_secrets("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
    assert out == "aws_secret_access_key = [redacted]"


def test_deep_json_is_not_json_and_bigints_are_doubles():
    pad = '"' + "p" * 3000 + '"'
    assert "array of 1 items" in squeeze_digest("t", "[" * 256 + pad + "]" * 256)
    assert "array of" not in squeeze_digest("t", "[" * 257 + pad + "]" * 257)
    d = squeeze_digest("t", '{"h": ' + "9" * 5000 + ', "pad": "' + "q" * 3000 + '"}')
    assert "h = null" in d


def test_squeeze_recursion_falls_back_to_hide(monkeypatch):
    import ctxfit.notes as n

    def boom(tool, text, **kw):
        raise RecursionError

    monkeypatch.setattr(n, "squeeze_digest", boom)
    out, r = mask_observations(anthropic_convo(8), mask_style="squeeze")
    assert r.masked_outputs == 4 and r.squeezed_outputs == 0
    assert r.reasons[:2] == ["masked", "squeeze_failed"]


def test_too_deep_input_is_passed_through():
    v = "x"
    for _ in range(3000):
        v = {"a": v}
    msgs = [
        {"role": "user", "content": "t"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "b", "input": v}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]},
    ]
    out, r = keeper(high_water_tokens=2, low_water_tokens=1).fit(msgs)
    assert r.reasons == ["too_deep"] and out[1] is msgs[1]


def test_recall_error_message_uses_normalised_id():
    res = handle_recall({"type": "tool_use", "id": "x", "input": {"id": "  ABC  "}}, lambda i: None)
    assert res["content"] == "ctxfit_recall: no stored output with id 'abc'"
