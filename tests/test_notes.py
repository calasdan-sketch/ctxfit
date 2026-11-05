"""Tests for running notes: observation masking (layer 1) + NotesKeeper (layer 2)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from types import SimpleNamespace

import pytest

import ctxfit
from ctxfit import Message, normalize, validate_pairing
from ctxfit.cli import main
from ctxfit.notes import (
    SUMMARISER_SYSTEM_PROMPT,
    NotesKeeper,
    anthropic_summarizer,
    build_summariser_prompts,
    canonical_json,
    estimate_tokens,
    mask_boundary,
    mask_observations,
    openai_summarizer,
    scrub_secrets,
)
from ctxfit.notes import _js_number as js_number

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def anthropic_convo(turns: int, out_chars: int = 2000, task: str = "Task: fix the build"):
    """Turn 0 starts with the task; every turn: assistant tool_use -> result -> reply."""
    msgs = [{"role": "user", "content": task}]
    for t in range(turns):
        if t:
            msgs.append({"role": "user", "content": f"next step {t}"})
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"running step {t}"},
                    {
                        "type": "tool_use",
                        "id": f"tu{t}",
                        "name": "bash",
                        "input": {"cmd": f"ls {t}"},
                    },
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"tu{t}", "content": f"{t}" * out_chars}
                ],
            }
        )
        msgs.append({"role": "assistant", "content": f"done {t}"})
    return msgs


def openai_convo(turns: int, out_chars: int = 2000):
    msgs = [
        {"role": "system", "content": "You are a careful agent."},
        {"role": "user", "content": "Task: migrate the database"},
    ]
    for t in range(turns):
        if t:
            msgs.append({"role": "user", "content": f"next step {t}"})
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call{t}",
                        "type": "function",
                        "function": {"name": "sql", "arguments": json.dumps({"q": f"select {t}"})},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"call{t}", "content": f"{t}" * out_chars})
        msgs.append({"role": "assistant", "content": f"done {t}"})
    return msgs


class FakeSummarizer:
    """Deterministic summariser that records every call."""

    def __init__(self, text="Goal: ship it\nNext step: continue", tokens=None):
        self.text = text
        self.tokens = tokens
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        if self.tokens is not None:
            return (self.text, *self.tokens)
        return self.text


def keeper(summ=None, **kw):
    cfg = dict(high_water_tokens=3000, low_water_tokens=1500, notes_max_tokens=100)
    cfg.update(kw)
    return NotesKeeper(summ or FakeSummarizer(), **cfg)


def dumps(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def no_consecutive_roles(msgs, start=0):
    return all(msgs[i]["role"] != msgs[i + 1]["role"] for i in range(start, len(msgs) - 1))


# ---------------------------------------------------------------------------
# layer 1: masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "turns,keep,chunk,expected",
    [(0, 4, 4, 0), (3, 4, 4, 0), (4, 4, 4, 0), (7, 4, 4, 0), (8, 4, 4, 4), (11, 4, 4, 4),
     (12, 4, 4, 8), (10, 2, 3, 6), (5, 0, 1, 5)],
)  # fmt: skip
def test_mask_boundary_formula(turns, keep, chunk, expected):
    assert mask_boundary(turns, keep, chunk) == expected


def test_exact_marker_text_and_offload():
    msgs = anthropic_convo(8)
    out, report = mask_observations(msgs)
    original = "0" * 2000
    id8 = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    block = out[2]["content"][0]
    assert block == {
        "type": "tool_result",
        "tool_use_id": "tu0",
        "content": f"[ctxfit: tool output hidden - bash - 2000 chars - id {id8}]",
    }
    assert report.mask_boundary == 4
    assert report.masked_outputs == 4
    assert report.masked_chars == 8000
    assert report.offload[id8] == original
    assert report.tokens_after_est < report.tokens_before_est
    # turn 4 onwards untouched
    assert out[4 * 4 + 1 :] == msgs[4 * 4 + 1 :]


def test_chunked_boundary_prefix_is_byte_identical_until_it_moves():
    prev = None
    for turns in range(8, 13):
        out, report = mask_observations(anthropic_convo(turns))
        if prev is not None:
            prev_out, prev_b = prev
            same = dumps(out[: len(prev_out)]) == dumps(prev_out)
            assert same is (report.mask_boundary == prev_b), turns
        prev = (out, report.mask_boundary)
    # boundary moved exactly once in that range (4 -> 8 at T=12)
    assert prev[1] == 8


def test_boundary_not_yet_moved_masks_nothing():
    # 7 turns, keep 4, chunk 4 -> B = 0 although turns 0-2 are "old".
    out, report = mask_observations(anthropic_convo(7))
    assert report.mask_boundary == 0
    assert report.masked_outputs == 0
    assert out == anthropic_convo(7)


def test_min_mask_chars_threshold():
    msgs = anthropic_convo(8, out_chars=399)
    _, r = mask_observations(msgs)
    assert r.masked_outputs == 0
    msgs = anthropic_convo(8, out_chars=400)
    _, r = mask_observations(msgs)
    assert r.masked_outputs == 4


def test_anthropic_list_content_is_error_and_extra_keys_preserved():
    msgs = anthropic_convo(8)
    msgs[2]["content"][0] = {
        "type": "tool_result",
        "tool_use_id": "tu0",
        "is_error": True,
        "cache_control": {"type": "ephemeral"},
        "content": [
            {"type": "text", "text": "E" * 500},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
            },
        ],
    }
    out, _ = mask_observations(msgs)
    blk = out[2]["content"][0]
    text = "E" * 500 + "\n[non-text]"
    id8 = hashlib.sha256(text.encode()).hexdigest()[:8]
    assert blk["is_error"] is True
    assert blk["cache_control"] == {"type": "ephemeral"}
    assert blk["tool_use_id"] == "tu0"
    assert blk["content"] == [
        {
            "type": "text",
            "text": f"[ctxfit: tool output hidden - bash - {len(text)} chars - id {id8}]",
        }
    ]


def test_tool_calls_never_modified_and_pairing_preserved_anthropic():
    msgs = anthropic_convo(12)
    out, _ = mask_observations(msgs)
    for a, b in zip(msgs, out):
        if a["role"] == "assistant" and isinstance(a["content"], list):
            assert a == b
    assert validate_pairing(normalize(out, flavor="anthropic")) == []


def test_openai_masking_and_tool_name_from_call():
    msgs = openai_convo(8)
    out, report = mask_observations(msgs)
    tool = out[3]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call0"
    assert tool["content"].startswith("[ctxfit: tool output hidden - sql - 2000 chars - id ")
    assert report.masked_outputs == 4
    for a, b in zip(msgs, out):
        if a.get("tool_calls"):
            assert a == b
    assert validate_pairing(normalize(out, flavor="openai")) == []


def test_openai_tool_name_field_fallback_and_unknown_tool():
    msgs = openai_convo(8)
    msgs[3]["tool_call_id"] = "unknown"
    msgs[2]["tool_calls"][0]["id"] = "unknown2"
    msgs[3]["name"] = "legacy_fn"
    out, _ = mask_observations(msgs, flavor="openai")
    assert " - legacy_fn - " in out[3]["content"]
    del msgs[3]["name"]
    out, _ = mask_observations(msgs, flavor="openai")
    assert " - tool - " in out[3]["content"]


def test_pinned_tool_outputs_are_not_masked():
    msgs = openai_convo(8)
    msgs[3]["pinned"] = True
    out, report = mask_observations(msgs)
    assert out[3] == msgs[3]
    assert report.masked_outputs == 3


def test_pin_tag_in_tool_output_does_not_protect_it():
    # Untrusted tool output must not be able to pin itself.
    msgs = openai_convo(8)
    msgs[3]["content"] = "[PIN] " + msgs[3]["content"]
    out, _ = mask_observations(msgs)
    assert out[3]["content"].startswith("[ctxfit: tool output hidden")


def test_pin_tag_in_anthropic_user_message_protects_its_results():
    msgs = anthropic_convo(8)
    # A text block makes the message a turn start (no longer "purely tool
    # results"), so turns renumber: 9 turns, B = 4, results of old turns 1-2
    # are masked, the pinned message is not.
    msgs[2]["content"].append({"type": "text", "text": "keep this [Pin]"})
    out, report = mask_observations(msgs)
    assert report.turns == 9
    assert out[2] == msgs[2]
    assert report.masked_outputs == 2


def test_pinned_flag_protects_anthropic_tool_result_message():
    msgs = anthropic_convo(8)
    msgs[2]["pinned"] = True
    out, report = mask_observations(msgs)
    assert out[2] == msgs[2] and report.masked_outputs == 3


def test_mask_does_not_mutate_input():
    for msgs in (anthropic_convo(12), openai_convo(12)):
        snapshot = copy.deepcopy(msgs)
        out, _ = mask_observations(msgs)
        assert msgs == snapshot
        out[0]["content"] = "changed"
        assert msgs == snapshot


def test_mask_accepts_message_objects():
    msgs = [Message(role="user", content="task")]
    for t in range(8):
        msgs += [
            Message(role="assistant", content="{}", kind="tool_call", name="f", call_id=f"c{t}"),
            Message(role="tool", content="z" * 600, kind="tool_result", call_id=f"c{t}"),
            Message(role="user", content=f"u{t}"),
        ]
    out, report = mask_observations(msgs)
    assert all(isinstance(m, Message) for m in out)
    assert report.mask_boundary == 4
    assert out[2].content.startswith("[ctxfit: tool output hidden - f - 600 chars")


def test_mask_rejects_bad_config_and_flavor():
    with pytest.raises(ValueError):
        mask_observations([], mask_chunk_turns=0)
    with pytest.raises(ValueError):
        mask_observations([], keep_recent_turns=-1)
    with pytest.raises(ValueError):
        mask_observations([{"role": "user", "content": "x"}], flavor="langchain")


def test_mask_report_summary_and_dict():
    _, r = mask_observations(anthropic_convo(8))
    assert "masked 4 tool output(s)" in r.summary()
    d = r.to_dict()
    assert d["saved_tokens_est"] == r.tokens_before_est - r.tokens_after_est
    assert "offload" not in d and len(d["offload_ids"]) == 4
    assert "offload" in r.to_dict(include_offload=True)
    json.dumps(r.to_dict(include_offload=True))


# ---------------------------------------------------------------------------
# layer 2: notes
# ---------------------------------------------------------------------------


def test_below_high_water_no_summariser_call():
    s = FakeSummarizer()
    k = keeper(s, high_water_tokens=1_000_000, low_water_tokens=10)
    out, r = k.fit(anthropic_convo(12))
    assert s.calls == []
    assert not r.notes_applied
    assert "below_high_water" in r.reasons and "masked" in r.reasons
    assert out == mask_observations(anthropic_convo(12))[0]


def test_anthropic_cut_merges_notes_into_following_user_message():
    s = FakeSummarizer("Goal: fix build")
    msgs = anthropic_convo(10)
    out, r = keeper(s).fit(msgs)
    assert len(s.calls) == 1
    assert r.notes_applied and r.cut_turn == 6 and r.evicted_messages == 23
    # pinned first user message, then the merged turn-6 user message
    assert out[0] == msgs[0]
    assert out[1]["role"] == "user"
    assert out[1]["content"][0] == {
        "type": "text",
        "text": "[ctxfit running notes - replaces 23 earlier messages; pinned rules above "
        "still apply]\nGoal: fix build",
    }
    assert out[1]["content"][1] == {"type": "text", "text": "next step 6"}
    assert no_consecutive_roles(out, start=1)
    assert validate_pairing(normalize(out, flavor="anthropic")) == []
    assert r.tokens_after_est < r.tokens_before_est


def test_anthropic_merge_keeps_leading_tool_results_first():
    # Only reachable with a result whose call is not in the history (orphan),
    # since cuts never split a pair; the Messages API wants tool_result first.
    s = FakeSummarizer("n")
    msgs = anthropic_convo(10)
    msgs[24] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "external", "content": "ok"},
            {"type": "text", "text": "next step 6"},
        ],
    }
    out, r = keeper(s).fit(msgs)
    assert r.notes_applied and r.cut_turn == 6
    assert [b["type"] for b in out[1]["content"]] == ["tool_result", "text", "text"]
    assert out[1]["content"][1]["text"].startswith("[ctxfit running notes")


def test_openai_cut_notes_standalone_user_message():
    s = FakeSummarizer("Goal: migrate db")
    msgs = openai_convo(10)
    out, r = keeper(s).fit(msgs)
    assert r.notes_applied
    assert out[0] == msgs[0] and out[1] == msgs[1]  # system + first user pinned
    assert out[2] == {
        "role": "user",
        "content": "[ctxfit running notes - replaces "
        f"{r.evicted_messages} earlier messages; pinned rules above still apply]\nGoal: migrate db",
    }
    assert out[3] == {"role": "user", "content": f"next step {r.cut_turn}"}
    assert validate_pairing(normalize(out, flavor="openai")) == []


def test_pinned_message_before_cut_is_kept_with_its_pair():
    s = FakeSummarizer("n")
    msgs = anthropic_convo(10)
    msgs[5]["content"][0]["text"] = "rule: never push to main [pin]"  # turn 1 assistant
    out, r = keeper(s).fit(msgs)
    assert r.notes_applied
    assert out[1] == msgs[5]  # pinned call kept verbatim, right after the task
    # ...together with its result (masked by layer 1, but still paired)
    assert out[2]["content"][0]["tool_use_id"] == "tu1"
    assert out[2]["content"][0]["content"].startswith("[ctxfit: tool output hidden - bash")
    # pinned content never reaches the summariser
    assert "never push to main" not in s.calls[0][1]
    assert validate_pairing(normalize(out, flavor="anthropic")) == []


def test_pinned_flag_and_system_role_never_summarised():
    s = FakeSummarizer("n")
    msgs = openai_convo(10)
    msgs[5]["pinned"] = True
    msgs[5]["content"] = "PINNED USER RULE"
    msgs.insert(6, {"role": "developer", "content": "DEV RULE"})
    out, r = keeper(s).fit(msgs)
    assert r.notes_applied
    user_prompt = s.calls[0][1]
    for secret_rule in ("PINNED USER RULE", "DEV RULE", "careful agent", "migrate the database"):
        assert secret_rule not in user_prompt
    assert {"role": "developer", "content": "DEV RULE"} in out


def test_prefix_hash_reuse_calls_summariser_zero_times_and_is_cache_stable():
    s = FakeSummarizer("notes v1")
    k = keeper(s)
    msgs = anthropic_convo(10)
    out1, r1 = k.fit(msgs)
    assert len(s.calls) == 1 and r1.cache_stable is None
    grown = msgs + [{"role": "user", "content": "one more thing"}]
    out2, r2 = k.fit(grown)
    assert len(s.calls) == 1  # no model call
    assert r2.state_reused and r2.summariser_calls == 0
    assert r2.cache_stable is True
    assert dumps(out2[: len(out1)]) == dumps(out1)


def test_history_fork_resets_state():
    s = FakeSummarizer("notes")
    k = keeper(s)
    msgs = anthropic_convo(10)
    k.fit(msgs)
    forked = copy.deepcopy(msgs)
    forked[1]["content"][0]["text"] = "edited history"
    s.text = "different notes"
    out, r = k.fit(forked)
    assert "history_forked" in r.reasons
    assert len(s.calls) == 2
    assert s.calls[1][1].startswith("CURRENT NOTES:\n(none yet)\n\nNEW EVENTS:\n")
    assert r.cache_stable is False


def test_incremental_update_passes_previous_notes_and_only_new_events():
    s = FakeSummarizer("notes v1")
    k = keeper(s, high_water_tokens=2500, low_water_tokens=1200)
    k.fit(anthropic_convo(10))
    first_cut = k.cut_turn
    s.text = "notes v2"
    out, r = k.fit(anthropic_convo(16))
    assert k.cut_turn > first_cut and r.previous_cut_turn == first_cut
    user = s.calls[1][1]
    assert user.startswith("CURRENT NOTES:\nnotes v1\n\nNEW EVENTS:\n")
    assert "running step 0" not in user  # already summarised
    assert f"running step {first_cut}" in user
    assert "notes v2" in json.dumps(out)
    assert k.state_dict()["summariser_calls"] == 2


@pytest.mark.parametrize(
    "behaviour,reason",
    [("raise", "exception"), ("empty", "empty"), ("oversized", "oversized"), ("none", "empty")],
)
def test_summariser_failure_keeps_previous_cut(behaviour, reason):
    s = FakeSummarizer("notes v1")
    k = keeper(s, high_water_tokens=2500, low_water_tokens=1200)
    out1, _ = k.fit(anthropic_convo(10))
    cut, notes = k.cut_turn, k.notes

    def bad(system, user):
        if behaviour == "raise":
            raise RuntimeError("boom")
        if behaviour == "empty":
            return "   \n"
        if behaviour == "none":
            return None
        return "x" * (4 * 2 * 100 + 4)  # > 2 x notes_max_tokens

    k.summarizer = bad
    msgs = anthropic_convo(16)
    out, r = k.fit(msgs)
    assert r.summariser_failed and r.failure_reason.startswith(reason)
    assert "summariser_failed" in r.reasons
    assert k.cut_turn == cut and k.notes == notes
    assert r.notes_applied and r.cut_turn == cut
    assert out[:2] == out1[:2]  # same pinned prefix + same notes, same cut
    assert validate_pairing(normalize(out, flavor="anthropic")) == []
    if behaviour == "raise":
        assert r.summariser_input_tokens == 0
    else:
        assert r.summariser_input_tokens > 0  # call happened: spend is real


def test_summariser_failure_without_state_falls_back_to_masking_only():
    def boom(system, user):
        raise ValueError("down")

    msgs = anthropic_convo(12)
    out, r = keeper(boom).fit(msgs)
    assert r.summariser_failed and not r.notes_applied
    assert out == mask_observations(msgs)[0]


def test_summariser_timeout():
    def slow(system, user):
        time.sleep(0.5)
        return "late"

    k = keeper(slow, summarizer_timeout_s=0.05)
    out, r = k.fit(anthropic_convo(10))
    assert r.summariser_failed and r.failure_reason == "timeout"
    assert k.cut_turn == 0 and not r.notes_applied
    assert r.summariser_tokens_estimated


def test_no_cut_allowed_when_too_few_turns():
    s = FakeSummarizer()
    out, r = keeper(s).fit(anthropic_convo(5, out_chars=20_000))
    assert "no_cut_allowed" in r.reasons
    assert s.calls == [] and not r.notes_applied


def test_low_water_reached_picks_earliest_boundary():
    from ctxfit.notes import _message_chars

    msgs = anthropic_convo(10)
    masked, _ = mask_observations(msgs)

    def est(c):  # pinned task + masked turns >= c, plus the notes allowance
        kept = [masked[0]] + masked[4 * c :]
        return estimate_tokens(sum(_message_chars("anthropic", m) for m in kept)) + 100

    low = 2700
    expected = min(c for c in range(1, 7) if est(c) <= low)
    assert expected == 5  # turns 1-4 are not enough, 5 is, 6 would over-evict
    s = FakeSummarizer("n")
    _, r = keeper(s, high_water_tokens=3000, low_water_tokens=low, min_evict_turns=1).fit(msgs)
    assert r.cut_turn == expected
    assert "low_water_not_reached" not in r.reasons


def test_low_water_unreachable_uses_latest_allowed_boundary():
    s = FakeSummarizer("n")
    _, r = keeper(s, high_water_tokens=3000, low_water_tokens=200, min_evict_turns=1).fit(
        anthropic_convo(10)
    )
    assert r.cut_turn == 6 and "low_water_not_reached" in r.reasons


def test_secret_scrubbing_keys_never_reach_summariser():
    secrets = [
        "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX",
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a1" * 18,
        "github_pat_" + "B2_" * 10,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        "hunter2secret",
    ]
    msgs = anthropic_convo(10)
    msgs[2]["content"][0]["content"] = (
        "env:\n" + "\n".join(secrets[:-1]) + "\npassword: hunter2secret"
    )
    msgs[1]["content"][1]["input"] = {"password": "hunter2secret", "token": secrets[0]}
    s = FakeSummarizer("n")
    keeper(s).fit(msgs)
    system, user = s.calls[0]
    for sec in secrets:
        for line in sec.splitlines():
            if len(line) > 12 and "PRIVATE KEY" not in line:
                assert line not in user, line
        assert sec not in user
    assert "[redacted]" in user


@pytest.mark.parametrize(
    "text,expected",
    [
        ("key sk-ant-abc123def456ghi789 end", "key [redacted] end"),
        ("task-abcdefghijklmnopqrstuv stays", "task-abcdefghijklmnopqrstuv stays"),
        ("AKIA1234567890ABCDEF", "[redacted]"),
        ("Password = s3cr3t! next", "[redacted] next"),
        ('{"password": "x1"}', '{"[redacted]'),
        ("plain text", "plain text"),
    ],
)
def test_scrub_secrets_patterns(text, expected):
    assert scrub_secrets(text) == expected


def test_net_savings_subtract_provider_reported_summariser_tokens():
    s = FakeSummarizer("notes", tokens=(1234, 56))
    _, r = keeper(s).fit(anthropic_convo(10))
    assert r.summariser_input_tokens == 1234 and r.summariser_output_tokens == 56
    assert not r.summariser_tokens_estimated
    assert r.net_saved_tokens_est == r.tokens_before_est - r.tokens_after_est - 1234 - 56
    assert r.to_dict()["net_saved_tokens_est"] == r.net_saved_tokens_est


def test_estimated_summariser_tokens_are_flagged_and_exact():
    s = FakeSummarizer("12345678")
    _, r = keeper(s).fit(anthropic_convo(10))
    system, user = s.calls[0]
    assert r.summariser_tokens_estimated
    assert r.summariser_input_tokens == math.ceil((len(system) + len(user)) / 4)
    assert r.summariser_output_tokens == 2
    assert "(estimated)" in r.summary()


def test_summariser_prompt_constants():
    system, user = build_summariser_prompts(None, "[user] hi", 1500)
    assert system.startswith(
        "You maintain compact running notes for an AI agent's long conversation."
    )
    assert system.endswith("Stay under 1500 tokens.")
    assert "{" not in system
    assert user == "CURRENT NOTES:\n(none yet)\n\nNEW EVENTS:\n[user] hi"
    assert "{notes_max_tokens}" in SUMMARISER_SYSTEM_PROMPT


def test_render_caps_args_and_results():
    s = FakeSummarizer("n")
    msgs = anthropic_convo(10, out_chars=3000)
    msgs[1]["content"][1]["input"] = {"blob": "q" * 2000}
    keeper(s).fit(msgs)
    user = s.calls[0][1]
    line_call = next(x for x in user.splitlines() if x.startswith("[assistant -> bash] "))
    assert len(line_call) == len("[assistant -> bash] ") + 500
    line_res = next(x for x in user.splitlines() if x.startswith("[bash result] "))
    assert len(line_res) == len("[bash result] ") + 1500
    assert "[assistant] running step 0" in user


def test_state_round_trip_through_json():
    s = FakeSummarizer("persisted notes")
    k1 = keeper(s)
    msgs = anthropic_convo(10)
    out1, _ = k1.fit(msgs)
    blob = json.dumps(k1.state_dict())
    s2 = FakeSummarizer("SHOULD NOT BE CALLED")
    k2 = keeper(s2)
    k2.load_state(json.loads(blob))
    out2, r2 = k2.fit(msgs)
    assert s2.calls == []
    assert out2 == out1 and r2.state_reused and r2.cache_stable is True
    assert k2.state_dict() == json.loads(blob) | {"updated_at": k2.state_dict()["updated_at"]}


def test_load_state_rejects_bad_input():
    k = keeper()
    with pytest.raises(ValueError):
        k.load_state({"version": 2})
    with pytest.raises(ValueError):
        k.load_state({"version": 1, "cut_turn": -1})
    with pytest.raises(ValueError):
        k.load_state({"version": 1, "cut_turn": 3, "notes": None, "prefix_hash": "x"})
    with pytest.raises(ValueError):
        k.load_state("nope")  # type: ignore[arg-type]


def test_state_dict_fields():
    k = keeper(clock=lambda: __import__("datetime").datetime(2026, 11, 5))
    k.fit(anthropic_convo(10))
    st = k.state_dict()
    assert set(st) >= {
        "version", "cut_turn", "prefix_hash", "notes", "notes_tokens", "evicted_messages",
        "summariser_calls", "summariser_input_tokens", "summariser_output_tokens", "updated_at",
    }  # fmt: skip
    assert st["version"] == 1 and st["updated_at"] == "2026-11-05T00:00:00"
    assert len(st["prefix_hash"]) == 64


def test_fit_does_not_mutate_input():
    for msgs in (anthropic_convo(12), openai_convo(12)):
        snap = copy.deepcopy(msgs)
        out, _ = keeper().fit(msgs)
        assert msgs == snap
        out[1]["content"] = "changed"
        assert msgs == snap


def test_system_param_counted_never_returned():
    msgs = anthropic_convo(6, out_chars=100)
    k = keeper(high_water_tokens=10_000_000, low_water_tokens=10)
    _, r0 = k.fit(msgs)
    out, r1 = keeper(high_water_tokens=10_000_000, low_water_tokens=10).fit(
        msgs, system=[{"type": "text", "text": "S" * 400}]
    )
    assert r1.tokens_before_est == r0.tokens_before_est + 100
    assert all(m.get("role") != "system" for m in out)


def test_offload_callback_and_recall():
    seen = {}
    k = keeper(
        offload=lambda i, t: seen.setdefault(i, t), high_water_tokens=10**7, low_water_tokens=1
    )
    _, r = k.fit(anthropic_convo(8))
    assert r.masked_outputs == 4 and set(seen) == set(r.offload)
    for id8, text in seen.items():
        assert k.recall(id8) == text


def test_offload_callback_failure_never_fails_request():
    def broken(i, t):
        raise OSError("disk full")

    k = keeper(offload=broken, high_water_tokens=10**7, low_water_tokens=1)
    _, r = k.fit(anthropic_convo(8))
    assert any(x.startswith("offload_callback_failed") for x in r.reasons)


def test_keeper_accepts_message_objects():
    msgs = normalize(openai_convo(10), flavor="openai")
    out, r = keeper().fit(msgs)
    assert all(isinstance(m, Message) for m in out)
    assert r.notes_applied


def test_keeper_config_validation():
    with pytest.raises(ValueError):
        NotesKeeper(FakeSummarizer(), high_water_tokens=10, low_water_tokens=10)
    with pytest.raises(ValueError):
        NotesKeeper(FakeSummarizer(), min_evict_turns=0)
    with pytest.raises(ValueError):
        NotesKeeper(FakeSummarizer(), notes_max_tokens=0)
    with pytest.raises(TypeError):
        NotesKeeper("not callable")  # type: ignore[arg-type]


def test_notes_report_summary_and_dict():
    _, r = keeper(FakeSummarizer("n")).fit(anthropic_convo(10))
    text = r.summary()
    assert "running notes replace" in text and "net saved" in text
    d = r.to_dict()
    json.dumps(d)
    for key in (
        "tokens_before_est", "tokens_after_est", "masked_outputs", "masked_chars",
        "notes_applied", "evicted_messages", "summariser_calls", "summariser_input_tokens",
        "summariser_output_tokens", "net_saved_tokens_est", "cache_stable", "reasons",
    ):  # fmt: skip
        assert key in d


def test_reset_forgets_notes():
    k = keeper(FakeSummarizer("n"))
    k.fit(anthropic_convo(10))
    assert k.notes == "n"
    k.reset()
    assert k.notes is None and k.cut_turn == 0


# ---------------------------------------------------------------------------
# summariser factories (fake clients; no SDK imports)
# ---------------------------------------------------------------------------


def test_anthropic_summarizer_factory():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="notes!")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    fn = anthropic_summarizer(client)
    assert fn("SYS", "USER") == ("notes!", 10, 3)
    assert captured["model"] == "claude-haiku-5-5"
    assert captured["system"] == "SYS"
    assert captured["messages"] == [{"role": "user", "content": "USER"}]


def test_openai_summarizer_factory():
    captured = {}

    def create(**kw):
        captured.update(kw)
        return {
            "choices": [{"message": {"content": "notes?"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fn = openai_summarizer(client, model="cheap-model", max_completion_tokens=99)
    assert fn("SYS", "USER") == ("notes?", 7, 2)
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["max_completion_tokens"] == 99


# ---------------------------------------------------------------------------
# shared definitions with the TypeScript port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, "1"), (0.5, "0.5"), (1e21, "1e+21"), (1e-7, "1e-7"), (1e-6, "0.000001"),
     (123456789012345680000.0, "123456789012345680000"), (-2.5e-10, "-2.5e-10"), (0.1, "0.1")],
)  # fmt: skip
def test_js_number_formatting(value, expected):
    assert js_number(value) == expected


def test_canonical_json_sorted_compact():
    assert canonical_json({"b": 1, "a": [1.0, "é", None, True]}) == '{"a":[1,"é",null,true],"b":1}'
    assert estimate_tokens(0) == 0 and estimate_tokens(1) == 1 and estimate_tokens(8) == 2


def test_top_level_exports():
    for name in ("mask_observations", "NotesKeeper", "NotesReport", "MaskReport",
                 "anthropic_summarizer", "openai_summarizer", "scrub_secrets"):  # fmt: skip
        assert hasattr(ctxfit, name)
    assert ctxfit.__version__ == "0.2.0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_mask(tmp_path, capsys):
    src = tmp_path / "c.json"
    src.write_text(json.dumps(anthropic_convo(8)), encoding="utf-8")
    out = tmp_path / "o.json"
    rep = tmp_path / "r.json"
    rc = main(["mask", str(src), "--out", str(out), "--report", str(rep)])
    assert rc == 0
    assert "masked 4 tool output(s)" in capsys.readouterr().out
    assert json.loads(rep.read_text())["masked_outputs"] == 4
    assert "[ctxfit: tool output hidden" in out.read_text()


def test_keeper_cache_stable_until_mask_boundary_moves():
    k = keeper(high_water_tokens=10**7, low_water_tokens=1)
    results = []
    for turns in range(8, 13):
        _, r = k.fit(anthropic_convo(turns))
        results.append((turns, r.mask_boundary, r.cache_stable))
    # T=8..11 share B=4 -> stable; T=12 moves B to 8 -> one cache break
    assert results[0][2] is None
    assert [c for _, _, c in results[1:4]] == [True, True, True]
    assert results[4][1] == 8 and results[4][2] is False
