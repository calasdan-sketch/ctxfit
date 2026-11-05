"""Probe eval: does a strategy keep what the model needs? (spec v6 #4, ``ctxfit eval``)

For each replayed conversation (see :mod:`ctxfit.replay`) take K checkpoint
requests. At each, find a tool output at least 8 turns old that holds a fact
token (rule I: a path, URL, code, version, id), blank that token out of one
of its lines, and ask a cheap model to fill the blank twice: once with the
full original context, once with the context the strategy would have sent.
Exact match scores both. ``--judge`` adds Factory-style open questions
(recall, artifact, continuation, decision) answered with both contexts and
graded 0-5 by the model against the full-context answer.

Calls cost money, so the run first prints an estimate in USD and refuses to
start without ``--max-usd`` at or above it. ``--batch`` sends everything
through the Message Batches API (half price, slower).
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .notes import (
    NotesConfig,
    _content_text,
    _Conversation,
    _estimate,
    estimate_tokens,
    extract_facts,
    finish_context,
    plan_context,
    scrub_secrets,
)
from .replay import STRATEGIES, Request, find_price, load_corpus, load_prices

MIN_AGE_TURNS = 8
CLOZE_MAX_TOKENS = 32
CLOZE_OUT_EST = 8
ANSWER_MAX_TOKENS = 300
ANSWER_OUT_EST = 120
GRADE_MAX_TOKENS = 4
GRADE_OUT_EST = 2
BLANK = "____"

CLOZE_QUESTION = (
    "Earlier in this conversation a tool output contained this line, with one value replaced "
    "by {blank}:\n{line}\nWhat exactly goes in the blank? Reply with only the missing text. "
    "If you cannot tell from the conversation, reply UNKNOWN."
)
JUDGE_QUESTIONS = {
    "recall": "What was the user's original task, in one sentence?",
    "artifact": "List every file path that has been read or changed so far.",
    "continuation": "What should happen next, in one or two sentences?",
    "decision": "What was the most recent decision taken, and why?",
}
GRADE_PROMPT = (
    "Reference answer:\n{reference}\n\nCandidate answer:\n{candidate}\n\nScore the candidate "
    "from 0 (unrelated or wrong) to 5 (same facts as the reference). Reply with one digit."
)


class EvalBudgetError(RuntimeError):
    """The estimated cost is not covered by --max-usd."""


@dataclass
class Call:
    """One model call: a conversation plus a question appended as the last user turn."""

    key: str
    messages: list[dict[str, Any]]
    max_tokens: int
    out_est: int


Backend = Callable[[list[Call]], dict[str, tuple[str, int, int]]]
"""calls -> {key: (text, input_tokens, output_tokens)}"""


def _ask(messages: Sequence[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """The context with ``question`` added to the final user turn (alternation stays valid)."""
    msgs = copy.deepcopy(list(messages))
    block = {"type": "text", "text": question}
    if msgs and msgs[-1].get("role") == "user":
        content = msgs[-1].get("content")
        if isinstance(content, list):
            msgs[-1]["content"] = [*content, block]
        else:
            msgs[-1]["content"] = [{"type": "text", "text": str(content or "")}, block]
    else:
        msgs.append({"role": "user", "content": [block]})
    return msgs


def _normal(answer: str) -> str:
    return answer.strip().strip("`'\"").strip()


@dataclass
class Probe:
    conversation: str
    request_index: int
    fact: str
    line: str


def find_probe(req: Request) -> Probe | None:
    """The cloze probe of one checkpoint: the oldest >= 8-turn-old output with a usable fact."""
    conv = _Conversation("anthropic", req.messages)
    for v in conv.views:
        if conv.turn_count - v.turn < MIN_AGE_TURNS + 1:
            continue
        for _cid, text in v.results:
            for fact in extract_facts(scrub_secrets(text)):
                for line in text.split("\n"):
                    if fact in line and BLANK not in line:
                        cloze = scrub_secrets(line.replace(fact, BLANK, 1))
                        if BLANK in cloze and fact not in cloze:
                            return Probe(req.conversation, 0, fact, cloze)
    return None


def _checkpoints(n: int, k: int) -> list[int]:
    return sorted({math.floor((j + 1) * n / (k + 1)) for j in range(k)} - {n}) if n else []


@dataclass
class Plan:
    calls: list[Call] = field(default_factory=list)
    probes: list[tuple[Probe, str, str]] = field(
        default_factory=list
    )  # (probe, full key, strategy key)
    judged: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (kind, full key, strategy key)


def build_plan(requests: Sequence[Request], strategy: str, checkpoints: int, judge: bool) -> Plan:
    modes = {name: (mode, over) for name, mode, over in STRATEGIES}
    if strategy not in modes or modes[strategy][0] is None:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose one of {[n for n in modes if n != 'none']}"
        )
    mode, over = modes[strategy]
    cfg = NotesConfig(**over)
    by_conv: dict[str, list[Request]] = {}
    for r in requests:
        by_conv.setdefault(r.conversation, []).append(r)
    plan = Plan()
    for conv in sorted(by_conv, key=lambda c: (min(r.ts_ms for r in by_conv[c]), c)):
        reqs = sorted(by_conv[conv], key=lambda r: (r.ts_ms, r.message_id))
        wanted = set(_checkpoints(len(reqs), checkpoints))
        state = None
        for i, r in enumerate(reqs):
            res = finish_context(
                plan_context(
                    r.messages,
                    flavor="anthropic",
                    mode=mode,
                    config=cfg,
                    state=state,
                    now=r.ts_ms / 1000,
                ),  # fmt: skip
                None,
            )
            state = res.state
            if i not in wanted:
                continue
            tag = f"{conv[:8]}-{i}"
            probe = find_probe(r)
            if probe is not None:
                probe.request_index = i
                q = CLOZE_QUESTION.format(blank=BLANK, line=probe.line)
                full = Call(
                    f"{tag}-cloze-full", _ask(r.messages, q), CLOZE_MAX_TOKENS, CLOZE_OUT_EST
                )
                strat = Call(
                    f"{tag}-cloze-{strategy}",
                    _ask(res.messages, q),
                    CLOZE_MAX_TOKENS,
                    CLOZE_OUT_EST,
                )
                plan.calls += [full, strat]
                plan.probes.append((probe, full.key, strat.key))
            if judge:
                for kind, q in JUDGE_QUESTIONS.items():
                    full = Call(
                        f"{tag}-{kind}-full", _ask(r.messages, q), ANSWER_MAX_TOKENS, ANSWER_OUT_EST
                    )
                    strat = Call(
                        f"{tag}-{kind}-{strategy}",
                        _ask(res.messages, q),
                        ANSWER_MAX_TOKENS,
                        ANSWER_OUT_EST,
                    )
                    plan.calls += [full, strat]
                    plan.judged.append((kind, full.key, strat.key))
    return plan


def _usd(
    table: dict[str, Any], model: str, tokens_in: float, tokens_out: float, batch: bool
) -> float:
    price = find_price(table, model)
    if price is None:
        raise ValueError(f"no price for eval model {model!r}")
    mult = table["providers"]["anthropic"].get("batch_multiplier", 0.5) if batch else 1
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6 * mult


def estimate_usd(
    plan: Plan, table: dict[str, Any], model: str, batch: bool, hidden: int = 0
) -> float:
    """Estimated cost: every call's input (chars/4 + an allowance for the system prompt) and
    typical output, plus one grading call per judged question."""
    tin = sum(hidden + _estimate("anthropic", c.messages, 0) for c in plan.calls)
    tout = sum(c.out_est for c in plan.calls)
    for _kind, _f, _s in plan.judged:
        tin += estimate_tokens(len(GRADE_PROMPT)) + 2 * ANSWER_OUT_EST
        tout += GRADE_OUT_EST
    return _usd(table, model, tin, tout, batch)


def run_eval(
    paths: Sequence[str],
    backend: Backend,
    *,
    model: str = "claude-haiku-4-5",
    strategy: str = "mask+arrival",
    checkpoints: int = 3,
    judge: bool = False,
    max_usd: float | None = None,
    batch: bool = False,
    prices: dict[str, Any] | None = None,
    on_estimate: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Build the probes, check the budget, make the calls, score them."""
    if checkpoints < 1:
        raise ValueError("checkpoints must be >= 1")
    table = prices if prices is not None else load_prices()
    requests = load_corpus(list(paths)).requests
    plan = build_plan(requests, strategy, checkpoints, judge)
    est = estimate_usd(plan, table, model, batch)
    if on_estimate is not None:
        on_estimate(est)
    if max_usd is None:
        raise EvalBudgetError(f"estimated cost ${est:.4f}: pass --max-usd to run")
    if est > max_usd:
        raise EvalBudgetError(f"estimated cost ${est:.4f} is over --max-usd {max_usd}")
    answers = backend(plan.calls) if plan.calls else {}
    spent_in = sum(a[1] for a in answers.values())
    spent_out = sum(a[2] for a in answers.values())
    full_ok = sum(_normal(answers[f][0]) == p.fact for p, f, _s in plan.probes)
    strat_ok = sum(_normal(answers[s][0]) == p.fact for p, _f, s in plan.probes)
    judge_out: dict[str, Any] | None = None
    if judge and plan.judged:
        grades = [
            Call(
                f"grade-{s}",
                [
                    {
                        "role": "user",
                        "content": GRADE_PROMPT.format(
                            reference=answers[f][0], candidate=answers[s][0]
                        ),
                    }
                ],  # fmt: skip
                GRADE_MAX_TOKENS,
                GRADE_OUT_EST,
            )
            for _k, f, s in plan.judged
        ]
        graded = backend(grades)
        spent_in += sum(a[1] for a in graded.values())
        spent_out += sum(a[2] for a in graded.values())
        by_kind: dict[str, list[int]] = {}
        for (kind, _f, s), g in zip(plan.judged, grades):
            m = re.search(r"[0-5]", graded[g.key][0])
            by_kind.setdefault(kind, []).append(int(m.group()) if m else 0)
        judge_out = {k: round(sum(v) / len(v), 2) for k, v in by_kind.items()}
    n = len(plan.probes)
    return {
        "eval": 1,
        "model": model,
        "strategy": strategy,
        "batch": batch,
        "probes": n,
        "cloze": {
            "full_correct": full_ok,
            "strategy_correct": strat_ok,
            "full_score": round(full_ok / n, 3) if n else None,
            "strategy_score": round(strat_ok / n, 3) if n else None,
        },
        "judge": judge_out,
        "estimated_usd": round(est, 6),
        "spent_usd": round(_usd(table, model, spent_in, spent_out, batch), 6),
        "probe_details": [
            {
                "conversation": p.conversation[:8],
                "request": p.request_index,
                "fact": p.fact,
                "full": _normal(answers[f][0]),
                "strategy": _normal(answers[s][0]),
            }  # fmt: skip
            for p, f, s in plan.probes
        ],
    }


# ---------------------------------------------------------------------------
# Anthropic backends (no SDK import: pass the client in)
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _reply(resp: Any) -> tuple[str, int, int]:
    text = "".join(
        str(_get(b, "text", "")) for b in (_get(resp, "content") or []) if _get(b, "type") == "text"
    )
    usage = _get(resp, "usage")
    return text, int(_get(usage, "input_tokens", 0) or 0), int(_get(usage, "output_tokens", 0) or 0)


def anthropic_backend(
    client: Any,
    model: str,
    *,
    batch: bool = False,
    poll_s: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Backend:
    """Calls through an ``anthropic.Anthropic`` client, one by one or as one Message Batch."""

    def params(c: Call) -> dict[str, Any]:
        return {"model": model, "max_tokens": c.max_tokens, "messages": c.messages}

    def direct(calls: list[Call]) -> dict[str, tuple[str, int, int]]:
        return {c.key: _reply(client.messages.create(**params(c))) for c in calls}

    def batched(calls: list[Call]) -> dict[str, tuple[str, int, int]]:
        ids = {f"p{i}": c.key for i, c in enumerate(calls)}
        job = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": params(c)} for cid, c in zip(ids, calls)]
        )
        while _get(job, "processing_status") != "ended":
            sleep(poll_s)
            job = client.messages.batches.retrieve(_get(job, "id"))
        out: dict[str, tuple[str, int, int]] = {}
        for item in client.messages.batches.results(_get(job, "id")):
            result = _get(item, "result")
            key = ids[_get(item, "custom_id")]
            if _get(result, "type") == "succeeded":
                out[key] = _reply(_get(result, "message"))
            else:
                out[key] = ("", 0, 0)
        for key in ids.values():
            out.setdefault(key, ("", 0, 0))
        return out

    return batched if batch else direct


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def context_text(messages: Sequence[dict[str, Any]]) -> str:
    """All text of a context (for fake models in tests and demos)."""
    parts: list[str] = []
    for m in messages:
        content = m.get("content")
        for b in content if isinstance(content, list) else [content]:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "tool_result":
                    parts.append(_content_text(b.get("content")))
                elif b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
    return "\n".join(parts)
