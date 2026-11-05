"""Replay synthetic agent transcripts through ctxfit running notes and report honest numbers.

    python benchmarks/running_notes_bench.py            # print + write benchmarks/RESULTS.md

Everything here is SYNTHETIC and deterministic (fixed seeds): the transcripts are
generated, not recorded, and the "summariser" is a fake that returns a fixed-size
digest. The numbers show what the mechanics do to prompt size and cache stability
on transcripts shaped like real agent runs; they say nothing about task quality.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from ctxfit import __version__  # noqa: E402
from ctxfit.notes import (  # noqa: E402
    NotesConfig,
    SummaryOutcome,
    _estimate,
    canonical_json,
    finish_context,
    plan_context,
    scrub_secrets,
)

FAKE_NOTES = ("Goal: finish the task. Facts: " + "fact; " * 200)[:1200]

WORDS = (
    "the of and to in is that for it as with was on be by this are from at or an "
    "which data model system result value test file error request user agent api "
    "config service report cache token budget price market growth research study"
).split()


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def python_file(rng: random.Random, name: str) -> str:
    out = [f'"""{name}: {words(rng, 12)}."""', "", "import json", "import os", ""]
    for f in range(rng.randint(8, 18)):
        out.append(f"def func_{f}(arg, *, flag=False):")
        out.append(f'    """{words(rng, 10)}."""')
        for _ in range(rng.randint(3, 9)):
            out.append(f"    value = arg.get('{rng.choice(WORDS)}', {rng.randint(0, 99)})")
        out.append("    return value")
        out.append("")
    return "\n".join(out)


def test_log(rng: random.Random, failing: bool) -> str:
    lines = ["============ test session starts ============", "platform linux -- Python 3.13"]
    n = rng.randint(40, 90)
    lines += [
        f"tests/test_{rng.choice(WORDS)}_{i}.py::test_{rng.choice(WORDS)} PASSED" for i in range(n)
    ]
    if failing:
        lines += [
            "tests/test_api.py::test_fetch FAILED",
            "E   AssertionError: expected 200, got 500",
        ]
        lines += [f"    at line {rng.randint(1, 400)}" for _ in range(6)]
    lines += [
        f"======= {'1 failed, ' if failing else ''}{n} passed in {rng.random() * 9:.2f}s ======="
    ]
    return "\n".join(lines)


def api_json(rng: random.Random) -> str:
    rows = [
        {"id": i, "name": f"{rng.choice(WORDS)}-{i}", "price": round(rng.random() * 100, 2),
         "tags": [rng.choice(WORDS) for _ in range(3)], "active": rng.random() > 0.3}
        for i in range(rng.randint(20, 60))
    ]  # fmt: skip
    return json.dumps(rows, indent=2)


def web_page(rng: random.Random) -> str:
    paras = [words(rng, rng.randint(60, 140)).capitalize() + "." for _ in range(rng.randint(8, 16))]
    return "\n\n".join(paras)


def search_results(rng: random.Random) -> str:
    return json.dumps(
        {"query": words(rng, 4), "results": [
            {"title": words(rng, 6), "url": f"https://example.com/{rng.randint(1, 9999)}",
             "snippet": words(rng, 30)} for _ in range(10)]}, indent=2,
    )  # fmt: skip


def coding_transcript(seed: int, turns: int = 32) -> list[tuple[str, dict[str, Any], str]]:
    rng = random.Random(seed)
    files = [f"app/mod_{i}.py" for i in range(8)]
    contents = {f: python_file(rng, f) for f in files}
    steps: list[tuple[str, dict[str, Any], str]] = []
    for t in range(turns):
        kind = rng.random()
        if kind < 0.45:
            f = rng.choice(files)  # the same files come back: re-reads happen
            steps.append(("read_file", {"path": f}, contents[f]))
        elif kind < 0.75:
            steps.append(("bash", {"command": "pytest -q"}, test_log(rng, rng.random() < 0.4)))
        else:
            steps.append(
                ("http_get", {"url": f"https://api.example.com/items?page={t}"}, api_json(rng))
            )
    return steps


def research_transcript(seed: int, turns: int = 26) -> list[tuple[str, dict[str, Any], str]]:
    rng = random.Random(seed)
    pages = [f"https://example.com/article/{i}" for i in range(10)]
    bodies = {p: web_page(rng) for p in pages}
    steps: list[tuple[str, dict[str, Any], str]] = []
    for t in range(turns):
        if rng.random() < 0.3:
            steps.append(("web_search", {"q": words(rng, 4)}, search_results(rng)))
        else:
            p = rng.choice(pages)
            steps.append(("web_fetch", {"url": p}, bodies[p]))
    return steps


def to_messages(steps: list[tuple[str, dict[str, Any], str]], task: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for t, (tool, inp, out) in enumerate(steps):
        if t:
            msgs.append({"role": "user", "content": f"Continue (step {t})."})
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"Step {t}: calling {tool}."},
            {"type": "tool_use", "id": f"toolu_{t:03d}", "name": tool, "input": inp}]})  # fmt: skip
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"toolu_{t:03d}", "content": out}]})  # fmt: skip
        msgs.append({"role": "assistant", "content": f"Step {t} done: {out[:60]!r}"})
    return msgs


@dataclass
class Totals:
    requests: int = 0
    before: int = 0
    after: int = 0
    summariser: int = 0
    summariser_calls: int = 0
    stable: int = 0
    comparable: int = 0
    rereads_hidden: int = 0
    rereads: int = 0
    last_after: int = 0
    cost: float = 0.0  # cache-priced input cost, in plain-input-token units
    facts_total: int = 0
    facts_visible: int = 0
    notes: list[str] = field(default_factory=list)


CACHE_WRITE, CACHE_READ = 1.25, 0.1  # Anthropic 5-minute cache prices vs plain input
SUMMARY_OUT_PRICE = 5.0  # output tokens typically cost ~5x input


def prompt_text(messages: list[dict[str, Any]]) -> str:
    """All text the model would read in these messages (tool inputs as JSON)."""
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
            continue
        for b in c or []:
            if b.get("type") == "text":
                parts.append(b["text"])
            elif b.get("type") == "tool_use":
                parts.append(json.dumps(b["input"], ensure_ascii=False))
            elif b.get("type") == "tool_result":
                rc = b.get("content")
                parts.append(
                    rc if isinstance(rc, str) else "\n".join(x.get("text", "") for x in rc)
                )
    return "\n".join(parts)


def all_facts(steps) -> set[str]:
    """Every distinct fact token (rule I tokenizer, no 8-per-output limit) in the tool outputs."""
    from ctxfit.notes import _clean_fact, _raw_fact_tokens

    facts: set[str] = set()
    for _tool, _inp, out in steps:
        for raw in _raw_fact_tokens(scrub_secrets(out)):
            tok = _clean_fact(raw)
            if tok is not None and len(tok) < 80:
                facts.add(tok)
    return facts


NO_MASK = {"keep_recent_turns": 10_000}
NOTES_LOW = {"high_water_tokens": 8_000, "low_water_tokens": 4_000}
ARRIVAL = {"arrival": "lossless"}
WARM_ONLY = {"cache_ttl_seconds": None}  # ignore cold-cache moments (v4 behaviour)

STRATEGIES = {
    "none (baseline)": None,
    "arrival lossless only (no masking)": ("mask", {**NO_MASK, **ARRIVAL}),
    "mask / hide": ("mask", {}),
    "mask / hide + arrival": ("mask", ARRIVAL),
    "mask / hide + clear_at_least 20k": ("mask", {"clear_at_least_chars": 20_000}),
    "mask / hide + clear_at_least 20k, cold moments off": (
        "mask",
        {"clear_at_least_chars": 20_000, **WARM_ONLY},
    ),
    "mask / hide + clear_at_least 20k + arrival": (
        "mask",
        {"clear_at_least_chars": 20_000, **ARRIVAL},
    ),
    "mask / hide + clear_at_least auto + arrival": (
        "mask",
        {"clear_at_least_chars": "auto", **ARRIVAL},
    ),
    "mask / hide, keep_facts off": ("mask", {"keep_facts": False}),
    "mask / squeeze": ("mask", {"mask_style": "squeeze"}),
    "notes, defaults (high water 60k)": ("notes", {}),
    "notes 8k/4k": ("notes", NOTES_LOW),
    "notes 8k/4k, cold moments off": ("notes", {**NOTES_LOW, **WARM_ONLY}),
    "notes 8k/4k + arrival": ("notes", {**NOTES_LOW, **ARRIVAL}),
}

CACHE_TTL = 300


def timestamps(seed: int, n: int, pauses: bool = True) -> list[float]:
    """Request times: mostly 10-60 s apart, with an occasional (12%) 6-20 minute pause."""
    rng = random.Random(seed * 7919)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        pause = rng.random() < 0.12
        gap = rng.uniform(360, 1200) if pause else rng.uniform(10, 60)
        t += gap if pauses or not pause else rng.uniform(10, 60)
    return out


def run(
    steps, task: str, strategy, base_cfg: dict[str, Any], seed: int = 0, pauses: bool = True
) -> Totals:
    tot = Totals()
    times = timestamps(seed, len(steps), pauses)
    state = None
    last_seen: dict[str, int] = {}  # call -> latest earlier turn with that exact call
    prev_boundary = 0  # previous request's applied mask boundary (turn t = step t here)
    prev_cut = 0  # previous request's cut: turns < prev_cut were summarised away
    prev_out: list[str] = []  # canonical JSON of the previous request's output messages
    final: list[dict[str, Any]] = []
    for t in range(len(steps)):
        tool, inp, out = steps[t]
        key = tool + json.dumps(inp, sort_keys=True)
        if key in last_seen:
            tot.rereads += 1
        msgs = to_messages(steps[: t + 1], task)

        cold = t == 0 or times[t] - times[t - 1] > CACHE_TTL

        def price(out: list[dict[str, Any]]) -> float:
            nonlocal prev_out
            cur = [canonical_json(m) for m in out]
            k = 0
            while not cold and k < min(len(cur), len(prev_out)) and cur[k] == prev_out[k]:
                k += 1
            prev_out = cur
            cached = _estimate("anthropic", out[:k], 0)
            total = _estimate("anthropic", out, 0)
            return CACHE_READ * cached + CACHE_WRITE * (total - cached)

        if strategy is None:
            est = _estimate("anthropic", msgs, 0)
            tot.requests += 1
            tot.before += est
            tot.after += est
            tot.last_after = est
            tot.cost += price(msgs)
            final = msgs
            if t:
                tot.comparable += 1
                tot.stable += 1
            last_seen[key] = t
            continue
        mode, extra = strategy
        cfg = NotesConfig(**{**base_cfg, **extra})
        plan = plan_context(
            msgs, flavor="anthropic", mode=mode, config=cfg, state=state, now=times[t]
        )
        outcome = SummaryOutcome.ok(FAKE_NOTES) if plan.summary is not None else None
        res = finish_context(plan, outcome)
        state = res.state
        r = res.report
        if key in last_seen:
            # Was the LATEST earlier copy of this exact call's output hidden from the prompt
            # the agent saw just before calling it again? (masked/squeezed: below the previous
            # request's boundary and eligible; or summarised away: below the previous cut)
            u = last_seen[key]
            eligible = (
                len(steps[u][2]) >= cfg.min_mask_chars and steps[u][0] not in cfg.exclude_tools
            )
            if (u < prev_boundary and eligible) or u < prev_cut:
                tot.rereads_hidden += 1
        tot.requests += 1
        tot.before += r.tokens_before_est
        tot.after += r.tokens_after_est
        tot.summariser += r.summariser_input_tokens + r.summariser_output_tokens
        tot.summariser_calls += r.summariser_calls
        tot.cost += price(res.messages)
        tot.cost += r.summariser_input_tokens + SUMMARY_OUT_PRICE * r.summariser_output_tokens
        final = res.messages
        tot.last_after = r.tokens_after_est
        if r.cache_stable is not None:
            tot.comparable += 1
            tot.stable += bool(r.cache_stable)
        prev_boundary = r.mask_boundary
        prev_cut = r.cut_turn if r.notes_applied else 0
        last_seen[key] = t
    facts = all_facts(steps)
    text = prompt_text(final)
    tot.facts_total = len(facts)
    tot.facts_visible = sum(1 for f in facts if f in text)
    return tot


def main() -> int:
    base_cfg = dict(keep_recent_turns=4, mask_chunk_turns=4, min_mask_chars=400,
                    high_water_tokens=60_000, low_water_tokens=30_000,
                    notes_max_tokens=1_500, min_evict_turns=4)  # fmt: skip
    corpora = {
        "coding agent (3 transcripts x 32 turns)": [
            (coding_transcript(s), "Fix the failing tests in ./app") for s in (1, 2, 3)
        ],
        "research agent (3 transcripts x 26 turns)": [
            (research_transcript(s), "Write a brief on the topic") for s in (11, 12, 13)
        ],
    }
    lines = [
        "# Running notes benchmark (synthetic)",
        "",
        f"Generated by `benchmarks/running_notes_bench.py` with ctxfit {__version__}. "
        "**All transcripts are synthetic** (seeded random generators shaped like agent runs: "
        "source files, test logs and JSON API responses for the coding agent; web pages and "
        "search results for the research agent). The summariser is a **fake** that returns "
        "a fixed 1,200-char digest, so notes quality is not measured at all.",
        "",
        "Method: each transcript is replayed one turn at a time; every turn is one request "
        "carrying the full history so far, and state is carried between requests like a real "
        "agent loop. Default config (keep 4 turns, chunk 4, mask outputs >= 400 chars, notes "
        "high water 60k / low water 30k est. tokens). Token figures are ctxfit's chars/4 "
        "estimate, summed over all requests of a run, not tokenizer counts. Requests get "
        "simulated timestamps (seeded): mostly 10-60 s apart, with a 12% chance of a 6-20 "
        "minute pause before a request; ctxfit is told the time, so its cold-cache rules "
        "(`cache_ttl_seconds` 300) see the same pauses.",
        "",
        "- **sent** = estimated input tokens actually sent, summed over requests (baseline = no rewriting).",
        "- **net saved** = (baseline − sent − summariser in/out) / baseline.",
        "- **cache-stable** = share of requests (after the first) whose output starts with the "
        "previous request's output byte for byte (so a provider prompt cache can be reused). "
        "The baseline is 100% by construction: without rewriting, every request extends the last.",
        "- **cache-priced cost** = input cost with Anthropic-style prompt caching, as a share "
        "of the baseline's cache-priced cost: per request, the leading messages identical to "
        "the previous request's are read from cache at 0.1x, the rest is written at 1.25x; "
        "summariser input counts 1x and its output 5x. After a pause longer than 300 s the "
        "cache has expired: the whole request is written at 1.25x, for every strategy "
        "including the baseline. The baseline itself benefits from caching (every request "
        "extends the last), so this is the fair comparison. **same, no pauses** = the same "
        "cost with every gap 10-60 s (the cache never expires), as a share of the baseline "
        "under those conditions.",
        "- **facts visible** = share of the distinct fact tokens (paths, URLs, error codes, "
        "versions, ids; the same tokenizer ctxfit uses for `keeps:`) found anywhere in the "
        "tool outputs that still appear verbatim in the last request. A crude proxy for "
        "'can the model still see the specifics'; it says nothing about prose content.",
        "- **last request** = estimated tokens of the final request of a transcript, averaged "
        "over the transcripts of the corpus.",
        "- **re-reads of hidden output** = times the agent re-ran an identical tool call whose "
        "latest earlier output was hidden, squeezed or summarised away in the prompt it had "
        "just seen (an older copy being hidden does not count if a newer one was visible). "
        "The transcripts are fixed, so this counts *opportunities* for a costly re-fetch, not "
        "re-fetches caused by ctxfit; a real agent may or may not re-run the call (or use "
        "`ctxfit_recall`).",
        "",
    ]
    for corpus, runs in corpora.items():
        lines += [
            f"## {corpus}",
            "",
            "| strategy | sent (est. tokens) | net saved | cache-priced cost | same, no pauses "
            "| summariser calls | cache-stable | last request | facts visible "
            "| re-reads of hidden output |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        base_sent = None
        base_cost = None
        base_steady = None
        for name, strat in STRATEGIES.items():
            agg = Totals()
            steady = 0.0
            for n, (steps, task) in enumerate(runs):
                t = run(steps, task, strat, base_cfg, seed=n + 100 * len(steps))
                steady += run(steps, task, strat, base_cfg, n + 100 * len(steps), False).cost
                agg.requests += t.requests
                agg.before += t.before
                agg.after += t.after
                agg.summariser += t.summariser
                agg.summariser_calls += t.summariser_calls
                agg.stable += t.stable
                agg.comparable += t.comparable
                agg.rereads += t.rereads
                agg.rereads_hidden += t.rereads_hidden
                agg.last_after += t.last_after
                agg.cost += t.cost
                agg.facts_total += t.facts_total
                agg.facts_visible += t.facts_visible
            if base_sent is None:
                base_sent, base_cost, base_steady = agg.after, agg.cost, steady
            net = (base_sent - agg.after - agg.summariser) / base_sent
            stable = agg.stable / agg.comparable if agg.comparable else 0
            facts = agg.facts_visible / agg.facts_total if agg.facts_total else 0
            lines.append(
                f"| {name} | {agg.after:,} | {net:.1%} | {agg.cost / base_cost:.0%} "
                f"| {steady / base_steady:.0%} "
                f"| {agg.summariser_calls} | {stable:.0%} | {agg.last_after // len(runs):,} "
                f"| {facts:.0%} | {agg.rereads_hidden} of {agg.rereads} |"
            )
        lines.append("")
    lines += [
        "## Reading this honestly",
        "",
        "- **What saves real money with caching:** lossless arrival compaction. It changes "
        "each output once, when it first arrives (before anything is cached), so every "
        "request still extends the last one (cache-stable 100%) and every fact stays "
        "visible. Here it cost 77-79% (coding) and 52-53% (research) of the baseline. Most "
        "of that is exact duplicates: these agents re-read the same files and pages, which "
        "real agents do too, but your mileage depends on how often yours does.",
        "- **Masking pays mostly through pauses.** With pauses, hide masking cost 65-76%; "
        "with the cache always warm, 92-98% (each boundary move rewrites the prefix at "
        "1.25x, which eats most of what it saves). Adding arrival and `clear_at_least_chars` "
        "20k is the cheapest setup here: 51-57% with pauses, 73-74% without.",
        "- **Cold-cache moments help a little.** Moving held-back boundaries during a pause "
        "(free, the cache is gone anyway) lowered `clear_at_least_chars` 20k from 64% to 56% "
        "(coding) and 74% to 72% (research). With `clear_at_least_chars` 0 it changes "
        "nothing, since every move already happens.",
        "- **Notes cost money.** At these low thresholds notes cost 75-93% with pauses and "
        "116-138% without: more than doing nothing. With the default high water (60k) they "
        "never engaged. Use notes to stay under a context limit, not to save money.",
        "- **Squeeze** keeps more of each output, so it saves less and, with a warm cache, "
        "can cost more than the baseline (116% on research).",
        "- `keeps:` facts cost 1-2 points and keep far more specifics visible; "
        "`clear_at_least_chars: auto` behaves like 0 here (these moves are big enough).",
        "- The coding transcripts' JSON rows have a list field, so they never become tables; "
        "the research corpus has no tables either. Tables are not what drives these numbers.",
        "- Nothing here measures answer quality. JetBrains Research (Dec 2025) is the evidence "
        "we rely on for quality; this script only checks the mechanics.",
        "",
    ]
    text = "\n".join(lines)
    (HERE / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
