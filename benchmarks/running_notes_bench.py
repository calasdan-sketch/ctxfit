"""Replay synthetic agent transcripts through ctxfit running notes and report honest numbers.

    python benchmarks/running_notes_bench.py            # print + write benchmarks/RESULTS.md

Everything here is SYNTHETIC and deterministic (fixed seeds): the transcripts are
generated, not recorded, and the "summariser" is a fake that returns a fixed-size
digest. The numbers show what the mechanics do to prompt size and cache stability
on transcripts shaped like real agent runs; they say nothing about task quality.
"""

from __future__ import annotations

import hashlib
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
    finish_context,
    plan_context,
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
    notes: list[str] = field(default_factory=list)


STRATEGIES = {
    "none (baseline)": None,
    "mask / hide": ("mask", {}),
    "mask / squeeze": ("mask", {"mask_style": "squeeze"}),
    "mask / hide + clear_at_least 20k": ("mask", {"clear_at_least_chars": 20_000}),
    "notes, defaults (high water 60k)": ("notes", {}),
    "notes, high water 8k / low 4k": (
        "notes",
        {"high_water_tokens": 8_000, "low_water_tokens": 4_000},
    ),
    "notes 8k/4k + squeeze": (
        "notes",
        {"high_water_tokens": 8_000, "low_water_tokens": 4_000, "mask_style": "squeeze"},
    ),
}


def run(steps, task: str, strategy, base_cfg: dict[str, Any]) -> Totals:
    tot = Totals()
    state = None
    first_seen: dict[str, int] = {}
    prev_hidden: set[str] = set()  # id8s hidden in the previous request's output
    prev_cut = 0  # turns < prev_cut were summarised away (turn t = step t here)
    for t in range(len(steps)):
        tool, inp, out = steps[t]
        key = tool + json.dumps(inp, sort_keys=True)
        if key in first_seen:
            tot.rereads += 1
        msgs = to_messages(steps[: t + 1], task)
        if strategy is None:
            from ctxfit.notes import _estimate

            est = _estimate("anthropic", msgs, 0)
            tot.requests += 1
            tot.before += est
            tot.after += est
            tot.last_after = est
            if t:
                tot.comparable += 1
                tot.stable += 1
            first_seen.setdefault(key, t)
            continue
        mode, extra = strategy
        cfg = NotesConfig(**{**base_cfg, **extra})
        plan = plan_context(msgs, flavor="anthropic", mode=mode, config=cfg, state=state)
        outcome = SummaryOutcome.ok(FAKE_NOTES) if plan.summary is not None else None
        res = finish_context(plan, outcome)
        state = res.state
        r = res.report
        if key in first_seen:
            # Was the earlier output of this exact call hidden from the prompt the agent saw
            # before deciding to call it again? (masked, squeezed or evicted into notes)
            u = first_seen[key]
            prev_out = steps[u][2]
            id8 = hashlib.sha256(prev_out.encode()).hexdigest()[:8]
            if id8 in prev_hidden or u < prev_cut:
                tot.rereads_hidden += 1
        tot.requests += 1
        tot.before += r.tokens_before_est
        tot.after += r.tokens_after_est
        tot.summariser += r.summariser_input_tokens + r.summariser_output_tokens
        tot.summariser_calls += r.summariser_calls
        tot.last_after = r.tokens_after_est
        if r.cache_stable is not None:
            tot.comparable += 1
            tot.stable += bool(r.cache_stable)
        prev_hidden = set(r.offload)
        prev_cut = r.cut_turn if r.notes_applied else 0
        first_seen.setdefault(key, t)
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
        "estimate, summed over all requests of a run, not tokenizer counts.",
        "",
        "- **sent** = estimated input tokens actually sent, summed over requests (baseline = no rewriting).",
        "- **net saved** = (baseline − sent − summariser in/out) / baseline.",
        "- **cache-stable** = share of requests (after the first) whose output starts with the "
        "previous request's output byte for byte (so a provider prompt cache can be reused).",
        "- **re-reads of hidden output** = times the agent re-ran an identical tool call whose "
        "earlier output was hidden, squeezed or summarised away in the prompt it had just seen. "
        "The transcripts are fixed, so this counts *opportunities* for a costly re-fetch, not "
        "re-fetches caused by ctxfit; a real agent may or may not re-run the call (or use "
        "`ctxfit_recall`).",
        "",
    ]
    for corpus, runs in corpora.items():
        lines += [
            f"## {corpus}",
            "",
            "| strategy | sent (est. tokens) | net saved | summariser calls | cache-stable | last request | re-reads of hidden output |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        base_sent = None
        for name, strat in STRATEGIES.items():
            agg = Totals()
            for steps, task in runs:
                t = run(steps, task, strat, base_cfg)
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
            if base_sent is None:
                base_sent = agg.after
            net = (base_sent - agg.after - agg.summariser) / base_sent
            stable = agg.stable / agg.comparable if agg.comparable else 0
            lines.append(
                f"| {name} | {agg.after:,} | {net:.1%} | {agg.summariser_calls} | {stable:.0%} "
                f"| {agg.last_after // len(runs):,} | {agg.rereads_hidden} of {agg.rereads} |"
            )
        lines.append("")
    lines += [
        "## Reading this honestly",
        "",
        "- Savings depend almost entirely on how much of the history is old tool output. "
        "These synthetic transcripts are tool-heavy on purpose; chat-heavy conversations "
        "save far less.",
        "- `cache-stable` below 100% means some requests rewrite earlier content and pay a "
        "fresh cache write; the chunked boundary keeps those moves rare, `clear_at_least_chars` "
        "makes them rarer.",
        "- With the default high water (60k est. tokens) notes never engage on these "
        "transcripts: masking alone keeps every request under it. The 8k/4k rows lower the "
        "marks only to show the notes mechanics at this scale; each summariser call costs "
        "its input (the evicted turns) and that cost is subtracted in `net saved`.",
        "- Squeeze keeps more of each output (head/tail, error lines, JSON shape) so it saves "
        "less than hide; whether that avoids re-reads is not something a replay can show.",
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
