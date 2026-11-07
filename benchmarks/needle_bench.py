"""Needle-retention benchmark: does the one fact the agent needs survive compression?

Seeded synthetic tool outputs (benchmarks/needle_corpus.py) each hide one
needle in the middle. Every available compressor gets the output; we report
tokens saved (tiktoken cl100k_base when installed), whether the needle string
is still in the result, and time per case. No network, no paid API.

    python benchmarks/needle_bench.py            # ctxfit variants (+ any competitor installed)
    python benchmarks/needle_bench.py --scale    # fit() time vs conversation length
    python benchmarks/needle_bench.py --big      # fit() time on one huge tool output (default estimator)

Competitors are used only if importable in this Python: langchain-core
(trim_messages), headroom-ai (headroom.compress), llmlingua (LLMLingua-2,
downloads a ~700 MB model from Hugging Face on first run, slow on CPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import needle_corpus as nc  # noqa: E402

import ctxfit  # noqa: E402
from ctxfit import CompactJson, FitPipeline, TruncateToolOutput  # noqa: E402

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
    COUNTER = "tiktoken cl100k_base"

    def tok(s: str) -> int:
        return len(_enc.encode(s))
except ImportError:  # estimate only; say so
    from ctxfit.estimate import PieceEstimator

    COUNTER = "ctxfit pieces ESTIMATE (install tiktoken for real counts)"
    tok = PieceEstimator().count


def _ctxfit(salient: bool):
    def run(text: str) -> str:
        pipe = FitPipeline(
            strategies=[
                CompactJson(max_outliers=10 if salient else 0, when="always"),
                TruncateToolOutput(salient=salient, when="always"),
            ],
            estimator="heuristic",
        )
        out, _ = pipe.fit(nc.as_messages(text))
        return out[3]["content"]

    return run


def _headroom(text: str) -> str:
    import headroom

    r = headroom.compress(nc.as_messages(text), model="gpt-4o")
    o = [m for m in r.messages if m.get("role") == "tool"][0]["content"]
    return o if isinstance(o, str) else json.dumps(o)


def _langchain(text: str) -> str:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        trim_messages,
    )

    lc = [
        SystemMessage("s"),
        HumanMessage("u"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "run", "args": {}}]),
        ToolMessage(text, tool_call_id="c1"),
        AIMessage("L"),
        HumanMessage("w"),
    ]
    out = trim_messages(
        lc,
        max_tokens=6000,
        token_counter=lambda ms: sum(tok(str(m.content)) for m in ms),
        strategy="last",
        start_on="human",
        include_system=True,
    )
    return "".join(str(m.content) for m in out if m.type == "tool")


_llmlingua = None


def _lingua(text: str) -> str:
    global _llmlingua
    if _llmlingua is None:
        from llmlingua import PromptCompressor

        _llmlingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _llmlingua.compress_prompt(text, rate=0.2, force_tokens=["\n"], drop_consecutive=True)[
        "compressed_prompt"
    ]


def _available():
    yield "ctxfit 0.7 behaviour (head+tail, 50 rows)", _ctxfit(False)
    yield "ctxfit 0.8 (salient lines + outlier rows)", _ctxfit(True)
    for name, mod, fn in (
        ("LangChain trim_messages (last, 6k)", "langchain_core", _langchain),
        ("Headroom compress()", "headroom", _headroom),
        ("LLMLingua-2 rate 0.2", "llmlingua", _lingua),
    ):
        try:
            __import__(mod)
        except ImportError:
            continue
        yield name, fn


def needles(per_kind: int) -> None:
    cases = nc.cases(per_kind=per_kind)
    kinds = list(dict.fromkeys(k for k, _, _ in cases))
    print(f"token counter: {COUNTER}; {len(cases)} cases ({per_kind} per kind), seed fixed\n")
    print("| compressor | " + " | ".join(kinds) + " | needles kept | tokens saved | ms/case |")
    print("|---|" + "---|" * (len(kinds) + 3))
    for name, fn in _available():
        kept = {k: 0 for k in kinds}
        before = after = 0
        t0 = time.perf_counter()
        for kind, text, needle in cases:
            out = fn(text)
            kept[kind] += needle in out
            before += tok(text)
            after += tok(out)
        ms = 1000 * (time.perf_counter() - t0) / len(cases)
        total = sum(kept.values())
        cells = " | ".join(f"{kept[k]}/{per_kind}" for k in kinds)
        print(
            f"| {name} | {cells} | {total}/{len(cases)} | "
            f"{100 * (1 - after / before):.1f}% | {ms:,.0f} |"
        )


def scale() -> None:
    def mk(n: int) -> list[dict]:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
        for i in range(n):
            cid = f"c{i}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": cid, "content": f"out {i} " * 300})
            msgs.append({"role": "user", "content": f"next {i}"})
        return msgs

    print("| tool calls | messages | fit() seconds |\n|---:|---:|---:|")
    for n in (100, 400, 1600):
        m = mk(n)
        t = time.perf_counter()
        ctxfit.fit(m, total_budget=20_000, estimator="heuristic")
        print(f"| {n} | {len(m)} | {time.perf_counter() - t:.2f} |")


def big() -> None:
    """One huge tool output through fit() with the default estimator (CPU only)."""
    import json

    cases = {
        "4.8 MB log, 400k repeated lines": "GET /health 200 12ms\n" * 230_000,
        "3 MB on one line": "x" * 3_000_000,
        "60k-row JSON array (2.6 MB)": json.dumps(
            [{"id": i, "v": "a" * 20} for i in range(60_000)]
        ),
    }
    print("| input | fit() seconds |\n|---|---:|")
    for name, text in cases.items():
        m = nc.as_messages(text)
        t = time.perf_counter()
        ctxfit.fit(m, total_budget=2_000)
        print(f"| {name} | {time.perf_counter() - t:.2f} |")


if __name__ == "__main__":
    if "--big" in sys.argv:
        big()
    elif "--scale" in sys.argv:
        scale()
    else:
        needles(5)
