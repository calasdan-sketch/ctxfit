"""Seeded synthetic tool outputs with planted 'needles' (shared by the bench scripts).

Each case: a single oversized tool result that holds one fact the agent must
still be able to see afterwards. Everything is generated from a seed; nothing
here is real data.
"""

from __future__ import annotations

import json
import random


def _log(rng: random.Random, n: int = 600) -> tuple[str, str]:
    lines = []
    for i in range(n):
        lines.append(
            f"2026-11-01T10:{i // 60 % 60:02d}:{i % 60:02d}Z INFO worker-{rng.randint(1, 4)} "
            f"processed batch {rng.randint(1000, 9999)} in {rng.randint(10, 99)}ms status=ok"
        )
    pos = rng.randint(n // 4, 3 * n // 4)
    needle = "FATAL migration 0042_add_index failed: relation idx_orders_user already exists"
    lines.insert(pos, f"2026-11-01T10:30:00Z ERROR db {needle}")
    return "\n".join(lines), "0042_add_index"


def _json_rows(rng: random.Random, n: int = 400) -> tuple[str, str]:
    rows = [
        {
            "id": i,
            "name": f"item-{i}",
            "price": round(rng.random() * 100, 2),
            "stock": rng.randint(0, 50),
            "status": "active",
        }
        for i in range(n)
    ]
    k = rng.randint(n // 4, 3 * n // 4)
    rows[k]["status"] = "recalled-batch-7731"
    return json.dumps(rows), "recalled-batch-7731"


def _source(rng: random.Random, n: int = 300) -> tuple[str, str]:
    lines = []
    for i in range(n):
        lines.append(f"def helper_{i}(x):\n    return x + {rng.randint(1, 9)}  # helper {i}\n")
    k = rng.randint(n // 4, 3 * n // 4)
    lines.insert(
        k, "def charge(card):\n    raise RuntimeError('LEDGER_SEED_9913 not initialised')\n"
    )
    return "\n".join(lines), "LEDGER_SEED_9913"


def _search(rng: random.Random, n: int = 160) -> tuple[str, str]:
    paras = []
    for i in range(n):
        paras.append(
            f"Result {i}: The committee reviewed the quarterly figures and noted general progress across "
            f"departments {rng.randint(1, 9)} and {rng.randint(1, 9)}. Further discussion was deferred."
        )
    k = rng.randint(n // 4, 3 * n // 4)
    paras.insert(
        k, "Result X: The deadline for the grant application is 14 March 2027 at 17:00 UTC."
    )
    return "\n\n".join(paras), "14 March 2027"


def _log_flood(rng: random.Random, n: int = 900) -> tuple[str, str]:
    """A log with 15 distinct WARN lines that mention 'error' (transient, harmless) around one real fatal."""
    lines = [
        f"2026-11-01T10:{i // 60 % 60:02d}:{i % 60:02d}Z INFO worker-{rng.randint(1, 4)} "
        f"processed batch {rng.randint(1000, 9999)} in {rng.randint(10, 99)}ms status=ok"
        for i in range(n)
    ]
    words = ["timeout", "reset", "busy", "stale", "slow", "retry", "eof", "dns"]
    for k, i in enumerate(sorted(rng.sample(range(50, n - 50), 15))):
        lines[i] = (
            f"2026-11-01T10:00:{k:02d}Z WARN net transient error {words[k % 8]}-{k} "
            f"on conn {rng.randint(100, 999)}, retrying"
        )
    pos = rng.randint(n * 6 // 10, n * 9 // 10)  # after most of the distractors
    needle = "migration 0042_add_index failed: relation idx_orders_user already exists"
    lines[pos] = f"2026-11-01T10:30:00Z ERROR db FATAL {needle}"
    return "\n".join(lines), "0042_add_index"


GENERATORS = {"log": _log, "json_rows": _json_rows, "source": _source, "search_text": _search}
# Added in round 5 and kept out of the default corpus so the published 20-case table stays
# comparable with the competitor rows; ``cases(flood=True)`` / ``needle_bench.py --flood``.
FLOOD_GENERATORS = {"log_flood": _log_flood}


def cases(seed: int = 7, per_kind: int = 5, flood: bool = False):
    """Yield (kind, text, needle) tuples."""
    out = []
    for kind, gen in {**GENERATORS, **(FLOOD_GENERATORS if flood else {})}.items():
        for j in range(per_kind):
            rng = random.Random(seed * 1000 + j)
            text, needle = gen(rng)
            out.append((kind, text, needle))
    return out


def as_messages(text: str) -> list[dict]:
    """A tiny OpenAI-shaped conversation whose single tool result is `text`."""
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Find the problem."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": text},
        {"role": "assistant", "content": "Looking."},
        {"role": "user", "content": "What did you find?"},
    ]
