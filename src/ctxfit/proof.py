"""Proof kit (spec v6 #5): the published synthetic corpus, its replay and the savings badge."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .replay import replay

DEMO_DIR = Path(__file__).with_name("demo")


def demo_paths() -> list[str]:
    return sorted(str(p) for p in DEMO_DIR.glob("*.jsonl"))


def demo_hash() -> str:
    """sha256 over the demo transcripts (name, NUL, bytes, NUL per file, sorted by name)."""
    h = hashlib.sha256()
    for p in sorted(DEMO_DIR.glob("*.jsonl")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def badge(corpus: str | None = None) -> dict[str, Any]:
    """A shields.io endpoint: the cheapest strategy's replayed cost vs doing nothing.

    Only ever computed from a published corpus (the bundled synthetic demo by
    default), never from a user's own transcripts.
    """
    report = replay([corpus] if corpus else demo_paths(), redact=True)
    best = min(report["strategies"][1:], key=lambda s: (s["cost_vs_baseline_bp"], s["name"]))
    change = (best["cost_vs_baseline_bp"] - 10000) / 100
    return {
        "schemaVersion": 1,
        "label": "replay cost (synthetic)",
        "message": f"{change:+.1f}% ({best['name']})",
        "color": "brightgreen" if change <= -10 else "green" if change < 0 else "orange",
    }
