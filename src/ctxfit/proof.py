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
    return corpus_hash(DEMO_DIR)


PUBLISHED_SHA256 = "2f8b17d0ab49a576f59da040fb336b522726e465925e5072be0b9dda5e456fde"
"""demo_hash() of the published synthetic corpus; badge() refuses any other corpus."""


def corpus_hash(folder: str | Path) -> str:
    """demo_hash() of the ``*.jsonl`` files directly in ``folder``."""
    h = hashlib.sha256()
    for p in sorted(Path(folder).glob("*.jsonl")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def badge_message(cost_vs_baseline_bp: int, name: str) -> tuple[str, str]:
    """(message, color): the change in tenths of a percent, rounded floor(x + 0.5).

    tenths = floor((bp - 10000 + 5) / 10) in integers; message =
    sign ("+" when >= 0) + |tenths| // 10 + "." + |tenths| % 10 + "% (" + name + ")";
    color brightgreen at <= -100 tenths, green below 0, else orange.
    """
    tenths = (cost_vs_baseline_bp - 10000 + 5) // 10
    sign = "+" if tenths >= 0 else "-"
    a = abs(tenths)
    color = "brightgreen" if tenths <= -100 else "green" if tenths < 0 else "orange"
    return f"{sign}{a // 10}.{a % 10}% ({name})", color


def badge(corpus: str | None = None) -> dict[str, Any]:
    """A shields.io endpoint: the cheapest strategy's replayed cost vs doing nothing.

    Only the published synthetic corpus is accepted (the bundled demo, or a
    folder whose files hash to the same value) -- never a user's own transcripts.
    """
    folder = Path(corpus) if corpus else DEMO_DIR
    if corpus_hash(folder) != PUBLISHED_SHA256:
        raise ValueError("badge only runs on the published synthetic corpus (hash mismatch)")
    report = replay(sorted(str(p) for p in folder.glob("*.jsonl")), redact=True)
    best = min(report["strategies"][1:], key=lambda s: (s["cost_vs_baseline_bp"], s["name"]))
    message, color = badge_message(best["cost_vs_baseline_bp"], best["name"])
    return {
        "schemaVersion": 1,
        "label": "replay cost (synthetic)",
        "message": message,
        "color": color,
    }
