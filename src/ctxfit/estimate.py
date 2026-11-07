"""Token estimation without API calls.

Two backends, one protocol:

- :class:`HeuristicEstimator` (default): chars/4 for Latin scripts, a
  heavier per-char rate for CJK, plus a small per-message overhead for
  role/format tokens. No dependencies. Honest error bounds: roughly
  ±25% on mixed prose/code; worse on CJK- or emoji-heavy text.
- :class:`TiktokenEstimator`: uses tiktoken's cl100k_base when the
  package is importable. Narrows error but does NOT equal any specific
  model's tokenizer — still an estimate.

Every public number ctxfit prints is labeled an estimate.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

from .messages import Message

# Per-message overhead approximating <|role|> framing tokens.
PER_MESSAGE_OVERHEAD = 4


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF  # CJK Extension A
        or 0x20000 <= o <= 0x2A6DF  # CJK Extension B
        or 0x3040 <= o <= 0x30FF  # Hiragana/Katakana
        or 0xAC00 <= o <= 0xD7AF  # Hangul Syllables
        or 0xFF00 <= o <= 0xFFEF  # Fullwidth forms
    )


_CJK_RE = re.compile(
    "[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]"
)


class TokenEstimator(Protocol):
    """Pluggable estimator: implement ``count(text)`` for your tokenizer."""

    def count(self, text: str) -> int: ...

    def count_message(self, msg: Message) -> int: ...


class HeuristicEstimator:
    """Chars/4 heuristic with CJK adjustment. No dependencies."""

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if text.isascii():  # fast path: no CJK possible (same result, ~50x faster)
            return max(1, math.ceil(len(text) / 4.0))
        cjk = len(_CJK_RE.findall(text))
        other = len(text) - cjk
        # Latin ~4 chars/token; CJK ~1.5 tokens/char (rough, documented).
        return max(1, math.ceil(other / 4.0 + cjk * 1.5))

    def count_message(self, msg: Message) -> int:
        return self.count(msg.content) + PER_MESSAGE_OVERHEAD


_WORD_RE = re.compile(r"[A-Za-z]+")
_LONG_WORD_RE = re.compile(r"[A-Za-z]{8,}")
_DIGITS_RE = re.compile(r"\d+")
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf]")


class PieceEstimator:
    """Structure-aware heuristic, no dependencies (the default fallback).

    Counts words, digit runs and punctuation separately instead of
    chars/4, which under-counts logs, JSON and id-heavy tool output by
    ~40% (measured against cl100k_base; see benchmarks/estimator_bench.py).
    Constants were fitted on cl100k_base over real Python source, real
    prose and synthetic logs/JSON: aggregate error stays within about
    -11%/+15% per content class. Still an estimate for *your* model.
    """

    name = "pieces"

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK_RE.findall(text)) if not text.isascii() else 0
        emoji = len(_EMOJI_RE.findall(text)) if not text.isascii() else 0
        if cjk:
            text = _CJK_RE.sub("", text)
        total = 0.0
        words = _WORD_RE.findall(text)
        total += len(words)
        for w in _LONG_WORD_RE.findall(text):
            total += math.ceil(len(w) / 7) - 1
        for d in _DIGITS_RE.findall(text):
            total += math.ceil(len(d) / 2)
        punct = sum(1 for ch in text if not (ch.isspace() or ch.isalnum() and ch.isascii()))
        total += 0.7 * (punct - emoji) + 2.0 * emoji
        total += 0.6 * text.count("\n") + cjk * 1.5
        return max(1, math.ceil(total))

    def count_message(self, msg: Message) -> int:
        return self.count(msg.content) + PER_MESSAGE_OVERHEAD


class TiktokenEstimator:
    """tiktoken cl100k_base backend. Optional dependency."""

    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tiktoken is not installed; pip install tiktoken or use estimator='heuristic'"
            ) from exc
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(self._enc.encode(text)))

    def count_message(self, msg: Message) -> int:
        return self.count(msg.content) + PER_MESSAGE_OVERHEAD


def get_estimator(kind: str | TokenEstimator = "auto") -> TokenEstimator:
    """Resolve an estimator spec.

    kind: "auto" (tiktoken if importable, else "pieces"), "pieces",
    "heuristic" (legacy chars/4), "tiktoken", or an object implementing
    TokenEstimator.
    """
    if isinstance(kind, str):
        if kind == "auto":
            try:
                return TiktokenEstimator()
            except ImportError:
                return PieceEstimator()
        if kind == "pieces":
            return PieceEstimator()
        if kind == "heuristic":
            return HeuristicEstimator()
        if kind == "tiktoken":
            return TiktokenEstimator()
        raise ValueError(
            f"unknown estimator {kind!r}; expected 'auto', 'pieces', 'heuristic', 'tiktoken', or a TokenEstimator"
        )
    if hasattr(kind, "count"):
        return kind
    raise TypeError(f"estimator must be a str or TokenEstimator, got {type(kind).__name__}")


def count_by_section(messages: Sequence[Message], estimator: TokenEstimator) -> dict[str, int]:
    """Estimated tokens per budget section."""
    totals: dict[str, int] = {"system": 0, "retrieved": 0, "history": 0, "scratchpad": 0}
    for m in messages:
        totals[m.section] = totals.get(m.section, 0) + estimator.count_message(m)
    return totals


def top_consumers(
    messages: Sequence[Message], estimator: TokenEstimator, n: int = 10
) -> list[dict[str, object]]:
    """The n most expensive messages, for CLI/report diagnostics."""
    scored = [
        {
            "index": i,
            "role": m.role,
            "kind": m.kind,
            "section": m.section,
            "est_tokens": estimator.count_message(m),
            "chars": len(m.content),
            "preview": (m.content[:80].replace("\n", " ") + ("…" if len(m.content) > 80 else "")),
        }
        for i, m in enumerate(messages)
    ]
    scored.sort(key=lambda d: d["est_tokens"], reverse=True)  # type: ignore[arg-type]
    return scored[:n]
