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
import threading
from collections.abc import Sequence
from typing import Protocol

from .messages import Message, printable

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


_BIG = 20_000  # texts at least this long are memoised (a fit() counts the same text many times)
_BIG_KEEP = 6


_memo_lock = threading.Lock()


def _memo(cache: dict, text: str, compute) -> int:
    """Count a long text once; repeat calls with the same text are dictionary lookups."""
    if len(text) < _BIG:
        return compute(text)
    hit = cache.get(text)
    if hit is None:
        hit = compute(text)
        with _memo_lock:  # a pipeline object may be shared between threads
            while len(cache) >= _BIG_KEEP:
                cache.pop(next(iter(cache)), None)
            cache[text] = hit
    return hit


class TokenEstimator(Protocol):
    """Pluggable estimator: implement ``count(text)`` for your tokenizer."""

    def count(self, text: str) -> int: ...

    def count_message(self, msg: Message) -> int: ...


class HeuristicEstimator:
    """Chars/4 heuristic with CJK adjustment. No dependencies."""

    name = "heuristic"

    def __init__(self) -> None:
        self._memo: dict[str, int] = {}

    def count(self, text: str) -> int:
        if not text:
            return 0
        return _memo(self._memo, text, self._count)

    def clear_cache(self) -> None:
        self._memo.clear()

    def _count(self, text: str) -> int:
        if text.isascii():  # fast path: no CJK possible (same result, ~50x faster)
            return max(1, math.ceil(len(text) / 4.0))
        cjk = len(_CJK_RE.findall(text))
        other = len(text) - cjk
        # Latin ~4 chars/token; CJK ~1.5 tokens/char (rough, documented).
        return max(1, math.ceil(other / 4.0 + cjk * 1.5))

    def count_message(self, msg: Message) -> int:
        return self.count(msg.content) + PER_MESSAGE_OVERHEAD


# Accented letters stay inside their word but each one makes the tokenizer cut the word (about
# 1.6 extra tokens per accented letter on cl100k_base over French/German/Spanish/Polish/Czech).
_ACCENT_RE = re.compile("[\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]")
# Letters with accents (French, German, Spanish, Polish ...) belong to the word they sit in.
_WORD_RE = re.compile("[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]+")
_LONG_WORD_RE = re.compile("[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]{8,}")
# Scripts whose tokens-per-character is far from the Latin rule (measured on cl100k_base over
# held-in samples, checked on held-out text by benchmarks/estimator_bench.py): Cyrillic is cheaper
# than the generic 0.7 per symbol, Greek, Hebrew, Arabic, the Indic scripts and Thai cost more.
_SCRIPT_RATES: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile("[\u0400-\u052f]"), 0.55),  # Cyrillic
    (re.compile("[\u0370-\u03ff\u1f00-\u1fff]"), 1.1),  # Greek
    (re.compile("[\u0590-\u05ff]"), 1.15),  # Hebrew
    (re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]"), 0.9),  # Arabic
    (re.compile("[\u0900-\u0dff]"), 1.3),  # Devanagari, Bengali ... Sinhala
    (re.compile("[\u0e00-\u0eff]"), 1.0),  # Thai, Lao
)
# A long run of mixed-case letters, digits and + / (a base64 blob, e.g. an inlined screenshot):
# about 0.72 tokens per character on cl100k_base. Requires upper, lower and a digit so ordinary
# words, hex hashes and paths (too many slashes) do not match.
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}")
_DIGITS_RE = re.compile(r"\d+")
_ASCII_NONPUNCT_RE = re.compile(r"[ \t\n\r\f\v\x1c-\x1f0-9A-Za-z]+")
_NONPUNCT_RE = re.compile("[\\s0-9A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]+")


_HAS_DIGIT, _HAS_UPPER, _HAS_LOWER = (re.compile(p) for p in ("[0-9]", "[A-Z]", "[a-z]"))


def _is_b64(run: str) -> bool:
    return bool(
        run.count("/") * 12 <= len(run)  # a path has far more slashes than base64 (about 1 in 64)
        and _HAS_DIGIT.search(run)
        and _HAS_UPPER.search(run)
        and _HAS_LOWER.search(run)
    )


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

    def __init__(self) -> None:
        self._memo: dict[str, int] = {}

    def count(self, text: str) -> int:
        if not text:
            return 0
        return _memo(self._memo, text, self._count)

    def clear_cache(self) -> None:
        self._memo.clear()

    def _count(self, text: str) -> int:
        cjk = len(_CJK_RE.findall(text)) if not text.isascii() else 0
        emoji = len(_EMOJI_RE.findall(text)) if not text.isascii() else 0
        if cjk:
            text = _CJK_RE.sub("", text)
        total = 0.0
        if not text.isascii():
            total += 1.6 * len(_ACCENT_RE.findall(text))
            for pat, rate in _SCRIPT_RATES:
                text, n = pat.subn("", text)
                total += n * rate
        if len(text) >= 40:
            blob_chars = 0

            def _blob(m: re.Match[str]) -> str:
                nonlocal blob_chars
                run = m.group()
                if not _is_b64(run):
                    return run
                blob_chars += len(run)
                return " "

            text = _B64_RE.sub(_blob, text)  # one pass: replacing match by match was quadratic
            total += 0.72 * blob_chars
        words = _WORD_RE.findall(text)
        total += len(words)
        for w in _LONG_WORD_RE.findall(text):
            total += math.ceil(len(w) / 7) - 1
        for d in _DIGITS_RE.findall(text):
            total += math.ceil(len(d) / 2)
        # Characters that are neither whitespace nor an ASCII letter/digit. One regex pass in C
        # for any text (a per-character Python loop was 10x slower, and any non-ASCII marker such
        # as "…" in a cut tool output sent the whole text down that path).
        punct = len((_ASCII_NONPUNCT_RE if text.isascii() else _NONPUNCT_RE).sub("", text))
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
        try:
            self._enc = tiktoken.get_encoding(encoding)
        except Exception as exc:  # tiktoken downloads its vocabulary on first use
            raise ImportError(
                f"tiktoken is installed but could not load {encoding!r} ({type(exc).__name__}). "
                "It downloads the vocabulary the first time; offline? Point TIKTOKEN_CACHE_DIR "
                "at a folder that already has it, or use estimator='pieces'"
            ) from exc

    def count(self, text: str) -> int:
        if not text:
            return 0
        # Text such as "<|endoftext|>" in a tool output is ordinary text here, not an error.
        return max(1, len(self._enc.encode(text, disallowed_special=())))

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
            "role": printable(m.role, 40),
            "kind": printable(m.kind, 40),
            "section": m.section,
            "est_tokens": estimator.count_message(m),
            "chars": len(m.content),
            "preview": printable(m.content[:80].replace("\n", " ").replace("\t", " "), 400)
            + ("…" if len(m.content) > 80 else ""),
        }
        for i, m in enumerate(messages)
    ]
    scored.sort(key=lambda d: d["est_tokens"], reverse=True)  # type: ignore[arg-type]
    return scored[:n]
