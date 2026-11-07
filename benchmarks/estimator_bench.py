"""How far off is each ctxfit estimator from tiktoken cl100k_base?

    pip install tiktoken && python benchmarks/estimator_bench.py

Ratio = estimate / real tokens, aggregated per content class (1.00 = exact).
Corpora: real Python source and real prose found in this Python's
site-packages (so the numbers depend on your install) plus the seeded
synthetic logs/JSON from needle_corpus.py. cl100k_base is one tokenizer;
other models differ.
"""

from __future__ import annotations

import glob
import site
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import needle_corpus as nc  # noqa: E402
import tiktoken  # noqa: E402

from ctxfit.estimate import HeuristicEstimator, PieceEstimator  # noqa: E402

enc = tiktoken.get_encoding("cl100k_base")


def _load(patterns: list[str], n: int) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            try:
                t = Path(f).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if 2000 < len(t) < 60000:
                out.append(t)
            if len(out) >= n:
                return out
    return out


sp = site.getsitepackages()[0]
sets = {
    "real Python source": _load([f"{sp}/**/*.py"], 40),
    "real prose (md/METADATA)": _load([f"{sp}/**/*.md", f"{sp}/**/METADATA"], 40),
}
for kind in ("log", "json_rows", "source", "search_text"):
    sets[f"synthetic {kind}"] = [t for k, t, _ in nc.cases(per_kind=6) if k == kind]

pieces, legacy = PieceEstimator(), HeuristicEstimator()
print("| content | samples | chars/4 (legacy) | pieces (default) |\n|---|---:|---:|---:|")
for name, texts in sets.items():
    if not texts:
        continue
    real = sum(len(enc.encode(t)) for t in texts)
    a = sum(legacy.count(t) for t in texts) / real
    b = sum(pieces.count(t) for t in texts) / real
    print(f"| {name} | {len(texts)} | {a:.2f} | {b:.2f} |")
