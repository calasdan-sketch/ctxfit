"""Mutation sanity check: break one condition at a time and make sure the tests notice.

    python scripts/mutation_check.py            # all mutants, 2 at a time, in throw-away copies
    python scripts/mutation_check.py -j 4 -k pinned

Every mutant is applied to a COPY of the repository in a temporary folder (your checkout is never
touched). A mutant is KILLED when the test suite fails and SURVIVED when it still passes; a
survivor means either a missing test or an equivalent change (listed with a reason). Takes a few
minutes. Needs the dev extra (pytest, hypothesis). Not part of the normal test run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, original text, replacement, why a survivor would be acceptable or "")
MUTANTS: list[tuple[str, str, str, str]] = [
    ("src/ctxfit/strategies.py", "and not m.pinned\n", "and m.pinned\n", ""),
    (
        "src/ctxfit/strategies.py",
        "or (m.pinned and not self.truncate_pinned)",
        "or (m.pinned and self.truncate_pinned)",
        "",
    ),
    (
        "src/ctxfit/strategies.py",
        "if any(messages[j].pinned for j in idxs):",
        "if False:",
        "equivalent: the next check covers a superset of these messages",
    ),
    ("src/ctxfit/strategies.py", "if any(messages[p].pinned for p in partners):", "if False:", ""),
    (
        "src/ctxfit/strategies.py",
        'if m.pinned or m.section == "system" or t == latest:',
        'if m.section == "system" or t == latest:',
        "",
    ),
    (
        "src/ctxfit/strategies.py",
        "if after >= before:",
        "if after > before:",
        "tie: JSON and table of exactly equal estimated size; either choice is as good",
    ),
    ("src/ctxfit/strategies.py", "if len(candidate) < len(m.content):", "if True:", ""),
    (
        "src/ctxfit/strategies.py",
        "if head_chars + tail_chars >= max_chars:",
        "if head_chars + tail_chars > max_chars:",
        "",
    ),
    ("src/ctxfit/strategies.py", "boiler / nonblank >= 0.3", "boiler / nonblank >= 0.9", ""),
    ("src/ctxfit/strategies.py", "if freq[col[i]] <= 2:", "if freq[col[i]] <= 0:", ""),
    ("src/ctxfit/strategies.py", "score -= 1", "score += 1", ""),
    ("src/ctxfit/strategies.py", "        score += 2\n    if _LEVEL_HIGH_RE", "        score += 0\n    if _LEVEL_HIGH_RE", ""),
    ("src/ctxfit/strategies.py", "    if _LEVEL_HIGH_RE.search(line):\n        score += 2", "    if _LEVEL_HIGH_RE.search(line):\n        score += 0", ""),
    ("src/ctxfit/strategies.py", "m.content[len(m.content) - tail_chars :]", "m.content[-tail_chars:]", ""),
    ("src/ctxfit/strategies.py", "max_chars = lim if lim else len(m.content)", "max_chars = lim", ""),
    ("src/ctxfit/strategies.py", "head_chars = min(head_chars, max_chars // 3)", "head_chars = head_chars", ""),
    ("src/ctxfit/strategies.py", "or (self.protect_tools and _tool_of(m, names) in self.protect_tools)", "", ""),
    ("src/ctxfit/strategies.py", "if self.protect_tools and _tool_of(m, names) in self.protect_tools:", "if False:", ""),
    ("src/ctxfit/pipeline.py", "never = [t for t, n in tool_limits.items() if n == 0]", "never = []", ""),
    ("src/ctxfit/pipeline.py", "if tool_limits and strategies is not None:", "if False:", ""),
    ("src/ctxfit/estimate.py", "PER_MESSAGE_OVERHEAD = 4", "PER_MESSAGE_OVERHEAD = 0", ""),
    ("src/ctxfit/estimate.py", "total += 1.6 * len(_ACCENT_RE.findall(text))", "total += 0", ""),
    ("src/ctxfit/estimate.py", "total += 0.72 * blob_chars", "total += 0", ""),
    ("src/ctxfit/estimate.py", "run.count(\"/\") * 12 <= len(run)", "True", ""),
    ("src/ctxfit/estimate.py", "(re.compile(\"[\\u0370-\\u03ff\\u1f00-\\u1fff]\"), 1.1)", "(re.compile(\"[\\u0370-\\u03ff\\u1f00-\\u1fff]\"), 0.7)", ""),
    ("src/ctxfit/estimate.py", "(re.compile(\"[\\u0900-\\u0dff]\"), 1.3)", "(re.compile(\"[\\u0900-\\u0dff]\"), 0.7)", ""),
    ("src/ctxfit/messages.py", 'if self.role == "system":\n            self.pinned = True', "if False:\n            self.pinned = True", ""),
    ("src/ctxfit/messages.py", "if self.section != assign_section(self.role, self.kind):", "if False:", ""),
    ("src/ctxfit/messages.py", 'if self.pinned and self.role != "system":', "if False:", ""),
    ("src/ctxfit/cli.py", "if any(\"role\" in r for r in rows):", "if False:", ""),
    ("src/ctxfit/cli.py", 'if path == "-":\n        raise CliError(\n            "a Claude Code', 'if False:\n        raise CliError(\n            "a Claude Code', ""),
    ("src/ctxfit/cli.py", "key=lambda r: (len(r.messages), r.ts_ms)", "key=lambda r: (-len(r.messages), r.ts_ms)", ""),
    ("src/ctxfit/cli.py", "or not num.strip().isdigit()", "or False", ""),
    ("src/ctxfit/adapters/langchain.py", 'name, args = get("name"), get("args")', 'name, args = None, ""', ""),
    ("src/ctxfit/adapters/langchain.py", "return value if isinstance(value, dict) else text", "return text", ""),
]  # fmt: skip


def run_one(idx: int, mutant: tuple[str, str, str, str], work: Path, timeout: int) -> str:
    rel, old, new, why = mutant
    copy = work / f"m{idx}"
    shutil.copytree(
        ROOT,
        copy,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )
    target = copy / rel
    text = target.read_text(encoding="utf-8")
    label = f"{rel.split('/')[-1]}: {old[:50]!r} -> {new[:30]!r}".replace("\n", "|")
    if old not in text:
        return f"BROKEN   {label}  (text not found: update the mutant list)"
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(copy / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    env["CTXFIT_HYPOTHESIS_EXAMPLES"] = "60"
    first = ["tests/test_round5.py", "tests/test_strategies.py", "tests/test_expert2.py"]
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *first, "tests"],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        verdict = "KILLED  " if r.returncode else "SURVIVED"
        note = f"  ({why})" if why and r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        verdict, note = "KILLED  ", "  (timed out: counts as noticed)"
    shutil.rmtree(copy, ignore_errors=True)
    return f"{verdict} {label}{note}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-j", type=int, default=2, help="mutants at a time")
    ap.add_argument("-k", default="", help="only mutants whose line contains this text")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    chosen = [(i, m) for i, m in enumerate(MUTANTS) if args.k in f"{m[0]} {m[1]} {m[2]}"]
    with tempfile.TemporaryDirectory(prefix="ctxfit-mut-") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(max_workers=args.j) as pool:
            results = list(pool.map(lambda im: run_one(im[0], im[1], work, args.timeout), chosen))
    for line in results:
        print(line)
    survived = [r for r in results if r.startswith(("SURVIVED", "BROKEN"))]
    print(f"\n{len(results) - len(survived)} of {len(results)} mutants noticed by the tests")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
