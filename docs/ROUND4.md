# Round 4: what a stranger gets from `pip install ctxfit`

Scope: the package as someone who has never seen the repo would meet it (README, wheel, CLI,
errors, consoles), not new compression research. Branch `expert-2`. Nothing was published,
tagged, pushed or uploaded.

## How this was run (honest)

There was no sub-agent tool in this session, so **one reviewer played every role in turn**.
The roles were still run for real (every finding below has the command that showed it), but
they are not independent minds; the second pass (Pass B) is the same reviewer re-attacking its
own change. Treat "found by N specialists" as "N angles", not "N people".

Environment: Linux, Python 3.10 / 3.11 / 3.12 / 3.13 (all four installed and used), network
available for read-only registry lookups. **Windows was never run.**

## Baseline (before any change, round-3 code, 3.13)

| check | result |
|---|---|
| `pytest` | 469 passed |
| `ruff check` | clean |
| `needle_bench.py` (no tiktoken, own estimator) | 20 / 20 needles, 91.9% saved, 0.7 behaviour 0 / 20 |
| `running_notes_bench.py --check` | RESULTS.md up to date |
| registry | PyPI `ctxfit` = **0.1.1** (Calas Automations Inc., 2026-10-20); repo = 0.7.0 |

## Specialists and findings

| # | Role | Finding | Proved by |
|---|---|---|---|
| 1 | First-time user (README only, fresh venv, folder with spaces) | `ctxfit --version` -> argparse "arguments are required"; no `python -m ctxfit`; no way to try it without writing a JSON file first | `v310/bin/ctxfit --version`; `python -m ctxfit` -> "No module named ctxfit.__main__" |
| 1 | same | Legacy console (cp1252/cp437/ASCII): `ctxfit fit` finished the work, then died printing `->`, `-`, `ok` glyphs | `PYTHONIOENCODING=cp1252 ctxfit fit convo.json` -> `'charmap' codec can't encode character '→'` |
| 1 | same | `ctxfit analyze f \| head` -> "ctxfit: error: [Errno 32] Broken pipe" | pipe to `head` |
| 1 | same | A sliced transcript (orphan tool result) printed a full Python traceback | `ctxfit fit orphan.json` |
| 1 | same | Errors named neither the file nor the fix ("Expecting value: line 2 column 1") | bad/empty/folder/latin-1 inputs |
| 1 | same | `ctxfit replay '*.jsonl'` in a folder of transcripts: 0 files | run in `rp/` |
| 2 | Release / packaging engineer | `LICENSE` declared (MIT) but **not in sdist or wheel**; no `py.typed`; setuptools deprecation warning for `license = {text=}`; version written in 3 places; no Python-version / OS classifiers; only a Homepage URL; sdist had no CHANGELOG/benchmarks | `python -m build`; `unzip -l`; `tar tzf` |
| 2 | same | README says npm `@decnutz/ctxfit`; **404** on the registry (and so is `calas-gateway`) | `npm view @decnutz/ctxfit version` |
| 2 | same | 1 sdist test (`demo.tape`) fails when tests run from the sdist | unpacked sdist, `pytest` on 3.10 |
| 3 | Supply-chain reviewer | Tests contain contiguous key-shaped strings (AWS, GitHub, Anthropic `sk-ant-`, Slack, JWT, PEM) that ship in the sdist and would trip GitHub push protection / scanners | `grep -E` over tests |
| 3 | same | Wheel: zero runtime dependencies (clean venv lists only ctxfit); no personal paths, no secrets in artifacts (demo transcripts use `/home/dev/app`) | `pip list`; grep over unpacked wheel + sdist |
| 4 | Docs / DX writer | README top example output was invented ("96,400 -> 41,150 [heuristic]", but the default is `pieces`); summary text said images are "pinned" while README says they are not (README right) | read vs run |
| 4 | same | "About 1 s per 5 MB" is optimistic: measured 0.3-0.6 s per MB here (4.5 MB single repeated line 2.4 s; a 109 MB / 2,000-output file took 66 s) | timing script, `ctxfit fit huge.json` |
| 5 | CI/CD integrator | CI ran Python 3.12 on Linux only; nothing built the package or installed the wheel; Windows never exercised | read `.github/workflows/proof.yml` |
| 6 | Hostile-input fuzzer | `[{"type":"text",...}, null]` or a string inside an Anthropic `content` list -> `AttributeError` traceback (31 + 24 of 3,000 fuzz cases) | a random-input loop over `ctxfit.fit` (3,000 cases) |
| 6 | same | **With tiktoken installed, any text containing `<|endoftext|>` crashes `fit()`** with `ValueError` (tool output from a web page / injection attempt) | `tiktoken` in a scratch venv; `ctxfit.fit([...<\|endoftext\|>...])` |
| 6 | same | **With tiktoken installed but offline, every `ctxfit` command fails** (`auto` only caught ImportError; tiktoken downloads its vocabulary on first use) | `HTTPS_PROXY=http://127.0.0.1:9 TIKTOKEN_CACHE_DIR=<empty> ctxfit demo` |
| 6 | same | Read-only output folder / output file replaced while another process holds it: fine on Linux (as non-root); Windows `os.replace` can raise `PermissionError` | `runuser -u nobody`; static review |
| 7 | Performance engineer | Any text with one non-ASCII character (every cut marker has `…`) was counted by a per-character Python loop (~2x slower on that path) | cProfile of `fit()` on 20 x 10 KB outputs |
| 7 | same | 8 threads x 20 `fit()` calls: no errors (estimator cache is locked) | threading loop |
| 8 | Competitor's maintainer | Headroom ships a copy-paste, zero-network smoke test and `headroom wrap`/proxy integrations; ctxfit had no equivalent first command. Its PyPI metadata: Python >= 3.11, a dozen-plus dependencies (litellm, pydantic, click, rich, ...); its README says the proxy/MCP server send an anonymous usage beacon unless turned off. ctxfit: zero deps, 3.10+, no telemetry (grep: no network imports except the lazy `anthropic` import in `eval`) | `curl https://pypi.org/pypi/headroom-ai/json`; `grep -n "import urllib\|socket\|requests" src` |

## What was built or fixed

All with tests in `tests/test_round4.py` (61 new tests) and the CHANGELOG entry "Round 4".

- CLI: `--version`, `python -m ctxfit`, `ctxfit demo` (also an install smoke test), `--json`
  on `analyze`/`fit`/`estimate`/`demo`, `estimate -`, exit codes documented, one-sentence errors
  for every ordinary file/JSON/permission problem, a catch-all that never prints a traceback
  (`CTXFIT_DEBUG=1` restores it), silent exit on a closed pipe, console-safe output
  (`->`/`ok` fallback, ASCII JSON for `--out -`), `os.replace` retry, `replay` bare-glob fix.
- Library: `pairing="repair"` / `--repair-pairing` (opt-in; default unchanged; refuses to
  drop a message that also holds an image), malformed Anthropic blocks -> clear `TypeError`,
  tiktoken special-token text and offline fallback, faster non-ASCII estimate path (same numbers,
  checked on 20,000 random strings), report wording fix.
- Packaging: LICENSE, `py.typed`, SPDX licence, classifiers, URLs, `MANIFEST.in`,
  single-sourced version, **0.8.0**, shared fixtures regenerated (only the `generator` string).
- CI: Python 3.10 + 3.13 + `windows-latest` matrix, and a build / `twine check` / clean-install job.
- Tests: key-shaped strings assembled at runtime; one repo-only test skips from an sdist.

## Packaging proof (offline build, clean installs)

Built with `python -m build` (setuptools 84) into a scratch directory; `twine check`: PASSED
for sdist and wheel. Wheel contents: 27 entries (package modules, `py.typed`, demo transcripts,
`prices.json`, `licenses/LICENSE`, metadata); no tests, no secrets, no personal paths.
Installed the **wheel (not editable)** into fresh venvs on 3.10, 3.11, 3.12, 3.13, each time from
a folder with a space in its name: `pip list` shows only ctxfit; every module imports under
`python -I`; `ctxfit --version`, `python -m ctxfit --version`, `--help` of all 8 sub-commands,
`demo`, `analyze`, `analyze --format json`, `fit --aggressive`, `--strict fit`, `mask`,
`replay <bundled demo> --json --redact`, `badge`, `estimate` and the README Python snippets
(`fit`, `offload=`) all ran. The test suite from the **sdist** passed against the installed
wheel on 3.10 (443 passed, 14 skipped for missing shared fixtures; that was before the final
small fixes, the full suite below is the final count).
Metadata consistent: author, Repository/Issues/Changelog URLs, MIT (file + SPDX), Python
3.10-3.13 classifiers, `python_requires >=3.10`, `__version__` 0.8.0 = CHANGELOG heading = wheel name.

## Registry check (read-only)

- PyPI `ctxfit`: **0.1.1** only (uploaded 2026-10-20, author Calas Automations Inc.). So today's
  `pip install ctxfit` gives an old release in which `demo`, `--version`, `mask`, `replay`, notes,
  `offload` ... do not exist; the README is for 0.8.0.
- Public API vs 0.1.1 (installed both): every 0.1.1 export and the `fit()` signature are still
  there; additions only (new keyword arguments have defaults). Behaviour differs (default
  estimator falls back to `pieces`, truncation keeps salient lines, JSON number spelling), so it
  is **not an API break**, but a minor bump, not a patch. **Recommended next release: 0.8.0.**
  Versions 0.2-0.7 never existed on PyPI; skipping them is allowed.
- npm: `@decnutz/ctxfit` and `calas-gateway` both 404 (nothing published). GitHub could not be
  checked from here (proxy returns 403), so the repo URL is Dan's word, unverified.

## Scored ideas (user value / differentiation / effort / risk; H M L)

| # | Idea | Value | Diff. | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | `ctxfit --version` | H | L | S | L | DONE |
| 2 | `python -m ctxfit` | M | L | S | L | DONE |
| 3 | `ctxfit demo` (zero-file first run, self-checking) | H | M | S | L | DONE |
| 4 | Ship LICENSE in sdist/wheel | H | L | S | L | DONE |
| 5 | `py.typed` | M | L | S | L | DONE (9 pre-existing mypy errors remain, internal) |
| 6 | SPDX licence + single-source version | M | L | S | L | DONE |
| 7 | Classifiers, URLs, MANIFEST.in | M | L | S | L | DONE |
| 8 | Plain-English file/JSON/permission errors | H | M | S | L | DONE |
| 9 | Catch-all: no traceback for bugs | H | L | S | L | DONE |
| 10 | Silent exit on closed pipe | M | L | S | L | DONE |
| 11 | Legacy console encodings (cp1252/ASCII) | H | M | S | L | DONE (simulated, not on Windows) |
| 12 | Retry `os.replace` on Windows lock | M | L | S | L | DONE (simulated) |
| 13 | `replay` bare-glob fix | M | L | S | L | DONE |
| 14 | `pairing="repair"` / `--repair-pairing` | H | H | M | M | DONE (opt-in, fail closed on images) |
| 15 | Clean error for malformed Anthropic blocks | M | L | S | L | DONE |
| 16 | `--json` everywhere + documented exit codes | H | M | S | L | DONE (`mask`, `replay` already JSON-capable; `badge` writes a file) |
| 17 | tiktoken: special tokens + offline fallback | H | M | S | L | DONE |
| 18 | CI matrix incl. Windows + package job | H | L | S | L | DONE (never run; first run is the test) |
| 19 | Estimator non-ASCII fast path | L | L | S | L | DONE (identical numbers) |
| 20 | Key-shaped test strings built at runtime | M | L | S | L | DONE |
| 21 | PUBLISH-CHECKLIST for Dan | H | L | S | L | DONE |
| 22 | Fix the 9 mypy errors / run mypy in CI | M | L | M | L | NOT-DONE |
| 23 | Shell completions | L | L | S | L | NOT-DONE |
| 24 | Async `afit()` | M | M | M | M | NOT-DONE (needs a design; no request seen) |
| 25 | Image / document token estimates | M | M | M | M | NOT-DONE (provider formulas unverified) |
| 26 | Per-tool policies (`exclude_tools`) | M | M | M | M | NOT-DONE |
| 27 | Accept JSONL / Claude Code transcripts in `fit` | M | H | M | M | NOT-DONE |
| 28 | Trusted-publishing release workflow, attestations | M | L | M | M | NOT-DONE (publishing is Dan's call) |
| 29 | Answer-quality evaluation of `fit()` | H | H | L | M | NOT-DONE (needs paid model calls) |
| 30 | Case-insensitive globbing / UNC test for `replay` on Windows | L | L | S | L | NOT-DONE (no Windows) |
| 31 | Faster salient-line scan + estimator (single pass) | M | L | M | M | NOT-DONE (0.3-0.6 s/MB is acceptable; no regression found) |
| 32 | TypeScript port published to npm | M | M | M | L | NOT-DONE (separate repo; README no longer claims it) |

## Pass B: attack the new code

- Re-fuzzed the CLI itself: 1,500 random conversations x 5 command lines = 7,500 runs through
  `main()` (analyze, fit, fit --repair-pairing --json, mask, anthropic flavour): exit codes only
  0 or 1, **0 tracebacks, 0 "bug in ctxfit"**.
- Escaping: a call id with RLO, zero-width and ESC characters, and a tool name with ESC, produce
  no raw control or bidi bytes in errors, `--json` or the summary (checked with `cat -v` / `od`).
- Re-ran the quick start from the freshly built wheel on 3.10-3.13 (above), the non-root
  read-only-folder case (`runuser -u nobody`), six processes writing the same `--out`
  (always valid JSON, no `.tmp` left), and the offline-tiktoken case with the real library.
- Found and fixed in this pass: `ctxfit.__main__` ran on import (now guarded; a module walker
  had exited silently), the sdist test above, the JWT/Slack-shaped strings that `ruff format` had
  re-joined after being split (now split with `+`).
- Re-verified every README / CHANGELOG number I touched: demo output pasted from a real run;
  estimator table and needle row re-run with tiktoken (20 / 20, 91.4%; matches); speed paragraph
  re-measured and softened; `RESULTS.md` regenerated (only the version in its header changed).

## Final numbers

| | before | after |
|---|---|---|
| tests | 469 passed | 529 passed, 1 skipped (needs a non-root user) |
| `ruff check` / `format --check` | clean | clean |
| needle bench (own estimator) | 20 / 20, 91.9% | 20 / 20, 91.9% (unchanged) |
| needle bench (tiktoken) | 20 / 20, 91.4% | 20 / 20, 91.4% |
| `--big` bench (this machine, noisy) | 2.0-2.4 s / 0.11-0.14 s / 0.7-0.9 s | 2.0-2.1 s / 0.12-0.15 s / 0.5-0.6 s |
| `running_notes_bench --check` | up to date | regenerated; only the version string differs |

## What could not be tested

- **Windows, at all.** Static review only (path separators, `os.replace` while open, cp1252
  console, CRLF, `shell=True`, signals: none of the last two are used). Console and CRLF cases are
  simulated on Linux. The new CI job on `windows-latest` is the first real test; read its result
  before claiming Windows support. Known Windows limits: `0600` file mode is ignored; `replay`
  globbing is case-sensitive.
- macOS. Python 3.14 (not installed).
- GitHub (403 through the proxy): repo URL, push protection, Actions.
- Competitors' behaviour: only their PyPI metadata/README text was read this round; the needle
  rows for Headroom, LangChain and LLMLingua are round-2/3 numbers, not re-run.
- Real provider calls, non-English transcripts, answer quality after compression.
- A real `pip install` from PyPI of 0.8.0 (it does not exist yet); the wheel was installed from file.
