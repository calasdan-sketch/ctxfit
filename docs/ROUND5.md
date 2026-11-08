# Round 5: the author line, other languages, and the lines that must survive

Scope: a fifth pass on `ctxfit` by specialists from its own field (context engineering, tokenizers,
long-context evaluation, agent harnesses), plus the author line Dan decided on. Branch `expert-2`.
Nothing was published, tagged, pushed or uploaded; no paid API was called.

## How this was run (honest)

There was no sub-agent tool in this session, so **one reviewer played each role in turn**, as in
round 4. Every finding below names the command that showed it. They are different angles, not
different minds. Pass B is the same reviewer attacking the new code again.

Environment: Linux, Python 3.11, 3.12 and 3.13 (3.10 is not installed here), network for
read-only registry lookups. **Windows was never run.** tiktoken (cl100k_base) was available in one
scratch environment; the main test environment has none, as a user without the extra would.

## 0. Author line

Dan's decision: the author and maintainer is exactly **Calas Automations Inc.**

| where | state |
|---|---|
| `pyproject.toml` | `authors = [{ name = "Calas Automations Inc." }]` (already right), `maintainers` added; no email anywhere (none was given for this project; none invented) |
| `LICENSE` | `Copyright (c) 2026 Calas Automations Inc.` (already right) |
| README / package | footer `MIT — Calas Automations Inc.`; new `ctxfit.__author__`, `__license__`; `ctxfit --help` ends with "ctxfit is MIT licensed software by Calas Automations Inc." |
| `package.json` | none: this repo is Python only |
| SARIF / JUnit tool metadata, generated-file headers | none exist in this package. The shared-fixture `generator` field is `ctxfit 0.8.0` (no author), unchanged |
| old handle / name variants | grep of the whole tree (src, tests, docs, benchmarks, CI, README): no `DC Enterprises`, `Dan Calas` or `@decnutz` label left (the only `@decnutz` mentions are a test that asserts it is absent and round-4 history). The `calasdan-sketch` hits are real GitHub URLs and were left alone |
| test that pins it | `tests/test_round5.py::test_author_is_the_company_everywhere_it_is_declared`, `test_installed_metadata_names_the_company`, `test_no_other_owner_label_in_the_tree` |

Proof from a freshly built wheel and sdist (not the editable install):

```
python -m build --outdir dist .          # ctxfit-0.8.0-py3-none-any.whl and ctxfit-0.8.0.tar.gz
twine check dist/*                       # both PASSED
unzip -p dist/ctxfit-0.8.0-py3-none-any.whl ctxfit-0.8.0.dist-info/METADATA | grep -E "^(Author|Maintainer)"
  Author: Calas Automations Inc.
  Maintainer: Calas Automations Inc.
tar xzf dist/ctxfit-0.8.0.tar.gz -O ctxfit-0.8.0/PKG-INFO | grep -E "^(Author|Maintainer)"   # same two lines
```

The package repository links point at `https://github.com/calasdan-sketch/ctxfit`. Whether that
repository is public was **not** checked (GitHub is blocked from here); the metadata only claims
what Dan said. If it is private, remove the Repository / Issues / Changelog lines from
`pyproject.toml` before uploading, or the PyPI page will show dead links.

## Baseline (before any change; round-4 code, Python 3.13)

| check | result |
|---|---|
| `pytest` | 529 passed, 1 skipped |
| `ruff check` | clean |
| `needle_bench.py` (own estimator) | 20 / 20 needles, 91.9% saved; 0.7 behaviour 0 / 20 |
| `needle_bench.py` (tiktoken) | 20 / 20, 91.4% |
| `running_notes_bench.py --check` | up to date |
| `needle_bench.py --big` | 1.04 s / 0.11 s / 0.39 s |
| registry | PyPI `ctxfit` = **0.1.1** (uploaded 2026-10-20, author Calas Automations Inc.); tree = 0.8.0 |

## Upgrade path from the registry release (0.1.1)

`pip download ctxfit==0.1.1 --no-deps`, installed into its own venv, next to this tree.

| check | result |
|---|---|
| 0.1.1 has persistent state or config files? | No. It reads a messages file and writes `--out` / `--report`; nothing else. Nothing to migrate |
| `ctxfit analyze --format json` output | same 9 keys in both versions |
| `ctxfit fit --report` JSON of 0.1.1 | all its keys are still in 0.8.0's report; 6 keys added (`facts_*`, `pairing_*`, `passthrough_messages`). Pinned by `test_report_keys_of_0_1_1_are_still_there` |
| 0.1.1's fitted output fed to 0.8.0 | accepted and fitted again (exit 2 = still over the 3,000-token test budget, exactly as 0.1.1 itself says) |
| `fit(budget=...)` in a 0.1.1 script | `TypeError` in both versions (the keyword is `total_budget`); not a regression |
| numbers differ | yes: default estimator `heuristic` (0.1.1) vs `pieces` (0.8.0) gave 14,218 vs 21,037 estimated tokens for the same file; documented since round 2. It is why the release is a minor bump, not a patch |

## Specialists and findings

| # | Role | Finding | Proved by |
|---|---|---|---|
| 1 | LLM context-engineering specialist | **A real `ERROR ... FATAL` line was dropped when 15 `WARN ... transient error ... retrying` lines came before it.** All 16 qualified as "salient"; ties went to the earlier line, 1,200 characters held 14 | `/tmp` flood script: 900-line log, 15 WARN distractors, needle at 90%: 1 of 20 seeds kept it. `needle_bench.py --flood` (git stash of the fix): 24 / 25 |
| 1 | same | Wants to keep the output of the file being edited verbatim while still cutting noisy `bash` output | no way existed; see "Built" (`tool_limits`) |
| 2 | Tokenizer engineer | **Non-English text was badly mis-counted.** Every non-ASCII symbol cost 0.7 token: Greek -32%, Hebrew -40%, Hindi -45%, Thai -26%, Arabic -17%, Russian +29%; accented Latin up to -33%; base64 about half the real count (an inlined screenshot in a tool output) | held-out sentences in `benchmarks/held_out_text.py` vs tiktoken cl100k_base; `estimator_bench.py` |
| 2 | same | My first fix of the base64 pass was **quadratic** (`str.replace` per match): 100k blob-like runs would take minutes | `perf.py`: 100k runs in a 5 MB text: 0.58 s after the rewrite to one `re.sub` pass |
| 2 | same | The path of a 3 MB single line got slower (0.11 to 0.19 s) because "has a digit/upper/lower" ran as a Python loop; now three C-level searches (0.09-0.12 s) | `needle_bench.py --big`, three runs |
| 3 | Long-context evaluation researcher | The needle benchmark had one distractor class (none). A flood of look-alike but harmless lines is the realistic failure | new `log_flood` generator, seeded; kept out of the 20-case table so competitor rows stay comparable |
| 3 | same | Limit that no ranking fixes: if the harmless lines are also ERROR-level, nothing structural tells them from the real one | stated in README ("use `audit_facts=True`"), not "fixed" |
| 4 | Agent-harness developer | **`fit()` changed messages it had nothing to do with.** A conversation with no tool calls (generic dict shape) came back with `"section": "system"` and `"pinned": true` added to the system message; an API that rejects unknown fields would refuse the request. (Whether OpenAI or Anthropic reject them was **not** tested: unverified) | hypothesis found it in under a second: `fit([{"role":"system","content":""},{"role":"user","content":""}])` |
| 4 | same | **LangChain messages lost their tool-call name and arguments.** The adapter read only `{"function": {...}}`; LangChain's own `{"name","args","id","type":"tool_call"}` came through with name `None` and arguments `""` | `tests/test_round5.py::test_langchain_native_tool_calls_keep_name_and_arguments` (found while writing the `tool_limits` test) |
| 4 | same | Claude Code users have the session as a `.jsonl` transcript and had to export it to a JSON array first | now read directly (see "Built") |
| 4 | same | A JSON-lines file with U+2028 / U+0085 / form feed *inside a string* would be cut in the wrong place by `str.splitlines()` (Node writes U+2028 unescaped) | read-through in Pass B; test `test_json_lines_with_a_raw_line_separator_inside_a_string` |
| 5 | Property-based and mutation-testing specialist | `TruncateToolOutput(tail_chars=0)` **made the message bigger** (15,000 to 15,595 characters): `text[-0:]` is the whole text | property-test probe; git stash of the fix |
| 5 | same | Mutation check (33 conditions flipped one by one in throw-away copies): first run 23 of 33 noticed. The 10 survivors led to 7 new tests and a corrected mutant list | `scripts/mutation_check.py` |
| 5 | same | Golden files: every output format (fit output x 3 shapes, report JSON, summary text, analyze JSON, analyze table) is now compared byte for byte | `tests/golden/`, `UPDATE_GOLDEN=1 pytest` to refresh on purpose |
| 6 | Release engineer | Putting `scripts/` in the sdist made the old `demo.tape` guard stop skipping, and a test that reads `.github/workflows/proof.yml` failed when the sdist's tests ran against an installed wheel | `pytest` from the unpacked sdist, against the wheel in a clean venv; guard moved to the workflow file |
| 7 | Technical writer | My first draft of the README and CHANGELOG numbers did not match what I had measured (it claimed accented French/German/Spanish were "20-60% low" before; the original code measured -33% to +3%). Rewritten from the re-run | `git stash` of `estimate.py`, `before.py` vs after |
| 8 | Competitor's maintainer (read-only) | Headroom `headroom-ai` 0.43.1 on PyPI (unchanged since the last check; released 2026-11-05): Python >= 3.11, 114 `requires_dist` entries (many behind extras), README advertises `headroom wrap` for many agents, a proxy and an MCP server, and an anonymous usage beacon on by default for the proxy and MCP server. No per-tool control, transcript input or script-aware counting is mentioned in its README (3,283 characters) | `curl https://pypi.org/pypi/headroom-ai/json` |
| 8 | same | LangChain `langchain-core` 1.7.6 `trim_messages(messages, *, max_tokens, token_counter, strategy, allow_partial, end_on, start_on, include_system, text_splitter)`: whole-message trimming, no per-tool setting | read from the downloaded wheel (`utils.py`) |

Competitor benchmark rows in the README were **not** re-run this round (only their versions and
public interfaces were re-read).

## Built or fixed (all with tests in `tests/test_round5.py`, 61 tests)

- Estimator: per-script rates (Cyrillic, Greek, Hebrew, Arabic, Indic scripts, Thai), accented
  Latin letters, base64 blobs, one-pass implementation. Held-out sentences 0.94-1.06 of cl100k_base.
- `Message.to_dict` / generic output: only keys that carry information (no invented `section` /
  `pinned`).
- Salient lines: ERROR/FATAL/CRITICAL level tokens and strong words rank above WARN/INFO/DEBUG lines.
- `TruncateToolOutput(tail_chars=0)` fixed.
- LangChain adapter reads both tool-call shapes; `args` returned as a dict.
- `tool_limits` on `fit()` / `FitPipeline` / CLI `--tool-limit NAME=N`, plus `protect_tools` on
  `CompactJson` and `Deduplicate`.
- JSON lines and Claude Code transcripts as input to `analyze` / `fit`.
- Author line pieces (above), `maintainers`, `hypothesis` in the dev extra, sdist includes tests,
  scripts and docs, `scripts/mutation_check.py`, `benchmarks/held_out_text.py`, `--flood`.

## Pass B: attacking the new code

- Read the whole diff again as an attacker. Found: the U+2028 line-splitting bug (fixed), the
  sdist guard regression (fixed), the LangChain claim in my own README that no test covered (which
  led to the LangChain defect above), and README / CHANGELOG numbers I had written from memory
  (rewritten from re-runs).
- Fuzzed the new input path: 1,500 random JSON-lines files (bad lines, NUL, 300-deep nesting,
  lone surrogates, transcript fragments, `--tool-limit` garbage including an ESC byte) through
  `main()`: exit code only 0 or 1, **0 tracebacks, 0 "bug in ctxfit", 0 raw ESC bytes on stderr**.
- Hostile estimator input (NUL, lone surrogates, bidi, 5,000 repeated accented letters, 999
  emoji): no error. Eight threads x 20 `fit(tool_limits=...)` calls: no error, one distinct output.
- `LC_ALL=C PYTHONIOENCODING=cp1252 ctxfit fit greek.jsonl --out -`: ASCII-escaped JSON, exit 0.
  No network (`HTTPS_PROXY` pointing nowhere): `fit` works with the built-in estimator.
- Built the sdist and wheel, `twine check` PASSED, installed the **wheel (not editable)** into
  clean venvs on 3.11, 3.12 and 3.13 from a folder with a space in its name: `--version`,
  `demo` (output identical to the README block), JSON-lines `fit`, transcript `fit --tool-limit`,
  the new README Python snippet. The tests of the unpacked sdist (round 4 + round 5 + the rest:
  all of them) passed against the installed wheel on 3.13.
- Mutation check, final run: see the table below.

## Scored ideas (user value / differentiation vs real competitors / effort / risk; H M L)

| # | Idea | Value | Diff. | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | Script-aware token estimator (Greek, Cyrillic, Hebrew, Arabic, Indic, Thai) | H | M | S | L | DONE |
| 2 | Accented Latin letters stay inside words | M | L | S | L | DONE |
| 3 | base64 blob counting | M | M | S | L | DONE |
| 4 | Severity-graded salient-line ranking | H | M | S | L | DONE |
| 5 | `log_flood` needle case (`--flood`) | M | M | S | L | DONE |
| 6 | No invented keys on dict output | H | L | S | L | DONE |
| 7 | `tail_chars=0` growth fix | L | L | S | L | DONE |
| 8 | LangChain-native tool calls | M | L | S | L | DONE |
| 9 | Per-tool limits / exclusions (carried from round 4 #26) | H | H | M | M | DONE (`tool_limits`, `protect_tools`) |
| 10 | JSON lines and Claude Code transcripts in `fit` (round 4 #27) | H | H | M | M | DONE |
| 11 | Hypothesis property tests | H | L | M | L | DONE |
| 12 | Golden files for every output format | M | L | S | L | DONE |
| 13 | Mutation check script | M | L | M | L | DONE |
| 14 | Author pins, `__author__`, `maintainers`, help epilog | H | L | S | L | DONE |
| 15 | `ctxfit calibrate`: fit the estimator's constants to the usage numbers recorded in a user's own transcripts (their model, their text) | H | H | L | M | NOT-DONE (best next idea; needs design for how much data is enough) |
| 16 | Claude Code hook / `PreCompact` example that calls `ctxfit fit` | H | H | M | M | NOT-DONE (the hook input/output contract was not re-verified from primary docs; will not guess spec fields) |
| 17 | MCP server exposing `fit` / `analyze` | M | M | M | M | NOT-DONE (adds a dependency; the zero-dependency promise is a selling point) |
| 18 | Glob patterns in `tool_limits` (`mcp__*`) | M | M | S | L | NOT-DONE |
| 19 | When many error lines tie, spread picks over first/middle/last instead of first | M | L | S | M | NOT-DONE (no ground truth on which is better) |
| 20 | Rates for Armenian, Georgian, Ethiopic, Khmer, Burmese | L | L | S | L | NOT-DONE (rare) |
| 21 | Fix the 9 mypy errors / run mypy in CI (round 4 #22) | M | L | M | L | NOT-DONE (mypy not installed here) |
| 22 | Shell completions (round 4 #23) | L | L | S | L | NOT-DONE |
| 23 | Async `afit()` (round 4 #24) | M | M | M | M | NOT-DONE |
| 24 | Image / document token estimates (round 4 #25) | M | M | M | M | NOT-DONE (provider formulas unverified) |
| 25 | Trusted-publishing release workflow (round 4 #28) | M | L | M | M | NOT-DONE (publishing is Dan's call) |
| 26 | Answer-quality evaluation of `fit()` (round 4 #29) | H | H | L | M | NOT-DONE (needs paid model calls) |
| 27 | Windows glob case-insensitivity / UNC for `replay` (round 4 #30) | L | L | S | L | NOT-DONE (no Windows) |
| 28 | Single-pass salient scan + estimator (round 4 #31) | M | L | M | M | NOT-DONE (the estimator got ~40% slower per MB; 0.08 s/MB is still small next to the 0.3-0.6 s/MB of the whole fit) |
| 29 | TypeScript port on npm (round 4 #32) | M | M | M | L | NOT-DONE (separate repo) |
| 30 | De-duplicate repeated identical tool *calls* (same name + arguments), not just identical outputs | M | M | M | M | NOT-DONE |

## Mutation check (final run)

`python scripts/mutation_check.py` breaks 35 conditions one at a time in throw-away copies of the
repository. Result: see "Final numbers". The survivors that remain are understood:

- `if any(messages[j].pinned for j in idxs)` in the sliding window: **equivalent**. The very next
  check covers a superset of the same messages, so removing the first changes only a log line.
- `if after >= before` in `CompactJson` -> `>`: differs only when the table and the JSON have
  exactly the same estimated size, where leaving the text alone or replacing it is equally good.

## Final numbers

| | before | after |
|---|---|---|
| tests | 529 passed, 1 skipped | 590 passed, 1 skipped (needs a non-root user) |
| `ruff check` / `ruff format --check` on `src tests benchmarks scripts` | clean | clean |
| needle bench, own estimator (20 cases) | 20 / 20, 91.9% | 20 / 20, 91.9% (unchanged) |
| needle bench, tiktoken (20 cases) | 20 / 20, 91.4% | 20 / 20, 91.4% (unchanged) |
| `--flood` case (5 cases) | 4 / 5 (24 / 25 overall) | 5 / 5 (25 / 25) |
| estimator vs cl100k, held-out non-English sentences | -45% to +29% | 0.94-1.06 of real |
| estimator, English / code / log / JSON | 0.89-1.11 | 0.89-1.11 (unchanged) |
| estimator speed (5 MB English, best of 3) | 0.28 s | 0.40 s (+40%) |
| `--big` bench (this machine, noisy) | 1.04 s / 0.11 s / 0.39 s | 1.15-1.33 s / 0.09-0.12 s / 0.34-0.40 s |
| `running_notes_bench.py --check` | up to date | up to date |
| mutation check | not run | 33 of 35 noticed (final run) |

## What could not be tested

- **Windows, at all** (static review only, as in round 4; the CI matrix job is still the first
  real test). Python 3.10 and 3.14. macOS.
- **GitHub** (403 through the proxy): whether the repository is public, push protection, Actions.
- Claude's own tokenizer: all estimator numbers are against OpenAI's cl100k_base, which is the only
  tokenizer available offline. They say how the estimator behaves, not how many tokens Claude bills.
- Real Claude Code transcripts from a user's machine: the reader was tested on a synthetic
  transcript built from the documented chain fields (`uuid`, `parentUuid`, `message.id`) and on the
  three bundled demo sessions (round 4's replay golden is byte-identical). Format drift in newer
  Claude Code versions would show up as "holds no model request yet".
- Whether OpenAI or Anthropic reject unknown message keys such as `pinned` (the fix removes the
  question; the rejection itself was not tried).
- Real provider calls, answer quality after compression, scripts beyond the ones in the table,
  languages tested with one or two sentences each.
- Competitor benchmark rows were not re-run.
- A real `pip install ctxfit` of 0.8.0: it does not exist on the registry yet.
