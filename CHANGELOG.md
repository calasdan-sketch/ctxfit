# Changelog

## 0.8.0 (not released yet: PyPI still has 0.1.1, the repo was 0.7.0)

Found by putting ctxfit through five expert reviews (LLM app engineer, token-cost
analyst, agent-memory researcher, performance engineer, sceptical agent developer).

### Fixed
- **`fit()` slowed down quadratically with conversation length** (800 tool calls: 10 s,
  2,000: 57 s). The sliding window recounted the whole conversation for every turn it
  considered, and `DropOldestUnpinned` recounted after every drop. Counts are now taken once
  and kept as a running total; the pair index is built once. Same output, measured:
  400 tool calls 1.24 s -> 0.03 s, 1,600 tool calls 0.13 s (was minutes).
  `benchmarks/needle_bench.py --scale` reproduces it.
- **Head+tail truncation silently lost the one line that mattered.** On the seeded needle
  benchmark the old behaviour kept 0 of 20 planted facts. `TruncateToolOutput` now also keeps,
  from the omitted middle, up to `salient_chars` (default 1,200) of error lines and
  odd-one-out lines with their original line numbers (`salient=False` restores the old
  output exactly). `CompactJson` keeps up to `max_outliers` (default 10) rows that differ
  from an otherwise uniform column, or contain an error word, past its 50-row cap
  (`max_outliers=0` restores the old output). Result: 20 of 20, with the saving
  92.0% -> 91.4%.

### Added
- `PieceEstimator` (`estimator="pieces"`), now what `"auto"` falls back to when tiktoken is
  not installed. chars/4 under-counted logs, JSON and id-heavy output by about 40% against
  cl100k_base (ratio 0.60-0.70); the new one is within -11%/+17% on every class tested.
  `"heuristic"` still gives the old chars/4 estimator. Budgets set against estimates in the
  fallback case will now be hit sooner on machine-generated text, which is the point.
- `audit_facts=True` (`ctxfit.fit`, `FitPipeline`, CLI `--audit-facts`): lists the
  paths, URLs, error codes, versions and ids from tool output that no longer appear anywhere
  in the fitted messages (`report.facts_lost`, shown in `summary()`; control and bidi
  characters are escaped). Off by default; skipped above 4M characters of tool output.
- `benchmarks/needle_bench.py`, `needle_corpus.py`, `estimator_bench.py`.

### Round 5: the author line, other languages, and the lines that must survive

Panel: an LLM context-engineering specialist, a tokenizer engineer, a long-context evaluation
researcher, an agent-harness developer, a property-based-testing / mutation-testing expert, a
Claude Code power user and a technical writer (details and the command behind each finding:
`docs/ROUND5.md`). Pass B re-attacked the new code and was run from a freshly built, non-editable
wheel.

#### Fixed
- **A system message came back with keys it never had.** With no tool calls in the conversation
  (so the generic dict shape), `fit()` returned the system message as
  `{"role": "system", ..., "section": "system", "pinned": true}`; a provider API may reject
  unknown fields. Dicts now come back with only the keys that carry information: `section` when it
  differs from what the role implies, `pinned` when the role does not pin by itself. A `fit()` that
  has nothing to do now returns the same text for every message (property-tested).
- **`TruncateToolOutput(tail_chars=0)` made the message bigger** (`text[-0:]` is the whole text):
  15,000 characters became 15,595. Fixed.
- **Error lines were ranked by position only.** When more lines qualified than the 1,200-character
  budget holds, the earliest won, so a real `ERROR ... FATAL` after fifteen `WARN ... transient
  error ... retrying` lines was dropped. A line that calls itself ERROR/FATAL/CRITICAL now outranks
  one that calls itself WARN/INFO/DEBUG, and fatal/panic/traceback/exception words rank higher
  still. New seeded case (`needle_bench.py --flood`): 24 / 25 before, 25 / 25 after. The 20-case
  table is unchanged (20 / 20, 91.4%).
- **The estimator was badly off outside English.** On held-out sentences Greek came out 32% low,
  Hebrew 40%, Hindi 45%, Thai 26%, Arabic 17% low and Russian 29% high; accented Latin text (Polish,
  Czech, Turkish probes) up to 33% low; a base64 blob (an inlined screenshot) about half the real
  count. Each of these scripts now has its own measured rate, accented letters stay inside their
  word, and base64 runs are counted by the character. The eight held-out sentences land at
  0.94-1.06 of cl100k_base, accented Latin at most 18% low, base64 within 2%. English, code, log
  and JSON numbers are unchanged; counting is about 40% slower per MB. Budgets set against
  estimates for such text will now be hit sooner, which is the point.
- **LangChain messages lost their tool-call name and arguments.** The adapter read only the
  OpenAI shape (`{"function": {...}}`); LangChain's own `AIMessage.tool_calls` entries
  (`{"name", "args", "id", "type": "tool_call"}`) came through with an empty name and empty
  arguments, so a fitted conversation sent back to LangChain had hollow tool calls. Both shapes
  are read now, and `args` goes back as a dict (what LangChain expects) when it is a JSON object.
- A JSON-lines file with U+2028 / U+0085 / form feed inside a string is split on `\n` only.
- The `demo.tape` sdist guard in `tests/test_notes_v6.py` now looks for the CI workflow it reads.

#### Added
- **`fit(..., tool_limits={"read_file": 0, "bash": 8000})`** and `ctxfit fit --tool-limit NAME=N`
  (repeatable): `0` leaves a tool's output alone (no truncation, compaction or de-duplication), a
  number of at least 300 sets that tool's cut point. `TruncateToolOutput(tool_limits=)`,
  `CompactJson(protect_tools=)`, `Deduplicate(protect_tools=)` for custom lists.
- **JSON lines and Claude Code transcripts as input** to `analyze` and `fit`: one message per line,
  or a `~/.claude/projects/**/*.jsonl` transcript (the longest rebuilt request is used; the CLI says
  so on stderr and that the system prompt and tool definitions are not in the file).
- `ctxfit.__author__` / `__license__`; `maintainers` in the package metadata; an "MIT licensed
  software by Calas Automations Inc." line in `ctxfit --help`.
- `tests/test_round5.py` (61 tests): author pin, hypothesis property tests (OpenAI and Anthropic
  shapes: input untouched, pairing valid, never bigger, newest message and system prompt
  unchanged, deterministic, re-fit never grows), golden files for every output format
  (`tests/golden/`), held-out estimator sentences, per-tool limits, JSON lines, transcripts.
- `scripts/mutation_check.py`: breaks 35 conditions one at a time in throw-away copies and reports
  which ones the tests miss (33 of 35 caught in the final run; the 2 survivors are explained in
  `docs/ROUND5.md`).
- `benchmarks/held_out_text.py`, estimator rows for eight scripts and base64 in
  `benchmarks/estimator_bench.py`, `needle_bench.py --flood`.
- `hypothesis` in the `dev` extra; `tests/`, `scripts/` and `docs/*.md` in the sdist.

#### Compatibility
Nothing is removed. Behaviour that changes: dict output no longer carries `section`/`pinned`
that the input did not have (and still carries them when they matter); token estimates for
non-English text and base64 are higher; the salient-line choice can differ when more lines
qualify than fit; `to_lc_messages` writes `args` as a dict when the arguments are a JSON object
(it wrote the JSON text). Reports from 0.1.1 and 0.7.x read the same (the report only gained keys).

### Round 4: what a stranger gets from `pip install ctxfit`

Panel: a first-hour user following only the README, a release/packaging engineer, a supply-chain
reviewer, a docs writer, a CI/CD integrator, a hostile-input fuzzer, a performance engineer and a
competitor's maintainer (details and the commands behind each finding: `docs/ROUND4.md`).

#### Fixed
- **`ctxfit --version` did not exist** and `python -m ctxfit` failed. Both work.
- **Crash on a console that cannot show `→ − ✓`** (Windows cp1252/cp437, `LC_ALL=C`):
  `ctxfit fit` died with a `UnicodeEncodeError` after doing the work. Output now falls back to
  `->`, `-`, `ok`; `--out -` writes plain-ASCII JSON on such a console.
- **Tracebacks for ordinary mistakes.** A conversation with an orphaned tool result (common in
  sliced transcripts), a `null` or string inside an Anthropic `content` list, or any unexpected
  error printed a Python traceback. They are now one plain sentence and exit code 1.
  Missing file, folder, empty file, invalid JSON, non-UTF-8 file, wrong JSON shape, read-only
  folder and missing output folder each get their own sentence naming the file.
- **With `tiktoken` installed (the recommended extra), any text containing `<|endoftext|>` (or
  another special-token spelling) made `fit()` raise `ValueError`.** A web page, a tokenizer
  discussion or a prompt-injection attempt in a tool output was enough. Such text is now
  counted as ordinary text.
- **With `tiktoken` installed but no network** (it downloads its vocabulary on first use),
  `ctxfit` and `fit(estimator="auto")` failed with a connection error. `auto` now falls back
  to the built-in estimator (the report names the one used); `estimator="tiktoken"` says what to do.
- `ctxfit analyze file | head` printed `Broken pipe` as an error. It exits quietly now.
- **`ctxfit replay '*.jsonl'`** (a bare pattern in the current folder) found 0 files.
- Replacing the `--out` file is retried on `PermissionError` (Windows holds files an editor or
  virus scanner has open), then reported as a sentence.
- The summary note for images and thinking blocks said "pinned"; they are handed back unchanged
  and are *not* pinned (README was right, the report text was wrong).
- Estimator: any text containing a non-ASCII character (such as the `…` in every cut marker) was
  counted by a per-character Python loop; now one regex pass. Same numbers (checked on 20,000
  random strings), about 2x faster on that path.
- Tests no longer contain contiguous key-shaped strings (AWS, GitHub, Anthropic, Slack, PEM, JWT
  shapes are assembled at runtime), so GitHub push protection and PyPI scanners have nothing to flag.

#### Added
- `ctxfit demo`: fits a built-in made-up agent run and checks three planted facts survived
  (exit 1 if one is lost). Needs no files, network or key; doubles as an install smoke test.
- `fit(..., pairing="repair")` / `FitPipeline(pairing=)` / `ctxfit fit --repair-pairing`:
  drops unmatched tool calls/results (and repeated ids) instead of raising; the report counts
  and lists them; a message that also holds images is never dropped (error instead).
  Default stays `"strict"`. `ctxfit.repair_pairing()` is exported.
- `--json` for `analyze`, `fit`, `estimate`, `demo`; `ctxfit estimate -` reads standard input.
  Exit codes are documented.
- Packaging: `LICENSE` file (the MIT licence was declared but not shipped), `py.typed`,
  SPDX `license = "MIT"` (needs setuptools 77+ to build), Python 3.10-3.13 / OS / Typing
  classifiers, Repository / Issues / Changelog URLs, `MANIFEST.in` (sdist now carries
  CHANGELOG, benchmarks and docs), single-sourced version.
- CI: tests on Python 3.10 and 3.13 and on `windows-latest`, plus a job that builds the sdist and
  wheel, runs `twine check`, installs the wheel into a clean environment and runs `ctxfit demo`.
- `tests/test_round4.py` (61 tests).

#### Not changed on purpose
`fit()` defaults, strategies and output are as in round 3 (benchmark rows identical).

### Round 3 (second expert pass, same unreleased line)

Panel: a computer-use/vision agent developer, an Anthropic-API compliance reviewer, a
security tester (terminal escapes, JSON injection), a Windows/ops engineer, a performance
engineer, a data-fidelity auditor, and a Headroom-side engineer (docs and release re-check).
Pass B re-read the new diff adversarially and fuzzed it (about 500 random conversations: no crash, no
input mutation, output re-fits cleanly).

#### Fixed
- **`fit()` silently deleted images, documents, thinking blocks (signatures) and
  `cache_control`**, and any provider field it did not model (`refusal`, `reasoning_content`,
  a participant `name` on an assistant message with tool calls). OpenAI `image_url` parts were
  replaced by the text "[non-text part omitted]". Such messages are now handed back unchanged
  (not rewritten, still droppable whole), the rest is copied onto the rebuilt message, and the
  report says how many messages were passed through (`passthrough_messages`).
- **Huge outputs were slow**: the estimator re-counted the same text about 30 times per
  `fit()` and the piece estimator looped per character. Counts are cached per text and the
  per-character work moved into one regex pass (identical numbers, checked on 3,000 random
  strings). 4.8 MB log: 6.1 s -> 1.0 s; 3 MB single line 1.7 s -> 0.14 s; 60k-row JSON 2.6 s -> 0.4 s.
- **Deeply nested JSON crashed `fit()`** with `RecursionError` (a `[[[...` tool output). Now left alone.
- **`CompactJson` altered data**: `10.50` became `10.5`, `1e400` became `inf`, an object with a
  repeated key silently kept one value, Python's `True`/`None` appeared in flat views, and a key
  or value containing a newline could forge a table row or a `- key: value` line. Numbers keep
  their spelling, repeated-key objects are left alone, JSON spelling is used, line breaks are flattened.
- **Terminal escapes**: `ctxfit analyze` printed raw control and bidi characters from message
  text, roles and kinds. They are now escaped as `\uXXXX`.
- **CLI**: files with a BOM or UTF-16 (PowerShell `>`) failed; `--flavor nope` was accepted; `--strict`
  overflow printed a traceback; `PermissionError` printed a traceback; `--out -` wrote a file named
  `-`; output files could be left half-written. Fixed (see README, CLI). `fit(flavor="nope")` is a `ValueError`.
- The summary printed `−-56.6%` when a custom strategy made the context bigger.

#### Added
- `fit(..., offload=dict)` / `FitPipeline(offload=)`: keeps the full text of every truncated tool
  output and names its id in the marker; `ctxfit.output_id`. Off by default.
- `benchmarks/needle_bench.py --big`; `tests/test_expert3.py` (32 tests; 437 -> 469 in all).

#### Behaviour changes (backward compatible in practice)
- Flat JSON views show `true`/`null` instead of `True`/`None`; floats keep their written form.
- Messages with non-text parts are no longer rebuilt from text (see Fixed).

#### Not built (ideas scored and left out, reasons)
Image/document token estimates (provider formulas unverified); stripping old screenshots on request;
substring JSON extraction inside mixed text; near-duplicate merging (unexplainable); a `fit()` recall
tool schema (the notes layer has one; an `offload` dict is enough for now); async `afit`; a
LangChain object passthrough; streaming input; per-tool policies (`exclude_tools` for `fit()`);
a PyPI/npm release and a fixtures regeneration (waits for the coordinated version bump);
answer-quality evaluation of `fit()` output (needs a paid model call; `ctxfit eval` covers the notes layer).
Still untested: a real Windows machine, non-English transcripts, real provider calls.

### Notes
- Version: bumped to 0.8.0 in round 4 (single source: `ctxfit.__version__`; `pyproject.toml`
  reads it). The shared conformance fixtures embed the library version in their `generator`
  field; they were regenerated with `fixtures/generate.py` and only that string changed (61 files).
  The TypeScript port only checks that the field looks like `ctxfit <digits>`.
- No change to running notes (`ctxfit.notes`), its state, or its spec. Re-running
  `benchmarks/running_notes_bench.py` changed only the version in the header of `RESULTS.md`.
