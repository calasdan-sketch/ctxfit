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
