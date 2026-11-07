# Changelog

## Unreleased (0.8.0 candidate; `__version__` is still 0.7.0 on purpose, see note)

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
- Version: the shared conformance fixtures (`../fixtures`, also run by the TypeScript port)
  embed the library version, so the bump waits for a coordinated fixture regeneration.
- No change to running notes (`ctxfit.notes`), its state, or its spec.
