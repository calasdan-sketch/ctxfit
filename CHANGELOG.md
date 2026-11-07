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

### Notes
- Version: the shared conformance fixtures (`../fixtures`, also run by the TypeScript port)
  embed the library version, so the bump waits for a coordinated fixture regeneration.
- No change to running notes (`ctxfit.notes`), its state, or its spec.
