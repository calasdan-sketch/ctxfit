# ctxfit — drop-in context-window optimizer for any agent loop

Attention cost scales quadratically: doubling context roughly quadruples
compute, and every extra chunk makes the agent slower, more expensive, and
*less accurate*. Fifty turns into a run you hit the limit — and each tool
result sits uncompressed in context for the rest of the run.

**ctxfit** is the standalone layer every agent team hand-rolls and nobody
ships: structural context compression with budget enforcement. No LLM calls,
no framework lock-in, deterministic.

```python
import ctxfit

fitted, report = ctxfit.fit(messages, total_budget=128_000)
print(report.summary())
# est. tokens: 96,400 → 41,150 (−57.3%) [heuristic]
#   retrieved   71,200 →   18,400
#   history     24,100 →   21,650
#   truncate_tool_output: −52,800 est. tokens, 3 message(s)
#   within budget ✓
```

Companion story with [ContextRouter](https://calasautomations.com):
*"ContextRouter picks what goes in, ctxfit keeps it lean."*

- PyPI: `ctxfit` · npm: `@decnutz/ctxfit`

---

## What it does

**Structural compression** (no API call, no latency, reproducible):

| Strategy | What it does |
|---|---|
| `Deduplicate` | Exact-duplicate message bodies → pointer to the first occurrence |
| `CompactJson` | JSON arrays → markdown tables; nested objects → flat key paths (only when it saves tokens) |
| `TruncateToolOutput` | Oversized tool results → head + tail with an explicit omission marker |
| `SlidingWindow` | Oldest turns dropped while history overflows (pinned + latest turn never dropped) |
| `DropOldestUnpinned` | Aggressive-mode last resort: oldest-first dropping until the total fits |

**Budget enforcement:** declare a total budget and per-section split —
default **system 10% / retrieved 55% / history 25% / scratchpad 10%**.
Strategies compress toward the budgets; surviving overruns produce warnings
(`strict=True` raises `BudgetOverflow` instead).

**Token estimation:** heuristic by default (chars/4, CJK-adjusted, zero
dependencies); automatically upgrades to tiktoken's `cl100k_base` when
installed. Pluggable — implement `count(text)` for your own tokenizer.

**Framework adapters:** raw message-list dicts are first-class
(`ctxfit.fit` accepts them directly). OpenAI-shaped (also covers
OpenAI-compatible endpoints like Ollama/freellmapi), Anthropic block-shaped,
and duck-typed LangChain-shaped helpers convert at the boundary; the core
never sees framework shapes.

## Install

```bash
pip install ctxfit
# optional: better estimates
pip install "ctxfit[tiktoken]"
```

## CLI

```bash
# Breakdown without changing anything (exit 2 if over budget — CI-friendly)
ctxfit analyze convo.json --budget 128000
ctxfit analyze convo.json --format json --top 20

# Compress to fit, write fitted messages + report
ctxfit fit convo.json --out fitted.json --report report.json --aggressive

# Section overrides + strict mode
ctxfit --budget 64000 --budgets retrieved=0.6,history=0.2 --strict fit convo.json

# Quick estimate
ctxfit estimate "some text"
```

`convo.json` is a JSON array of message dicts (generic `{"role","content"}` or
OpenAI-shaped; `--flavor` overrides auto-detection).

## Python API

```python
import ctxfit
from ctxfit import Budget, FitPipeline, TruncateToolOutput

# One call
fitted, report = ctxfit.fit(
    messages,
    total_budget=128_000,
    budgets={"retrieved": 0.6, "history": 0.2},  # partial overrides OK
    strategies="default",      # or "aggressive", or [Strategy(...), ...]
    estimator="auto",          # "heuristic" | "tiktoken" | your own
    strict=False,              # True -> raise instead of warn
    flavor="auto",             # "openai" | "anthropic" | "langchain" | "generic"
)

# Full control
pipeline = FitPipeline(
    strategies=[TruncateToolOutput(max_chars=8000, when="always")],
    estimator="heuristic",
    budget=Budget(total=64_000),
    aggressive=True,
)
fitted, report = pipeline.fit(messages, flavor="openai")

# Measure-only
info = pipeline.analyze(messages)   # section usage, warnings, top consumers
```

`FitReport` carries `tokens_before/after` per section, per-strategy savings,
`warnings`, `over_budget`, plus `summary()` and `to_dict()`.

### Custom strategies

```python
from ctxfit import Strategy, Message

class RedactSecrets(Strategy):
    name = "redact_secrets"
    target_sections = ("retrieved", "history")

    def _apply(self, messages, ctx, report):
        out = []
        for m in messages:
            new = m.content.replace("sk-live-", "sk-live-[REDACTED]")
            if new != m.content:
                report.log("redacted secret-like token", 0)
            out.append(Message(role=m.role, content=new, kind=m.kind,
                               name=m.name, call_id=m.call_id,
                               section=m.section, pinned=m.pinned, meta=m.meta))
        return out, report
```

Contract: pure function (never mutate inputs — return new objects), preserve
message order, never modify `tool_call` content, never orphan half of a
tool_call/tool_result pair (the pipeline raises `PairingBroken` if you do),
and respect `pinned`.

### Pinning

System messages are pinned automatically. Pin anything that must survive:

```python
{"role": "user", "content": task_brief, "pinned": True}
```

Pinned messages are never dropped; truncated only with explicit opt-in
(`TruncateToolOutput(truncate_pinned=True)`).

## Safety rules (load-bearing)

1. **Pair-atomicity.** Tool calls and results are linked by ID and kept,
   truncated, or dropped *together*. `validate_pairing()` runs before and
   after the pipeline; violations raise `PairingBroken`.
2. **Tool-call arguments are sacred.** No strategy ever modifies `tool_call`
   content — API validity is non-negotiable.
3. **Soft enforcement by default.** ctxfit degrades to warnings, never crashes
   your agent loop. Opt into `strict=True` if you want exceptions.
4. **Honest numbers.** Every figure is an estimate, labeled as such.

## Limitations (read before trusting it blindly)

- **Estimates, not tokens.** The heuristic is roughly ±25% on mixed prose/code
  and worse on CJK/emoji-heavy text. **Budget 80–90% of your model's true
  window** as a safety margin. tiktoken (optional) narrows the error but no
  heuristic equals your model's actual tokenizer.
- **Lossy by design.** ctxfit deletes and rewrites content to fit budgets.
  Pin anything that must survive verbatim.
- **Head+tail truncation assumes the middle matters least.** If the critical
  line is item #500 of 1,000, it will be lost. Pin or pre-filter such payloads.
- **JSON compaction is whole-message only** (no substring extraction);
  non-uniform object arrays are left alone; table cells truncate at 120 chars
  (marked `…`); only applied when it actually saves tokens.
- **Exact-match dedup only.** Near-duplicates are not merged — deliberate;
  fuzzy matching is unexplainable and risky.
- **No LLM summarization, ever.** Structural compression only. (Subclass
  `Strategy` with your own summarizer if you want semantic compression —
  that's your API bill, not ours.)
- **Turn counting is user-message-led.** Tool-heavy loops with few user turns
  should use `aggressive=True`.
- **`to_anthropic` emits system messages as `{"role": "system"}` dicts** —
  hoist them to the API's top-level `system` parameter yourself.
- **Section overrides are a power tool.** Mislabeling sections mostly hurts
  you (wrong protection, wrong truncation eligibility); assignment from
  role/kind is deterministic and never inferred from content length.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest          # 104 tests
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

## License

MIT — Calas Automations Inc.
