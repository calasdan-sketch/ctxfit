# ctxfit — drop-in context-window optimizer for any agent loop

Attention cost scales quadratically: doubling context roughly quadruples
compute, and every extra chunk makes the agent slower, more expensive, and
*less accurate*. Fifty turns into a run you hit the limit — and each tool
result sits uncompressed in context for the rest of the run.

**ctxfit** is the standalone layer every agent team hand-rolls and nobody
ships: structural context compression with budget enforcement. No LLM calls,
no framework lock-in, deterministic. Opt-in running notes for long agent runs.

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
| `CompactJson` | JSON arrays → markdown tables (first 50 rows plus up to 10 outlier rows); nested objects → flat key paths (only when it saves tokens) |
| `TruncateToolOutput` | Oversized tool results → head + tail, plus the error lines and odd-one-out lines of the omitted middle (with original line numbers), and an explicit omission marker |
| `SlidingWindow` | Oldest turns dropped while history overflows (pinned + latest turn never dropped) |
| `DropOldestUnpinned` | Aggressive-mode last resort: oldest-first dropping until the total fits |

**Budget enforcement:** declare a total budget and per-section split —
default **system 10% / retrieved 55% / history 25% / scratchpad 10%**.
Strategies compress toward the budgets; surviving overruns produce warnings
(`strict=True` raises `BudgetOverflow` instead).

**Token estimation:** a dependency-free estimator by default (`pieces`: counts
words, digit runs and punctuation separately, CJK-adjusted); automatically
upgrades to tiktoken's `cl100k_base` when installed. `estimator="heuristic"`
is the old chars/4. Pluggable — implement `count(text)` for your own
tokenizer. Measured against cl100k_base (`benchmarks/estimator_bench.py`):

| content | chars/4 | pieces |
|---|---:|---:|
| real Python source (varies with the files on the machine) | 1.10 | 1.03 |
| real prose | 0.97 | 1.11 |
| logs (synthetic) | 0.62 | 0.89 |
| JSON rows (synthetic) | 0.60 | 0.97 |

(estimate / real tokens; 1.00 is exact. chars/4 under-counts machine output by
about 40%, so a "within budget" answer could be 1.6x over in real tokens.)

**Framework adapters:** raw message-list dicts are first-class
(`ctxfit.fit` accepts them directly). OpenAI-shaped (also covers
OpenAI-compatible endpoints like Ollama/freellmapi), Anthropic block-shaped,
and duck-typed LangChain-shaped helpers convert at the boundary; the core
never sees framework shapes.

## Does the one important line survive?

Tool output is where context goes, and the line the agent needs (the failing
migration, the recalled batch, the deadline) is rarely at the head or tail.
`benchmarks/needle_bench.py` plants one fact in the middle of 20 seeded
synthetic tool outputs (logs, JSON rows, source, search text) and checks
whether it is still there afterwards. Real token counts (tiktoken
cl100k_base), CPU only, no paid API; competitors run only if installed:

| compressor | needles kept | tokens saved | ms/case |
|---|---:|---:|---:|
| ctxfit 0.7 behaviour (head+tail, 50 rows) | 0 / 20 | 92.0% | 4 |
| **ctxfit unreleased 0.8** (salient lines, outlier rows) | **20 / 20** | 91.4% | 6 |
| [Headroom](https://github.com/chopratejas/headroom) 0.43.1 `compress()` | 20 / 20 | 63.2% | 54 |
| [LangChain `trim_messages`](https://python.langchain.com/docs/how_to/trim_messages/) (langchain-core 1.7.6, last, 6k) | 10 / 20 | 76.4% | 30 |
| [LLMLingua-2](https://github.com/microsoft/LLMLingua) 0.2.2, rate 0.2, CPU | 5 / 20 | 69.9% | 17,843 |

The corpus is ours and its needles are the kinds salient-line selection is
built for (error lines, odd-one-out rows), so read it as "this failure mode is
fixed", not "ctxfit wins everywhere". Headroom (installed from PyPI, default settings) kept every needle too:
98.6% saved on logs and 61.6% on JSON, but 0% on the source and prose cases.
LangChain `trim_messages` drops whole messages, so a single
large tool result is either kept whole or gone; LLMLingua-2 is a token-dropping
model (one model, one rate, one run; other settings will differ). Timings are
one shared cloud CPU, indicative only. Not tested: answer quality
after compression (see JetBrains Research on masking, cited below),
non-English text, real transcripts (`ctxfit replay` does that on yours).
Want to know what a run lost? `ctxfit.fit(..., audit_facts=True)` or
`--audit-facts` lists the paths, codes, versions and ids that no longer appear.
Competitor versions were re-checked on PyPI on 2026-11-07 (still the ones above); their
rows were not re-run. Headroom's README also lists reversible retrieval (`headroom_retrieve`)
and ML image compression, which `fit()` only partly matches (see `offload=` below; images are
left untouched, not shrunk).

Speed on one huge output (`benchmarks/needle_bench.py --big`, default estimator, one shared
cloud CPU, indicative): a 4.8 MB log of 400k repeated lines 6.1 s -> 1.0 s, 3 MB on one line
1.7 s -> 0.14 s, a 60,000-row JSON array 2.6 s -> 0.4 s.

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
OpenAI-shaped; `--flavor` overrides auto-detection, and an unknown flavor is an error).
Files saved by Windows tools (UTF-8 with BOM, UTF-16 from PowerShell `>`) are read as is.
`-` means standard input (`ctxfit analyze -`) or, for `--out`, standard output
(`ctxfit fit c.json --out -`, the summary then goes to stderr). Output files are written
whole or not at all, so `--out` may name the input file. `--strict` overflow exits 2.

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

### Images, thinking blocks and provider fields

`fit()` works on text. A message that carries something it cannot represent, such as an
image part, a document, an Anthropic `thinking` block with its signature, `cache_control`, or a
provider field on a block or tool call, is **handed back exactly as it came in**: no strategy
rewrites it, and it is never half-rebuilt. It is not pinned, so the sliding window and
aggressive mode may still drop it whole (with its turn). Fields on the message itself that ctxfit
does not model (`refusal`, `reasoning_content`, `name`, your own ids) are copied back onto the
rebuilt message. `report.passthrough_messages` and `summary()` say how many messages were
handled this way; **their non-text parts are not counted in the token estimates** (image token
cost depends on the provider and is not guessed). Older versions silently deleted these parts.

### Getting the original back (`offload=`)

```python
store = {}                                     # any dict-like; you decide how long it lives
fitted, report = ctxfit.fit(messages, total_budget=60_000, offload=store)
# a cut tool output now ends its marker with "full text id 3f9a1c07"; store["3f9a1c07"] is the original
```

Ids are the first 8 hex of the output's sha256 (`ctxfit.output_id(text)`), the same ids running
notes use. ctxfit only fills the dict; serving it to the model (a tool, a retry) is your loop's
job. Off by default, nothing changes when it is off. Only `TruncateToolOutput` offloads; rows
dropped by `CompactJson` and turns dropped by the window are not stored.

## Running notes (opt-in, 0.7.0)

Long agent runs re-send the whole conversation every turn. Most of that is old
tool output nobody will read again. `ctxfit.notes` cuts it down, all of it off
unless you call it.

**What saves real money if your provider caches prompts** (Claude, OpenAI):
a cached prefix is already 10x cheaper, and rewriting old history makes you
pay to cache it again. So, in order:

1. **`arrival="lossless"`** — compacts each tool output once, when it first
   arrives, without losing anything: a repeat of an output the model can
   still see becomes `[ctxfit: identical to earlier output id 3f9a1c07]`, flat
   JSON arrays become tables, runs of identical lines become `line  [x12]`.
   The same output always gets the same form, so the prompt cache is never
   broken. In our synthetic benchmark this alone cost 52-79% of doing nothing,
   with every fact still visible (most of it from repeated file and page
   reads; your agent may repeat less).
2. **Masking with `clear_at_least_chars`** (e.g. 20,000) — hides old outputs
   only in big steps, and **during pauses**: after `cache_ttl_seconds` (300)
   without a request the cache has expired anyway, so ctxfit takes that
   moment to catch up. With arrival on: 51-57% of baseline cost with
   realistic pauses, 73-74% when the cache never goes cold.
3. **Running notes** — only to stay under a context limit. They cost more
   than they save on cached traffic (summariser calls plus re-caching).

`benchmarks/RESULTS.md` has the numbers and the method (synthetic transcripts).

The two layers:

1. **Observation masking** — no model call. Old tool outputs (≥ 400 chars,
   older than the last 4 turns) become one line:
   `[ctxfit: tool output hidden - bash - 18234 chars - id 3f9a1c07]`.
   The id is the first 8 hex of the output's sha256, so you can keep the
   original and look it up.
2. **Running notes** — once the masked prompt still passes a high-water mark
   (default 60,000 est. tokens), the oldest turns are replaced by ONE notes
   message written by a cheap summariser model *you* supply. Pinned rules stay
   outside the notes, verbatim.

```python
import json

import anthropic
from ctxfit import NotesKeeper, anthropic_summarizer, mask_observations

# Layer 1 only: deterministic, free
masked, report = mask_observations(messages)          # Anthropic, OpenAI or generic shape
print(report.summary())

# Both layers: keep one NotesKeeper per conversation
keeper = NotesKeeper(anthropic_summarizer(anthropic.Anthropic()))  # model="claude-haiku-5-5"
fitted, report = keeper.fit(body["messages"], system=body.get("system"),
                            context_management=body.get("context_management"))
print(report.summary())   # example output:
# est. tokens: 71,300 → 24,900 (−65.1%) [chars/4]
#   masked 9 tool output(s), 96,420 chars (boundary turn 20 of 26)
#   running notes replace 61 earlier message(s) (cut at turn 12, notes ~900 est. tokens)
#   net saved: 46,400 est. tokens
#   prompt-cache prefix: stable

state = json.dumps(keeper.state_dict())   # persist between processes; keeper.load_state(...)
```

Always pass the **full original history** to `keeper.fit()`; it re-applies
the same cut every time. `openai_summarizer(client, model=...)` works with
any OpenAI-compatible endpoint, or pass any `(system_prompt, user_prompt) ->
text` function (return `(text, input_tokens, output_tokens)` to get exact
spend in the report; raise `SummaryTruncated` if the provider cut the answer
off). ctxfit imports no SDK; you hand it the client.
CLI preview of layer 1: `ctxfit mask convo.json --report r.json`
(`--offload` adds the original outputs; that report file is written owner-only, mode 0600).

**Knobs added in 0.4.0** (all off by default, so 0.3.0 behaviour is unchanged):

- `exclude_tools=["read_file"]` — outputs of these tools are never masked.
- `clear_at_least_chars=20_000` — the mask boundary only moves when that
  move hides at least this much. A move rewrites earlier messages and costs a
  fresh cache write, so small moves are skipped. The applied boundary is
  remembered in the state and never moves backwards.
- `mask_style="squeeze"` — instead of a one-line marker, keep a short,
  deterministic digest: for JSON, its shape and first item or scalar fields;
  for logs, the first and last lines plus error/fail/warning lines; for
  other text, the head and tail. Used only when the digest is at most 40% of
  the original, otherwise the marker. Secrets are scrubbed from it.
- **Recall.** Hidden outputs stay retrievable by their id. Add
  `recall_tool("anthropic")` (or `"openai"`) to your request's `tools`; when
  the model calls `ctxfit_recall`, answer with `keeper.handle_recall(call)`,
  which returns the ready-made tool result. The text comes back exactly as the tool
  produced it, **unscrubbed** (secrets included): it is that conversation's own
  data. Never share a keeper or its offload store across users or
  conversations.

**Changed in 0.5.0** (state from 0.4.0 is discarded once, `spec_changed`):

- **Markers keep the key facts.** A hidden output's marker now ends with up to
  8 specifics found in it — file paths, URLs, error codes, versions, ids —
  e.g. `[ctxfit: tool output hidden - bash - 18234 chars - id 3f9a1c07] - keeps: app/main.py; E1234; v2.3.1`.
  They are taken from the secret-scrubbed text. On by default
  (`keep_facts=False` for the old one-liner).
- **Notes say what can be recalled.** Outputs evicted by a cut stay
  retrievable by id, and the notes message ends with
  `[hidden outputs you can recall: 3f9a1c07 bash; ...]` (newest 20). The
  summariser sees each result's id and is asked to cite it.
- **Lossless tables.** With `mask_style="squeeze"`, a JSON array of flat
  objects becomes a pipe table that keeps every value (only secrets are
  redacted) when that is at most 75% of the original.
  `squeeze_recent_json=True` does this for the recent turns too.
- **Prose digest.** Squeezed prose keeps the first and last sentence and the
  sentences with numbers or facts, instead of a blind head and tail.
- **`clear_at_least_chars="auto"`** works out per move whether hiding is
  worth the cache write it causes, from cache prices
  (`cache_write_multiplier=1.25`, `cache_read_multiplier=0.1`) and how many
  more requests you expect to reuse the prefix (`expected_reuse=4`).

**New in 0.7.0** (state from 0.6.0 is discarded once, `spec_changed`):

- **Check it on your own history first: `ctxfit replay`.** Reads Claude
  Code's local transcripts (`~/.claude/projects/**/*.jsonl`, or paths you
  give, `--since 2026-11-01`; symlinked folders are never followed),
  rebuilds every request, prices what was
  really recorded, and shows what each strategy would have cost with prompt
  caching, how often a hidden output was fetched again, and whether any
  file read was altered. Nothing leaves your machine; `--redact` prints
  numbers only; `--json` prints the exact report the TypeScript gateway's
  `calas-gateway replay` also prints. The system prompt and tools are not in
  the files, so their size is estimated from the first request's recorded
  tokens (the report says so), and the simulation is checked against the
  recorded bill (`calibration_off` if it misses by more than 5%). Running
  notes are not simulated (they need a summariser).
- **Anchor tools.** Outputs the model may copy from verbatim — `Read`,
  `NotebookRead`, `Grep` by default (`anchor_tools=`) — are never hidden,
  squeezed or reshaped; only an exact repeat of one may become a
  reference to the earlier copy.
- **OpenAI Responses API** (`flavor="responses"`, the body's `input`):
  `function_call_output` items are masked and compacted like tool results;
  reasoning, compaction and anything encrypted pass through untouched;
  requests with `previous_response_id` are left alone.
- **`ctxfit eval`** checks whether a strategy keeps what the model needs:
  at a few checkpoints per conversation it blanks a fact (a path, code,
  version, id) out of an output at least 8 turns old and asks a cheap model
  to fill it in, with the full context and with the strategy's context
  (`--judge` adds open recall/artifact/next-step/decision questions graded
  by the model). It prints an estimated price first and will not run
  without `--max-usd`; `--batch` uses the Message Batches API at half price.
  **It sends transcript content to the model provider** (tool outputs, file
  contents, prompts; secrets are scrubbed by pattern, which is not a
  guarantee) and says so every time. Contexts are sent as plain text turns:
  tool calls and results become text, thinking blocks are dropped.
- **Proof kit**: `ctxfit badge --out savings.json` (a shields.io endpoint
  computed only from the bundled synthetic demo corpus; any other corpus is
  refused), `scripts/demo.tape`
  (vhs recording of a replay; the GIF is built in CI, not committed), and a
  CI job that fails when `benchmarks/RESULTS.md` or `docs/demo-replay.json`
  no longer match the code.

**Changed in 0.6.0**:

- `arrival="lossless"` (see above; off by default, recommended).
- **Cold-cache moments.** ctxfit remembers when the last request went out
  (`clock=` on the keeper, `now=` on `plan_context`). After more than
  `cache_ttl_seconds` (default 300; `None` turns this off), boundary moves
  happen regardless of `clear_at_least_chars`, and when running notes need a
  cut (over high water, as always) it goes as far back as allowed — rewriting
  is free then. The first request of a conversation counts as cold.
- Fix: an HTTP status code next to `HTTP`/`status` was counted from the wrong
  position (now matches the TypeScript port).
- Lossless tables keep every number exactly as written (big ids, `0.10`,
  `-0.0`) and are not used when the JSON has duplicate keys.
- The secret scrubber also catches `sk-` keys after `_`, npm tokens, Slack
  webhook URLs, secret URL query parameters (`sig`, `token`, `X-Amz-Signature`,
  ...), 40-hex values after a key name like `GITHUB_TOKEN`, and AWS secrets
  after a space. URL facts on markers drop their `?query` and `#fragment`.
- `now=` accepts only `2026-11-05T12:00:00.000Z`-style times (with `Z` or an
  offset) or epoch seconds; anything else is an error.

**No I/O in the core.** `NotesKeeper` is a thin wrapper. If you want to make
the summariser call yourself (async, a queue, another process), use the pure
functions the TypeScript port (calas-gateway) mirrors:

```python
from ctxfit import NotesConfig, SummaryOutcome, finish_context, plan_context

plan = plan_context(messages, system=system, config=NotesConfig(), state=saved_state)
outcome = None
if plan.summary:                       # at most one call per request
    text = my_cheap_model(plan.summary.system, plan.summary.user)
    outcome = SummaryOutcome.ok(text, input_tokens, output_tokens)   # or .timeout() / .error("...")
result = finish_context(plan, outcome)  # result.messages, result.state, result.report
saved_state = result.state.to_dict()
```

**What it saves, and the evidence.** Masking old observations first cut agent
cost by more than half with equal or better task success in JetBrains
Research's study of coding agents (Dec 2025); summarising on top of masking
only added a further 7–11%, and the summariser's own spend was over 7% of the
bill. So masking does the heavy lifting, notes are the last resort, and they
are updated incrementally: one summariser call per cut move, none while the
history prefix is unchanged.

**What it protects.** System/developer messages, `pinned: True` messages,
the first user message (the task) and any user/system/developer message whose
own text contains `[pin]` are never masked and never summarised: summaries
were shown to drop rules (0% → 30% violations, arXiv 2606.22528) and keeping
rules outside the summary brought that back to 0%. An assistant message or a
tool output cannot pin itself with `[pin]` (use `pinned: True`). The
`pinned` key is a ctxfit directive, so it is removed from the Anthropic/OpenAI
messages ctxfit returns. Tool calls are never modified; a cut never separates
a call from its result (pairing is checked before any summariser call, even
with reused or missing tool ids). Inputs are never mutated.

**Prompt injection.** Tool output is untrusted. It reaches the summariser
labelled `[bash result - untrusted data]` with every line indented, so no
line of it can pose as a message or a rule, and the summariser is told to
record facts from it but never instructions or permissions. Its answer is
cleaned before use: `[pin]` tags and anything that looks like a notes header
are removed and secrets scrubbed. This lowers the risk; it does not remove it
(a summariser can still be talked into writing a false "fact").

**Trade-offs, plainly.**

- **Prompt cache.** Any rewrite of an earlier part of the prompt breaks
  provider prompt caching. The masking boundary moves only every 4 turns and
  the notes only when the high-water mark is passed, so between moves the
  prefix is byte-identical (`report.cache_stable` tells you). Each move costs
  one full cache write.
- **Rules can still drop** if they are not pinned. Pin them (`[pin]` in the
  text, or `pinned: True`).
- **The summariser costs money.** It is real spend and the report always
  subtracts it: `net_saved_tokens_est = before − after − summariser_in −
  summariser_out`. The request that moves the cut usually shows a *negative*
  net (the evicted turns are read once by the summariser); it pays back on
  the requests after it.
- **Failures fall back, never fail.** An exception, an empty answer, an
  answer over 2× `notes_max_tokens`, a truncated answer, or a timeout keeps
  the previous notes and cut and reports `summariser_failed` with a short
  `failure_reason`. Retries back off: after a failure no new summary is tried
  for `min_evict_turns` turns, doubling per failure up to 64 turns
  (`summariser_backoff`), so a broken summariser is not called on every request.
- **Secrets.** Each message's full text is scrubbed before it is cut down for
  the summariser: PEM private keys (also unterminated), `sk-`/`sk-ant-`/
  `sk-proj-` keys, Stripe, AWS, GitHub, GitLab, Slack, Google and Hugging Face
  tokens, JWTs, `Bearer` tokens, `user:pass@` in URLs and `password: …` /
  `api_key=…` style values (the key name is kept). Tool-call arguments are
  scrubbed value by value before they are encoded, and values under keys
  like `password`, `api_key` or `db_token` are replaced outright. Previous
  notes and the summariser's answer are scrubbed too.
- **Provider-managed context is left alone.** If the request uses
  `context_management`, or carries thinking/compaction/server-tool blocks,
  ctxfit returns it unchanged (`provider_context`); pass `force=True` to
  rewrite anyway.

**Limits.**

- Estimates are chars/4 (not your tokenizer) so the Python and TypeScript
  implementations trigger at exactly the same point; budget with a margin.
- A masked output is gone from the prompt. If the model needs it again, it has
  to re-run the tool, or your loop has to serve it from `keeper.recall(id8)`
  / the `offload=` callback. The in-memory store is capped
  (`offload_max_bytes`, default 16 MB, oldest dropped first; `0` keeps
  nothing in memory and only calls your callback).
- Notes are lossy and only as good as your summariser model. ctxfit does not
  check them for accuracy, only for length and emptiness.
- The scrubber is a regex safety net, not a guarantee; secrets in unusual
  formats get through.
- `fit` is synchronous: it blocks for the summariser call. The call has a
  20 s timeout by default (`summarizer_timeout_s`) and runs on a small shared
  pool of daemon threads; after a timeout ctxfit stops waiting and ignores the
  late answer, but cannot cancel a call already in flight (it may still be
  billed). `await keeper.afit(...)` runs `fit` in a worker thread for asyncio code.
- A `NotesKeeper` is **not thread-safe**: one keeper per conversation, and
  don't call it from two threads at once (`plan_context`/`finish_context`
  themselves are pure and safe anywhere).
- Editing or forking earlier history discards the notes and rebuilds them
  (one more summariser call); state is per conversation. State from ctxfit
  0.2.0 (`version: 1`) is accepted and discarded on the next call
  (`spec_changed`): the notes are rebuilt once.
- Conversations over 5,000 messages are passed through untouched
  (`too_many_messages`).
- Layer 2 needs the whole history each call; it is not a streaming API.
- Shapes: Anthropic Messages, OpenAI chat.completions, ctxfit generic dicts
  and `Message` lists (which come back as `Message` objects with their `meta`
  kept). Not LangChain objects. The notes layer works on the raw provider JSON
  with its own turn and pin rules; it does not go through `normalize()`.

**How it compares.** Short and factual; read each project's own docs.

| | ctxfit running notes | [Anthropic context editing](https://platform.claude.com/docs/build-with-claude/context-editing) | [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction.md) | [Headroom](https://github.com/chopratejas/headroom) | [LLMLingua](https://github.com/microsoft/llmlingua) |
|---|---|---|---|---|---|
| Where it runs | Your process | Anthropic's servers (beta header) | OpenAI's servers (Responses API) or `/responses/compact` | Your machine (library, proxy, MCP) | Your machine |
| Providers | Anthropic + OpenAI-shaped JSON | Claude only | OpenAI only | Several | Any (works on text) |
| What it does to old tool output | Hides, or squeezes to a digest | Clears oldest results to a placeholder | Folds history into an encrypted compaction item | Compresses by content type (JSON, code, prose) | Drops low-importance tokens |
| Needs a model to compress | Only for notes (your cheap model) | No (clearing) | Yes, OpenAI's | Text model for prose; JSON/code rules otherwise | Yes (small LM / encoder) |
| Readable result | Yes, plain text | Yes | No ("opaque") | Yes | Yes, but lossy wording |
| Get the original back | `ctxfit_recall` / offload | No (your client keeps history) | No | Yes (`headroom_retrieve`) | No |
| Prompt-cache care | Chunked boundary, `clear_at_least_chars`, `cache_stable` report | `clear_at_least` | n/a (server-managed) | CacheAligner flags volatile content | Not a focus |

ctxfit borrows `exclude_tools` and `clear_at_least` from Anthropic's context
editing and leaves requests that already use it alone (`provider_context`).
If you are on Claude only and want zero code, context editing is simpler. If
you need a summary that a human can read, can audit or that works across
providers, or a record you can recall from, that is where ctxfit fits.
One honest caveat from our own benchmark: priced with Claude prompt caching,
masking saves far less money than tokens while the cache stays warm (92-98%
of baseline cost), squeeze can cost more than doing nothing, and running notes
cost more — they are for staying under the context limit. Lossless arrival
compaction and pause-time catch-up are where the money is.
Savings claims of other projects are their own; ours, on synthetic
transcripts only, are in `benchmarks/RESULTS.md` (method stated there).

Exact behaviour, including how every ambiguous case is resolved, is pinned by
shared conformance fixtures (spec v4) that the TypeScript port (calas-gateway)
runs too.

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

- **Estimates, not tokens.** `pieces` landed within about -11% / +17% of
  cl100k_base on every class we tested (table above); chars/4 was 40% low on
  logs and JSON, and emoji-heavy text is still under-counted. **Budget 80–90%
  of your model's true window** as a safety margin. tiktoken (optional)
  narrows the error but no estimate equals your model's actual tokenizer.
- **Lossy by design.** ctxfit deletes and rewrites content to fit budgets.
  Pin anything that must survive verbatim.
- **Head+tail truncation assumes the middle matters least.** Since the
  unreleased 0.8 it also keeps error lines and lines unlike their neighbours,
  and outlier JSON rows, but a critical line that looks like every other line
  (or sits in output with no repeated structure) is still lost. Pin or
  pre-filter such payloads, and use `audit_facts=True` to see what went.
- **JSON compaction is whole-message only** (no substring extraction);
  non-uniform object arrays are left alone; objects with a repeated key are left alone
  (a table would silently keep only one value); numbers keep their written spelling
  (`10.50` stays `10.50`); table cells truncate at 120 chars
  (marked `…`); only applied when it actually saves tokens.
- **Exact-match dedup only.** Near-duplicates are not merged — deliberate;
  fuzzy matching is unexplainable and risky.
- **No LLM calls in `fit()`, ever.** The core pipeline is structural only.
  Summarisation exists only in the opt-in running notes, through a
  summariser you supply and pay for (see "Running notes").
- **Non-text parts are not estimated.** Images, documents and thinking blocks pass through
  unchanged and uncounted (see "Images, thinking blocks and provider fields"). A budget on an
  image-heavy conversation is therefore too optimistic by the image cost; leave margin.
- **Very large inputs are linear, not instant.** About 1 s per 5 MB of tool output on one core;
  the running-notes layer is separate and unchanged.
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
.venv/bin/python -m pytest          # 469 tests
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

## License

MIT — Calas Automations Inc.
