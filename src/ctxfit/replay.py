"""Replay Claude Code transcripts through the running-notes strategies (spec v6 #1).

``ctxfit replay`` reads Claude Code's local JSONL transcripts
(``~/.claude/projects/**/*.jsonl``), rebuilds every model request from the
``parentUuid`` chains, prices what was actually recorded, and simulates what
each strategy would have cost with the same prompt-cache model the
benchmark uses. Nothing leaves the machine. The JSON output is defined
byte for byte in the shared ``fixtures/README.md`` ("Replay"), so the
TypeScript port (``calas-gateway replay``) prints the identical document.

How a transcript becomes requests:

- Every ``user``/``assistant`` entry has a ``uuid`` and a ``parentUuid``.
  Assistant entries that share ``message.id`` are ONE API response (Claude
  Code writes one line per content block); the earliest of them marks the
  request, and the request is every message on the chain before it.
- A chain stops at a ``isCompactSummary`` entry (that summary starts a new
  segment) and where ``isSidechain`` changes (subagent runs are separate
  conversations). Entries of other types are skipped but still link.
- A conversation is all requests whose chain starts at the same entry;
  resumed sessions continue a chain across files.
- The system prompt and tool definitions are not in the files. Their size is
  estimated once per conversation: the first request's recorded prompt
  tokens (input + cache read + cache write) minus the estimated messages.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .notes import (
    ALGO,
    NotesConfig,
    _clean_fact,
    _content_text,
    _estimate,
    _js_int,
    _json_depth_ok,
    _raw_fact_tokens,
    _reject_constant,
    _sha256,
    canonical_json,
    finish_context,
    parse_now,
    plan_context,
    scrub_secrets,
)

REPLAY_FORMAT = 1
CACHE_TTL_MS = 300_000
DUPLICATE_PREFIX = "[ctxfit: identical to earlier output id "
HIDDEN_PREFIX_METHOD = (
    "per conversation: the first request's recorded prompt tokens (input + cache read + "
    "cache write) minus the estimated messages, at least 0"
)
NOTES_NOT_SIMULATED = "running notes are not simulated (they need a summariser model)"

#: The strategies replayed, in output order: (name, mode, config overrides).
STRATEGIES: tuple[tuple[str, str | None, dict[str, Any]], ...] = (
    ("none", None, {}),
    ("arrival", "mask", {"arrival": "lossless", "keep_recent_turns": 1_000_000}),
    ("mask", "mask", {}),
    ("mask+arrival", "mask", {"arrival": "lossless"}),
    (
        "mask+clear20k+arrival",
        "mask",
        {"arrival": "lossless", "clear_at_least_chars": 20_000},
    ),
)

_SNAPSHOT_SUFFIX = re.compile(
    r"-(?:latest|20\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r"|20\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))",
    re.ASCII,
)


# ---------------------------------------------------------------------------
# prices (the gateway's data/prices.json format and lookup rule)
# ---------------------------------------------------------------------------


def default_prices_path() -> Path:
    return Path(__file__).with_name("prices.json")


def load_prices(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_prices_path()
    return json.loads(p.read_text(encoding="utf-8"))


def find_price(table: dict[str, Any], model: str) -> dict[str, Any] | None:
    """The gateway's rule: the exact key, else a key + ``-latest``/a date (longest key)."""
    models = table.get("providers", {}).get("anthropic", {}).get("models", {})
    if model in models:
        return models[model]
    best = None
    for key in models:
        if (
            model.startswith(key)
            and _SNAPSHOT_SUFFIX.fullmatch(model[len(key) :])
            and (best is None or len(key) > len(best))
        ):
            best = key
    return models[best] if best is not None else None


def _tier(table: dict[str, Any], price: dict[str, Any], prompt_tokens: int) -> dict[str, float]:
    prov = table["providers"]["anthropic"]
    lc = price.get("long_context")
    tier = lc if lc and prompt_tokens > lc["threshold_tokens"] else price
    write5 = tier.get("cache_write", tier["input"] * prov.get("cache_write_multiplier", 1.25))
    write1h = tier.get("cache_write_1h", tier["input"] * prov.get("cache_write_1h_multiplier", 2))
    return {
        "input": tier["input"],
        "output": tier["output"],
        "read": tier["cache_read"],
        "write5": write5,
        "write1h": write1h,
    }


# ---------------------------------------------------------------------------
# transcripts -> requests
# ---------------------------------------------------------------------------


@dataclass
class Request:
    conversation: str  # uuid of the chain's first entry
    session: str
    ts_ms: int
    message_id: str
    model: str
    usage: dict[str, int]
    messages: list[dict[str, Any]]
    response_tool_uses: list[tuple[str, str]]  # (name, canonical input) the response issued
    sidechain: bool
    segment_start_is_summary: bool


@dataclass
class Corpus:
    files: int = 0
    entries: int = 0
    skipped_lines: int = 0
    requests: list[Request] = field(default_factory=list)


def default_paths() -> list[str]:
    return [os.path.join(os.path.expanduser("~"), ".claude", "projects", "**", "*.jsonl")]


_WILD = re.compile(r"[*?]")


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """``**/`` = any number of directories, ``*`` = any characters but ``/``, ``?`` = one
    such character; everything else literal (no ``[...]`` classes)."""
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:[^/]+/)*"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(out, re.DOTALL)


def _walk_files(root: str) -> list[str]:
    """Every file under ``root``, never following a symlinked directory (no loops)."""
    found: list[str] = []
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for f in files:
            full = os.path.join(dirpath, f)
            if os.path.isfile(full):
                found.append(full)
    return found


def expand_paths(paths: Sequence[str]) -> list[str]:
    """Files from paths, directories (every ``*.jsonl`` below) and ``*``/``**``/``?`` patterns.

    Symlinked directories are never followed; files are deduplicated by their
    real path (the first in sorted order is kept). Sorted by path.
    """
    out: set[str] = set()
    for raw in paths:
        p = os.path.expanduser(raw)
        if _WILD.search(p):
            parts = p.replace(os.sep, "/").split("/")
            fixed = []
            for part in parts:
                if _WILD.search(part):
                    break
                fixed.append(part)
            base = "/".join(fixed) or "."
            if not os.path.isdir(base):
                continue
            pat = p.replace(os.sep, "/")
            rx = _glob_regex(pat)
            bare = base == "." and not pat.startswith("./")  # `*.jsonl` meets `./x.jsonl`
            for f in _walk_files(base):
                name = f.replace(os.sep, "/")
                if bare and name.startswith("./"):
                    name = name[2:]
                if rx.fullmatch(name):
                    out.add(f[2:] if bare and f.startswith("." + os.sep) else f)
        elif os.path.isdir(p):
            out.update(f for f in _walk_files(p) if f.endswith(".jsonl"))
        elif os.path.isfile(p):
            out.add(p)
    unique: dict[str, str] = {}
    for f in sorted(out):
        unique.setdefault(os.path.realpath(f), f)
    return sorted(unique.values())


def _usage(u: Any) -> dict[str, int]:
    u = u if isinstance(u, dict) else {}

    def n(v: Any) -> int:
        # JSON integers (an integral float like 3.0 counts: JSON.parse cannot tell them
        # apart); above 2^53 through the double, as JavaScript reads it.
        if isinstance(v, bool) or not isinstance(v, int | float):
            return 0
        if isinstance(v, float) and not (math.isfinite(v) and v.is_integer()):
            return 0
        v = int(v)
        if v < 0:
            return 0
        return int(float(v)) if v > 2**53 else v

    write = n(u.get("cache_creation_input_tokens"))
    cc = u.get("cache_creation")
    w1h = n(cc.get("ephemeral_1h_input_tokens")) if isinstance(cc, dict) else 0
    w1h = min(w1h, write)
    return {
        "input": n(u.get("input_tokens")),
        "read": n(u.get("cache_read_input_tokens")),
        "write5": write - w1h,
        "write1h": w1h,
        "output": n(u.get("output_tokens")),
    }


def _blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return list(content)
    if content is None or content == "":
        return []
    return [
        {"type": "text", "text": content if isinstance(content, str) else canonical_json(content)}
    ]


_BLANK_LINE = re.compile(r"[ \t\n\r\f\v]*")
MAX_LINE_DEPTH = 200


def _session(v: Any) -> str:
    """A sessionId as text: a string as is, missing/null as "", anything else canonical JSON."""
    if v is None:
        return ""
    return v if isinstance(v, str) else canonical_json(v)


def _ts(entry: dict[str, Any]) -> int | None:
    t = entry.get("timestamp")
    if not isinstance(t, str):
        return None
    try:
        return parse_now(t)
    except ValueError:
        return None


def load_corpus(paths: Sequence[str]) -> Corpus:
    corpus = Corpus()
    files = expand_paths(paths)
    corpus.files = len(files)
    entries: list[dict[str, Any]] = []
    for path in files:
        with open(path, "rb") as fh:
            data = fh.read().decode("utf-8", errors="replace")
        for line in data.split("\n"):
            if _BLANK_LINE.fullmatch(line):
                continue
            # A line nested deeper than 200 levels is skipped before parsing (keeps every
            # later step within the stack on both sides); NaN/Infinity are not JSON.
            if not _json_depth_ok(line, MAX_LINE_DEPTH):
                corpus.skipped_lines += 1
                continue
            try:
                e = json.loads(line, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                corpus.skipped_lines += 1
                continue
            if not isinstance(e, dict):
                corpus.skipped_lines += 1
                continue
            corpus.entries += 1
            if not isinstance(e.get("uuid"), str):
                continue  # summaries, snapshots: no place in a chain
            if e.get("type") in ("user", "assistant") and (
                _ts(e) is None or not isinstance(e.get("message"), dict)
            ):
                corpus.skipped_lines += 1
                continue
            entries.append(e)  # other entry types still link the chain
    by_uuid: dict[str, dict[str, Any]] = {}
    for e in entries:
        by_uuid.setdefault(e["uuid"], e)

    def msg_id(e: dict[str, Any]) -> str:
        mid = e["message"].get("id")
        return mid if isinstance(mid, str) and mid else "uuid:" + e["uuid"]

    # Group assistant entries by message.id; the earliest (timestamp, uuid) marks the request.
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in by_uuid.values():
        if e["type"] == "assistant":
            groups.setdefault(msg_id(e), []).append(e)
    for mid, group in groups.items():
        group.sort(key=lambda e: (_ts(e), e["uuid"]))
        head = group[0]
        chain, root, summary_root = _walk(head, by_uuid)
        tool_uses: list[tuple[str, str]] = []
        for e in group:
            for b in _blocks(e["message"].get("content")):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_uses.append(
                        (str(b.get("name") or "tool"), canonical_json(b.get("input", {})))
                    )
        model = head["message"].get("model")
        usage = next((e["message"].get("usage") for e in group if e["message"].get("usage")), None)
        corpus.requests.append(
            Request(
                conversation=root,
                session=_session(head.get("sessionId")),
                ts_ms=_ts(head) or 0,
                message_id=mid,
                model=model if isinstance(model, str) else "",
                usage=_usage(usage),
                messages=_to_messages(chain),
                response_tool_uses=tool_uses,
                sidechain=head.get("isSidechain") is True,
                segment_start_is_summary=summary_root,
            )
        )
    return corpus


def _walk(
    head: dict[str, Any], by_uuid: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, bool]:
    """Entries before ``head`` back to the segment start, oldest first; (chain, root, summary?)."""
    side = head.get("isSidechain") is True
    chain: list[dict[str, Any]] = []
    root = head["uuid"]
    summary = False
    seen = {head["uuid"]}
    parent = head.get("parentUuid")
    while isinstance(parent, str) and parent in by_uuid and parent not in seen:
        e = by_uuid[parent]
        if (e.get("isSidechain") is True) != side:
            break
        seen.add(parent)
        if e.get("type") in ("user", "assistant"):
            chain.append(e)
            root = e["uuid"]
        if e.get("isCompactSummary") is True:
            summary = True
            break
        parent = e.get("parentUuid")
    chain.reverse()
    return chain, root, summary


def _to_messages(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entries -> API messages: consecutive entries of one role become one message."""
    out: list[dict[str, Any]] = []
    for e in chain:
        role = "assistant" if e["type"] == "assistant" else "user"
        blocks = _blocks(e["message"].get("content"))
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return out


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


def _bp(a: float, b: float) -> int:
    return math.floor(10000 * a / b + 0.5) if b else 0


def _round(x: float) -> int:
    return math.floor(x + 0.5)


def _outputs_by_id(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in messages:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if isinstance(tid, str):
                        out.setdefault(tid, b.get("content"))
    return out


def _calls(messages: Sequence[dict[str, Any]]) -> list[tuple[str, str, str]]:
    out = []
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    out.append(
                        (
                            str(b.get("name") or "tool"),
                            canonical_json(b.get("input", {})),
                            str(b.get("id")),
                        )
                    )
    return out


def _text(content: Any) -> str:
    """Rule 6 text of a tool_result's content (what the id8 hashes)."""
    return _content_text(content)


def _prompt_text(messages: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        for b in m["content"] if isinstance(m.get("content"), list) else [m.get("content")]:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(str(b.get("text", "")))
                elif b.get("type") == "tool_use":
                    parts.append(canonical_json(b.get("input", {})))
                elif b.get("type") == "tool_result":
                    parts.append(_text(b.get("content")))
    return "\n".join(parts)


def _facts(messages: Sequence[dict[str, Any]]) -> set[str]:
    facts: set[str] = set()
    for content in _outputs_by_id(messages).values():
        for raw in _raw_fact_tokens(scrub_secrets(_text(content))):
            tok = _clean_fact(raw)
            if tok is not None:
                facts.add(tok)
    return facts


@dataclass
class _Totals:
    requests: int = 0
    sent: int = 0
    cost: float = 0.0
    stable: int = 0
    comparable: int = 0
    facts_total: int = 0
    facts_visible: int = 0
    refetches: int = 0
    refetch_after_hide: int = 0
    anchors: set[tuple[str, str]] = field(default_factory=set)
    last_sum: int = 0
    conversations: int = 0


def _simulate(
    convs: list[tuple[str, list[Request]]],
    hidden: dict[str, int],
    table: dict[str, Any],
    mode: str | None,
    overrides: dict[str, Any],
) -> _Totals:
    tot = _Totals()
    cfg = NotesConfig(**overrides) if mode else None
    anchors = set(cfg.anchor_tools) if cfg else set(NotesConfig().anchor_tools)
    for conv, reqs in convs:
        state = None
        prev_out: list[str] | None = None
        prev_ts: int | None = None
        last_p = 0
        out: list[dict[str, Any]] = []
        for r in reqs:
            if mode is None:
                out = r.messages
                offload: dict[str, str] = {}
            else:
                plan = plan_context(
                    r.messages,
                    flavor="anthropic",
                    mode=mode,
                    config=cfg,
                    state=state,
                    now=r.ts_ms / 1000,
                )
                res = finish_context(plan, None)
                state, out, offload = res.state, res.messages, res.offload
            canon = [canonical_json(m) for m in out]
            cold = prev_ts is None or r.ts_ms - prev_ts > CACHE_TTL_MS
            k = 0
            if prev_out is not None:
                while k < min(len(canon), len(prev_out)) and canon[k] == prev_out[k]:
                    k += 1
                tot.comparable += 1
                tot.stable += k == len(prev_out)
            h = hidden[conv]
            p_tokens = h + _estimate("anthropic", out, 0)
            cached = 0 if cold else h + _estimate("anthropic", out[:k], 0)
            price = find_price(table, r.model)
            if price is not None:
                t = _tier(table, price, p_tokens)
                tot.cost += (
                    cached * t["read"]
                    + (p_tokens - cached) * t["write5"]
                    + r.usage["output"] * t["output"]
                )
            tot.requests += 1
            tot.sent += p_tokens
            last_p = p_tokens
            # refetch: the response re-issues an earlier call whose latest output was hidden
            calls = _calls(r.messages)
            originals = _outputs_by_id(r.messages)
            for name, inp in r.response_tool_uses:
                earlier = [c for c in calls if c[0] == name and c[1] == inp]
                if not earlier:
                    continue
                tot.refetches += 1
                content = originals.get(earlier[-1][2])
                if content is not None:
                    if _sha256(_text(content))[:8] in offload:
                        tot.refetch_after_hide += 1
            # anchor outputs altered (other than an exact-duplicate reference)
            names = {cid: n for n, _i, cid in calls}
            shown = _outputs_by_id(out)
            for tid, content in originals.items():
                if names.get(tid) in anchors and tid in shown:
                    new = shown[tid]
                    if new != content and not _text(new).startswith(DUPLICATE_PREFIX):
                        tot.anchors.add((conv, tid))
            prev_out, prev_ts = canon, r.ts_ms
        if reqs:
            tot.conversations += 1
            tot.last_sum += last_p
            facts = _facts(reqs[-1].messages)
            text = _prompt_text(out)
            tot.facts_total += len(facts)
            tot.facts_visible += sum(1 for f in facts if f in text)
    return tot


def replay(
    paths: Sequence[str],
    *,
    prices: dict[str, Any] | None = None,
    since: str | None = None,
    redact: bool = False,
) -> dict[str, Any]:
    """The replay report as a JSON-ready dict (key order is part of the format)."""
    table = prices if prices is not None else load_prices()
    corpus = load_corpus(paths)
    since_ms = None
    if since:
        since_ms = parse_now(since + "T00:00:00Z" if len(since) == 10 else since)
    reqs = [r for r in corpus.requests if since_ms is None or r.ts_ms >= since_ms]
    by_conv: dict[str, list[Request]] = {}
    for r in reqs:
        by_conv.setdefault(r.conversation, []).append(r)
    convs = []
    for conv, rs in by_conv.items():
        rs.sort(key=lambda r: (r.ts_ms, r.message_id))
        convs.append((conv, rs))
    convs.sort(key=lambda c: (c[1][0].ts_ms, c[0]))
    hidden: dict[str, int] = {}
    recorded = 0.0
    unpriced = 0
    for conv, rs in convs:
        first = rs[0]
        u = first.usage
        prompt = u["input"] + u["read"] + u["write5"] + u["write1h"]
        hidden[conv] = max(0, prompt - _estimate("anthropic", first.messages, 0)) if prompt else 0
        for r in rs:
            price = find_price(table, r.model)
            if price is None:
                unpriced += 1
                continue
            u = r.usage
            t = _tier(table, price, u["input"] + u["read"] + u["write5"] + u["write1h"])
            recorded += (
                u["input"] * t["input"]
                + u["read"] * t["read"]
                + u["write5"] * t["write5"]
                + u["write1h"] * t["write1h"]
                + u["output"] * t["output"]
            )
    results = [
        (name, _simulate(convs, hidden, table, mode, overrides))
        for name, mode, overrides in STRATEGIES
    ]
    none = results[0][1]
    sessions: dict[str, list[Any]] = {}
    for conv, rs in convs:
        for r in rs:
            s = sessions.setdefault(r.session, [set(), 0])
            s[0].add(conv)
            s[1] += 1
    report: dict[str, Any] = {
        "replay": REPLAY_FORMAT,
        "algo": ALGO,
        "prices_as_of": table.get("as_of"),
        "corpus": {
            "files": corpus.files,
            "entries": corpus.entries,
            "skipped_lines": corpus.skipped_lines,
            "sessions": len(sessions),
            "conversations": len(convs),
            "sidechain_conversations": sum(1 for _c, rs in convs if rs[0].sidechain),
            "compaction_segments": sum(1 for _c, rs in convs if rs[0].segment_start_is_summary),
            "requests": len(reqs),
            "unpriced_requests": unpriced,
        },
        "hidden_prefix": {"method": HIDDEN_PREFIX_METHOD, "tokens_total": sum(hidden.values())},
        "baseline": {
            "recorded_cost_micro_usd": _round(recorded),
            "simulated_none_micro_usd": _round(none.cost),
            "calibration_bp": _bp(none.cost, recorded),
            "calibration_off": abs(_bp(none.cost, recorded) - 10000) > 500,
        },
        "strategies": [
            {
                "name": name,
                "requests": t.requests,
                "sent_tokens_est": t.sent,
                "net_saved_bp": _bp(none.sent - t.sent, none.sent),
                "cost_micro_usd": _round(t.cost * recorded / none.cost) if none.cost else 0,
                "cost_vs_baseline_bp": _bp(t.cost, none.cost),
                "cache_stable_bp": _bp(t.stable, t.comparable),
                "last_request_tokens_avg": _round(t.last_sum / t.conversations)
                if t.conversations
                else 0,
                "facts_visible_bp": _bp(t.facts_visible, t.facts_total),
                "refetches": t.refetches,
                "refetch_after_hide": t.refetch_after_hide,
                "anchor_touched": len(t.anchors),
                "summariser_calls": 0,
            }
            for name, t in results
        ],
        "notes": NOTES_NOT_SIMULATED,
    }
    if not redact:
        report["sessions"] = [
            {
                "session": sid[:8],
                "conversations": len(v[0]),
                "requests": v[1],
            }
            for sid, v in sorted(sessions.items())
        ]
    return report


def dumps(report: dict[str, Any]) -> str:
    """The exact bytes of ``ctxfit replay --json``: 2-space JSON, UTF-8, trailing newline.

    Integers beyond 2^53 are printed as JavaScript prints the double they become
    (``JSON.stringify``), so both implementations write the same digits.
    """
    big: list[str] = []

    def mark(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: mark(x) for k, x in v.items()}
        if isinstance(v, list):
            return [mark(x) for x in v]
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) > 2**53:
            big.append(_js_int(v))
            return f"\x00jsnum{len(big) - 1}\x00"
        return v

    text = json.dumps(mark(report), indent=2, ensure_ascii=False)
    for i, digits in enumerate(big):
        text = text.replace(f'"\\u0000jsnum{i}\\u0000"', digits)
    return text + "\n"


def summary(report: dict[str, Any]) -> str:
    """A short human-readable table."""
    c, b = report["corpus"], report["baseline"]
    lines = [
        f"replay: {c['requests']} requests in {c['conversations']} conversations "
        f"({c['sessions']} sessions, {c['files']} files)",
        f"recorded cost ${b['recorded_cost_micro_usd'] / 1e6:,.2f}; simulated 'none' "
        f"{b['calibration_bp'] / 100:.1f}% of it"
        + ("  ! calibration_off" if b["calibration_off"] else ""),
        "",
        f"{'strategy':<24}{'cost':>8}{'stable':>8}{'facts':>8}{'refetch':>9}{'anchor':>8}",
    ]
    for s in report["strategies"]:
        lines.append(
            f"{s['name']:<24}{s['cost_vs_baseline_bp'] / 100:>7.1f}%"
            f"{s['cache_stable_bp'] / 100:>7.0f}%{s['facts_visible_bp'] / 100:>7.0f}%"
            f"{s['refetch_after_hide']:>5}/{s['refetches']:<3}{s['anchor_touched']:>8}"
        )
    lines.append("")
    lines.append(report["notes"] + "; costs are estimates, nothing left this machine.")
    return "\n".join(lines)


def iter_requests(paths: Iterable[str]) -> list[Request]:
    """All requests of the transcripts (for ``ctxfit eval``)."""
    return load_corpus(list(paths)).requests
