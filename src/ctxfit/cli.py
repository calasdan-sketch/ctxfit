"""ctxfit CLI: analyze and fit saved conversations.

Exit codes: 0 done, 1 error (message on stderr, never a traceback), 2 over budget
(``analyze``, ``fit --strict``) or wrong usage. Data goes to stdout, progress and
messages to stderr whenever stdout carries data (``--json``, ``--out -``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import __version__
from .budgets import Budget, BudgetOverflow
from .messages import printable
from .pipeline import FitPipeline, PairingBroken

ISSUES_URL = "https://github.com/calasdan-sketch/ctxfit/issues"

# What a console that cannot show unicode gets instead (Windows code pages, LC_ALL=C).
_ASCII_SYMBOLS = str.maketrans(
    {"\u2192": "->", "\u2212": "-", "\u2713": "ok", "\u2026": "...", "\u2014": "-", "\u00b7": "-"}
)


class CliError(Exception):
    """A problem the user can fix; printed as ``ctxfit: error: ...`` with exit code 1."""


def _say(text: str = "", file: Any = None) -> None:
    """``print`` that survives a console which cannot encode the text (no crash, no data loss)."""
    f = file if file is not None else sys.stdout
    enc = getattr(f, "encoding", None) or "ascii"
    try:
        text.encode(enc)  # stderr may be set to escape silently; check ourselves
    except (UnicodeEncodeError, LookupError):
        text = text.translate(_ASCII_SYMBOLS)
        try:
            text.encode(enc)
        except (UnicodeEncodeError, LookupError):
            text = text.encode("ascii", "backslashreplace").decode("ascii")
    f.write(text + "\n")


def _json_text(data: Any, *, indent: int | None = 2, stream: Any = None) -> str:
    """JSON for a stream: unicode kept as is when the stream can show it, else \\uXXXX escapes."""
    enc = (getattr(stream if stream is not None else sys.stdout, "encoding", None) or "").lower()
    return json.dumps(data, indent=indent, ensure_ascii=enc.replace("-", "") not in ("utf8",))


def _decode(raw: bytes) -> str:
    """Text of a file as Windows tools write it: UTF-8 with or without BOM, or UTF-16/32 with BOM."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # PowerShell 5 `>` writes UTF-16 LE with BOM
        if raw.startswith(b"\xff\xfe\x00\x00"):
            return raw.decode("utf-32")
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def _read(path: str) -> str:
    """File text; ``-`` reads standard input."""
    shown = "standard input" if path == "-" else repr(printable(path, 200))
    try:
        if path == "-":
            return _decode(sys.stdin.buffer.read())
        with open(path, "rb") as fh:
            return _decode(fh.read())
    except FileNotFoundError:
        raise CliError(f"there is no file {shown} (check the name and the folder)") from None
    except IsADirectoryError:
        raise CliError(f"{shown} is a folder; give the path of a JSON file") from None
    except PermissionError:
        raise CliError(f"not allowed to read {shown} (permission denied)") from None
    except UnicodeDecodeError:
        raise CliError(
            f"{shown} is not UTF-8 text; JSON files must be UTF-8 (save it again as UTF-8)"
        ) from None
    except OSError as exc:
        raise CliError(f"cannot read {shown}: {exc.strerror or exc}") from None


def _replace(tmp: str, target: str) -> None:
    """``os.replace``; on Windows it fails while an editor or virus scanner holds the target."""
    for attempt in range(4):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.1 * (attempt + 1))


def _write(path: str, text: str, private: bool = False) -> None:
    """Write a file all at once (a failure never leaves a half-written one); ``-`` is stdout."""
    if path == "-":
        sys.stdout.write(text)
        return
    shown = repr(printable(path, 200))
    try:
        _write_file(path, text, private)
    except PermissionError:
        raise CliError(f"not allowed to write {shown} (permission denied)") from None
    except IsADirectoryError:
        raise CliError(f"{shown} is a folder; give a file name") from None
    except FileNotFoundError:
        raise CliError(f"cannot write {shown}: the folder does not exist") from None
    except OSError as exc:
        raise CliError(f"cannot write {shown}: {exc.strerror or exc}") from None


def _write_file(path: str, text: str, private: bool) -> None:
    target = os.path.realpath(path)
    tmp = f"{target}.ctxfit-{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o666)
    except OSError:  # directory not writable: fall back to writing the file itself
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o666)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if private:
            os.chmod(target, 0o600)
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if private:
            os.chmod(tmp, 0o600)
        _replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _note(msg: str, out: str | None, quiet: bool = False) -> None:
    """Progress lines go to stderr when stdout carries the data (``--out -``, ``--json``)."""
    _say(msg, sys.stderr if (out == "-" or quiet) else sys.stdout)


def _load_jsonl(text: str, path: str, shown: str) -> list[Any] | None:
    """Messages from a JSON-lines file, or None when the text is not JSON lines.

    Two shapes: one message per line (``{"role": ..., "content": ...}``), or a Claude Code
    transcript (``~/.claude/projects/**/*.jsonl``), from which the longest request is rebuilt.
    """
    # Split on "\n" only: str.splitlines() also cuts at U+2028 / U+0085 / form feed, which JSON
    # allows unescaped inside a string (Node writes U+2028 as is).
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None
    rows: list[Any] = []
    try:
        for ln in lines:
            rows.append(json.loads(ln))
    except (ValueError, RecursionError):
        return None
    if not all(isinstance(r, dict) for r in rows):
        return None
    if any("role" in r for r in rows):
        return rows
    if not any(isinstance(r.get("uuid"), str) and "message" in r for r in rows):
        return None
    if path == "-":
        raise CliError(
            "a Claude Code transcript has to be read from a file, not standard input; give its path"
        )
    if any(ch in path for ch in "*?"):
        raise CliError(f"{shown}: give the transcript's own path (no * or ? patterns here)")
    from .replay import load_corpus

    requests = load_corpus([path]).requests
    if not requests:
        raise CliError(f"{shown} is a Claude Code transcript but holds no model request yet")
    best = max(requests, key=lambda r: (len(r.messages), r.ts_ms))
    _say(
        f"ctxfit: read a Claude Code transcript: using its longest request ({len(best.messages)} "
        f"messages, of {len(requests)} requests). The system prompt and tool definitions are not "
        "in the file, so they are not counted.",
        sys.stderr,
    )
    return best.messages


def _load(path: str) -> list[Any]:
    text = _read(path)
    shown = "standard input" if path == "-" else repr(printable(path, 200))
    if not text.strip():
        raise CliError(f"{shown} is empty; expected a JSON array of messages")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        rows = _load_jsonl(text, path, shown)
        if rows is None:
            raise CliError(
                f"{shown} is not valid JSON ({exc.msg}, line {exc.lineno}, column {exc.colno})"
            ) from None
        data = rows
    except RecursionError:
        raise CliError(f"{shown} is nested too deeply to read") from None
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise CliError(
            f"{shown} must hold a JSON array of messages like "
            '[{"role": "user", "content": "..."}], or an object with a "messages" array'
        )
    return data


def _parse_budgets(spec: str | None) -> dict[str, float] | None:
    if not spec:
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f"bad --budgets entry: {part!r} (expected section=fraction)")
        out[k] = float(v)
    return out


def _parse_tool_limits(specs: list[str] | None) -> dict[str, int] | None:
    if not specs:
        return None
    out: dict[str, int] = {}
    for spec in specs:
        name, _, num = spec.rpartition("=")
        if not name or not num.strip().isdigit():
            raise CliError(
                f"bad --tool-limit {printable(spec, 80)!r}: write NAME=CHARACTERS, e.g. "
                "read_file=0 (never shorten) or bash=8000"
            )
        out[name] = int(num)
    return out


def _build_pipeline(args: argparse.Namespace) -> FitPipeline:
    return FitPipeline(
        estimator=args.estimator,
        budget=Budget.from_overrides(total=args.budget, overrides=_parse_budgets(args.budgets)),
        strict=args.strict,
        aggressive=args.aggressive,
        audit_facts=getattr(args, "audit_facts", False),
        pairing="repair" if getattr(args, "repair_pairing", False) else "strict",
        tool_limits=_parse_tool_limits(getattr(args, "tool_limit", None)),
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    raw = _load(args.input)
    pipeline = _build_pipeline(args)
    info = pipeline.analyze(raw, flavor=args.flavor, top_n=args.top)
    if args.format == "json" or args.json:
        _say(_json_text(info))
    else:
        _print_table(info)
    return 2 if info["over_budget"] else 0


def _print_table(info: dict[str, Any]) -> None:
    _say(f"ctxfit analyze  [{info['estimator']}]  budget {info['total_budget']:,} est. tokens")
    _say(f"messages: {info['message_count']}   total used: {info['total_used_est']:,} est. tokens")
    _say()
    _say(f"{'section':10s} {'used (est)':>11s} {'budget':>11s} {'status':>8s}")
    for s in ("system", "retrieved", "history", "scratchpad"):
        used = info["usage_est"].get(s, 0)
        allowed = info["allowances"][s]
        status = "OVER" if used > allowed else "ok"
        _say(f"{s:10s} {used:>11,} {allowed:>11,} {status:>8s}")
    _say()
    if info["warnings"]:
        _say("warnings:")
        for w in info["warnings"]:
            _say(f"  ! {w}")
        _say()
    _say(f"top {len(info['top_consumers'])} messages by est. tokens:")
    for t in info["top_consumers"]:
        _say(
            f"  #{t['index']:<4d} {t['est_tokens']:>8,}  {t['role']}/{t['kind']} "
            f"[{t['section']}]  {t['preview']}"
        )


def cmd_fit(args: argparse.Namespace) -> int:
    raw = _load(args.input)
    pipeline = _build_pipeline(args)
    if args.out == "-" and args.report == "-":
        raise CliError("--out - and --report - would both use standard output")
    if args.json and "-" in (args.out, args.report):
        raise CliError("--json prints the report on standard output; do not also use '-' there")
    try:
        fitted, report = pipeline.fit(raw, flavor=args.flavor)
    except BudgetOverflow as exc:  # --strict
        _say(f"ctxfit: {exc}", sys.stderr)
        return 2
    except PairingBroken as exc:
        raise CliError(
            f"{str(exc).split(' (pairing=')[0]}. The conversation looks cut in the middle or incomplete. "
            "Add --repair-pairing to drop the unmatched tool messages and continue."
        ) from None
    quiet = bool(args.json)
    if args.out:
        if args.out == "-":
            body = _json_text(fitted)  # unicode only if this console can show it
        else:
            body = json.dumps(fitted, indent=2, ensure_ascii=False)
        _write(args.out, body + "\n")
        _note(f"wrote {args.out}", args.out, quiet)
    if args.report:
        _write(args.report, json.dumps(report.to_dict(), indent=2) + "\n")
        _note(f"wrote {args.report}", args.out, quiet)
    if args.json:
        _say(json.dumps(report.to_dict(), indent=2))
    else:
        _say(report.summary(), sys.stderr if args.out == "-" else sys.stdout)
    return 2 if report.over_budget else 0


def cmd_mask(args: argparse.Namespace) -> int:
    from .notes import mask_observations

    raw = _load(args.input)
    masked, report = mask_observations(
        raw,
        keep_recent_turns=args.keep_recent_turns,
        mask_chunk_turns=args.chunk_turns,
        min_mask_chars=args.min_chars,
        flavor=args.flavor,
    )
    if args.out:
        body = (
            _json_text(masked)
            if args.out == "-"
            else json.dumps(masked, indent=2, ensure_ascii=False)
        )
        _write(args.out, body + "\n")
        _note(f"wrote {args.out}", args.out)
    if args.report:
        data = report.to_dict(include_offload=args.offload)
        # With --offload the report holds the original tool outputs: owner-only.
        _write(args.report, json.dumps(data, indent=2) + "\n", private=bool(args.offload))
        _note(f"wrote {args.report}", args.out)
    _say(report.summary(), sys.stderr if args.out == "-" else sys.stdout)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .replay import default_paths, dumps, load_prices, replay, summary

    prices = load_prices(args.prices)
    report = replay(
        args.paths or default_paths(), prices=prices, since=args.since, redact=args.redact
    )
    text = dumps(report) if args.json else summary(report) + "\n"
    if args.out:
        _write(args.out, dumps(report))
        _note(f"wrote {args.out}", None, quiet=bool(args.json))
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        _say(text.translate(_ASCII_SYMBOLS).rstrip("\n"))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import PRIVACY_WARNING, EvalBudgetError, anthropic_backend, dumps, run_eval
    from .replay import default_paths

    _say(f"warning: {PRIVACY_WARNING}", sys.stderr)

    def show(usd: float) -> None:
        _say(
            f"estimated cost: ${usd:.4f} ({args.model}{', batch' if args.batch else ''})",
            sys.stderr,
        )

    def backend(calls):  # created lazily: no SDK import unless the run goes ahead
        import anthropic  # type: ignore[import-not-found]

        return anthropic_backend(anthropic.Anthropic(), args.model, batch=args.batch)(calls)

    try:
        result = run_eval(
            args.paths or default_paths(),
            backend,
            model=args.model,
            strategy=args.strategy,
            checkpoints=args.checkpoints,
            judge=args.judge,
            max_usd=args.max_usd,
            batch=args.batch,
            on_estimate=show,
        )
    except EvalBudgetError as exc:
        _say(f"ctxfit: {exc}", sys.stderr)
        return 2
    sys.stdout.write(dumps(result))
    return 0


def cmd_badge(args: argparse.Namespace) -> int:
    from .proof import badge

    data = badge(args.corpus or None)
    _write(args.out, json.dumps(data, indent=2) + "\n")
    _say(f"wrote {args.out}: {data['message']}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from .estimate import get_estimator

    est = get_estimator(args.estimator)
    text = args.text
    if args.file:
        text = _read(args.file)
    elif text == "-":
        text = _read("-")
    n, name = est.count(text), getattr(est, "name", "?")
    if args.json:
        _say(json.dumps({"tokens": n, "chars": len(text), "estimator": name}))
    else:
        _say(f"{n:,} est. tokens [{name}] ({len(text):,} chars)")
    return 0


DEMO_BUDGET = 6000


def cmd_demo(args: argparse.Namespace) -> int:
    """A made-up agent run, fitted: shows what ctxfit does with no files and no network."""
    from .pipeline import fit

    convo, needles = _demo_conversation()
    fitted, report = fit(convo, total_budget=DEMO_BUDGET, estimator=args.estimator)
    text = "\n".join(m["content"] for m in fitted if isinstance(m.get("content"), str))
    kept = {label: needle in text for label, needle in needles.items()}
    if args.json:
        data = report.to_dict()
        data["needles_kept"] = kept
        _say(json.dumps(data, indent=2))
    else:
        _say(
            "ctxfit demo: a made-up coding-agent run with 3 big tool outputs (nothing is sent anywhere)"
        )
        _say(report.summary())
        _say()
        for label, ok in kept.items():
            _say(f"  {'kept' if ok else 'LOST'}: {label}")
    return 0 if all(kept.values()) else 1


def _demo_conversation() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Deterministic sample run; the one line that matters sits in the middle of each output."""
    log = [
        f"[build] step {i:04d} compiled module_{i % 97}.o in {i % 13 + 1} ms" for i in range(1500)
    ]
    log[760] = "ERROR: migration 0042_add_index failed: lock timeout after 30s"
    rows = [{"id": i, "host": f"web-{i % 20}", "status": "ok", "latency_ms": 20 + i % 7}
            for i in range(300)]  # fmt: skip
    rows[141]["status"] = "failed: disk full"
    src = [f"def helper_{i}(x):\n    return x + {i}\n" for i in range(400)]
    src[222] = "def rotate_keys(x):\n    raise RuntimeError('TODO: rotation not implemented')\n"
    outputs = ["\n".join(log), json.dumps(rows), "\n".join(src)]
    convo: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a careful coding agent."},
        {"role": "user", "content": "The nightly build is failing. Find out why."},
    ]
    for i, (name, out) in enumerate(zip(("run_build", "query_hosts", "read_file"), outputs)):
        convo.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": "{}"}}]})  # fmt: skip
        convo.append({"role": "tool", "tool_call_id": f"call_{i}", "content": out})
    needles = {
        "the failing migration line in a 1,500-line log": "migration 0042_add_index failed",
        "the one failed row in 300 JSON rows": "failed: disk full",
        "the unfinished function in 400 lines of source": "rotation not implemented",
    }
    return convo, needles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ctxfit",
        description="Drop-in context-window optimizer: fit any message list within budget.",
        epilog="ctxfit is MIT licensed software by Calas Automations Inc.",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=128_000,
        help="total context budget in est. tokens (default 128000)",
    )
    p.add_argument(
        "--budgets", default=None, help="section overrides, e.g. 'retrieved=0.6,history=0.2'"
    )
    p.add_argument(
        "--estimator",
        default="auto",
        help="'auto' (tiktoken if installed, else pieces), 'pieces', 'heuristic' (chars/4), 'tiktoken'",
    )
    p.add_argument(
        "--flavor", default="auto", help="'auto', 'openai', 'anthropic', 'langchain', 'generic'"
    )
    p.add_argument(
        "--aggressive", action="store_true", help="enable last-resort drop-oldest-unpinned strategy"
    )
    p.add_argument(
        "--audit-facts",
        action="store_true",
        help="report which paths/URLs/codes/versions/ids from tool output did not survive",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="raise on surviving budget overflow instead of warning",
    )

    p.add_argument("--version", action="version", version=f"ctxfit {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    a = sub.add_parser("analyze", help="token breakdown by section; no changes made")
    a.add_argument(
        "input", help="JSON file: array of messages (or JSON lines / a Claude Code transcript)"
    )
    a.add_argument("--format", choices=("table", "json"), default="table")
    a.add_argument("--top", type=int, default=10, help="top-N consumers to list")
    a.add_argument("--json", action="store_true", help="same as --format json")
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("fit", help="compress the conversation to fit the budget")
    f.add_argument(
        "input", help="JSON file: array of messages (or JSON lines / a Claude Code transcript)"
    )
    f.add_argument("--out", default=None, help="write fitted messages here (JSON)")
    f.add_argument("--report", default=None, help="write FitReport JSON here")
    f.add_argument("--json", action="store_true", help="print the report as JSON on stdout")
    f.add_argument(
        "--repair-pairing",
        action="store_true",
        help="drop tool calls without results and results without calls instead of failing",
    )
    f.add_argument(
        "--tool-limit",
        action="append",
        metavar="NAME=N",
        help="per-tool limit in characters, repeatable; 0 = never shorten that tool's output "
        "(e.g. --tool-limit read_file=0 --tool-limit bash=8000)",
    )
    f.set_defaults(func=cmd_fit)

    mk = sub.add_parser(
        "mask", help="preview observation masking (running notes layer 1); no model calls"
    )
    mk.add_argument("input", help="JSON file: array of messages (or a body with 'messages')")
    mk.add_argument("--keep-recent-turns", type=int, default=4)
    mk.add_argument("--chunk-turns", type=int, default=4, help="boundary moves in chunks of N")
    mk.add_argument("--min-chars", type=int, default=400, help="only mask outputs this long")
    mk.add_argument("--out", default=None, help="write masked messages here (JSON)")
    mk.add_argument("--report", default=None, help="write MaskReport JSON here")
    mk.add_argument(
        "--offload",
        action="store_true",
        help="include original outputs (id8 -> text) in the report (written with mode 0600)",
    )
    mk.set_defaults(func=cmd_mask)

    rp = sub.add_parser(
        "replay",
        help="replay Claude Code transcripts through the running-notes strategies (local only)",
    )
    rp.add_argument("paths", nargs="*", help="JSONL files, folders or globs "
                    "(default ~/.claude/projects/**/*.jsonl)")  # fmt: skip
    rp.add_argument("--since", default=None, help="only requests on or after this date/time")
    rp.add_argument("--prices", default=None, help="price table JSON (gateway format)")
    rp.add_argument("--redact", action="store_true", help="numbers only (no session ids)")
    rp.add_argument("--json", action="store_true", help="print the exact JSON report")
    rp.add_argument("--out", default=None, help="also write the JSON report here")
    rp.set_defaults(func=cmd_replay)

    ev = sub.add_parser("eval", help="probe eval: does a strategy keep facts? (paid model calls)")
    ev.add_argument("paths", nargs="*", help="transcripts, as for replay")
    ev.add_argument("--strategy", default="mask+arrival")
    ev.add_argument("--checkpoints", type=int, default=3, help="checkpoints per conversation")
    ev.add_argument("--model", default="claude-haiku-4-5", help="cheap model that answers")
    ev.add_argument(
        "--judge", action="store_true", help="add recall/artifact/continuation/decision probes"
    )
    ev.add_argument("--batch", action="store_true", help="use the Message Batches API (half price)")
    ev.add_argument("--max-usd", type=float, default=None, help="required: refuse above this")
    ev.set_defaults(func=cmd_eval)

    bd = sub.add_parser(
        "badge", help="shields.io endpoint JSON from the published synthetic corpus"
    )
    bd.add_argument("--out", required=True)
    bd.add_argument("--corpus", default=None, help="corpus folder (default: the bundled demo)")
    bd.set_defaults(func=cmd_badge)

    e = sub.add_parser("estimate", help="quick token estimate for text")
    e.add_argument("text", nargs="?", default="", help="text to estimate")
    e.add_argument("--file", default=None, help="read text from file instead")
    e.add_argument("--json", action="store_true", help="print the result as JSON")
    e.set_defaults(func=cmd_estimate)

    dm = sub.add_parser(
        "demo", help="fit a built-in made-up agent run: shows what ctxfit does (no files needed)"
    )
    dm.add_argument("--json", action="store_true", help="print the report as JSON")
    dm.set_defaults(func=cmd_demo)
    return p


_GLOBAL_VALUE_FLAGS = ("--budget", "--budgets", "--estimator", "--flavor")
_GLOBAL_BOOL_FLAGS = ("--aggressive", "--audit-facts", "--strict")
_SUBCOMMANDS = ("analyze", "fit", "mask", "replay", "eval", "badge", "estimate", "demo")


def _hoist_global_options(argv: list[str]) -> list[str]:
    """Accept the global options after the subcommand too.

    The README shows ``ctxfit analyze convo.json --budget 128000``; argparse
    only takes them before the subcommand, so move them there.
    """
    try:
        at = next(i for i, a in enumerate(argv) if a in _SUBCOMMANDS)
    except StopIteration:
        return argv
    head, rest, hoisted = argv[:at], argv[at + 1 :], []
    keep: list[str] = []
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--":
            keep.extend(rest[i:])
            break
        if a in _GLOBAL_BOOL_FLAGS:
            hoisted.append(a)
        elif a in _GLOBAL_VALUE_FLAGS and i + 1 < len(rest):
            hoisted.extend([a, rest[i + 1]])
            i += 1
        elif a.split("=", 1)[0] in _GLOBAL_VALUE_FLAGS and "=" in a:
            hoisted.append(a)
        else:
            keep.append(a)
        i += 1
    return head + hoisted + [argv[at]] + keep


def main(argv: list[str] | None = None) -> int:
    argv = _hoist_global_options(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        _say(f"ctxfit: error: {exc}", sys.stderr)
        return 1
    except BrokenPipeError:  # `ctxfit ... | head`: the reader left on purpose
        try:  # keep the interpreter's exit-time flush from printing a second error
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, OSError, ImportError) as exc:
        _say(f"ctxfit: error: {printable(str(exc), 500)}", sys.stderr)
        return 1
    except Exception as exc:  # a bug, not the user's fault: say so, keep the screen clean
        if os.environ.get("CTXFIT_DEBUG"):
            raise
        _say(
            f"ctxfit: unexpected {type(exc).__name__}: {printable(str(exc), 300)}\n"
            f"This is a bug in ctxfit, not in your file. Please report it: {ISSUES_URL}\n"
            "(set CTXFIT_DEBUG=1 to see the full traceback)",
            sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
