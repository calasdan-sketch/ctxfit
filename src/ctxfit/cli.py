"""ctxfit CLI: analyze and fit saved conversations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .budgets import Budget
from .pipeline import FitPipeline


def _load(path: str) -> list[Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of messages")
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


def _build_pipeline(args: argparse.Namespace) -> FitPipeline:
    return FitPipeline(
        estimator=args.estimator,
        budget=Budget.from_overrides(total=args.budget, overrides=_parse_budgets(args.budgets)),
        strict=args.strict,
        aggressive=args.aggressive,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    raw = _load(args.input)
    pipeline = _build_pipeline(args)
    info = pipeline.analyze(raw, flavor=args.flavor, top_n=args.top)
    if args.format == "json":
        print(json.dumps(info, indent=2))
    else:
        _print_table(info)
    return 2 if info["over_budget"] else 0


def _print_table(info: dict[str, Any]) -> None:
    print(f"ctxfit analyze  [{info['estimator']}]  budget {info['total_budget']:,} est. tokens")
    print(f"messages: {info['message_count']}   total used: {info['total_used_est']:,} est. tokens")
    print()
    print(f"{'section':10s} {'used (est)':>11s} {'budget':>11s} {'status':>8s}")
    for s in ("system", "retrieved", "history", "scratchpad"):
        used = info["usage_est"].get(s, 0)
        allowed = info["allowances"][s]
        status = "OVER" if used > allowed else "ok"
        print(f"{s:10s} {used:>11,} {allowed:>11,} {status:>8s}")
    print()
    if info["warnings"]:
        print("warnings:")
        for w in info["warnings"]:
            print(f"  ! {w}")
        print()
    print(f"top {len(info['top_consumers'])} messages by est. tokens:")
    for t in info["top_consumers"]:
        print(
            f"  #{t['index']:<4d} {t['est_tokens']:>8,}  {t['role']}/{t['kind']} "
            f"[{t['section']}]  {t['preview']}"
        )


def cmd_fit(args: argparse.Namespace) -> int:
    raw = _load(args.input)
    pipeline = _build_pipeline(args)
    fitted, report = pipeline.fit(raw, flavor=args.flavor)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(fitted, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"wrote {args.report}")
    print(report.summary())
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
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(masked, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    if args.report:
        data = report.to_dict(include_offload=args.offload)
        if args.offload:
            # The report then holds the original tool outputs: owner-only.
            fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.chmod(args.report, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        else:
            fh = open(args.report, "w", encoding="utf-8")
        with fh:
            json.dump(data, fh, indent=2)
        print(f"wrote {args.report}")
    print(report.summary())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .replay import default_paths, dumps, load_prices, replay, summary

    prices = load_prices(args.prices)
    report = replay(
        args.paths or default_paths(), prices=prices, since=args.since, redact=args.redact
    )
    text = dumps(report) if args.json else summary(report) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(dumps(report))
        print(f"wrote {args.out}")
    sys.stdout.write(text)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import PRIVACY_WARNING, EvalBudgetError, anthropic_backend, dumps, run_eval
    from .replay import default_paths

    print(f"warning: {PRIVACY_WARNING}", file=sys.stderr)

    def show(usd: float) -> None:
        print(f"estimated cost: ${usd:.4f} ({args.model}{', batch' if args.batch else ''})")

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
        print(f"ctxfit: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(dumps(result))
    return 0


def cmd_badge(args: argparse.Namespace) -> int:
    from .proof import badge

    data = badge(args.corpus or None)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}: {data['message']}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from .estimate import get_estimator

    est = get_estimator(args.estimator)
    text = args.text
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    print(f"{est.count(text):,} est. tokens [{getattr(est, 'name', '?')}] ({len(text):,} chars)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ctxfit",
        description="Drop-in context-window optimizer: fit any message list within budget.",
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
        help="'auto' (tiktoken if installed, else heuristic), 'heuristic', 'tiktoken'",
    )
    p.add_argument(
        "--flavor", default="auto", help="'auto', 'openai', 'anthropic', 'langchain', 'generic'"
    )
    p.add_argument(
        "--aggressive", action="store_true", help="enable last-resort drop-oldest-unpinned strategy"
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="raise on surviving budget overflow instead of warning",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="token breakdown by section; no changes made")
    a.add_argument("input", help="JSON file: array of messages")
    a.add_argument("--format", choices=("table", "json"), default="table")
    a.add_argument("--top", type=int, default=10, help="top-N consumers to list")
    a.set_defaults(func=cmd_analyze)

    f = sub.add_parser("fit", help="compress the conversation to fit the budget")
    f.add_argument("input", help="JSON file: array of messages")
    f.add_argument("--out", default=None, help="write fitted messages here (JSON)")
    f.add_argument("--report", default=None, help="write FitReport JSON here")
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
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        print(f"ctxfit: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
