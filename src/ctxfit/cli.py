"""ctxfit CLI: analyze and fit saved conversations."""

from __future__ import annotations

import argparse
import json
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

    e = sub.add_parser("estimate", help="quick token estimate for text")
    e.add_argument("text", nargs="?", default="", help="text to estimate")
    e.add_argument("--file", default=None, help="read text from file instead")
    e.set_defaults(func=cmd_estimate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"ctxfit: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
