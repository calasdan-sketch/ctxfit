"""FitPipeline: ordered strategies + budget ledger + FitReport."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .adapters import to_anthropic, to_lc_messages, to_openai
from .budgets import Budget, BudgetLedger, BudgetOverflow, OverflowWarning
from .estimate import TokenEstimator, count_by_section, get_estimator, top_consumers
from .messages import (
    KIND_TOOL_RESULT,
    Message,
    count_opaque,
    normalize,
    printable,
    raw_passthrough,
    repair_pairing,
    validate_pairing,
    with_extras,
)
from .strategies import (
    AGGRESSIVE_STRATEGIES,
    DEFAULT_STRATEGIES,
    Strategy,
    StrategyContext,
    StrategyReport,
)


@dataclass
class FitReport:
    """Everything ctxfit did, in one inspectable object."""

    tokens_before: dict[str, int]
    tokens_after: dict[str, int]
    strategy_reports: list[StrategyReport] = field(default_factory=list)
    warnings: list[OverflowWarning] = field(default_factory=list)
    estimator_name: str = "heuristic"
    total_budget: int = 128_000
    aggressive: bool = False
    # Fact audit (opt-in, ``audit_facts=True``): specifics (paths, URLs, error
    # codes, versions, ids) seen in tool output before, and which of them no
    # longer appear anywhere in the fitted messages. None = audit not run.
    facts_seen: int | None = None
    facts_lost: list[str] = field(default_factory=list)
    facts_lost_count: int = 0
    # Source messages passed through unchanged because they hold images,
    # thinking blocks, cache_control marks or other parts the text model
    # cannot represent. Their non-text parts are not counted in the estimates.
    passthrough_messages: int = 0
    # ``pairing="repair"``: unmatched tool_call / tool_result messages dropped before fitting.
    pairing_repaired: int = 0
    pairing_repairs: list[str] = field(default_factory=list)

    @property
    def total_before(self) -> int:
        return sum(self.tokens_before.values())

    @property
    def total_after(self) -> int:
        return sum(self.tokens_after.values())

    @property
    def saved_est(self) -> int:
        return self.total_before - self.total_after

    @property
    def savings_pct(self) -> float:
        if self.total_before == 0:
            return 0.0
        return 100.0 * self.saved_est / self.total_before

    @property
    def over_budget(self) -> bool:
        return bool(self.warnings)

    def summary(self) -> str:
        lines = [
            f"est. tokens: {self.total_before:,} → {self.total_after:,} "
            f"({'−' if self.savings_pct >= 0 else '+'}{abs(self.savings_pct):.1f}%) "
            f"[{self.estimator_name}]",
        ]
        for section in ("system", "retrieved", "history", "scratchpad"):
            b, a = self.tokens_before.get(section, 0), self.tokens_after.get(section, 0)
            if b or a:
                lines.append(f"  {section:10s} {b:>9,} → {a:>9,}")
        for sr in self.strategy_reports:
            if sr.actions:
                lines.append(
                    f"  {sr.strategy}: −{sr.saved_tokens_est:,} est. tokens, "
                    f"{sr.messages_affected} message(s)"
                )
        if self.warnings:
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
            if not self.aggressive:
                lines.append(
                    "  hint: sections still overflow — retry with aggressive=True "
                    "to enable last-resort dropping, or raise the budget"
                )
        else:
            lines.append("within budget ✓")
        if self.passthrough_messages:
            lines.append(
                f"note: {self.passthrough_messages} message(s) hold images, thinking blocks or "
                "other non-text parts; handed back unchanged (not rewritten), and those parts "
                "are not counted in the estimates"
            )
        if self.pairing_repaired:
            lines.append(
                f"note: dropped {self.pairing_repaired} unmatched tool_call/tool_result "
                "message(s) first (pairing='repair')"
            )
        if self.facts_seen is not None:
            if self.facts_lost_count:
                sample = "; ".join(_safe(t) for t in self.facts_lost[:5])
                lines.append(
                    f"facts: {self.facts_lost_count} of {self.facts_seen} specifics from tool "
                    f"output are gone (e.g. {sample})"
                )
            else:
                lines.append(
                    f"facts: all {self.facts_seen} specifics from tool output still present"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator_name,
            "total_budget": self.total_budget,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "saved_est": self.saved_est,
            "savings_pct": round(self.savings_pct, 2),
            "over_budget": self.over_budget,
            "facts_seen": self.facts_seen,
            "facts_lost_count": self.facts_lost_count,
            "facts_lost": self.facts_lost,
            "passthrough_messages": self.passthrough_messages,
            "pairing_repaired": self.pairing_repaired,
            "pairing_repairs": self.pairing_repairs,
            "warnings": [str(w) for w in self.warnings],
            "strategies": [
                {
                    "strategy": sr.strategy,
                    "saved_tokens_est": sr.saved_tokens_est,
                    "messages_affected": sr.messages_affected,
                    "actions": sr.actions,
                }
                for sr in self.strategy_reports
            ],
        }


_AUDIT_MAX_CHARS = 4_000_000


def _safe(token: str) -> str:
    """Make untrusted text printable: control, bidi and zero-width chars -> \\uXXXX."""
    return printable(token, 120)


def _audit_facts(
    before: Sequence[Message], after: Sequence[Message]
) -> tuple[int | None, list[str]]:
    """(facts seen in tool output before, those absent from every message after)."""
    from .notes import _clean_fact, _raw_fact_tokens

    sources = [m.content for m in before if m.kind == KIND_TOOL_RESULT or m.section == "retrieved"]
    if sum(len(t) for t in sources) > _AUDIT_MAX_CHARS:
        return None, []  # too large to audit cheaply; report says "not run"
    seen: dict[str, None] = {}
    for text in sources:
        for raw in _raw_fact_tokens(text):
            tok = _clean_fact(raw)
            if tok:
                seen.setdefault(tok)
    haystack = "\n".join(m.content for m in after)
    lost = [t for t in seen if t not in haystack]
    return len(seen), lost


class PairingBroken(Exception):
    """A strategy violated the tool_call/tool_result pairing invariant."""


class FitPipeline:
    """Run an ordered strategy pipeline against section budgets."""

    def __init__(
        self,
        strategies: Sequence[Strategy | type] | None = None,
        estimator: str | TokenEstimator = "auto",
        budget: Budget | None = None,
        strict: bool = False,
        aggressive: bool = False,
        audit_facts: bool = False,
        offload: MutableMapping[str, str] | None = None,
        pairing: str = "strict",
    ) -> None:
        if pairing not in ("strict", "repair"):
            raise ValueError(f"unknown pairing mode {pairing!r}; expected 'strict' or 'repair'")
        self.pairing = pairing
        self.audit_facts = audit_facts
        self.offload = offload
        if strategies is None:
            kinds = AGGRESSIVE_STRATEGIES if aggressive else DEFAULT_STRATEGIES
            strategies = [k() for k in kinds]
        self.strategies: list[Strategy] = [s() if isinstance(s, type) else s for s in strategies]
        self.estimator = get_estimator(estimator)
        self.budget = budget or Budget()
        self.strict = strict
        self.aggressive = aggressive

    def _ledger(self, messages: list[Message]) -> BudgetLedger:
        return BudgetLedger(
            budget=self.budget,
            usage_est=count_by_section(messages, self.estimator),
        )

    def fit_messages(self, messages: Sequence[Message]) -> tuple[list[Message], FitReport]:
        """Fit an already-normalized message list. Returns (fitted, report)."""
        try:
            return self._fit_messages(messages)
        finally:  # do not keep big tool outputs alive inside a long-lived pipeline object
            clear = getattr(self.estimator, "clear_cache", None)
            if callable(clear):
                clear()

    def _fit_messages(self, messages: Sequence[Message]) -> tuple[list[Message], FitReport]:
        current = list(messages)
        repairs: list[str] = []
        original = current
        if self.pairing == "repair":
            current, repairs = repair_pairing(current)
        input_problems = validate_pairing(current)
        if input_problems:
            raise PairingBroken(
                "input messages already break tool_call/tool_result pairing: "
                + "; ".join(input_problems)
                + (
                    " (pairing='repair' drops the unmatched ones)"
                    if self.pairing == "strict"
                    else ""
                )
            )
        before = count_by_section(original, self.estimator)
        strategy_reports: list[StrategyReport] = []

        for strategy in self.strategies:
            ctx = StrategyContext(
                estimator=self.estimator,
                budget=self.budget,
                ledger=self._ledger(current),
                aggressive=self.aggressive,
                offload=self.offload,
            )
            current, sreport = strategy(current, ctx)
            strategy_reports.append(sreport)

        pairing_problems = validate_pairing(current)
        if pairing_problems:
            raise PairingBroken(
                "pipeline broke tool_call/tool_result pairing: " + "; ".join(pairing_problems)
            )

        after = count_by_section(current, self.estimator)
        ledger = BudgetLedger(budget=self.budget, usage_est=after)
        warnings = ledger.overflows()
        if self.strict and warnings:
            raise BudgetOverflow(
                "over budget after pipeline: " + "; ".join(str(w) for w in warnings)
            )
        facts_seen: int | None = None
        lost: list[str] = []
        if self.audit_facts:
            facts_seen, lost = _audit_facts(messages, current)
        report = FitReport(
            facts_seen=facts_seen,
            facts_lost=lost[:20],
            facts_lost_count=len(lost),
            passthrough_messages=count_opaque(current),
            pairing_repaired=len(repairs),
            pairing_repairs=repairs[:20],
            tokens_before=before,
            tokens_after=after,
            strategy_reports=strategy_reports,
            warnings=warnings,
            estimator_name=getattr(self.estimator, "name", type(self.estimator).__name__),
            total_budget=self.budget.total,
            aggressive=self.aggressive,
        )
        return current, report

    def fit(self, raw: Sequence[Any], flavor: str = "auto") -> tuple[list[Any], FitReport]:
        """Fit raw messages in any supported shape.

        Returns ``(fitted_in_original_shape, report)``.
        """
        detected = flavor
        if flavor == "auto":
            if raw and all(isinstance(m, Message) for m in raw):
                detected = "generic"
            else:
                from .messages import detect_flavor

                detected = detect_flavor(raw)
        messages = normalize(raw, flavor=flavor)
        fitted, report = self.fit_messages(messages)
        if detected == "openai":
            return to_openai(fitted), report
        if detected == "anthropic":
            return to_anthropic(fitted), report
        if detected == "langchain":
            return to_lc_messages(fitted), report
        return [raw_passthrough([m]) or with_extras(m.to_dict(), [m]) for m in fitted], report

    def analyze(self, raw: Sequence[Any], flavor: str = "auto", top_n: int = 10) -> dict[str, Any]:
        """Measure without modifying: section breakdown + top consumers."""
        messages = normalize(raw, flavor=flavor)
        usage = count_by_section(messages, self.estimator)
        ledger = BudgetLedger(budget=self.budget, usage_est=usage)
        return {
            "estimator": getattr(self.estimator, "name", type(self.estimator).__name__),
            "total_budget": self.budget.total,
            "allowances": self.budget.allowances(),
            "usage_est": usage,
            "total_used_est": sum(usage.values()),
            "over_budget": ledger.over_budget,
            "warnings": [str(w) for w in ledger.overflows()],
            "top_consumers": top_consumers(messages, self.estimator, n=top_n),
            "message_count": len(messages),
        }


def fit(
    raw: Sequence[Any],
    total_budget: int = 128_000,
    budgets: Mapping[str, float] | None = None,
    strategies: str | Sequence[Strategy | type] = "default",
    estimator: str | TokenEstimator = "auto",
    strict: bool = False,
    aggressive: bool = False,
    flavor: str = "auto",
    audit_facts: bool = False,
    offload: MutableMapping[str, str] | None = None,
    pairing: str = "strict",
) -> tuple[list[Any], FitReport]:
    """One-call API: fit raw messages within budget.

    strategies: "default" | "aggressive" | a list of Strategy instances.
    Returns ``(fitted_messages_in_original_shape, FitReport)``.

    offload: pass a dict to keep the full text of every truncated tool output
    (``{id: text}``, id = :func:`ctxfit.output_id`); the cut marker names the id so your
    loop can serve the original back when the model asks for it.

    pairing: ``"strict"`` (default) raises :class:`PairingBroken` when a tool call has no result
    or the reverse; ``"repair"`` drops those unmatched messages first and says so in the report.

    Note: token figures are estimates (heuristic ±~25%). Budget 80-90% of
    your model's true context window as a safety margin.
    """
    if isinstance(strategies, str):
        if strategies == "default":
            strat_list: Sequence[Strategy | type] | None = None
        elif strategies == "aggressive":
            strat_list = None
            aggressive = True
        else:
            raise ValueError(f"unknown strategies preset {strategies!r}")
    else:
        strat_list = strategies
    pipeline = FitPipeline(
        strategies=strat_list,
        estimator=estimator,
        budget=Budget.from_overrides(total=total_budget, overrides=budgets),
        strict=strict,
        aggressive=aggressive,
        audit_facts=audit_facts,
        offload=offload,
        pairing=pairing,
    )
    return pipeline.fit(raw, flavor=flavor)
