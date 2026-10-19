"""FitPipeline: ordered strategies + budget ledger + FitReport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .adapters import to_anthropic, to_lc_messages, to_openai
from .budgets import Budget, BudgetLedger, BudgetOverflow, OverflowWarning
from .estimate import TokenEstimator, count_by_section, get_estimator, top_consumers
from .messages import Message, normalize, validate_pairing
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
            f"(−{self.savings_pct:.1f}%) [{self.estimator_name}]",
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
    ) -> None:
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
        current = list(messages)
        input_problems = validate_pairing(current)
        if input_problems:
            raise PairingBroken(
                "input messages already break tool_call/tool_result pairing: "
                + "; ".join(input_problems)
            )
        before = count_by_section(current, self.estimator)
        strategy_reports: list[StrategyReport] = []

        for strategy in self.strategies:
            ctx = StrategyContext(
                estimator=self.estimator,
                budget=self.budget,
                ledger=self._ledger(current),
                aggressive=self.aggressive,
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
        report = FitReport(
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
        return [m.to_dict() for m in fitted], report

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
) -> tuple[list[Any], FitReport]:
    """One-call API: fit raw messages within budget.

    strategies: "default" | "aggressive" | a list of Strategy instances.
    Returns ``(fitted_messages_in_original_shape, FitReport)``.

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
    )
    return pipeline.fit(raw, flavor=flavor)
