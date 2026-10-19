"""Section budgets with overflow warnings.

Default split (panel-tuned): system 10% / retrieved 55% / history 25% /
scratchpad 10%. Retrieved dominates real agent runs; history beyond ~25%
is where lost-in-the-middle accuracy decay bites hardest.

Enforcement is soft by default: strategies compress toward budgets and
surviving overruns surface as warnings. ``strict=True`` raises instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .messages import SECTIONS

DEFAULT_SPLIT: dict[str, float] = {
    "system": 0.10,
    "retrieved": 0.55,
    "history": 0.25,
    "scratchpad": 0.10,
}


@dataclass
class OverflowWarning:
    section: str
    used_est: int
    budget_est: int

    @property
    def over_by_est(self) -> int:
        return self.used_est - self.budget_est

    def __str__(self) -> str:
        return (
            f"section '{self.section}' over budget: "
            f"{self.used_est:,} est. tokens used vs {self.budget_est:,} allowed "
            f"(over by {self.over_by_est:,})"
        )


class BudgetOverflow(Exception):
    """Raised in strict mode when a section still overflows after the pipeline."""


@dataclass
class Budget:
    """Total context budget and per-section split."""

    total: int = 128_000
    split: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPLIT))

    def __post_init__(self) -> None:
        unknown = set(self.split) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown budget sections: {sorted(unknown)}")
        # Fill unspecified sections with defaults so partial overrides work.
        for s in SECTIONS:
            self.split.setdefault(s, DEFAULT_SPLIT[s])
        total_frac = sum(self.split[s] for s in SECTIONS)
        if not 0.999 <= total_frac <= 1.001:
            raise ValueError(f"budget split must sum to 1.0, got {total_frac:.4f}")
        if self.total <= 0:
            raise ValueError("budget total must be positive")

    @classmethod
    def from_overrides(
        cls, total: int = 128_000, overrides: Mapping[str, float] | None = None
    ) -> Budget:
        split = dict(DEFAULT_SPLIT)
        if overrides:
            for k, v in overrides.items():
                if k not in SECTIONS:
                    raise ValueError(f"unknown budget section: {k!r}")
                split[k] = float(v)
        return cls(total=total, split=split)

    def allowance(self, section: str) -> int:
        return int(self.total * self.split[section])

    def allowances(self) -> dict[str, int]:
        return {s: self.allowance(s) for s in SECTIONS}


@dataclass
class BudgetLedger:
    """Measured per-section usage against a Budget."""

    budget: Budget
    usage_est: dict[str, int]

    @property
    def total_used_est(self) -> int:
        return sum(self.usage_est.get(s, 0) for s in SECTIONS)

    def overflows(self) -> list[OverflowWarning]:
        out: list[OverflowWarning] = []
        for s in SECTIONS:
            used = self.usage_est.get(s, 0)
            allowed = self.budget.allowance(s)
            if used > allowed:
                out.append(OverflowWarning(section=s, used_est=used, budget_est=allowed))
        return out

    @property
    def over_budget(self) -> bool:
        return bool(self.overflows())
