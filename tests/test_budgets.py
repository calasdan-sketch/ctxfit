"""Tests for budgets and overflow warnings."""

import pytest

from ctxfit.budgets import Budget, BudgetLedger, BudgetOverflow, OverflowWarning


def test_default_split_sums_to_one():
    b = Budget()
    assert abs(sum(b.split.values()) - 1.0) < 1e-9


def test_default_allowances():
    b = Budget(total=100_000)
    assert b.allowance("system") == 10_000
    assert b.allowance("retrieved") == 55_000
    assert b.allowance("history") == 25_000
    assert b.allowance("scratchpad") == 10_000


def test_partial_overrides_fill_defaults():
    b = Budget.from_overrides(total=1000, overrides={"retrieved": 0.60, "history": 0.20})
    assert b.split["retrieved"] == 0.60
    assert b.split["system"] == 0.10  # default filled in
    assert b.split["scratchpad"] == 0.10


def test_unbalanced_overrides_rejected():
    # 0.10 + 0.60 + 0.25 + 0.10 = 1.05
    with pytest.raises(ValueError, match="must sum to 1.0"):
        Budget.from_overrides(total=1000, overrides={"retrieved": 0.60})


def test_balanced_overrides_ok():
    b = Budget.from_overrides(total=1000, overrides={"retrieved": 0.50, "history": 0.30})
    assert b.allowance("retrieved") == 500
    assert b.allowance("history") == 300


def test_unknown_section_rejected():
    with pytest.raises(ValueError, match="unknown budget section"):
        Budget.from_overrides(overrides={"nope": 0.5})
    with pytest.raises(ValueError, match="unknown budget sections"):
        Budget(total=100, split={"nope": 1.0})


def test_nonpositive_total_rejected():
    with pytest.raises(ValueError, match="positive"):
        Budget(total=0)


def test_ledger_no_overflow():
    b = Budget(total=1000)
    ledger = BudgetLedger(
        budget=b, usage_est={"system": 50, "retrieved": 100, "history": 100, "scratchpad": 10}
    )
    assert ledger.overflows() == []
    assert not ledger.over_budget


def test_ledger_overflow_warning():
    b = Budget(total=1000)  # retrieved allowance = 550
    ledger = BudgetLedger(
        budget=b, usage_est={"system": 50, "retrieved": 900, "history": 100, "scratchpad": 10}
    )
    over = ledger.overflows()
    assert len(over) == 1
    w = over[0]
    assert isinstance(w, OverflowWarning)
    assert w.section == "retrieved"
    assert w.used_est == 900 and w.budget_est == 550
    assert w.over_by_est == 350
    assert "retrieved" in str(w) and "350" in str(w)
    assert ledger.over_budget


def test_ledger_total_used():
    b = Budget(total=1000)
    ledger = BudgetLedger(
        budget=b, usage_est={"system": 1, "retrieved": 2, "history": 3, "scratchpad": 4}
    )
    assert ledger.total_used_est == 10


def test_budget_overflow_is_exception():
    assert issubclass(BudgetOverflow, Exception)
