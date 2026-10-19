"""ctxfit — drop-in context-window optimizer for any agent loop.

Structural compression, no LLM calls: sliding windows, tool-output
truncation with head/tail, JSON→markdown compaction, dedup. Section
budgets with overflow warnings. Framework-agnostic core + adapters.
"""

from .budgets import Budget, BudgetLedger, BudgetOverflow, OverflowWarning
from .estimate import HeuristicEstimator, TiktokenEstimator, TokenEstimator, get_estimator
from .messages import Message, normalize, validate_pairing
from .pipeline import FitPipeline, FitReport, PairingBroken, fit
from .strategies import (
    CompactJson,
    Deduplicate,
    DropOldestUnpinned,
    SlidingWindow,
    Strategy,
    StrategyContext,
    StrategyReport,
    TruncateToolOutput,
)

__version__ = "0.1.1"

__all__ = [
    "Budget",
    "BudgetLedger",
    "BudgetOverflow",
    "CompactJson",
    "Deduplicate",
    "DropOldestUnpinned",
    "FitPipeline",
    "FitReport",
    "HeuristicEstimator",
    "Message",
    "OverflowWarning",
    "PairingBroken",
    "SlidingWindow",
    "Strategy",
    "StrategyContext",
    "StrategyReport",
    "TiktokenEstimator",
    "TokenEstimator",
    "TruncateToolOutput",
    "__version__",
    "fit",
    "get_estimator",
    "normalize",
    "validate_pairing",
]
