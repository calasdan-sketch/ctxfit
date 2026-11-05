"""ctxfit — drop-in context-window optimizer for any agent loop.

Structural compression, no LLM calls: sliding windows, tool-output
truncation with head/tail, JSON→markdown compaction, dedup. Section
budgets with overflow warnings. Framework-agnostic core + adapters.

Opt-in running notes (``ctxfit.notes``): deterministic observation masking
plus incremental notes written by a summariser model you supply.
"""

from .budgets import Budget, BudgetLedger, BudgetOverflow, OverflowWarning
from .estimate import HeuristicEstimator, TiktokenEstimator, TokenEstimator, get_estimator
from .messages import Message, normalize, validate_pairing
from .notes import (
    MaskReport,
    NotesConfig,
    NotesKeeper,
    NotesReport,
    NotesState,
    SummaryOutcome,
    SummaryRequest,
    SummaryTruncated,
    anthropic_summarizer,
    finish_context,
    mask_observations,
    openai_summarizer,
    plan_context,
    scrub_secrets,
)
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

__version__ = "0.3.0"

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
    "MaskReport",
    "Message",
    "NotesConfig",
    "NotesKeeper",
    "NotesReport",
    "NotesState",
    "OverflowWarning",
    "PairingBroken",
    "SlidingWindow",
    "Strategy",
    "StrategyContext",
    "StrategyReport",
    "SummaryOutcome",
    "SummaryRequest",
    "SummaryTruncated",
    "TiktokenEstimator",
    "TokenEstimator",
    "TruncateToolOutput",
    "__version__",
    "anthropic_summarizer",
    "finish_context",
    "fit",
    "get_estimator",
    "mask_observations",
    "normalize",
    "openai_summarizer",
    "plan_context",
    "scrub_secrets",
    "validate_pairing",
]
