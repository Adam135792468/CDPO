"""Critical Step DPO utilities and dataset builders.

This package contains the reusable algorithmic components needed to reproduce
the paper pipeline inside this repository:

1. PubMed-style critical step selection via multi-agent voting.
2. Partial-context construction for prevention/correction rollouts.
3. Rubric-based rollout scoring.
4. Step-grouped CDPO dataset construction.
5. Grouped all-pairs rubric CDPO loss.

The implementation is intentionally modular so that rollout collection can be
plugged into different agent backends while preserving a stable data schema.
"""

from .context import (
    DEFAULT_PUBMED_SYSTEM_PROMPT,
    build_correction_context,
    build_partial_context_records,
    build_prevention_context,
    parse_interleaved_turns,
)
from .dataset import (
    build_cdpo_step_records,
    flatten_step_record_to_pair_records,
    summarize_step_records,
)
from .loss import rubric_cdpo_loss
from .mcq_localization import localize_mcq_critical_turn
from .scoring import LiteLLMRubricScorer, RubricScorerConfig
from .types import (
    CDPOStepRecord,
    PartialRolloutRecord,
    RolloutScoreRecord,
    RolloutTrace,
    RubricItem,
    TrajectoryTurn,
)
from .voting import (
    CriticalStepJudgeConfig,
    MultiAgentCriticalStepSelector,
    select_pubmed_critical_steps,
)

__all__ = [
    "CDPOStepRecord",
    "CriticalStepJudgeConfig",
    "DEFAULT_PUBMED_SYSTEM_PROMPT",
    "LiteLLMRubricScorer",
    "MultiAgentCriticalStepSelector",
    "PartialRolloutRecord",
    "RolloutScoreRecord",
    "RolloutTrace",
    "RubricItem",
    "RubricScorerConfig",
    "TrajectoryTurn",
    "build_cdpo_step_records",
    "build_correction_context",
    "build_partial_context_records",
    "build_prevention_context",
    "flatten_step_record_to_pair_records",
    "localize_mcq_critical_turn",
    "parse_interleaved_turns",
    "rubric_cdpo_loss",
    "select_pubmed_critical_steps",
    "summarize_step_records",
]
