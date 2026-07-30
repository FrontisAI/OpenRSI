"""
Services module for decoupled code generation and evaluation.

This module implements a decoupled architecture similar to SGLang's prefill/decode
separation, where code generation and sandbox evaluation are handled by separate
services coordinated through a scheduler.
"""

from tts_search.services.evaluator import EvaluatorService
from tts_search.services.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopController,
)
from tts_search.services.generator import GeneratorService
from tts_search.services.rejection import (
    AcceptAllPolicy,
    AcceptScoredPolicy,
    BaselinePostprocessPolicy,
    BetterThanReferencePolicy,
    MedalPolicy,
    MixedLeaderboardBaselinePolicy,
    RejectionDecision,
    RewardThresholdPolicy,
    ScoreThresholdPolicy,
    build_rejection_policy,
)
from tts_search.services.scheduler import Scheduler, SchedulerConfig

__all__ = [
    "GeneratorService",
    "EvaluatorService",
    "GenerationLoopConfig",
    "GenerationLoopController",
    "AcceptAllPolicy",
    "AcceptScoredPolicy",
    "BaselinePostprocessPolicy",
    "BetterThanReferencePolicy",
    "MedalPolicy",
    "MixedLeaderboardBaselinePolicy",
    "RejectionDecision",
    "RewardThresholdPolicy",
    "ScoreThresholdPolicy",
    "build_rejection_policy",
    "Scheduler",
    "SchedulerConfig",
]
