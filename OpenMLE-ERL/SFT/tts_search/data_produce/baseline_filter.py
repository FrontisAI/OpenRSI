"""Shared baseline token and validation/test gap filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tts_search.data_produce.common import maybe_float
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    feedback_is_success,
    relative_gap_from_feedback,
)
from tts_search.data_produce.token_filter import token_count_within_limit


@dataclass(frozen=True)
class BaselineTokenGapConfig:
    """Configuration shared by generation-time and postprocess filtering."""

    max_total_tokens: int = 32768
    gap_config: GapFilterConfig = field(default_factory=GapFilterConfig)
    keep_equal: bool = True


@dataclass(frozen=True)
class BaselineTokenGapDecision:
    """Decision returned by the shared baseline token/gap filter."""

    accepted: bool
    reason: str
    token_length: float | None = None
    gap_info: dict[str, Any] = field(default_factory=dict)


def evaluate_baseline_token_gap(
    *,
    slime_message_tokens: Any,
    feedback_text: str | None,
    metadata: Mapping[str, Any] | None,
    config: BaselineTokenGapConfig | None = None,
    task_score_scale: float | None = None,
) -> BaselineTokenGapDecision:
    """Apply the baseline token and validation/test gap filters.

    ``slime_message_tokens`` is intentionally the only accepted token count.
    API usage fields such as ``generation_total_tokens`` are not equivalent to
    SLIME chat-template tokens and must be computed before calling this helper.
    """

    config = config or BaselineTokenGapConfig()
    token_value = maybe_float(slime_message_tokens)
    if token_value is None:
        return BaselineTokenGapDecision(False, "missing_slime_message_tokens")
    if not token_count_within_limit(
        token_value,
        max_total_tokens=config.max_total_tokens,
        keep_equal=config.keep_equal,
    ):
        return BaselineTokenGapDecision(
            False,
            f"token_length>{config.max_total_tokens}",
            token_length=token_value,
        )

    gap_config = config.gap_config
    feedback = feedback_text or ""
    if gap_config.require_feedback_success and not feedback_is_success(feedback):
        return BaselineTokenGapDecision(
            False,
            "feedback_not_success",
            token_length=token_value,
        )

    relative_gap, gap_info = relative_gap_from_feedback(
        feedback,
        dict(metadata or {}),
        gap_config,
        task_score_scale=task_score_scale,
    )
    gap_threshold = float(gap_info.get("gap_threshold") or gap_config.max_relative_gap)
    if relative_gap is None:
        if gap_config.require_comparable:
            return BaselineTokenGapDecision(
                False,
                "missing_valid_test_gap",
                token_length=token_value,
                gap_info=gap_info,
            )
    elif relative_gap > gap_threshold:
        return BaselineTokenGapDecision(
            False,
            f"relative_gap>{gap_threshold}",
            token_length=token_value,
            gap_info=gap_info,
        )

    return BaselineTokenGapDecision(
        True,
        "baseline_token_gap_ok",
        token_length=token_value,
        gap_info=gap_info,
    )
