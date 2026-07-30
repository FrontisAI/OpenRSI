"""Rejection sampling policies for generated-code selection.

The generation loop only needs one question from this module: should an
evaluated generation be kept for the final dataset?  Keeping that decision
behind a policy makes it easy to swap filtering methods without changing
generation or sandbox evaluation code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from tts_search.data_produce.baseline_filter import (
    BaselineTokenGapConfig,
    evaluate_baseline_token_gap,
)
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
)

MEDAL_LABELS = frozenset({"gold", "silver", "bronze"})


@dataclass(frozen=True)
class RejectionDecision:
    """Decision returned by a rejection policy."""

    accepted: bool
    reason: str


class RejectionPolicy(Protocol):
    """Interface for deciding whether an evaluated sample is accepted."""

    name: str

    def accepts_result(self, result: Any) -> RejectionDecision:
        """Return whether an EvaluationResult-like object should be accepted."""

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        """Return whether a serialized eval_results.jsonl record is accepted."""


class AcceptAllPolicy:
    """Accept every evaluated sample, including failed sandbox runs."""

    name = "all"

    def accepts_result(self, result: Any) -> RejectionDecision:
        return RejectionDecision(True, "all")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        return RejectionDecision(True, "all")


class AcceptScoredPolicy:
    """Accept every sample that has a numeric sandbox score."""

    name = "accept_scored"

    def accepts_result(self, result: Any) -> RejectionDecision:
        if getattr(result, "score", None) is not None:
            return RejectionDecision(True, "has_score")
        return RejectionDecision(False, "missing_score")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        if record.get("score") is not None:
            return RejectionDecision(True, "has_score")
        return RejectionDecision(False, "missing_score")


class MedalPolicy:
    """Accept samples that earned one of the requested medal labels."""

    name = "medal"

    def __init__(self, accepted_medals: set[str] | None = None):
        medals = accepted_medals or set(MEDAL_LABELS)
        self._accepted_medals = {str(medal).lower() for medal in medals}

    def accepts_result(self, result: Any) -> RejectionDecision:
        medal = str(getattr(result, "submit_medal", "")).lower()
        if medal in self._accepted_medals:
            return RejectionDecision(True, f"medal:{medal}")
        return RejectionDecision(False, f"medal:{medal or 'missing'}")

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        medal = str(record.get("submit_medal", "")).lower()
        if medal in self._accepted_medals:
            return RejectionDecision(True, f"medal:{medal}")
        return RejectionDecision(False, f"medal:{medal or 'missing'}")


class ScoreThresholdPolicy:
    """Accept samples whose sandbox score is at least a fixed threshold."""

    name = "score_threshold"

    def __init__(self, threshold: float):
        self._threshold = float(threshold)

    def accepts_result(self, result: Any) -> RejectionDecision:
        return self._decide(getattr(result, "score", None))

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        return self._decide(record.get("score"))

    def _decide(self, score: Any) -> RejectionDecision:
        if score is None:
            return RejectionDecision(False, "missing_score")
        score_float = float(score)
        if score_float >= self._threshold:
            return RejectionDecision(True, f"score>={self._threshold}")
        return RejectionDecision(False, f"score<{self._threshold}")


class RewardThresholdPolicy:
    """Accept samples whose reward is at least a fixed threshold."""

    name = "reward_threshold"

    def __init__(self, threshold: float):
        self._threshold = float(threshold)

    def accepts_result(self, result: Any) -> RejectionDecision:
        return self._decide(getattr(result, "reward", None))

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        return self._decide(record.get("reward"))

    def _decide(self, reward: Any) -> RejectionDecision:
        if reward is None:
            return RejectionDecision(False, "missing_reward")
        reward_float = float(reward)
        if reward_float >= self._threshold:
            return RejectionDecision(True, f"reward>={self._threshold}")
        return RejectionDecision(False, f"reward<{self._threshold}")


class BetterThanReferencePolicy:
    """Accept samples whose score beats a reference model score for the task."""

    name = "better_than_reference"

    def __init__(self, reference_scores: Mapping[str, float]):
        self._reference_scores = {
            str(task_id): float(score) for task_id, score in reference_scores.items()
        }

    def accepts_result(self, result: Any) -> RejectionDecision:
        return self._decide(
            task_id=getattr(result, "task_id", None),
            task_name=getattr(result, "task_name", None),
            score=getattr(result, "score", None),
            metadata=getattr(result, "metadata", None),
        )

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        return self._decide(
            task_id=record.get("task_id"),
            task_name=record.get("task_name"),
            score=record.get("score"),
            metadata=metadata,
        )

    def _decide(
        self,
        *,
        task_id: Any,
        task_name: Any,
        score: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        if score is None:
            return RejectionDecision(False, "missing_score")

        key_candidates = [str(task_id), str(task_name)]
        reference = None
        for key in key_candidates:
            if key in self._reference_scores:
                reference = self._reference_scores[key]
                break
        if reference is None:
            return RejectionDecision(False, "missing_reference_score")

        score_float = float(score)
        if _score_beats_reference(score_float, reference, metadata):
            op = "<" if _lower_is_better(metadata) else ">"
            return RejectionDecision(True, f"score{op}{reference}")
        op = ">=" if _lower_is_better(metadata) else "<="
        return RejectionDecision(False, f"score{op}{reference}")


class MixedLeaderboardBaselinePolicy:
    """Use medal rejection for leaderboard tasks and baseline-best otherwise."""

    name = "mixed"

    def __init__(
        self,
        reference_scores: Mapping[str, float],
        leaderboard_target: int = 2,
        no_leaderboard_target: int = 4,
        accepted_medals: set[str] | None = None,
    ):
        self._reference_policy = BetterThanReferencePolicy(reference_scores)
        medals = accepted_medals or set(MEDAL_LABELS)
        self._medal_policy = MedalPolicy({str(medal).lower() for medal in medals})
        self._leaderboard_target = int(leaderboard_target)
        self._no_leaderboard_target = int(no_leaderboard_target)

    def has_leaderboard(self, metadata: Mapping[str, Any] | None) -> bool:
        return _metadata_has_leaderboard(metadata)

    def target_for_metadata(self, metadata: Mapping[str, Any] | None) -> int:
        return (
            self._leaderboard_target
            if self.has_leaderboard(metadata)
            else self._no_leaderboard_target
        )

    def accepts_result(self, result: Any) -> RejectionDecision:
        metadata = getattr(result, "metadata", None)
        if self.has_leaderboard(metadata):
            return self._medal_policy.accepts_result(result)
        return self._reference_policy._decide(
            task_id=getattr(result, "task_id", None),
            task_name=getattr(result, "task_name", None),
            score=getattr(result, "score", None),
            metadata=metadata,
        )

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        if self.has_leaderboard(metadata):
            return self._medal_policy.accepts_record(record, metadata)
        return self._reference_policy._decide(
            task_id=record.get("task_id"),
            task_name=record.get("task_name"),
            score=record.get("score"),
            metadata=metadata,
        )


class BaselinePostprocessPolicy:
    """Apply GLM-4.7-style token and valid-test gap filters before a policy."""

    def __init__(
        self,
        inner: RejectionPolicy,
        *,
        max_total_tokens: int = 32768,
        max_relative_gap: float = 0.12,
        require_comparable_gap: bool = True,
    ):
        self._inner = inner
        self._baseline_config = BaselineTokenGapConfig(
            max_total_tokens=int(max_total_tokens),
            gap_config=GapFilterConfig(
                max_relative_gap=float(max_relative_gap),
                require_comparable=bool(require_comparable_gap),
            ),
        )
        self.name = f"baseline_{inner.name}"

    def accepts_result(self, result: Any) -> RejectionDecision:
        baseline = self._baseline_decision(
            slime_message_tokens=getattr(result, "slime_message_tokens", None),
            feedback=getattr(result, "feedback", None),
            metadata=getattr(result, "metadata", None),
        )
        if not baseline.accepted:
            return baseline
        return self._inner.accepts_result(result)

    def accepts_record(
        self,
        record: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> RejectionDecision:
        baseline = self._baseline_decision(
            slime_message_tokens=record.get("slime_message_tokens"),
            feedback=_feedback_from_record(record),
            metadata=metadata,
        )
        if not baseline.accepted:
            return baseline
        return self._inner.accepts_record(record, metadata)

    def target_for_metadata(self, metadata: Mapping[str, Any] | None) -> int | None:
        target_fn = getattr(self._inner, "target_for_metadata", None)
        if target_fn is None:
            return None
        return int(target_fn(metadata))

    def _baseline_decision(
        self,
        *,
        slime_message_tokens: Any,
        feedback: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> RejectionDecision:
        decision = evaluate_baseline_token_gap(
            slime_message_tokens=slime_message_tokens,
            feedback_text=feedback,
            metadata=metadata,
            config=self._baseline_config,
        )
        return RejectionDecision(decision.accepted, decision.reason)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _range_size(low: Any, high: Any) -> float | None:
    low_float = _maybe_float(low)
    high_float = _maybe_float(high)
    if low_float is None or high_float is None:
        return None
    value = abs(high_float - low_float)
    return value if value > 0 else None


def _feedback_from_record(record: Mapping[str, Any]) -> str | None:
    feedback = record.get("feedback")
    if feedback is not None:
        return str(feedback)
    feedback_path = record.get("feedback_path")
    if feedback_path:
        path = Path(str(feedback_path))
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _lower_is_better(metadata: Mapping[str, Any] | None) -> bool:
    if metadata is None:
        return False
    higher = metadata.get("higher_is_better")
    if isinstance(higher, str):
        return higher.strip().lower() in {"false", "0", "no"}
    return higher is False


def _score_beats_reference(
    score: float,
    reference: float,
    metadata: Mapping[str, Any] | None,
) -> bool:
    return score < reference if _lower_is_better(metadata) else score > reference


def _metadata_has_leaderboard(metadata: Mapping[str, Any] | None) -> bool:
    if metadata is None:
        return False
    if _range_size(metadata.get("leaderboard_min"), metadata.get("leaderboard_max")):
        return True

    leaderboard_dir = metadata.get("leaderboard_dir")
    task_name = metadata.get("task_name")
    data_dir = metadata.get("data_dir")
    candidates: list[Path] = []
    if leaderboard_dir and task_name:
        task_root = Path(str(leaderboard_dir)) / str(task_name)
        candidates.extend(
            [
                Path(str(leaderboard_dir)) / f"{task_name}.csv",
                task_root / "info" / "public_leaderboard.csv",
                task_root / "public_leaderboard.csv",
            ]
        )
    if data_dir:
        candidates.append(Path(str(data_dir)) / "info" / "public_leaderboard.csv")
    return any(path.exists() for path in candidates)


def load_reference_scores(path: str | Path) -> dict[str, float]:
    """Load reference scores from JSON/JSONL.

    JSON may be either ``{"task_id": score}`` or a list of records.  JSONL
    records should contain ``task_id`` or ``task_name`` and ``score``.
    """

    reference_path = Path(path)
    if reference_path.suffix == ".jsonl":
        scores: dict[str, float] = {}
        with open(reference_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("task_id") or row.get("task_name")
                if key is not None and row.get("score") is not None:
                    scores[str(key)] = float(row["score"])
        return scores

    data = json.loads(reference_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): float(value) for key, value in data.items()}
    if isinstance(data, list):
        scores = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            key = row.get("task_id") or row.get("task_name")
            if key is not None and row.get("score") is not None:
                scores[str(key)] = float(row["score"])
        return scores
    raise ValueError(f"Unsupported reference score format: {reference_path}")


def build_rejection_policy(
    *,
    name: str | None,
    score_threshold: float | None = None,
    reward_threshold: float | None = None,
    reference_scores_path: str | Path | None = None,
    accepted_medals: list[str] | tuple[str, ...] | set[str] | None = None,
    apply_baseline_filters: bool = False,
    baseline_token_limit: int = 32768,
    baseline_relative_gap_limit: float = 0.12,
    mixed_leaderboard_target: int = 2,
    mixed_no_leaderboard_target: int = 4,
) -> RejectionPolicy:
    """Build a rejection policy from config values."""

    policy_name = (name or "accept_scored").lower()
    if policy_name in {"all", "accept_all", "none"}:
        return AcceptAllPolicy()

    policy: RejectionPolicy
    if policy_name in {"success", "scored", "accept_scored", "all_scored"}:
        policy = AcceptScoredPolicy()
    elif policy_name in {"medal", "has_medal"}:
        medals = (
            {str(medal).lower() for medal in accepted_medals}
            if accepted_medals
            else None
        )
        policy = MedalPolicy(accepted_medals=medals)
    elif policy_name in {"score_threshold", "threshold"}:
        if score_threshold is None:
            raise ValueError("score_threshold rejection policy requires a threshold")
        policy = ScoreThresholdPolicy(score_threshold)
    elif policy_name in {"reward_threshold", "reward"}:
        threshold = (
            reward_threshold if reward_threshold is not None else score_threshold
        )
        if threshold is None:
            raise ValueError("reward_threshold rejection policy requires a threshold")
        policy = RewardThresholdPolicy(threshold)
    elif policy_name in {"better_than_reference", "compare_reference"}:
        if reference_scores_path is None:
            raise ValueError(
                "better_than_reference rejection policy requires reference_scores_path"
            )
        policy = BetterThanReferencePolicy(load_reference_scores(reference_scores_path))
    elif policy_name in {"mixed", "mixed_leaderboard_baseline"}:
        medals = (
            {str(medal).lower() for medal in accepted_medals}
            if accepted_medals
            else None
        )
        reference_scores = (
            load_reference_scores(reference_scores_path)
            if reference_scores_path is not None
            else {}
        )
        policy = MixedLeaderboardBaselinePolicy(
            reference_scores,
            leaderboard_target=mixed_leaderboard_target,
            no_leaderboard_target=mixed_no_leaderboard_target,
            accepted_medals=medals,
        )
    else:
        raise ValueError(f"Unknown rejection policy: {name}")

    if apply_baseline_filters:
        return BaselinePostprocessPolicy(
            policy,
            max_total_tokens=baseline_token_limit,
            max_relative_gap=baseline_relative_gap_limit,
        )
    return policy
