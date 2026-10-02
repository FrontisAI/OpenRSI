from __future__ import annotations

import pytest

from tts_search import reward_func_utils

NEUTRAL = 0.5


def _meta(**overrides):
    metadata = {
        "higher_is_better": True,
        "theoretical_min": 0.0,
        "theoretical_max": 1.0,
        "leaderboard_min": 0.5,
        "leaderboard_max": 0.9,
    }
    metadata.update(overrides)
    return metadata


def _reward(score, metadata):
    return reward_func_utils.score2reward(score, metadata)


def test_higher_is_better_maps_theoretical_worst_to_neutral():
    """The worst achievable score must anchor the neutral reward of 0.5."""
    metadata = _meta()
    assert _reward(metadata["theoretical_min"], metadata) == pytest.approx(NEUTRAL)


def test_higher_is_better_anchors_on_lowest_achievable_bound():
    """Rewards must span the theoretical range, not just the leaderboard range.

    With theoretical [0.0, 1.0] and leaderboard [0.5, 0.9] the worst score is the
    theoretical minimum (0.0), so sub-leaderboard solutions must receive a reward
    strictly between neutral and best rather than being crushed against neutral.
    """
    metadata = _meta()
    worst = _reward(metadata["theoretical_min"], metadata)
    worst_leaderboard = _reward(metadata["leaderboard_min"], metadata)
    best = _reward(metadata["theoretical_max"], metadata)

    assert worst == pytest.approx(NEUTRAL)
    assert worst < worst_leaderboard < best
    assert best > NEUTRAL


def test_higher_is_better_is_monotonic_in_score():
    metadata = _meta()
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    rewards = [_reward(score, metadata) for score in scores]
    assert rewards == sorted(rewards)


def test_lower_is_better_maps_theoretical_worst_to_neutral():
    metadata = _meta(
        higher_is_better=False,
        theoretical_min=0.0,
        theoretical_max=100.0,
        leaderboard_min=10.0,
        leaderboard_max=50.0,
    )
    assert _reward(metadata["theoretical_max"], metadata) == pytest.approx(NEUTRAL)


def test_lower_is_better_is_monotonic_in_score():
    metadata = _meta(
        higher_is_better=False,
        theoretical_min=0.0,
        theoretical_max=100.0,
        leaderboard_min=10.0,
        leaderboard_max=50.0,
    )
    scores = [0.0, 25.0, 50.0, 75.0, 100.0]
    rewards = [_reward(score, metadata) for score in scores]
    assert rewards == sorted(rewards, reverse=True)


@pytest.mark.parametrize(
    ("higher_is_better", "bounds", "score"),
    [
        (True, {"theoretical_min": None, "leaderboard_min": None}, 0.5),
        (False, {"theoretical_max": None, "leaderboard_max": None}, 0.5),
    ],
)
def test_unknown_bounds_still_return_neutral(higher_is_better, bounds, score):
    """An undeterminable interval must keep returning the neutral reward."""
    metadata = _meta(higher_is_better=higher_is_better, **bounds)
    assert _reward(score, metadata) == pytest.approx(NEUTRAL)


def test_lower_bound_is_floored_at_minus_100():
    """A theoretical floor below -100 is clamped, keeping the range finite."""
    metadata = _meta(theoretical_min=-500.0, leaderboard_min=-400.0)
    assert _reward(-100.0, metadata) == pytest.approx(NEUTRAL)


def test_upper_bound_is_capped_at_100_for_lower_is_better():
    """A theoretical ceiling above 100 is clamped, keeping the range finite."""
    metadata = _meta(
        higher_is_better=False,
        theoretical_min=-500.0,
        theoretical_max=500.0,
        leaderboard_min=-400.0,
        leaderboard_max=400.0,
    )
    assert _reward(100.0, metadata) == pytest.approx(NEUTRAL)
