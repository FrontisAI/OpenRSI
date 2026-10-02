from __future__ import annotations

import sys
from pathlib import Path

import pytest

OPENMLE_ERL_RL = Path(__file__).resolve().parents[1]
if str(OPENMLE_ERL_RL) not in sys.path:
    sys.path.insert(0, str(OPENMLE_ERL_RL))

import adaptive_reward_advantage_utils as utils  # noqa: E402


class _Sample:
    def __init__(self, group_index):
        self.group_index = group_index


class _Args:
    def __init__(self, n_samples_per_prompt=2):
        self.n_samples_per_prompt = n_samples_per_prompt


def _total(groups):
    return sum(len(group) for group in groups)


def test_mixed_group_index_never_drops_samples():
    """Samples without a group_index must not vanish when others have one."""
    samples = [_Sample(1), _Sample(1), _Sample(2), _Sample(None)]
    groups = utils.group_samples_by_group_index(_Args(), samples)
    assert _total(groups) == len(samples)


def test_all_missing_group_index_never_drops_samples():
    samples = [_Sample(None) for _ in range(4)]
    groups = utils.group_samples_by_group_index(_Args(), samples)
    assert _total(groups) == len(samples)


def test_all_missing_group_index_uses_contiguous_chunks():
    """With no group_index the samples must be chunked, not made singletons."""
    samples = [_Sample(None) for _ in range(4)]
    groups = utils.group_samples_by_group_index(_Args(n_samples_per_prompt=2), samples)
    assert [len(group) for group in groups] == [2, 2]
    assert [s for group in groups for s in group] == samples


def test_all_missing_group_index_does_not_neutralise_advantages():
    """Singleton groups made every advantage exactly 0.0, silently flattening the batch."""
    samples = [_Sample(None) for _ in range(4)]
    groups = utils.group_samples_by_group_index(_Args(n_samples_per_prompt=2), samples)

    advantages = []
    for group in groups:
        advantages.extend(
            utils.compute_ttt_entropic_advantages([0.1, 0.9, 0.5, 0.7][: len(group)])
        )

    assert len(advantages) == len(samples)
    assert any(value != 0.0 for value in advantages)


def test_indexed_samples_still_group_by_group_index():
    samples = [_Sample(1), _Sample(2), _Sample(1), _Sample(2)]
    groups = utils.group_samples_by_group_index(_Args(), samples)
    assert [len(group) for group in groups] == [2, 2]
    assert all(group[0].group_index == key for key, group in zip((1, 2), groups))


def test_nested_list_input_is_returned_unchanged():
    already_grouped = [[_Sample(1)], [_Sample(2)]]
    assert utils.group_samples_by_group_index(_Args(), already_grouped) is already_grouped


def test_empty_input_returns_empty_groups():
    assert utils.group_samples_by_group_index(_Args(), []) == []


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_chunking_never_drops_or_duplicates_samples(n):
    samples = [_Sample(None) for _ in range(7)]
    groups = utils.group_samples_by_group_index(_Args(n_samples_per_prompt=n), samples)
    assert _total(groups) == len(samples)
    assert [s for group in groups for s in group] == samples