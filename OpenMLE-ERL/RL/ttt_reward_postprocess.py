import math

from adaptive_reward_advantage_utils import (
    compute_ttt_entropic_advantages,
    group_samples_by_group_index,
)


def post_process_rewards(args, samples):
    grouped_samples = group_samples_by_group_index(args, samples)

    raw_rewards: list[float] = []
    processed_rewards: list[float] = []

    for group in grouped_samples:
        group_rewards = [float(sample.get_reward_value(args)) for sample in group]
        group_advantages = compute_ttt_entropic_advantages(
            group_rewards,
            target_kl=float(math.log(2.0)),
            beta_max=1e6,
            iters=60,
            eps=1e-12,
        )
        raw_rewards.extend(group_rewards)
        processed_rewards.extend(group_advantages)

    return raw_rewards, processed_rewards
