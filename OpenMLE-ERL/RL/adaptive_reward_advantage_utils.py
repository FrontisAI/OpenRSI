import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from reward_func_utils import get_bound_limits_signed, score2reward, signed_score


BOUND_MODE_TOP1_TOP8 = "top1_top8"
BOUND_MODE_TOP1_TOP16 = "top1_top16"
BOUND_MODE_TOP1_ALL_MEAN = "top1_all_mean"
BOUND_MODE_TOP1_ALL_MEDIAN = "top1_all_median"
BOUND_MODE_THEORETICAL = "theoretical"

SUPPORTED_ADAPTIVE_BOUND_MODES = {
    BOUND_MODE_TOP1_TOP8,
    BOUND_MODE_TOP1_TOP16,
    BOUND_MODE_TOP1_ALL_MEAN,
    BOUND_MODE_TOP1_ALL_MEDIAN,
    BOUND_MODE_THEORETICAL,
}


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _sanitize_bound_pair(
    best_signed: float | None,
    worst_signed: float | None,
    *,
    metadata: dict,
    min_span: float = 1e-6,
) -> tuple[float | None, float | None]:
    static_best, static_worst = get_bound_limits_signed(metadata)

    if _finite(best_signed) and _finite(static_best):
        best_signed = min(float(best_signed), float(static_best))
    if _finite(worst_signed) and _finite(static_worst):
        worst_signed = max(float(worst_signed), float(static_worst))

    if not (_finite(best_signed) and _finite(worst_signed)):
        if _finite(static_best) and _finite(static_worst):
            best_signed = float(static_best)
            worst_signed = float(static_worst)
        else:
            return None, None

    best_signed = float(best_signed)
    worst_signed = float(worst_signed)
    if best_signed <= worst_signed:
        best_signed = worst_signed + max(float(min_span), 1e-12)
    return best_signed, worst_signed


def compute_adaptive_bound_pair(
    *,
    metadata: dict,
    historical_scores: Iterable[float],
    current_group_scores: Iterable[float],
    mode: str,
    min_span: float = 1e-6,
    lower_shift_ratio: float = 0.0,
) -> tuple[float | None, float | None]:
    mode = (mode or BOUND_MODE_TOP1_TOP8).strip().lower()
    if mode not in SUPPORTED_ADAPTIVE_BOUND_MODES:
        raise ValueError(f"Unsupported adaptive reward bound mode: {mode}")

    if mode == BOUND_MODE_THEORETICAL:
        static_best, static_worst = get_bound_limits_signed(metadata)
        return _sanitize_bound_pair(static_best, static_worst, metadata=metadata, min_span=min_span)

    signed_scores = [
        signed_score(score, metadata)
        for score in list(historical_scores) + list(current_group_scores)
        if _finite(score)
    ]
    if not signed_scores:
        static_best, static_worst = get_bound_limits_signed(metadata)
        return _sanitize_bound_pair(static_best, static_worst, metadata=metadata, min_span=min_span)

    signed_scores = sorted(float(s) for s in signed_scores)[::-1]
    best_signed = signed_scores[0]

    if mode == BOUND_MODE_TOP1_TOP8:
        worst_signed = signed_scores[min(7, len(signed_scores) - 1)]
    elif mode == BOUND_MODE_TOP1_TOP16:
        worst_signed = signed_scores[min(15, len(signed_scores) - 1)]
    elif mode == BOUND_MODE_TOP1_ALL_MEAN:
        worst_signed = float(np.mean(signed_scores))
    else:
        assert mode == BOUND_MODE_TOP1_ALL_MEDIAN
        worst_signed = float(np.median(signed_scores))

    shift_ratio = max(float(lower_shift_ratio), 0.0)
    if shift_ratio > 0.0:
        worst_signed = float(worst_signed) - shift_ratio * max(float(best_signed) - float(worst_signed), 0.0)

    return _sanitize_bound_pair(best_signed, worst_signed, metadata=metadata, min_span=min_span)


def score_to_group_adaptive_reward(
    *,
    score: float | None,
    metadata: dict,
    historical_scores: Iterable[float],
    current_group_scores: Iterable[float],
    mode: str,
    reward_mapping_mode: str,
    use_score2reward: bool,
    default_reward: float = 0.0,
    min_span: float = 1e-6,
    lower_shift_ratio: float = 0.0,
    single_finite_reward_one: bool = False,
) -> tuple[float, dict[str, float | str | bool | None]]:
    context: dict[str, float | str | bool | None] = {
        "enabled": False,
        "bound_mode": mode,
        "best_signed": None,
        "worst_signed": None,
        "lower_shift_ratio": float(max(lower_shift_ratio, 0.0)),
        "single_finite_reward_one_applied": False,
    }
    if not _finite(score):
        return float(default_reward), context
    if not use_score2reward:
        return float(score), context

    available_signed_scores = [
        signed_score(value, metadata)
        for value in list(historical_scores) + list(current_group_scores)
        if _finite(value)
    ]
    if single_finite_reward_one and len(available_signed_scores) == 1:
        only_signed = float(available_signed_scores[0])
        context["enabled"] = True
        context["best_signed"] = only_signed
        context["worst_signed"] = only_signed
        context["single_finite_reward_one_applied"] = True
        return 1.0, context

    best_signed, worst_signed = compute_adaptive_bound_pair(
        metadata=metadata,
        historical_scores=historical_scores,
        current_group_scores=current_group_scores,
        mode=mode,
        min_span=min_span,
        lower_shift_ratio=lower_shift_ratio,
    )
    context["best_signed"] = best_signed
    context["worst_signed"] = worst_signed
    context["enabled"] = _finite(best_signed) and _finite(worst_signed)

    reward_metadata = dict(metadata)
    if context["enabled"]:
        reward_metadata["dynamic_bound_best_signed"] = best_signed
        reward_metadata["dynamic_bound_worst_signed"] = worst_signed
    reward = score2reward(float(score), reward_metadata, mode=reward_mapping_mode)
    return float(reward), context


def compute_gspo_group_advantages(
    rewards: list[float],
    *,
    std_normalization: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    arr = arr - float(arr.mean())
    if std_normalization:
        std = float(arr.std(ddof=1 if arr.size > 1 else 0))
        arr = arr / (std + float(eps))
    return arr.astype(np.float64).tolist()


def _solve_entropic_beta(
    rewards: np.ndarray,
    *,
    target_kl: float,
    beta_max: float,
    iters: int,
) -> float:
    k = int(rewards.size)
    if k < 2:
        return 0.0

    log_k = math.log(k)
    rewards = rewards.astype(np.float64)
    centered = rewards - float(rewards.max())

    def kl_hat(beta_value: float) -> float:
        logits = beta_value * centered
        log_denom = float(np.logaddexp.reduce(logits))
        log_q = logits - log_denom
        q = np.exp(log_q)
        return float(np.sum(q * (log_q + log_k)))

    lo, hi = 0.0, 1.0
    if kl_hat(hi) < target_kl:
        while hi < beta_max and kl_hat(hi) < target_kl:
            hi *= 2.0
        if kl_hat(hi) < target_kl:
            return float(hi)

    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if kl_hat(mid) < target_kl:
            lo = mid
        else:
            hi = mid
    return float(hi)


def compute_ttt_entropic_advantages(
    rewards: list[float],
    *,
    target_kl: float = math.log(2.0),
    beta_max: float = 1e6,
    iters: int = 60,
    eps: float = 1e-12,
) -> list[float]:
    if not rewards:
        return []
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    if rewards_arr.size < 2:
        return [0.0 for _ in rewards]
    if np.allclose(rewards_arr, rewards_arr[0]):
        return [0.0 for _ in rewards]

    beta = _solve_entropic_beta(rewards_arr, target_kl=target_kl, beta_max=beta_max, iters=iters)
    centered = rewards_arr - float(rewards_arr.max())
    exp_vals = np.exp(beta * centered)
    if rewards_arr.size == 1:
        loo_z = exp_vals
    else:
        loo_z = (exp_vals.sum() - exp_vals) / float(rewards_arr.size - 1)
    weights = exp_vals / (loo_z + float(eps))
    advantages = weights - 1.0
    return advantages.astype(np.float64).tolist()


def group_samples_by_group_index(args, samples) -> list[list]:
    if samples and isinstance(samples[0], list):
        return samples

    grouped: dict[int, list] = defaultdict(list)
    fallback_groups: list[list] = []
    for idx, sample in enumerate(samples):
        group_index = getattr(sample, "group_index", None)
        if group_index is None:
            fallback_groups.append([sample])
            continue
        grouped[int(group_index)].append(sample)

    if grouped:
        return [grouped[key] for key in sorted(grouped)]
    if fallback_groups:
        return fallback_groups

    n = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    return [samples[i : i + n] for i in range(0, len(samples), n)]
