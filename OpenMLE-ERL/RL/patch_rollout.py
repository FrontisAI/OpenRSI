"""
Monkey patch for custom rollout behaviors.
This file should be imported BEFORE slime modules are loaded.
"""

import asyncio
import inspect
import os

import numpy as np
import torch
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step


REWARD_VIEW_KEYS = (
    "metric_static_base_reward",
    "dynamic_raw_reward",
)

QUALITY_PROXY_KEYS = (
    "metric_static_base_reward",
    "metric_static_base_reward_is_task_best_improvement",
    "history_best_static_base_reward",
)


def _to_finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _has_static_bounds(metadata):
    flag = metadata.get("metric_static_base_reward_has_bounds")
    if flag is not None:
        try:
            return bool(int(flag))
        except (TypeError, ValueError):
            pass
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = _to_finite_float(metadata.get("theoretical_min"))
    theoretical_max = _to_finite_float(metadata.get("theoretical_max"))
    leaderboard_min = _to_finite_float(metadata.get("leaderboard_min"))
    leaderboard_max = _to_finite_float(metadata.get("leaderboard_max"))
    if higher_is_better is False:
        best = leaderboard_min if leaderboard_min is not None else theoretical_min
        worst = leaderboard_max if leaderboard_max is not None else theoretical_max
    else:
        best = leaderboard_max if leaderboard_max is not None else theoretical_max
        worst = leaderboard_min if leaderboard_min is not None else theoretical_min
    return best is not None and worst is not None


def _build_mode_bucket():
    return {
        "samples": [],
        "reward_views": {key: [] for key in REWARD_VIEW_KEYS},
        "hack_count": 0,
        "empty_count": 0,
        "success_count": 0,
        "network_success_count": 0,
        "scored_count": 0,
        "total_count": 0,
    }


def _log_reward_view_means(log_dict, prefix, reward_views):
    for reward_key, values in reward_views.items():
        mean_value = _mean_or_none(values)
        if mean_value is not None:
            output_key = "static_base_reward" if reward_key == "metric_static_base_reward" else reward_key
            log_dict[f"{prefix}/{output_key}"] = mean_value
            log_dict[f"{prefix}/{output_key}_count"] = len(values)
            if reward_key == "metric_static_base_reward":
                log_dict[f"{prefix}/reward"] = mean_value
                log_dict[f"{prefix}/reward_count"] = len(values)


def _log_quality_proxy_metrics(log_dict, prefix, samples):
    if not samples:
        return

    group_buckets = {}
    for sample_idx, sample in enumerate(samples):
        metadata = sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}
        group_key = metadata.get("group_index")
        if group_key is None and hasattr(sample, "group_index"):
            group_key = sample.group_index
        if group_key is None:
            group_key = f"sample_{sample_idx}"
        bucket = group_buckets.setdefault(
            group_key,
            {
                "metric_static_base_reward": [],
                "metric_static_base_reward_is_task_best_improvement": [],
            },
        )
        for key in bucket:
            if key == "metric_static_base_reward" and not _has_static_bounds(metadata):
                continue
            value = _to_finite_float(metadata.get(key))
            if value is not None:
                bucket[key].append(value)

    if not group_buckets:
        return

    reward_key = "metric_static_base_reward"
    group_bests = [max(bucket[reward_key]) for bucket in group_buckets.values() if bucket[reward_key]]
    if group_bests:
        log_dict[f"{prefix}/group_best_static_base_reward"] = float(np.mean(group_bests))
        log_dict[f"{prefix}/group_best_static_base_reward_count"] = len(group_bests)
        log_dict[f"{prefix}/group_best_static_base_reward_ge_0p95_rate"] = float(
            np.mean([best >= 0.95 for best in group_bests])
        )

    group_improvement_counts = [
        float(sum(bucket["metric_static_base_reward_is_task_best_improvement"]))
        for bucket in group_buckets.values()
        if bucket["metric_static_base_reward"]
    ]
    if group_improvement_counts:
        log_dict[f"{prefix}/static_base_reward_is_task_best_improvement"] = float(np.mean(group_improvement_counts))

    history_values = [
        _to_finite_float(
            (sample.metadata if hasattr(sample, "metadata") and isinstance(sample.metadata, dict) else {}).get(
                "history_best_static_base_reward"
            )
        )
        for sample in samples
    ]
    history_values = [value for value in history_values if value is not None]
    if history_values:
        log_dict[f"{prefix}/history_best_static_base_reward"] = float(np.mean(history_values))


def _compute_debug_rewards(args, samples):
    import slime.ray.rollout as rollout_mod

    if getattr(args, "custom_reward_post_process_path", None):
        custom_post_process = rollout_mod.load_function(args.custom_reward_post_process_path)
        raw_rewards, processed_rewards = custom_post_process(args, samples)
        return list(raw_rewards), list(processed_rewards)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


def patched_save_db_snapshot(self, rollout_id):
    """Save a snapshot of the program database if enabled."""
    # Check if DB snapshot saving is enabled
    if os.environ.get("SAVE_DB_SNAPSHOTS", "0") != "1":
        return

    try:
        import slime.ray.rollout as rollout_mod

        custom_generate_module_path = self.args.custom_generate_function_path
        if custom_generate_module_path:
            module_path, _ = custom_generate_module_path.rsplit(".", 1)
            save_db_snapshot_func = rollout_mod.load_function(f"{module_path}.save_db_snapshot")

            run_dir = os.environ.get("RUN_DIR", ".")
            snapshot_path = os.path.join(run_dir, f"program_database_iter_{rollout_id}.db")

            result = save_db_snapshot_func(snapshot_path)
            if inspect.isawaitable(result):
                asyncio.run(result)
            print(f"[RolloutManager] Database snapshot saved to {snapshot_path}")
    except Exception as e:
        print(f"[RolloutManager] Warning: Failed to save database snapshot: {e}")


def patched_log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.load_debug_rollout_data:
        return
    log_dict = {**(rollout_extra_metrics or {})}
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time

    import slime.ray.rollout as rollout_mod

    log_dict |= rollout_mod.dict_add_prefix(rollout_mod.compute_statistics(response_lengths), "rollout/response_len/")
    log_dict |= rollout_mod._compute_zero_std_metrics(args, samples)
    log_dict |= rollout_mod._compute_spec_metrics(args, samples)
    log_dict |= rollout_mod.dict_add_prefix(rollout_mod._compute_reward_cat_metrics(args, samples), "rollout/")

    mode_buckets = {
        "draft": _build_mode_bucket(),
        "debug": _build_mode_bucket(),
        "improve": _build_mode_bucket(),
        "crossover": _build_mode_bucket(),
    }
    overall_reward_views = {key: [] for key in REWARD_VIEW_KEYS}
    for sample in samples:
        if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
            generation_mode = sample.metadata.get("generation_mode", "draft")
            if generation_mode == "improve":
                if sample.metadata.get("parent_reward", None) == 0.0:
                    generation_mode = "debug"
            if generation_mode not in mode_buckets:
                generation_mode = "draft"
            mode_bucket = mode_buckets[generation_mode]
            mode_bucket["samples"].append(sample)
            mode_bucket["total_count"] += 1
            hack_value = sample.metadata.get("hack", 0)
            code_category = sample.metadata.get("code_category", "valid")
            score = sample.metadata.get("score")
            status_codes = sample.metadata.get("status_codes", [])
            latest_status_code = status_codes[-1] if isinstance(status_codes, list) and status_codes else None
            success_flag = latest_status_code == 200 or hack_value == 1
            network_success_flag = latest_status_code == 200 or latest_status_code in (-1, -2, -3, -4)

            if hack_value == 1:
                mode_bucket["hack_count"] += 1
            if code_category == "empty":
                mode_bucket["empty_count"] += 1
            if success_flag:
                mode_bucket["success_count"] += 1
            if network_success_flag:
                mode_bucket["network_success_count"] += 1
            if score is not None:
                mode_bucket["scored_count"] += 1

            for reward_key in REWARD_VIEW_KEYS:
                if reward_key == "metric_static_base_reward" and not _has_static_bounds(sample.metadata):
                    continue
                reward_value = _to_finite_float(sample.metadata.get(reward_key))
                if reward_value is not None:
                    mode_bucket["reward_views"][reward_key].append(reward_value)
                    overall_reward_views[reward_key].append(reward_value)

    for mode_name, mode_bucket in mode_buckets.items():
        _log_reward_view_means(log_dict, f"rollout/{mode_name}", mode_bucket["reward_views"])
        _log_quality_proxy_metrics(log_dict, f"rollout/{mode_name}/quality", mode_bucket["samples"])
        total_count = mode_bucket["total_count"]
        if total_count > 0:
            log_dict[f"rollout/{mode_name}/hack_rate"] = mode_bucket["hack_count"] / total_count
            log_dict[f"rollout/{mode_name}/hack_count"] = mode_bucket["hack_count"]
            log_dict[f"rollout/{mode_name}/empty_rate"] = mode_bucket["empty_count"] / total_count
            log_dict[f"rollout/{mode_name}/empty_count"] = mode_bucket["empty_count"]
            log_dict[f"rollout/{mode_name}/success_rate"] = mode_bucket["success_count"] / total_count
            log_dict[f"rollout/{mode_name}/success_count"] = mode_bucket["success_count"]
            log_dict[f"rollout/{mode_name}/network_success_rate"] = mode_bucket["network_success_count"] / total_count
            log_dict[f"rollout/{mode_name}/network_success_count"] = mode_bucket["network_success_count"]
            log_dict[f"rollout/{mode_name}/scored_rate"] = mode_bucket["scored_count"] / total_count
            log_dict[f"rollout/{mode_name}/scored_count"] = mode_bucket["scored_count"]
            log_dict[f"rollout/{mode_name}/total_count"] = total_count

    total_samples = len(samples)
    code_category_counts = {
        "hack": 0,
        "hack_verify": 0,
        "no_verify": 0,
        "empty": 0,
        "valid": 0,
    }
    for sample in samples:
        if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
            code_category = sample.metadata.get("code_category", "valid")
            if code_category in code_category_counts:
                code_category_counts[code_category] += 1
    if total_samples > 0:
        for code_category, count in code_category_counts.items():
            log_dict[f"rollout/code_category/{code_category}/count"] = count
            log_dict[f"rollout/code_category/{code_category}/ratio"] = count / total_samples

    total_draft_count = mode_buckets["draft"]["total_count"]
    total_debug_count = mode_buckets["debug"]["total_count"]
    total_improve_count = mode_buckets["improve"]["total_count"]
    total_crossover_count = mode_buckets["crossover"]["total_count"]
    total_samples_with_mode = total_draft_count + total_debug_count + total_improve_count + total_crossover_count
    if total_samples_with_mode > 0:
        log_dict["rollout/draft/ratio"] = total_draft_count / total_samples_with_mode
        log_dict["rollout/debug/ratio"] = total_debug_count / total_samples_with_mode
        log_dict["rollout/improve/ratio"] = total_improve_count / total_samples_with_mode
        log_dict["rollout/crossover/ratio"] = total_crossover_count / total_samples_with_mode
        total_hack_count = sum(bucket["hack_count"] for bucket in mode_buckets.values())
        total_empty_count = sum(bucket["empty_count"] for bucket in mode_buckets.values())
        total_success_count = sum(bucket["success_count"] for bucket in mode_buckets.values())
        total_network_success_count = (
            sum(bucket["network_success_count"] for bucket in mode_buckets.values())
        )
        total_scored_count = sum(bucket["scored_count"] for bucket in mode_buckets.values())
        log_dict["rollout/hack_rate"] = total_hack_count / total_samples_with_mode
        log_dict["rollout/hack_count"] = total_hack_count
        log_dict["rollout/empty_rate"] = total_empty_count / total_samples_with_mode
        log_dict["rollout/empty_count"] = total_empty_count
        log_dict["rollout/success_rate"] = total_success_count / total_samples_with_mode
        log_dict["rollout/success_count"] = total_success_count
        log_dict["rollout/network_success_rate"] = total_network_success_count / total_samples_with_mode
        log_dict["rollout/network_success_count"] = total_network_success_count
        log_dict["rollout/scored_rate"] = total_scored_count / total_samples_with_mode
        log_dict["rollout/scored_count"] = total_scored_count

    _log_reward_view_means(log_dict, "rollout", overall_reward_views)
    _log_quality_proxy_metrics(log_dict, "rollout/quality", samples)

    raw_rewards, processed_rewards = _compute_debug_rewards(args, samples)
    raw_reward_mean = _mean_or_none([value for value in raw_rewards if _to_finite_float(value) is not None])
    processed_reward_mean = _mean_or_none([value for value in processed_rewards if _to_finite_float(value) is not None])
    if raw_reward_mean is not None:
        log_dict["rollout/raw_reward_debug"] = raw_reward_mean
    if processed_reward_mean is not None:
        log_dict["rollout/processed_reward"] = processed_reward_mean
    printable_log_dict = {
        **log_dict,
        "rollout/raw_rewards": raw_rewards,
        "rollout/processed_rewards": processed_rewards,
    }
    print(f"perf {rollout_id}: {printable_log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    # Keep list metrics out of wandb/tensorboard (can be huge); still printed above.
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # Persist program DB snapshot once per rollout when enabled.
    # (Previously RolloutManager.save_db_snapshot was patched but never called.)
    try:
        patched_save_db_snapshot(type("_RolloutArgsProxy", (), {"args": args})(), rollout_id)
    except Exception as e:
        print(f"[PATCH_ROLLOUT] Warning: Failed to save database snapshot: {e}")

    # Return True so --custom-rollout-log-function-path skips stock logging.
    return True


def apply_patch():
    """Apply monkey patch to rollout internals."""
    try:
        import slime.ray.rollout as rollout_mod

        rollout_mod.RolloutManager.save_db_snapshot = patched_save_db_snapshot
        rollout_mod._log_rollout_data = patched_log_rollout_data
        print("✅ [PATCH_ROLLOUT] Successfully patched RolloutManager.save_db_snapshot and _log_rollout_data")
        return True
    except Exception as e:
        print(f"⚠️ [PATCH_ROLLOUT] Failed to patch: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ != "__main__":
    apply_patch()
