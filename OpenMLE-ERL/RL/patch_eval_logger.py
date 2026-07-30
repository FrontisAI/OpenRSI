"""
Monkey patch for custom eval logger.
This file should be imported BEFORE slime modules are loaded.

Usage: Add this at the very beginning of train.py or main entry point:
    import sys
    sys.path.insert(0, '/path/to/examples/mle-agent-rl')
    import patch_eval_logger
"""

import sys
import os
import pandas as pd
import numpy as np
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

# Make sure we can import from current directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEFAULT_LEADERBOARD_ROOTS: tuple[str, ...] = ()
LEADERBOARD_PTH = os.getenv("LEADERBOARD_ROOT", "").strip()
REWARD_VIEW_KEYS = (
    "metric_static_base_reward",
    "dynamic_raw_reward",
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


def _build_mode_bucket():
    return {
        "reward_views": {key: [] for key in REWARD_VIEW_KEYS},
        "scores": [],
        "scores_every_tasks": {},
        "response_lengths": [],
        "status_codes": [],
        "hack_count": 0,
        "empty_count": 0,
        "success_count": 0,
        "total_count": 0,
        "samples": [],
    }


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


def _log_reward_view_means(log_dict, prefix, reward_views):
    for reward_key, values in reward_views.items():
        if values:
            output_key = "static_base_reward" if reward_key == "metric_static_base_reward" else reward_key
            mean_value = float(np.mean(values))
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


def is_lower_better(leaderboard: pd.DataFrame) -> bool:
    """
    Determine from the leaderboard whether lower scores are better.
    Returns True if lower scores are better, False otherwise.
    """
    scores = leaderboard["score"]
    top_score = scores.iloc[0]
    bottom_score = scores.iloc[-1]
    return bool(top_score < bottom_score)


def get_medal_for_score(score: float | None, leaderboard: pd.DataFrame) -> str:
    """
    Determine the medal level for a score using the leaderboard.

    Args:
        score: Score to evaluate.
        leaderboard: DataFrame containing a score column.

    Returns:
        str: Medal level ("Gold", "Silver", "Bronze", or "None").
    """
    if score is None or np.isnan(score):
        return "None"

    lower_better = is_lower_better(leaderboard)
    num_teams = len(leaderboard)
    scores = leaderboard["score"]

    def get_score_at_position(position: int) -> float:
        if position - 1 >= len(scores) or position < 1:
            raise IndexError("Position out of bounds in the leaderboard.")
        return scores.iloc[position - 1]

    def get_thresholds(num_teams: int) -> tuple:
        """Return the gold, silver, and bronze score thresholds."""
        if 1 <= num_teams < 100:
            gold_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
        elif 100 <= num_teams < 250:
            gold_threshold = get_score_at_position(10)
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.2)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.4)))
        elif 250 <= num_teams < 1000:
            gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
            silver_threshold = get_score_at_position(50)
            bronze_threshold = get_score_at_position(100)
        elif num_teams >= 1000:
            gold_threshold = get_score_at_position(10 + int(num_teams * 0.002))
            silver_threshold = get_score_at_position(max(1, int(num_teams * 0.05)))
            bronze_threshold = get_score_at_position(max(1, int(num_teams * 0.1)))
        else:
            raise ValueError("Number of teams in leaderboard must be greater than 0.")

        return float(gold_threshold), float(silver_threshold), float(bronze_threshold)

    gold_threshold, silver_threshold, bronze_threshold = get_thresholds(num_teams)

    # Determine the medal level.
    if lower_better:
        if score <= gold_threshold:
            return "Gold"
        elif score <= silver_threshold:
            return "Silver"
        elif score <= bronze_threshold:
            return "Bronze"
    else:
        if score >= gold_threshold:
            return "Gold"
        elif score >= silver_threshold:
            return "Silver"
        elif score >= bronze_threshold:
            return "Bronze"

    return "None"


def load_public_leaderboard(task_name: str) -> pd.DataFrame | None:
    if os.getenv("LEADERBOARD_ROOTS"):
        roots = [item for item in os.getenv("LEADERBOARD_ROOTS", "").split(os.pathsep) if item]
    else:
        roots = [LEADERBOARD_PTH] if LEADERBOARD_PTH else []

    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates = [
            os.path.join(root, task_name, "info", "public_leaderboard.csv"),
            os.path.join(root, task_name, "leaderboard.csv"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            leaderboard = pd.read_csv(path)
            if "score" in leaderboard.columns:
                leaderboard = leaderboard.copy()
                leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
                leaderboard = leaderboard[leaderboard["score"].notna()].reset_index(drop=True)
                if "rank" in leaderboard.columns:
                    leaderboard = leaderboard.sort_values("rank", kind="stable").reset_index(drop=True)
                return leaderboard
    return None


def custom_log_eval_rollout_data(rollout_id, args, data, extra_metrics=None):
    """Extended eval logger with custom metrics for MLE-Agent."""
    from slime.utils.metric_utils import compute_pass_rate, dict_add_prefix
    from slime.ray.rollout import _compute_reward_cat_metrics

    # Keep compatibility with upstream signature:
    # _log_eval_rollout_data(rollout_id, args, data, extra_metrics)
    # and preserve any framework-provided eval metrics.
    log_dict = {**(extra_metrics or {})}

    for key in data.keys():
        rewards = data[key]["rewards"]
        # log_dict[f"eval-avg-rewards/{key}"] = sum(rewards) / len(rewards)

        # Custom metrics from samples
        if (samples := data[key].get("samples")) is not None:
            # Original reward category metrics
            log_dict |= dict_add_prefix(_compute_reward_cat_metrics(args, samples), f"eval/{key}-")

            # 🆕 Separate debug/draft/improve/crossover mode statistics by prompt_type
            debug_data = _build_mode_bucket()
            draft_data = _build_mode_bucket()
            improve_data = _build_mode_bucket()
            crossover_data = _build_mode_bucket()

            # 🆕 Custom metrics - Status code distribution (split by prompt_type)
            status_codes = []
            scores = []
            scores_every_tasks = {}
            overall_reward_views = {key: [] for key in REWARD_VIEW_KEYS}
            for sample in samples:
                if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
                    prompt_type = sample.metadata.get("prompt_type", "draft")

                    # Select target data dict based on prompt_type
                    if prompt_type == "debug":
                        target_data = debug_data
                    elif prompt_type == "improve":
                        target_data = improve_data
                    elif prompt_type == "crossover":
                        target_data = crossover_data
                    else:  # draft or default
                        target_data = draft_data

                    # Collect hack statistics
                    target_data["total_count"] += 1
                    target_data["samples"].append(sample)
                    hack_value = sample.metadata.get("hack", 0)
                    code_category = sample.metadata.get("code_category", "valid")
                    if hack_value == 1:
                        target_data["hack_count"] += 1
                    if code_category == "empty":
                        target_data["empty_count"] += 1

                    for reward_key in REWARD_VIEW_KEYS:
                        if reward_key == "metric_static_base_reward" and not _has_static_bounds(sample.metadata):
                            continue
                        reward_value = _to_finite_float(sample.metadata.get(reward_key))
                        if reward_value is not None:
                            target_data["reward_views"][reward_key].append(reward_value)
                            overall_reward_views[reward_key].append(reward_value)

                    # Collect status codes
                    if "status_codes" in sample.metadata:
                        target_data["status_codes"].extend(sample.metadata["status_codes"])
                        status_codes.extend(sample.metadata["status_codes"])
                    sample_status_codes = sample.metadata.get("status_codes", [])
                    latest_status_code = (
                        sample_status_codes[-1]
                        if isinstance(sample_status_codes, list) and sample_status_codes
                        else None
                    )
                    if latest_status_code == 200 or hack_value == 1:
                        target_data["success_count"] += 1

                    # Collect scores
                    if "score" in sample.metadata:
                        score_val = sample.metadata["score"]
                        if score_val is not None:
                            target_data["scores"].append(score_val)
                            scores.append(score_val)

                    # Collect scores per task
                    if "task_name" in sample.metadata and "score" in sample.metadata:
                        task_name = sample.metadata["task_name"]
                        score_val = sample.metadata["score"]
                        if task_name not in target_data["scores_every_tasks"]:
                            target_data["scores_every_tasks"][task_name] = []
                        target_data["scores_every_tasks"][task_name].append(score_val)
                        if task_name not in scores_every_tasks:
                            scores_every_tasks[task_name] = []
                        scores_every_tasks[task_name].append(score_val)

                    # Collect response lengths (using framework's method, same as training)
                    response_length = sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length
                    target_data["response_lengths"].append(response_length)

            print(f"scores_every_tasks: {scores_every_tasks}")
            # Compute overall statistics (all modes combined)
            all_response_lengths = (
                debug_data["response_lengths"]
                + draft_data["response_lengths"]
                + improve_data["response_lengths"]
                + crossover_data["response_lengths"]
            )
            total_hack_count = (
                debug_data["hack_count"]
                + draft_data["hack_count"]
                + improve_data["hack_count"]
                + crossover_data["hack_count"]
            )
            total_empty_count = (
                debug_data["empty_count"]
                + draft_data["empty_count"]
                + improve_data["empty_count"]
                + crossover_data["empty_count"]
            )
            total_success_count = (
                debug_data["success_count"]
                + draft_data["success_count"]
                + improve_data["success_count"]
                + crossover_data["success_count"]
            )
            total_sample_count = (
                debug_data["total_count"]
                + draft_data["total_count"]
                + improve_data["total_count"]
                + crossover_data["total_count"]
            )

            # Code category statistics
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
                    log_dict[f"eval/{key}-code_category/{code_category}/count"] = count
                    log_dict[f"eval/{key}-code_category/{code_category}/ratio"] = count / total_samples

            # 🆕 Log statistics for each prompt_type (debug/draft/improve/crossover)
            for mode_name, mode_data in [
                ("debug", debug_data),
                ("draft", draft_data),
                ("improve", improve_data),
                ("crossover", crossover_data),
            ]:
                _log_reward_view_means(log_dict, f"eval/{key}-{mode_name}", mode_data["reward_views"])
                _log_quality_proxy_metrics(log_dict, f"eval/{key}-{mode_name}/quality", mode_data["samples"])

                if mode_data["status_codes"]:
                    total_status = len(mode_data["status_codes"])
                    log_dict[f"eval/{key}-{mode_name}/http_200_rate"] = (
                        mode_data["status_codes"].count(200) / total_status
                    )
                    log_dict[f"eval/{key}-{mode_name}/connection_error_rate"] = (
                        mode_data["status_codes"].count(503) / total_status
                    )
                    log_dict[f"eval/{key}-{mode_name}/other_error_rate"] = (
                        sum(1 for c in mode_data["status_codes"] if c not in [200, 503]) / total_status
                    )  # A 504 response indicates a timeout.

                if mode_data["response_lengths"]:
                    log_dict[f"eval/{key}-{mode_name}/avg_response_length"] = sum(mode_data["response_lengths"]) / len(
                        mode_data["response_lengths"]
                    )
                    log_dict[f"eval/{key}-{mode_name}/max_response_length"] = max(mode_data["response_lengths"])

                # Log hack statistics per mode
                if mode_data["total_count"] > 0:
                    log_dict[f"eval/{key}-{mode_name}/hack_rate"] = mode_data["hack_count"] / mode_data["total_count"]
                    log_dict[f"eval/{key}-{mode_name}/hack_count"] = mode_data["hack_count"]
                    log_dict[f"eval/{key}-{mode_name}/empty_rate"] = mode_data["empty_count"] / mode_data["total_count"]
                    log_dict[f"eval/{key}-{mode_name}/empty_count"] = mode_data["empty_count"]
                    log_dict[f"eval/{key}-{mode_name}/success_rate"] = (
                        mode_data["success_count"] / mode_data["total_count"]
                    )
                    log_dict[f"eval/{key}-{mode_name}/success_count"] = mode_data["success_count"]
                    log_dict[f"eval/{key}-{mode_name}/total_count"] = mode_data["total_count"]

                # Medal statistics per mode - count medals for EACH sample, not just best score
                # Distinguish scored samples without medals from samples without scores.
                mode_medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
                mode_scored_count = 0  # Total number of samples with valid scores.
                mode_task_medal_counts = {}

                for task_name, task_scores in mode_data["scores_every_tasks"].items():
                    try:
                        leaderboard = load_public_leaderboard(task_name)
                        if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                            # Count medal for EACH score (each sample), not just the best
                            for score in task_scores:
                                if score is not None:
                                    medal_per_score = get_medal_for_score(score, leaderboard)
                                    mode_medal_cnt[medal_per_score] += 1
                                    mode_scored_count += 1
                                    if medal_per_score != "None":
                                        mode_task_medal_counts[task_name] = (
                                            mode_task_medal_counts.get(task_name, 0) + 1
                                        )
                                    print(f"MEDAL: {medal_per_score} for score: {score} in task: {task_name}")
                        else:
                            if task_scores is None:
                                print(f"Warning: Task scores are None for task '{task_name}'")
                            if leaderboard is None:
                                print(f"Warning: Leaderboard is None for task '{task_name}'")
                            elif "score" not in leaderboard.columns:
                                print(f"Warning: Score column not found in leaderboard for task '{task_name}'")
                            valid_score_count = len([s for s in (task_scores or []) if s is not None])
                            mode_medal_cnt["None"] += valid_score_count
                            mode_scored_count += valid_score_count
                            print(f"MEDAL: Cannot get medal for task '{task_name}'")
                    except Exception as e:
                        # If leaderboard not found, count all scores as None
                        valid_score_count = len([s for s in (task_scores or []) if s is not None])
                        mode_medal_cnt["None"] += valid_score_count
                        mode_scored_count += valid_score_count
                        print(f"Warning: Exception when getting medal '{task_name}': {e}")

                # Calculate metrics
                mode_no_score_count = mode_data["total_count"] - mode_scored_count  # Samples without scores.

                # Always record scored and unscored counts, regardless of medal availability.
                log_dict[f"eval/{key}-{mode_name}/scored_count"] = mode_scored_count  # Number of scored samples.

                log_dict[f"eval/{key}-{mode_name}/medal_gold_count"] = mode_medal_cnt["Gold"]
                log_dict[f"eval/{key}-{mode_name}/medal_silver_count"] = mode_medal_cnt["Silver"]
                log_dict[f"eval/{key}-{mode_name}/medal_bronze_count"] = mode_medal_cnt["Bronze"]
                log_dict[f"eval/{key}-{mode_name}/medal_count"] = (
                    mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]
                )  # Total awarded medals.
                log_dict[f"eval/{key}-{mode_name}/medal_none_count"] = mode_medal_cnt["None"]  # Scored but no medal.
                # Medal rate across all samples, including samples without scores.
                if mode_data["total_count"] > 0:
                    log_dict[f"eval/{key}-{mode_name}/medal_rate_total"] = (
                        mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]
                    ) / mode_data["total_count"]
                    log_dict[f"eval/{key}-{mode_name}/scored_rate_total"] = (
                        mode_scored_count / mode_data["total_count"]
                    )

                # Log per-task medal counts for this mode
                for task_name, task_medal_count in mode_task_medal_counts.items():
                    log_dict[f"eval/{key}-{mode_name}-{task_name}/task_medal_count"] = task_medal_count

            # Overall medal statistics - count medals for EACH sample, not just best score
            # Distinguish scored samples without medals from samples without scores.
            medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
            overall_scored_count = 0  # Total number of samples with valid scores.
            overall_task_medal_counts = {}

            for task_name, task_scores in scores_every_tasks.items():
                try:
                    leaderboard = load_public_leaderboard(task_name)
                    if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                        # Count medal for EACH score (each sample), not just the best
                        for score in task_scores:
                            if score is not None:
                                medal_per_score = get_medal_for_score(score, leaderboard)
                                medal_cnt[medal_per_score] += 1
                                overall_scored_count += 1
                                if medal_per_score != "None":
                                    overall_task_medal_counts[task_name] = (
                                        overall_task_medal_counts.get(task_name, 0) + 1
                                    )
                    else:
                        if task_scores is None:
                            print(f"  Warning: Task scores are None for task '{task_name}'")
                        if leaderboard is None:
                            print(f"  Warning: Leaderboard is None for task '{task_name}'")
                        elif "score" not in leaderboard.columns:
                            print(f"  Warning: Score column not found in leaderboard for task '{task_name}'")
                        valid_score_count = len([s for s in (task_scores or []) if s is not None])
                        medal_cnt["None"] += valid_score_count
                        overall_scored_count += valid_score_count
                        print(f"  Warning: Cannot load leaderboard for task '{task_name}'")
                except Exception as e:
                    # If leaderboard not found, count all scores as None
                    valid_score_count = len([s for s in (task_scores or []) if s is not None])
                    medal_cnt["None"] += valid_score_count
                    overall_scored_count += valid_score_count
                    print(f"  Warning: Cannot load leaderboard for task '{task_name}': {e}")

            # Always record the base statistics.
            log_dict[f"eval/{key}-scored_count"] = overall_scored_count  # Number of scored samples.
            # Record medal statistics.
            log_dict[f"eval/{key}-medal_gold_count"] = medal_cnt["Gold"]
            log_dict[f"eval/{key}-medal_silver_count"] = medal_cnt["Silver"]
            log_dict[f"eval/{key}-medal_bronze_count"] = medal_cnt["Bronze"]
            log_dict[f"eval/{key}-medal_count"] = (
                medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]
            )  # Total awarded medals.
            log_dict[f"eval/{key}-medal_none_count"] = medal_cnt["None"]  # Scored but no medal.

            # Medal rate across all samples, including samples without scores.
            if total_sample_count > 0:
                medal_rate_total = (
                    medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]
                ) / total_sample_count
                log_dict[f"eval/{key}-medal_rate_total"] = medal_rate_total
                log_dict[f"eval/{key}/medal_rate"] = medal_rate_total
                log_dict[f"eval/{key}-scored_rate_total"] = overall_scored_count / total_sample_count

            # Log per-task total medal counts (all modes combined)
            for task_name, task_medal_count in overall_task_medal_counts.items():
                log_dict[f"eval/{key}-{task_name}/task_medal_count"] = task_medal_count

            _log_reward_view_means(log_dict, f"eval/{key}", overall_reward_views)
            _log_quality_proxy_metrics(log_dict, f"eval/{key}/quality", samples)

            # Log status code metrics
            if status_codes:
                total_status = len(status_codes)
                log_dict[f"eval/{key}-http_200_rate"] = status_codes.count(200) / total_status
                log_dict[f"eval/{key}-connection_error_rate"] = status_codes.count(503) / total_status
                log_dict[f"eval/{key}-other_error_rate"] = (
                    sum(1 for c in status_codes if c not in [200, 503]) / total_status
                )

            # Log score statistics
            # if scores:
            # log_dict[f"eval/{key}-avg_score"] = sum(scores) / len(scores)
            # log_dict[f"eval/{key}-max_score"] = max(scores)
            # log_dict[f"eval/{key}-min_score"] = min(scores)
            # log_dict[f"eval/{key}-score_std"] = (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5

            # Log overall response length statistics
            if all_response_lengths:
                log_dict[f"eval/{key}-avg_response_length"] = sum(all_response_lengths) / len(all_response_lengths)
                log_dict[f"eval/{key}-max_response_length"] = max(all_response_lengths)

            # Log overall hack statistics
            if total_sample_count > 0:
                log_dict[f"eval/{key}-hack_rate"] = total_hack_count / total_sample_count
                log_dict[f"eval/{key}-hack_count"] = total_hack_count
                log_dict[f"eval/{key}-empty_rate"] = total_empty_count / total_sample_count
                log_dict[f"eval/{key}-empty_count"] = total_empty_count
                log_dict[f"eval/{key}-success_rate"] = total_success_count / total_sample_count
                log_dict[f"eval/{key}-success_count"] = total_success_count
                log_dict[f"eval/{key}-total_sample_count"] = total_sample_count

        # Original truncated ratio
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

        # Original pass rate metrics
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    print(f"[CUSTOM EVAL LOGGER] eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    # Return True so --custom-eval-rollout-log-function-path skips stock logging.
    return True


# Apply the monkey patch
def apply_patch():
    """Apply monkey patch to _log_eval_rollout_data"""
    try:
        import slime.ray.rollout

        slime.ray.rollout._log_eval_rollout_data = custom_log_eval_rollout_data
        # Stock RolloutManager.eval already forwards metrics to _log_eval_rollout_data;
        # keep a thin wrapper only for compatibility with older call sites.
        original_eval = slime.ray.rollout.RolloutManager.eval

        def eval_with_log_dict(self, rollout_id):
            if self.args.debug_train_only:
                return None
            self.health_monitoring_resume()

            result = slime.ray.rollout.call_rollout_fn(
                self.eval_generate_rollout,
                self.args,
                rollout_id,
                self.data_source,
                evaluation=True,
            )
            data = result.data
            self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
            return slime.ray.rollout._log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

        if getattr(original_eval, "__name__", "") != "eval_with_log_dict":
            slime.ray.rollout.RolloutManager.eval = eval_with_log_dict
        print("✅ [PATCH_EVAL_LOGGER] Successfully patched _log_eval_rollout_data")
        return True
    except Exception as e:
        print(f"⚠️ [PATCH_EVAL_LOGGER] Failed to patch: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_medal_statistics_integration():
    """Integration test for medal statistics with simulated samples"""
    import tempfile
    import shutil
    from collections import namedtuple

    print("\n" + "=" * 80)
    print("TESTING MEDAL STATISTICS INTEGRATION")
    print("=" * 80)

    # Create a temporary leaderboard for testing
    temp_dir = tempfile.mkdtemp()
    test_task_name = "tabular-playground-series-dec-2021"
    test_leaderboard_dir = os.path.join(temp_dir, test_task_name)
    os.makedirs(test_leaderboard_dir, exist_ok=True)

    # Create mock samples with different prompt_types and scores
    MockSample = namedtuple("MockSample", ["metadata"])

    test_samples = [
        # Debug samples (3 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.99}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.85}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name, "score": 0.70}),
        MockSample(metadata={"prompt_type": "debug", "task_name": test_task_name}),  # No score
        # Draft samples (4 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.955}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.82}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.65}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name, "score": 0.50}),
        MockSample(metadata={"prompt_type": "draft", "task_name": test_task_name}),  # No score
        # Improve samples (5 with scores + 1 without score)
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.93}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.88}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.75}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.62}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name, "score": 0.45}),
        MockSample(metadata={"prompt_type": "improve", "task_name": test_task_name}),  # No score
    ]

    # Process samples and collect statistics (simulating the logic in custom_log_eval_rollout_data)
    debug_data = {"scores_every_tasks": {}, "total_count": 0}
    draft_data = {"scores_every_tasks": {}, "total_count": 0}
    improve_data = {"scores_every_tasks": {}, "total_count": 0}
    crossover_data = {"scores_every_tasks": {}, "total_count": 0}
    scores_every_tasks = {}

    for sample in test_samples:
        prompt_type = sample.metadata.get("prompt_type", "draft")
        task_name = sample.metadata["task_name"]
        score = sample.metadata.get("score")  # Use .get() to handle missing scores

        if prompt_type == "debug":
            target_data = debug_data
        elif prompt_type == "improve":
            target_data = improve_data
        elif prompt_type == "crossover":
            target_data = crossover_data
        else:
            target_data = draft_data

        # Count all samples
        target_data["total_count"] += 1

        # Only add scores if they exist and are valid
        if score is not None and not (isinstance(score, float) and np.isnan(score)):
            if task_name not in target_data["scores_every_tasks"]:
                target_data["scores_every_tasks"][task_name] = []
            target_data["scores_every_tasks"][task_name].append(score)

            if task_name not in scores_every_tasks:
                scores_every_tasks[task_name] = []
            scores_every_tasks[task_name].append(score)

    # Calculate medals for each mode
    all_passed = True

    for mode_name, mode_data in [
        ("debug", debug_data),
        ("draft", draft_data),
        ("improve", improve_data),
        ("crossover", crossover_data),
    ]:
        mode_medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
        mode_scored_count = 0

        for task_name, task_scores in mode_data["scores_every_tasks"].items():
            try:
                leaderboard = load_public_leaderboard(task_name)
                if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                    for score in task_scores:
                        medal_per_score = get_medal_for_score(score, leaderboard)
                        mode_medal_cnt[medal_per_score] += 1
                        mode_scored_count += 1
            except Exception as e:
                mode_medal_cnt["None"] += len(task_scores)
                mode_scored_count += len(task_scores)

        mode_no_score_count = mode_data["total_count"] - mode_scored_count
        total = sum(mode_medal_cnt.values())
        medal_rate = (
            (mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]) / total
            if total > 0
            else 0.0
        )
        medal_rate_total = (
            (mode_medal_cnt["Gold"] + mode_medal_cnt["Silver"] + mode_medal_cnt["Bronze"]) / mode_data["total_count"]
            if mode_data["total_count"] > 0
            else 0.0
        )

        print(f"\n{mode_name.upper()} Results:")
        print(f"  Gold: {mode_medal_cnt['Gold']} ")
        print(f"  Silver: {mode_medal_cnt['Silver']} ")
        print(f"  Bronze: {mode_medal_cnt['Bronze']} ")
        print(f"  None (scored, no medal): {mode_medal_cnt['None']} ")
        print(f"  No Score: {mode_no_score_count} ")
        print(f"  Scored Count: {mode_scored_count} ")
        print(f"  Total Count: {mode_data['total_count']} ")
        print(f"  Medal Rate (scored samples): {medal_rate:.4f} ")
        print(f"  Medal Rate Total (all samples): {medal_rate_total:.4f} ")

    # Calculate overall medals
    medal_cnt = {"Gold": 0, "Silver": 0, "Bronze": 0, "None": 0}
    overall_scored_count = 0

    for task_name, task_scores in scores_every_tasks.items():
        try:
            leaderboard = load_public_leaderboard(task_name)
            if task_scores and leaderboard is not None and "score" in leaderboard.columns:
                for score in task_scores:
                    if score is not None:
                        medal_per_score = get_medal_for_score(score, leaderboard)
                        medal_cnt[medal_per_score] += 1
                        overall_scored_count += 1
        except Exception as e:
            medal_cnt["None"] += len([s for s in task_scores if s is not None])
            overall_scored_count += len([s for s in task_scores if s is not None])

    total_sample_count = (
        debug_data["total_count"] + draft_data["total_count"] + improve_data["total_count"] + crossover_data["total_count"]
    )
    overall_no_score_count = total_sample_count - overall_scored_count
    total = sum(medal_cnt.values())
    overall_medal_rate = (medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]) / total if total > 0 else 0.0
    overall_medal_rate_total = (
        (medal_cnt["Gold"] + medal_cnt["Silver"] + medal_cnt["Bronze"]) / total_sample_count
        if total_sample_count > 0
        else 0.0
    )

    print(f"\nOVERALL Results:")
    print(f"  Gold: {medal_cnt['Gold']}")
    print(f"  Silver: {medal_cnt['Silver']} ")
    print(f"  Bronze: {medal_cnt['Bronze']} ")
    print(f"  None (scored, no medal): {medal_cnt['None']} ")
    print(f"  No Score: {overall_no_score_count} ")
    print(f"  Scored Count: {overall_scored_count} ")
    print(f"  Total Count: {total_sample_count} ")
    print(f"  Medal Rate (scored samples): {overall_medal_rate:.4f} ")
    print(f"  Medal Rate Total (all samples): {overall_medal_rate_total:.4f} ")

    # Clean up
    shutil.rmtree(temp_dir)

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ INTEGRATION TEST PASSED")
    else:
        print("❌ INTEGRATION TEST FAILED")
    print("=" * 80 + "\n")

    return all_passed


# Auto-apply patch when this module is imported
if __name__ != "__main__":
    apply_patch()
else:
    # Run tests when executed directly
    print("Running medal calculation tests...")
    test2_passed = test_medal_statistics_integration()

    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"Integration statistics test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 80)
