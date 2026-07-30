"""
Pass@k Evaluation Script with Decoupled Generator/Evaluator Services.

This script implements a pass@k evaluation system where:
1. k code samples are generated per task
2. All samples are evaluated on the test set (no validation set)
3. Rejection policy decides which evaluated samples count toward the loop target
4. Generation stops when the target is met or the max generation budget is reached

Supports three modes:
- Full mode: Generate and evaluate (default)
- Generation only: Set enable_sandbox_eval=False
- Evaluation only: Set evaluate_only=True (evaluates existing gen_results.jsonl)

Checkpoint resume:
- Automatically resumes from gen_results.jsonl and eval_results.jsonl
- Skips already-generated samples
- Evaluates all unevaluated samples (including those from previous runs)

Usage:
    # Full generation + evaluation
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47

    # Generation only (no evaluation)
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47 enable_sandbox_eval=False

    # Evaluation only (evaluate existing generations)
    python -m tts_search.evaluate_pass_k --config-name experiment/parallel_glm47 evaluate_only=True resume_from=<OUTPUT_DIR>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

from tts_search.runtime_storage import configure_runtime_storage

configure_runtime_storage(default_run_id="evaluate_pass_k")

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from litellm.router import Router
from omegaconf import DictConfig, OmegaConf

# from tts_search import eval_utils
from tts_search.data_produce.gap_filter import enrich_tasks_with_metric_protocol
from tts_search.prompt_builder import build_pass_k_draft_messages
from tts_search.services import Scheduler, SchedulerConfig
from tts_search.services.scheduler import (
    build_task_states_from_progress,
    load_checkpoint,
    load_progress_snapshot,
    progress_snapshot_has_medal_counts,
    rebuild_task_states_from_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
logger = logging.getLogger(__name__)

# Load .env before Hydra parses config (so oc.env can resolve)
load_dotenv(REPO_ROOT / ".env")


def _is_minimax_config(cfg: DictConfig) -> bool:
    """Check if config uses MiniMax provider."""
    try:
        model = str(cfg.litellm.model_list[0].litellm_params.get("model", ""))
        return "minimax" in model.lower()
    except (AttributeError, IndexError, KeyError):
        return False


def _parse_minimax_api_keys(cfg: DictConfig | None = None) -> list[str]:
    """Parse MiniMax API keys only if config uses MiniMax."""
    if cfg is not None and not _is_minimax_config(cfg):
        return []

    keys_env = os.getenv("MINIMAX_API_KEYS", "").strip()
    if keys_env:
        try:
            if keys_env.startswith("["):
                parsed = json.loads(keys_env)
                if not isinstance(parsed, list):
                    raise ValueError("MINIMAX_API_KEYS must be a JSON list")
                keys = [str(k).strip() for k in parsed if str(k).strip()]
            else:
                keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        except Exception as exc:
            raise ValueError(
                "Failed to parse MINIMAX_API_KEYS. Use JSON list or comma-separated."
            ) from exc
    else:
        single_key = os.getenv("MINIMAX_API_KEY", "").strip()
        keys = [single_key] if single_key else []

    if not keys:
        raise ValueError("No API keys found in MINIMAX_API_KEYS or MINIMAX_API_KEY")

    return keys


def _split_tasks_evenly(
    tasks: list[dict[str, Any]], n_shards: int
) -> list[list[dict[str, Any]]]:
    """Split task records across shards.

    Args:
        tasks: Task dictionaries loaded from the eval data source.
        n_shards: Number of output shards to create.

    Returns:
        A list of task lists, assigned round-robin for balanced shard sizes.
    """
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    for idx, task in enumerate(tasks):
        shards[idx % n_shards].append(task)
    return shards


def _build_router_with_key(cfg: DictConfig, api_key: str) -> Router:
    """Build a LiteLLM router for one API key.

    Args:
        cfg: Hydra config containing the LiteLLM model list.
        api_key: Provider API key for this shard.

    Returns:
        A Router instance with the first model's API key overridden.
    """
    litellm_cfg = OmegaConf.create(OmegaConf.to_container(cfg.litellm, resolve=False))
    litellm_cfg.model_list[0].litellm_params.api_key = api_key
    return instantiate(litellm_cfg)


def _merge_dirs(src: Path, dst: Path) -> None:
    """Merge src directory into dst without overwriting existing files."""
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _merge_dirs(item, target)
        else:
            if target.exists():
                logger.warning("Skip existing file during merge: %s", target)
                continue
            shutil.move(str(item), str(target))


def _migrate_task_dirs(output_dir: Path, task_subdir: str = "tasks") -> None:
    """Move task folders under output_dir into task_subdir."""
    tasks_dir = output_dir / task_subdir
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == task_subdir:
            continue

        target_dir = tasks_dir / entry.name
        target_dir.mkdir(parents=True, exist_ok=True)
        _merge_dirs(entry, target_dir)
        try:
            entry.rmdir()
        except OSError:
            logger.warning("Directory not empty after merge: %s", entry)


async def run_pass_k_evaluation(
    cfg: DictConfig,
    router: Router,
    tasks: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run pass@k evaluation for all tasks.

    Args:
        cfg: Hydra configuration
        router: LiteLLM router
        tasks: List of task configurations
        output_dir: Output directory

    Returns:
        Global statistics
    """
    _migrate_task_dirs(output_dir, task_subdir="tasks")

    # Resume from previous run if specified
    resume_from = cfg.get("resume_from", None)
    print(f"resume_from: {resume_from}")
    completed_tasks: set[str] = set()
    resumed_results: dict[str, dict[str, Any]] = {}

    if resume_from is not None:
        resume_dir = Path(resume_from)
        logger.info(f"Attempting to resume from: {resume_dir}")

        if bool(cfg.get("loop_enabled", False)):
            resume_task_list = []
            for task in tasks:
                metadata = task[cfg.data.metadata_key]
                resume_task_list.append(
                    {
                        "task_id": metadata["uuid"],
                        "task_name": metadata["task_name"],
                        "metadata": metadata,
                    }
                )

            loop_increment = int(
                cfg.get(
                    "loop_target_increment",
                    int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
                )
            )
            loop_success_target = int(cfg.get("loop_success_target", 1))
            loop_success_metric = str(cfg.get("loop_success_metric", "success"))
            loop_medal_target = (
                int(cfg.get("loop_medal_target"))
                if cfg.get("loop_medal_target") is not None
                else loop_success_target
            )
            loop_max_target = (
                int(cfg.get("loop_max_target"))
                if cfg.get("loop_max_target") is not None
                else None
            )

            progress_snapshot = load_progress_snapshot(resume_dir)
            if bool(cfg.get("resume_rebuild_task_states", False)):
                logger.info(
                    "Ignoring progress.json and rebuilding task states from gen/eval "
                    "results because resume_rebuild_task_states=true"
                )
                progress_snapshot = None
            task_states = build_task_states_from_progress(
                progress_snapshot=progress_snapshot,
                tasks=resume_task_list,
                loop_increment=loop_increment,
                loop_max_target=loop_max_target,
                rejection_policy_name=str(
                    cfg.get("rejection_policy", loop_success_metric)
                ).lower(),
            )
            if (
                loop_success_metric == "medal"
                and task_states
                and not progress_snapshot_has_medal_counts(progress_snapshot)
            ):
                task_states = {}

            # Fallback: rebuild task_states from gen/eval results if missing/empty
            if not task_states:
                _, _, gen_results_data, eval_results_data = load_checkpoint(resume_dir)
                task_states = rebuild_task_states_from_results(
                    tasks=resume_task_list,
                    gen_results_data=gen_results_data,
                    eval_results_data=eval_results_data,
                    loop_increment=loop_increment,
                    loop_success_target=loop_success_target,
                    loop_success_metric=loop_success_metric,
                    loop_medal_target=loop_medal_target,
                    loop_max_target=loop_max_target,
                    rejection_target=cfg.get("rejection_target"),
                )
                logger.info(
                    f"Rebuilt task_states from gen/eval results: {len(task_states)} tasks"
                )

            for state in task_states.values():
                if state.done and state.task_name:
                    completed_tasks.add(state.task_name)

            if completed_tasks:
                logger.info(
                    f"Loaded {len(completed_tasks)} completed tasks from resume data"
                )

        logger.info(f"Resumed {len(completed_tasks)} completed tasks from {resume_dir}")

    # Create scheduler configuration with decoupled pipeline for max efficiency
    logger.info(f"sandbox_base_url: {cfg.sandbox.base_url}")
    logger.info(
        f"sandbox_cpu_base_url: {cfg.sandbox.get('cpu_base_url', cfg.sandbox.base_url)}"
    )
    logger.info(
        f"sandbox_resource_type_override: {cfg.sandbox.get('resource_type_override')}"
    )
    scheduler_config = SchedulerConfig(
        max_steps=int(cfg.max_steps),
        n_samples_per_task=int(cfg.get("n_samples_per_task", 1)),
        llm_concurrency=int(cfg.llm_concurrency),
        llm_max_retries_per_step=int(cfg.get("llm_max_retries_per_step", 8)),
        sandbox_base_url=str(cfg.sandbox.base_url),
        sandbox_cpu_base_url=str(cfg.sandbox.get("cpu_base_url", cfg.sandbox.base_url)),
        sandbox_resource_type_override=cfg.sandbox.get("resource_type_override"),
        sandbox_concurrency=int(cfg.sandbox.concurrency),
        job_timeout=int(cfg.sandbox.job_timeout),
        wait_timeout=int(cfg.sandbox.wait_timeout),
        poll_interval=int(cfg.sandbox.poll_interval),
        use_score2reward=bool(cfg.sandbox.get("use_score2reward", True)),
        batch_size=int(cfg.get("batch_size", 16)),
        task_concurrency=int(cfg.get("task_concurrency", 1)),
        enable_sandbox_eval=bool(cfg.get("enable_sandbox_eval", True)),
        skip_eval_for_done_tasks=bool(
            cfg.get(
                "skip_eval_for_done_tasks",
                SchedulerConfig.skip_eval_for_done_tasks,
            )
        ),
        # Decoupled pipeline settings
        eval_queue_maxsize=int(cfg.get("eval_queue_maxsize", 0)),
        eval_batch_timeout=float(cfg.get("eval_batch_timeout", 1.0)),
        task_subdir="tasks",
        # Looping evaluation settings
        loop_enabled=bool(cfg.get("loop_enabled", False)),
        loop_success_target=int(cfg.get("loop_success_target", 1)),
        loop_success_metric=str(cfg.get("loop_success_metric", "success")),
        loop_medal_target=(
            int(cfg.get("loop_medal_target"))
            if cfg.get("loop_medal_target") is not None
            else None
        ),
        loop_target_increment=int(
            cfg.get(
                "loop_target_increment",
                int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
            )
        ),
        loop_max_target=(
            int(cfg.get("loop_max_target"))
            if cfg.get("loop_max_target") is not None
            else int(
                cfg.get(
                    "loop_target_increment",
                    int(cfg.max_steps) * int(cfg.get("n_samples_per_task", 1)),
                )
            )
        ),
        resume_rebuild_task_states=bool(
            cfg.get(
                "resume_rebuild_task_states",
                SchedulerConfig.resume_rebuild_task_states,
            )
        ),
        rejection_policy=(
            str(cfg.get("rejection_policy"))
            if cfg.get("rejection_policy") is not None
            else None
        ),
        rejection_target=(
            int(cfg.get("rejection_target"))
            if cfg.get("rejection_target") is not None
            else None
        ),
        rejection_score_threshold=(
            float(cfg.get("rejection_score_threshold"))
            if cfg.get("rejection_score_threshold") is not None
            else None
        ),
        rejection_reward_threshold=(
            float(cfg.get("rejection_reward_threshold"))
            if cfg.get("rejection_reward_threshold") is not None
            else None
        ),
        rejection_reference_scores_path=(
            str(cfg.get("rejection_reference_scores_path"))
            if cfg.get("rejection_reference_scores_path") is not None
            else None
        ),
        rejection_accepted_medals=(
            tuple(cfg.get("rejection_accepted_medals"))
            if cfg.get("rejection_accepted_medals") is not None
            else None
        ),
        rejection_apply_baseline_filters=bool(
            cfg.get(
                "rejection_apply_baseline_filters",
                SchedulerConfig.rejection_apply_baseline_filters,
            )
        ),
        rejection_baseline_token_limit=int(
            cfg.get(
                "rejection_baseline_token_limit",
                SchedulerConfig.rejection_baseline_token_limit,
            )
        ),
        rejection_baseline_tokenizer_model=(
            str(cfg.get("rejection_baseline_tokenizer_model"))
            if cfg.get("rejection_baseline_tokenizer_model") is not None
            else SchedulerConfig.rejection_baseline_tokenizer_model
        ),
        rejection_baseline_relative_gap_limit=float(
            cfg.get(
                "rejection_baseline_relative_gap_limit",
                SchedulerConfig.rejection_baseline_relative_gap_limit,
            )
        ),
        rejection_mixed_leaderboard_target=int(
            cfg.get(
                "rejection_mixed_leaderboard_target",
                SchedulerConfig.rejection_mixed_leaderboard_target,
            )
        ),
        rejection_mixed_no_leaderboard_target=int(
            cfg.get(
                "rejection_mixed_no_leaderboard_target",
                SchedulerConfig.rejection_mixed_no_leaderboard_target,
            )
        ),
        tree_search_enabled=bool(cfg.get("tree_search_enabled", False)),
        tree_search_algorithm=str(cfg.get("tree_search_algorithm", "greedy")),
        tree_search_max_pending_per_task=int(
            cfg.get("tree_search_max_pending_per_task", 1)
        ),
        tree_search_db_path=(
            str(cfg.get("tree_search_db_path"))
            if cfg.get("tree_search_db_path") is not None
            else None
        ),
        tree_search_db_max_per_task=(
            int(cfg.get("tree_search_db_max_per_task"))
            if cfg.get("tree_search_db_max_per_task") is not None
            else None
        ),
        greedy_num_drafts=int(cfg.get("greedy_num_drafts", 5)),
        greedy_debug_prob=float(cfg.get("greedy_debug_prob", 0.5)),
        greedy_draft_prob=float(cfg.get("greedy_draft_prob", 0.0)),
        gen_retry_max=(
            int(cfg.get("gen_retry_max"))
            if cfg.get("gen_retry_max") is not None
            else SchedulerConfig.gen_retry_max
        ),
    )

    # Create scheduler. Tree-search mode initializes its ProgramDatabase lazily.
    scheduler = Scheduler(
        router=router,
        config=scheduler_config,
        database=None,
        output_dir=output_dir,  # Initialize gen/eval writers immediately
        litellm_model_list=cfg.litellm.model_list,
    )

    # Prepare task inputs (with optional limit)
    max_tasks = cfg.get("max_tasks", None)
    if max_tasks is not None:
        max_tasks = int(max_tasks)
        tasks = tasks[:max_tasks]
        logger.info(f"Limited to first {max_tasks} tasks")

    use_draft_prompt_builder = bool(cfg.get("use_draft_prompt_builder", False))
    draft_time_limit = bool(cfg.get("draft_time_limit", True))
    draft_time_limit_value = cfg.get(
        "draft_time_limit_value",
        cfg.sandbox.get("job_timeout", None) if "sandbox" in cfg else None,
    )
    logger.info(
        "use_draft_prompt_builder=%s draft_time_limit=%s draft_time_limit_value=%s",
        use_draft_prompt_builder,
        draft_time_limit,
        draft_time_limit_value,
    )

    task_inputs: list[dict[str, Any]] = []
    for task in tasks:
        metadata = task[cfg.data.metadata_key]
        raw_messages: list[dict[str, str]] = task[cfg.data.input_key].tolist()
        messages = (
            build_pass_k_draft_messages(
                metadata,
                raw_messages,
                time_limit=draft_time_limit,
                time_limit_value=draft_time_limit_value,
            )
            if use_draft_prompt_builder
            else raw_messages
        )
        task_name = metadata["task_name"]
        task_id = metadata["uuid"]
        data_dir = metadata["data_dir"]

        task_inputs.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "messages": messages,
                "metadata": metadata,
                "data_dir": data_dir,
            }
        )

    # Get model name
    model_name = str(cfg.litellm.model_list[0].model_name)

    logger.info(f"Starting pass@{scheduler_config.max_steps} evaluation")
    logger.info(
        f"Total tasks: {len(tasks)}, Completed: {len(completed_tasks)}, Remaining: {len(task_inputs)}"
    )
    logger.info(f"Using model: {model_name}")

    # Progress file for external monitoring
    progress_file = output_dir / "progress.json"

    # Start progress monitor task
    async def monitor_progress():
        """Monitor and log progress every 30 seconds with separate gen/eval tracking."""
        while True:
            await asyncio.sleep(30)
            progress = scheduler.get_progress()
            if progress["total_tasks"] == 0:
                continue

            task_pct = progress["completed_tasks"] / progress["total_tasks"] * 100

            # Get generation progress
            gen_progress = progress.get("generation", {})
            gen_total = gen_progress.get("total", 0)
            gen_completed = gen_progress.get("completed", 0)
            gen_in_progress = gen_progress.get("in_progress", 0)
            gen_rate = gen_progress.get("rate_per_second", 0)
            gen_pct = gen_completed / gen_total * 100 if gen_total > 0 else 0

            # Get evaluation progress
            eval_progress = progress.get("evaluation", {})
            eval_total = eval_progress.get("total", 0)
            eval_completed = eval_progress.get("completed", 0)
            eval_in_progress = eval_progress.get("in_progress", 0)
            eval_pending = eval_progress.get("pending_in_queue", 0)
            eval_rate = eval_progress.get("rate_per_second", 0)
            eval_pct = eval_completed / eval_total * 100 if eval_total > 0 else 0

            logger.info(
                f"📊 Progress Update: "
                f"Tasks {progress['completed_tasks']}/{progress['total_tasks']} ({task_pct:.1f}%)"
            )
            logger.info(
                f"   Generation: {gen_completed}/{gen_total} ({gen_pct:.1f}%) | "
                f"In-progress: {gen_in_progress} | Rate: {gen_rate:.2f}/s"
            )
            if scheduler_config.enable_sandbox_eval:
                active_slots = eval_progress.get("active_sandbox_slots", 0)
                max_slots = eval_progress.get("sandbox_concurrency", 1)
                utilization = eval_progress.get("sandbox_utilization", 0)
                logger.info(
                    f"   Evaluation: {eval_completed}/{eval_total} ({eval_pct:.1f}%) | "
                    f"In-progress: {eval_in_progress} | Queue: {eval_pending} | Rate: {eval_rate:.2f}/s"
                )
                logger.info(
                    f"   Sandbox: {active_slots}/{max_slots} slots ({utilization:.0%} utilization)"
                )
            logger.info(
                f"   Streamed: {progress.get('streaming_samples_written', 0)} samples"
            )

            # Save progress to file for external monitoring
            progress_file.write_text(
                json.dumps(progress, indent=2, default=str), encoding="utf-8"
            )

            # Stop monitoring when all tasks are completed
            if progress["completed_tasks"] >= progress["total_tasks"]:
                break

    # Check for evaluation-only mode
    evaluate_only = bool(cfg.get("evaluate_only", False))

    if evaluate_only:
        # Evaluation-only mode: evaluate existing gen_results.jsonl
        logger.info("=" * 60)
        logger.info("EVALUATION ONLY MODE")
        logger.info("=" * 60)

        # Build task list for metadata lookup
        all_task_inputs = []
        for task in tasks:
            metadata = task[cfg.data.metadata_key]
            task_id = metadata["uuid"]
            all_task_inputs.append(
                {
                    "task_id": task_id,
                    "metadata": metadata,
                }
            )

        # Start progress monitor for eval-only mode
        monitor_task = asyncio.create_task(monitor_progress())

        try:
            eval_results = await scheduler.run_evaluation_only(
                output_dir=output_dir,
                tasks=all_task_inputs,
            )

            logger.info(f"Evaluated {len(eval_results)} samples")
        finally:
            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        results = []  # No TaskResult objects in eval-only mode
    else:
        # Normal mode: run generation and evaluation
        results = []
        if task_inputs:  # Only run if there are tasks to process
            monitor_task = asyncio.create_task(monitor_progress())

            try:
                results = await scheduler.run_tasks(
                    tasks=task_inputs,
                    model_name=model_name,
                    output_dir=output_dir,
                    resume=True,  # Enable checkpoint resume
                )
            finally:
                # Cancel monitor task
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
        else:
            logger.info("All tasks already completed, skipping evaluation")

    # Aggregate results (merge new and resumed)
    task_results_map: dict[str, dict[str, Any]] = {}

    # Add new results
    for result in results:
        task_results_map[result.task_name] = {
            "task_name": result.task_name,
            "task_id": result.task_id,
            "best_score": result.best_score,
            "best_reward": result.best_reward,
            "num_samples": result.num_samples,
            "total_cost": result.total_cost,
            "total_tokens": result.total_tokens,
            "status_counts": result.status_counts,
            "success_count": result.status_counts.get("success", 0),
            "medal_count": result.medal_count,
            "medal_counts": result.medal_counts,
            "test_time": result.test_time,
        }

    # Merge resumed results
    for task_name, resumed_stat in resumed_results.items():
        if task_name not in task_results_map:
            task_results_map[task_name] = resumed_stat
            logger.info(f"Merged resumed result for task: {task_name}")

    # Calculate global statistics
    total_cost = sum(r["total_cost"] for r in task_results_map.values())
    total_tokens = sum(r["total_tokens"] for r in task_results_map.values())
    total_samples = sum(r["num_samples"] for r in task_results_map.values())

    global_stats = {
        "experiment_name": str(cfg.experiment_name),
        "k": scheduler_config.max_steps,
        "num_tasks": len(task_results_map),
        "total_samples": total_samples,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "scheduler_stats": scheduler.get_stats(),
    }

    # Save global statistics
    (output_dir / "global_stat.json").write_text(
        json.dumps(global_stats, indent=2, default=str), encoding="utf-8"
    )

    # Save detailed task results
    (output_dir / "task_results.json").write_text(
        json.dumps(task_results_map, indent=2), encoding="utf-8"
    )

    return global_stats


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/parallel_glm47",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry for pass@k generation/evaluation.

    Args:
        cfg: Hydra config resolved from ``tts_search/configs`` and CLI overrides.

    Returns:
        None. Writes generation, evaluation, progress, and summary files to disk.
    """
    logging.basicConfig(level=logging.INFO)
    random.seed(cfg.seed)

    api_keys = _parse_minimax_api_keys(cfg)
    if api_keys:
        logger.info("Using %d MINIMAX API key(s)", len(api_keys))
    else:
        # Not MiniMax config, use single key from config
        api_keys = [None]  # Will use key from config
        logger.info("Not MiniMax config, using API key from config")

    data_path = Path(cfg.data.eval_data)
    tasks = (
        json.loads(data_path.read_text(encoding="utf-8"))
        if data_path.suffix == ".json"
        else pd.read_parquet(data_path).to_dict(orient="records")
    )

    leaderboard_dir = (
        str(cfg.data.leaderboard_dir) if cfg.data.leaderboard_dir is not None else None
    )
    for task in tasks:
        metadata = task[cfg.data.metadata_key]
        metadata["leaderboard_dir"] = leaderboard_dir
    metric_metadata_parquet = cfg.data.get("metric_metadata_parquet", None)
    enriched_metric_count = enrich_tasks_with_metric_protocol(
        tasks,
        source_parquet=data_path if data_path.suffix != ".json" else None,
        metric_metadata_parquet=(
            Path(str(metric_metadata_parquet))
            if metric_metadata_parquet is not None
            else None
        ),
        metadata_key=str(cfg.data.metadata_key),
    )
    if enriched_metric_count:
        logger.info(
            "Enriched %d task metadata records with self_valid_protocol",
            enriched_metric_count,
        )

    logger.info("Loaded %d tasks from %s", len(tasks), data_path)

    # Filter tasks by task list file if specified
    task_filter_file = cfg.get("task_filter_file", None)
    if task_filter_file is not None:
        task_filter_path = Path(task_filter_file)
        if task_filter_path.exists():
            with open(task_filter_path, "r") as f:
                allowed_task_names = {line.strip() for line in f if line.strip()}

            original_count = len(tasks)
            tasks = [
                task
                for task in tasks
                if task[cfg.data.metadata_key]["task_name"] in allowed_task_names
            ]
            logger.info(
                f"Task filter applied: {original_count} -> {len(tasks)} tasks "
                f"(from {task_filter_path})"
            )
        else:
            logger.warning(f"Task filter file not found: {task_filter_path}")

    # Use resume_from as output_dir if specified
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        base_output_dir = Path(resume_from)
        logger.info(f"Using resume directory as output: {base_output_dir}")
    else:
        base_output_dir = Path(cfg.output_dir)

    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_output = base_output_dir / "config.yaml"
    with open(config_output, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Run evaluation
    if len(api_keys) == 1 and api_keys[0] is None:
        # Not MiniMax, use router from config directly
        router: Router = instantiate(cfg.litellm)
        output_dir = base_output_dir
        global_stats = asyncio.run(
            run_pass_k_evaluation(
                cfg=cfg,
                router=router,
                tasks=tasks,
                output_dir=output_dir,
            )
        )
    elif len(api_keys) == 1:
        router = _build_router_with_key(cfg, api_keys[0])
        output_dir = base_output_dir
        global_stats = asyncio.run(
            run_pass_k_evaluation(
                cfg=cfg,
                router=router,
                tasks=tasks,
                output_dir=output_dir,
            )
        )
    else:
        task_shards = _split_tasks_evenly(tasks, len(api_keys))
        shard_dirs: list[Path] = []
        for idx, shard_tasks in enumerate(task_shards, start=1):
            shard_dir = base_output_dir / f"shard_{idx:02d}_of_{len(api_keys)}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(shard_dir)
            logger.info(
                "Shard %d/%d: %d task(s) -> %s",
                idx,
                len(api_keys),
                len(shard_tasks),
                shard_dir,
            )

        async def run_sharded_evals() -> list[dict[str, Any]]:
            """Run one pass@k evaluation coroutine per API-key shard.

            Args:
                None; closes over shard tasks, shard dirs, API keys, and config.

            Returns:
                Per-shard global statistics returned by ``run_pass_k_evaluation``.
            """
            sandbox_concurrency = int(cfg.sandbox.concurrency)
            per_shard_sandbox = max(1, sandbox_concurrency // len(api_keys))
            if per_shard_sandbox * len(api_keys) != sandbox_concurrency:
                logger.info(
                    "Sandbox concurrency %d split across %d keys -> %d per shard",
                    sandbox_concurrency,
                    len(api_keys),
                    per_shard_sandbox,
                )
            coros = []
            for api_key, shard_tasks, shard_dir in zip(
                api_keys, task_shards, shard_dirs, strict=True
            ):
                router = _build_router_with_key(cfg, api_key)
                shard_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                shard_cfg.sandbox.concurrency = per_shard_sandbox
                if resume_from is not None:
                    shard_cfg.resume_from = str(shard_dir)
                coros.append(
                    run_pass_k_evaluation(
                        cfg=shard_cfg,
                        router=router,
                        tasks=shard_tasks,
                        output_dir=shard_dir,
                    )
                )
            return await asyncio.gather(*coros)

        shard_stats = asyncio.run(run_sharded_evals())
        global_stats = {
            "experiment_name": str(cfg.experiment_name),
            "k": int(cfg.max_steps),
            "num_tasks": sum(s.get("num_tasks", 0) for s in shard_stats),
            "total_samples": sum(s.get("total_samples", 0) for s in shard_stats),
            "total_cost": sum(s.get("total_cost", 0) for s in shard_stats),
            "total_tokens": sum(s.get("total_tokens", 0) for s in shard_stats),
            "avg_grade": None,
        }

        (base_output_dir / "global_stat_combined.json").write_text(
            json.dumps(global_stats, indent=2, default=str), encoding="utf-8"
        )
        output_dir = base_output_dir

    # Log summary
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Tasks: {global_stats['num_tasks']}")
    logger.info(f"Total samples: {global_stats['total_samples']}")
    logger.info(f"Total cost: ${global_stats['total_cost']:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
