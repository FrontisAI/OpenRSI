"""
Scheduler - Coordinates generation and evaluation services.

This scheduler manages the workflow between the generator and evaluator
services, similar to how SGLang coordinates prefill and decode operations.
It supports pass@k evaluation where k samples are generated and all
evaluated on the test set.

Key Features:
- Decoupled generation and evaluation pipelines
- Generation runs independently without waiting for evaluation
- Evaluation is optional and consumes from async queue
- Separate progress tracking for generation and evaluation
- Streaming output to JSONL files for durability
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from litellm.router import Router

# Optional matplotlib for progress trend chart (use Agg for headless)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from tts_search import eval_utils
from tts_search.data_produce.token_filter import (
    count_chat_template_tokens,
    load_tokenizer,
)
from tts_search.greedy import AlgoConfig, GreedySearch
from tts_search.program_database import Program, ProgramDatabase
from tts_search.services.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    EvaluatorService,
)
from tts_search.services.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopController,
)
from tts_search.services.generator import (
    GenerationRequest,
    GenerationResult,
    GeneratorService,
)
from tts_search.services.rejection import (
    MEDAL_LABELS,
    RejectionPolicy,
    build_rejection_policy,
)
from tts_search.services.result_persistence import StreamingJSONLWriter
from tts_search.services.tree_search_state import (
    NODE_EVALUATED,
    NODE_GENERATED,
    SearchEvent,
    SearchState,
    append_search_event,
    atomic_write_search_state,
)

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Generation settings
    max_steps: int = 64
    n_samples_per_task: int = 1
    llm_concurrency: int = 64
    llm_max_retries_per_step: int = 10

    # Evaluation settings
    sandbox_base_url: str = ""
    sandbox_cpu_base_url: str = ""
    sandbox_resource_type_override: str | None = None
    sandbox_concurrency: int = 32
    job_timeout: int = 3600
    wait_timeout: int = 86400
    poll_interval: int = 5
    use_score2reward: bool = True

    # Pipeline settings
    batch_size: int = 16  # Batch size for evaluation collection
    task_concurrency: int = 1  # Number of tasks to run concurrently
    enable_sandbox_eval: bool = True  # Enable/disable sandbox evaluation
    skip_eval_for_done_tasks: bool = False  # Skip queued evals after a task is done

    # Decoupled pipeline settings
    eval_queue_maxsize: int = 0  # 0 = unlimited queue size
    eval_batch_timeout: float = 1.0  # Timeout for collecting evaluation batch

    # Task output settings
    task_subdir: str | None = None  # Optional subdirectory for per-task outputs

    # Looping task evaluation settings
    loop_enabled: bool = False
    loop_success_target: int = 1  # x: required success count
    loop_success_metric: str = "success"  # success or medal
    loop_medal_target: int | None = None  # required medal count; defaults to success target
    loop_target_increment: int = 1  # y: target increment per round
    loop_max_target: int | None = None  # z: stop when target reaches this
    resume_rebuild_task_states: bool = False
    rejection_policy: str | None = None
    rejection_target: int | None = None
    rejection_score_threshold: float | None = None
    rejection_reward_threshold: float | None = None
    rejection_reference_scores_path: str | None = None
    rejection_accepted_medals: tuple[str, ...] | None = None
    rejection_apply_baseline_filters: bool = True
    rejection_baseline_token_limit: int = 32768
    rejection_baseline_tokenizer_model: str | None = None
    rejection_baseline_relative_gap_limit: float = 0.12
    rejection_mixed_leaderboard_target: int = 2
    rejection_mixed_no_leaderboard_target: int = 4

    # Tree-search generation settings. Disabled by default to preserve pass@k.
    tree_search_enabled: bool = False
    tree_search_algorithm: str = "greedy"
    tree_search_max_pending_per_task: int = 1
    tree_search_db_path: str | None = None
    tree_search_db_max_per_task: int | None = None
    greedy_num_drafts: int = 5
    greedy_debug_prob: float = 0.5
    greedy_draft_prob: float = 0.0

    # Generation retry: on gen failure, retry up to gen_retry_max times (None = infinite)
    gen_retry_max: int | None = 10

    # Progress history: record gen/eval stats and completed_tasks every N seconds, write JSONL + trend chart
    progress_history_interval_seconds: float = 600.0  # 10 minutes
    # Progress trend chart: width in [min, max] (inches). Full trend shown by compressing x-axis within max width
    progress_trend_fig_width_min: float = 10.0  # minimum figure width (inches)
    progress_trend_fig_width_max: float = 40.0  # maximum figure width; beyond this, x-axis is compressed
    progress_trend_inches_per_hour: float = 2.0  # width per hour of time span (until hitting max)


@dataclass
class GenerationProgress:
    """Progress tracking for generation pipeline."""

    total: int = 0
    success: int = 0
    in_progress: int = 0
    failed: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def pending(self) -> int:
        return self.total - self.success - self.in_progress - self.failed


@dataclass
class EvaluationProgress:
    """Progress tracking for evaluation pipeline."""

    total: int = 0
    success: int = 0
    in_progress: int = 0
    pending_in_queue: int = 0
    failed: int = 0
    active_sandbox_slots: int = 0  # Number of sandbox evaluations currently running
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class TaskResult:
    """Result for a single task after pass@k evaluation."""

    task_id: str
    task_name: str
    samples: list[Program]
    best_program: Program
    best_score: float | None
    best_reward: float
    total_cost: float
    total_tokens: int
    num_samples: int
    status_counts: dict[str, int]
    medal_count: int
    medal_counts: dict[str, int]
    val_time: float
    test_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskLoopState:
    """Per-task state for looping evaluation."""

    task_id: str
    task_name: str
    success_count: int = 0
    medal_count: int = 0
    accepted_count: int = 0
    completed_count: int = 0
    generated_count: int = 0
    pending_gen: int = 0
    target_count: int = 0
    max_target: int | None = None
    accepted_target: int | None = None
    next_step_index: int = 0
    done: bool = False


def load_checkpoint(
    output_dir: Path,
) -> tuple[set[str], set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load checkpoint state from output directory.

    Returns:
        Tuple of (completed_gen_ids, completed_eval_ids, gen_results_data,
                  eval_results_data)
    """
    completed_gen_ids: set[str] = set()
    completed_eval_ids: set[str] = set()
    gen_results_data: list[dict[str, Any]] = []
    eval_results_data: list[dict[str, Any]] = []

    gen_file = output_dir / "gen_results.jsonl"
    eval_file = output_dir / "eval_results.jsonl"

    # Load generation results
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        request_id = data.get("request_id")
                        if request_id:
                            completed_gen_ids.add(request_id)
                            gen_results_data.append(data)
                    except json.JSONDecodeError:
                        continue

    # Load evaluation results (ids + full records)
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                request_id = data.get("request_id")
                if request_id:
                    completed_eval_ids.add(request_id)
                eval_results_data.append(data)

    gen_by_request = {
        str(data.get("request_id")): data
        for data in gen_results_data
        if data.get("request_id") is not None
    }
    for data in eval_results_data:
        request_id = data.get("request_id")
        gen_data = gen_by_request.get(str(request_id)) if request_id is not None else None
        if gen_data is not None:
            for gen_key, eval_key in (
                ("prompt_tokens", "generation_prompt_tokens"),
                ("completion_tokens", "generation_completion_tokens"),
                ("total_tokens", "generation_total_tokens"),
            ):
                if data.get(eval_key) is None and gen_data.get(gen_key) is not None:
                    data[eval_key] = gen_data[gen_key]
            task_name = data.get("task_name") or gen_data.get("task_name")
            task_id = data.get("task_id") or gen_data.get("task_id")
            step_index = data.get("step_index", gen_data.get("step_index"))
        else:
            task_name = data.get("task_name")
            task_id = data.get("task_id")
            step_index = data.get("step_index")

        if data.get("feedback_path") is None and task_name and step_index is not None:
            safe_name = safe_task_output_name(str(task_name), task_id)
            candidates = [
                output_dir / "tasks" / safe_name / f"step_{step_index}" / "feedback.txt",
                output_dir / safe_name / f"step_{step_index}" / "feedback.txt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    data["feedback_path"] = str(candidate)
                    break

    return completed_gen_ids, completed_eval_ids, gen_results_data, eval_results_data


def load_progress_snapshot(output_dir: Path) -> dict[str, Any] | None:
    """Load progress.json if available for resume."""
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def safe_task_output_name(task_name: str, task_id: str | None) -> str:
    """Return a task output directory name that is unique for duplicate task names."""
    safe_task_name = str(task_name).replace("/", "_").replace("\\", "_")
    safe_task_id = str(task_id or "missing_task_id").replace("/", "_").replace("\\", "_")
    return f"{safe_task_name}_{safe_task_id}"


def _eval_result_has_medal(
    data: dict[str, Any], metadata: dict[str, Any] | None = None
) -> bool:
    submit_medal = data.get("submit_medal")
    if isinstance(submit_medal, str) and submit_medal.lower() in MEDAL_LABELS:
        return True

    if metadata is None or data.get("score") is None:
        return False
    try:
        leaderboard = eval_utils.load_leaderboard(metadata)
        if leaderboard is None:
            return False
        medal = eval_utils.get_medal_for_score(float(data["score"]), leaderboard)
    except Exception as exc:
        logger.warning(
            "Failed to rebuild medal status for task=%s score=%s: %s",
            data.get("task_name") or data.get("task_id"),
            data.get("score"),
            exc,
        )
        return False
    return medal in MEDAL_LABELS


def _loop_target_met(
    state: TaskLoopState,
    rejection_target: int,
) -> bool:
    return state.accepted_count >= rejection_target


def progress_snapshot_has_medal_counts(progress_snapshot: dict[str, Any] | None) -> bool:
    snapshot_states = (progress_snapshot or {}).get("task_states") or {}
    return bool(snapshot_states) and all(
        "medal" in state_data for state_data in snapshot_states.values()
    )


def build_task_states_from_progress(
    progress_snapshot: dict[str, Any] | None,
    tasks: list[dict[str, Any]],
    loop_increment: int,
    loop_max_target: int | None,
    rejection_policy_name: str = "accept_scored",
) -> dict[str, TaskLoopState]:
    """Build task states from progress.json snapshot if available."""
    if not progress_snapshot:
        return {}
    snapshot_states = progress_snapshot.get("task_states") or {}
    if not snapshot_states:
        return {}

    task_states: dict[str, TaskLoopState] = {}
    for task in tasks:
        task_id = task["task_id"]
        task_name = task["task_name"]
        state_data = snapshot_states.get(task_id, {})
        state = TaskLoopState(
            task_id=task_id,
            task_name=task_name,
            max_target=state_data.get("max_target", loop_max_target),
            target_count=int(state_data.get("target", loop_increment)),
        )
        state.success_count = int(state_data.get("success", 0))
        state.medal_count = int(state_data.get("medal", 0))
        if "accepted" in state_data:
            state.accepted_count = int(state_data["accepted"])
        elif rejection_policy_name in {"medal", "has_medal"}:
            state.accepted_count = state.medal_count
        else:
            state.accepted_count = state.success_count
        state.completed_count = int(state_data.get("completed", 0))
        state.generated_count = int(state_data.get("generated", 0))
        if state_data.get("accepted_target") is not None:
            state.accepted_target = int(state_data["accepted_target"])
        state.done = bool(state_data.get("done", False))
        task_states[task_id] = state
    return task_states


def rebuild_task_states_from_results(
    tasks: list[dict[str, Any]],
    gen_results_data: list[dict[str, Any]],
    eval_results_data: list[dict[str, Any]],
    loop_increment: int,
    loop_success_target: int,
    loop_success_metric: str = "success",
    loop_medal_target: int | None = None,
    loop_max_target: int | None = None,
    rejection_policy: RejectionPolicy | None = None,
    rejection_target: int | None = None,
    accepted_targets: Mapping[str, int] | None = None,
    count_successful_generation_as_completed: bool = False,
) -> dict[str, TaskLoopState]:
    """Rebuild task states from gen/eval result records."""
    policy = rejection_policy or build_rejection_policy(name=loop_success_metric)
    target_accept_count = int(
        rejection_target
        if rejection_target is not None
        else (
            loop_medal_target
            if loop_success_metric == "medal" and loop_medal_target is not None
            else loop_success_target
        )
    )
    task_states: dict[str, TaskLoopState] = {}
    task_metadata_map: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_states[task["task_id"]] = TaskLoopState(
            task_id=task["task_id"],
            task_name=task["task_name"],
            max_target=loop_max_target,
            target_count=loop_increment,
            accepted_target=(
                int(accepted_targets[task["task_id"]])
                if accepted_targets and task["task_id"] in accepted_targets
                else target_accept_count
            ),
        )
        if task.get("metadata") is not None:
            task_metadata_map[task["task_id"]] = task["metadata"]

    for data in gen_results_data:
        task_id = data.get("task_id")
        if task_id not in task_states:
            continue
        state = task_states[task_id]
        step_index = int(data.get("step_index", 0))
        if step_index >= state.next_step_index:
            state.next_step_index = step_index + 1
        state.generated_count += 1
        if count_successful_generation_as_completed or not data.get("success", True):
            state.completed_count += 1

    for data in eval_results_data:
        task_id = data.get("task_id")
        if task_id not in task_states:
            continue
        state = task_states[task_id]
        state.completed_count += 1
        if data.get("success", False):
            state.success_count += 1
        if _eval_result_has_medal(data, task_metadata_map.get(task_id)):
            state.medal_count += 1
        if policy.accepts_record(data, task_metadata_map.get(task_id)).accepted:
            state.accepted_count += 1

    for state in task_states.values():
        state_accept_target = (
            int(accepted_targets[state.task_id])
            if accepted_targets and state.task_id in accepted_targets
            else target_accept_count
        )
        state.accepted_target = state_accept_target
        if _loop_target_met(state, state_accept_target):
            state.done = True
            state.target_count = max(state.completed_count, loop_increment)
            continue
        if loop_success_metric == "medal":
            completed_rounds = max(
                1,
                (
                    max(state.generated_count, state.completed_count)
                    + loop_increment
                    - 1
                )
                // loop_increment,
            )
            target = loop_increment * completed_rounds
        else:
            target = loop_increment * (state.generated_count // loop_increment + 1)
        if loop_max_target is not None:
            target = min(target, loop_max_target)
        state.target_count = max(target, loop_increment)
        while state.completed_count >= state.target_count:
            if _loop_target_met(
                state,
                state_accept_target,
            ) or (
                loop_max_target is not None and state.target_count >= loop_max_target
            ):
                state.done = True
                break
            if loop_success_metric != "medal":
                break
            next_target = state.target_count + loop_increment
            if loop_max_target is not None:
                next_target = min(next_target, loop_max_target)
            if next_target <= state.target_count:
                state.done = True
                break
            state.target_count = next_target

    return task_states


class Scheduler:
    """
    Scheduler for coordinating generation and evaluation.

    This scheduler implements a decoupled pipeline architecture:
    - Generator produces code samples independently (producer)
    - Evaluator consumes and scores samples (consumer)
    - Both pipelines run concurrently via async queues
    - Progress is tracked separately for each pipeline
    """

    def __init__(
        self,
        router: Router,
        config: SchedulerConfig,
        database: ProgramDatabase | None = None,
        output_dir: Path | str | None = None,
        litellm_model_list: list[Any] | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            router: LiteLLM router for model requests
            config: Scheduler configuration
            database: Optional program database for persistence
            output_dir: Optional output directory for gen/eval result files
        """
        self._router = router
        self._config = config
        self._database = database
        self._output_dir = Path(output_dir) if output_dir else None

        # Initialize generator service
        self._generator = GeneratorService(
            router=router,
            concurrency=config.llm_concurrency,
            max_retries=config.llm_max_retries_per_step,
            model_list=litellm_model_list,
        )

        # Evaluator is created as context manager in run()
        self._evaluator: EvaluatorService | None = None

        # Decoupled pipeline queues
        self._eval_queue: asyncio.Queue[GenerationResult | None] = asyncio.Queue(
            maxsize=config.eval_queue_maxsize
        )

        # Progress tracking (separate for gen and eval)
        self._gen_progress = GenerationProgress()
        self._eval_progress = EvaluationProgress()
        self._progress_lock = asyncio.Lock()

        # Task-level tracking
        self._total_tasks = 0
        self._completed_tasks = 0
        self._task_states: dict[str, TaskLoopState] | None = None

        # Start/end times
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # Generation results storage (for creating Programs after eval)
        self._gen_results: dict[str, GenerationResult] = {}
        self._eval_results: dict[str, EvaluationResult] = {}
        self._results_lock = asyncio.Lock()
        self._eval_result_hook: Callable[[EvaluationResult], Awaitable[None]] | None = (
            None
        )
        self._eval_record_metadata_hook: Callable[[EvaluationResult], dict[str, Any]] | None = None
        self._baseline_tokenizer: Any | None = None

        # Streaming writers (will be lazily initialized in run_tasks)
        self._gen_writer: StreamingJSONLWriter | None = None
        self._eval_writer: StreamingJSONLWriter | None = None
        self._search_algorithm = None
        if self._config.tree_search_enabled:
            algorithm = str(self._config.tree_search_algorithm).lower()
            if algorithm != "greedy":
                raise ValueError(
                    "Scheduler tree_search currently supports only algorithm='greedy'"
                )
            if int(self._config.tree_search_max_pending_per_task) != 1:
                raise ValueError(
                    "tree_search_max_pending_per_task must be 1 in the v1 scheduler"
                )
            if self._database is None:
                db_path = (
                    Path(self._config.tree_search_db_path)
                    if self._config.tree_search_db_path
                    else (self._output_dir or Path(".")) / "tree_search_programs.db"
                )
                self._database = ProgramDatabase(
                    str(db_path),
                    max_per_task=self._config.tree_search_db_max_per_task,
                )
            self._search_algorithm = GreedySearch(
                AlgoConfig(
                    num_drafts=self._config.greedy_num_drafts,
                    debug_prob=self._config.greedy_debug_prob,
                    draft_prob=self._config.greedy_draft_prob,
                )
            )

    async def run_evaluation_only(
        self,
        output_dir: Path,
        tasks: list[dict[str, Any]] | None = None,
    ) -> list[EvaluationResult]:
        """
        Run evaluation only on existing generation results.

        This method reads gen_results.jsonl, identifies unevaluated samples,
        and evaluates them using fire-and-forget pattern (same as run_evaluation).

        Useful for:
        1. Pure evaluation mode (generation was done separately)
        2. Resuming evaluation after interruption

        Args:
            output_dir: Directory containing gen_results.jsonl
            tasks: Optional task list for metadata lookup

        Returns:
            List of new evaluation results
        """
        progress_snapshot = load_progress_snapshot(output_dir)
        self._start_time = datetime.now()

        # Initialize task-level tracking
        # Count unique tasks in generation results
        unique_task_names = set()
        if tasks:
            unique_task_names = {
                task.get("metadata", {}).get("task_name") for task in tasks
            }
            unique_task_names.discard(None)
        self._total_tasks = len(unique_task_names) if unique_task_names else 0
        self._completed_tasks = 0  # Will be updated as evaluation completes

        # Load checkpoint
        completed_gen_ids, completed_eval_ids, gen_results_data, _ = load_checkpoint(
            output_dir
        )

        # Find unevaluated samples
        unevaluated = [
            data
            for data in gen_results_data
            if data["request_id"] not in completed_eval_ids
            and data.get("success", True)
        ]

        if not unevaluated:
            logger.info("No unevaluated samples found")
            return []

        logger.info(
            f"Found {len(unevaluated)} unevaluated samples out of {len(gen_results_data)} total"
        )
        logger.info(f"Already evaluated: {len(completed_eval_ids)} samples")

        # Build task metadata map if provided
        task_metadata_map: dict[str, dict[str, Any]] = {}
        if tasks:
            for task in tasks:
                metadata = task.get("metadata", {})
                task_id = metadata.get("uuid") or task.get("task_id")
                if task_id:
                    task_metadata_map[task_id] = metadata

        # Initialize progress - set total to all gen results, completed to already evaluated
        eval_elapsed = (
            progress_snapshot.get("evaluation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        self._eval_progress = EvaluationProgress(
            total=len(gen_results_data),
            success=len(completed_eval_ids),
            start_time=datetime.now() - timedelta(seconds=float(eval_elapsed)),
        )

        # Initialize generation progress from gen_results
        self._gen_progress = GenerationProgress()
        self._gen_progress.total = len(gen_results_data)
        self._gen_progress.success = len(gen_results_data)
        self._gen_progress.start_time = self._start_time
        self._gen_progress.end_time = self._start_time

        # Setup eval writer in append mode
        self._eval_writer = StreamingJSONLWriter(
            output_dir / "eval_results.jsonl", append=True
        )

        # Create evaluator
        evaluator_ctx = EvaluatorService(
            base_url=self._config.sandbox_base_url,
            cpu_base_url=self._config.sandbox_cpu_base_url,
            resource_type_override=self._config.sandbox_resource_type_override,
            concurrency=self._config.sandbox_concurrency,
            job_timeout=self._config.job_timeout,
            wait_timeout=self._config.wait_timeout,
            poll_interval=self._config.poll_interval,
        )

        all_eval_results: list[EvaluationResult] = []

        async with evaluator_ctx as evaluator:
            self._evaluator = evaluator

            # Queue all unevaluated samples to eval queue
            logger.info(f"Queueing {len(unevaluated)} samples for evaluation")
            for data in unevaluated:
                task_id = data["task_id"]
                task_name = data["task_name"]
                metadata = task_metadata_map.get(task_id, {})

                # Build output_dir for this task (same as in run_evaluation)
                base_task_root = (
                    output_dir / self._config.task_subdir
                    if getattr(self._config, "task_subdir", None)
                    else output_dir
                )
                base_task_root.mkdir(parents=True, exist_ok=True)
                task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # Create GenerationResult with proper output_dir
                gen_result = GenerationResult(
                    request_id=data["request_id"],
                    task_id=task_id,
                    task_name=task_name,
                    code=data.get("code", ""),
                    raw_text=data.get("raw_text", ""),
                    reasoning_content=data.get("reasoning_content"),
                    prompt_tokens=data.get("prompt_tokens", 0),
                    completion_tokens=data.get("completion_tokens", 0),
                    total_tokens=data.get("total_tokens", 0),
                    cost=data.get("cost", 0.0),
                    model_name=data.get("model_name", ""),
                    messages=[],
                    metadata=metadata,
                    step_index=data.get("step_index", 0),
                    data_dir=metadata.get("data_dir", ""),
                    output_dir=task_output_dir,
                )
                await self._eval_queue.put(gen_result)
                async with self._progress_lock:
                    self._eval_progress.pending_in_queue += 1

            # Signal end of queue
            await self._eval_queue.put(None)

            # Run evaluation loop with fire-and-forget pattern
            # store_results=False since we don't need to keep results in memory for eval-only mode
            all_eval_results = await self._run_eval_loop(
                output_dir, store_results=False
            )

        self._end_time = datetime.now()
        self._evaluator = None

        # Update completed tasks count
        # Count tasks where all samples are evaluated
        task_samples_count: dict[str, tuple[int, int]] = (
            {}
        )  # task_name -> (total, evaluated)
        for data in gen_results_data:
            task_name = data.get("task_name")
            if task_name:
                if task_name not in task_samples_count:
                    task_samples_count[task_name] = (0, 0)
                total, evaluated = task_samples_count[task_name]
                task_samples_count[task_name] = (total + 1, evaluated)

        # Count evaluated samples per task (include newly evaluated)
        all_completed_eval_ids = completed_eval_ids.union(
            {r.request_id for r in all_eval_results}
        )
        for request_id in all_completed_eval_ids:
            for data in gen_results_data:
                if data.get("request_id") == request_id:
                    task_name = data.get("task_name")
                    if task_name and task_name in task_samples_count:
                        total, evaluated = task_samples_count[task_name]
                        task_samples_count[task_name] = (total, evaluated + 1)
                    break

        # Count fully evaluated tasks
        self._completed_tasks = sum(
            1 for total, evaluated in task_samples_count.values() if total == evaluated
        )

        logger.info(
            f"Evaluation only completed: {len(all_eval_results)} samples evaluated"
        )
        logger.info(
            f"Tasks with all samples evaluated: {self._completed_tasks}/{self._total_tasks}"
        )

        return all_eval_results

    async def run_tasks(
        self,
        tasks: list[dict[str, Any]],
        model_name: str,
        output_dir: Path | None = None,
        resume: bool = True,
    ) -> list[TaskResult]:
        """
        Run pass@k evaluation for multiple tasks.

        Args:
            tasks: List of task configurations
            model_name: Model to use for generation
            output_dir: Output directory for artifacts (uses self._output_dir if None)
            resume: If True, resume from checkpoint (skip completed gen/eval)

        Returns:
            List of TaskResults
        """
        # Use provided output_dir or fall back to the one from __init__
        if output_dir is None:
            if self._output_dir is None:
                raise ValueError(
                    "output_dir must be provided either in __init__ or run_tasks"
                )
            output_dir = self._output_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = datetime.now()

        if self._config.loop_enabled:
            return await self._run_looping_tasks(
                tasks=tasks,
                model_name=model_name,
                output_dir=output_dir,
                resume=resume,
            )
        else:
            raise ValueError("Loop is not enabled")

    async def _run_looping_tasks(
        self,
        tasks: list[dict[str, Any]],
        model_name: str,
        output_dir: Path,
        resume: bool = True,
    ) -> list[TaskResult]:
        """Run looping evaluation with per-task success/target tracking."""
        if self._config.loop_target_increment <= 0:
            raise ValueError("loop_target_increment must be > 0")

        loop_increment = int(self._config.loop_target_increment)
        if self._tree_search_enabled():
            if int(self._config.tree_search_max_pending_per_task) != 1:
                raise ValueError(
                    "tree_search_max_pending_per_task must be 1 in the v1 scheduler"
                )
            loop_increment = 1
        loop_success_target = int(self._config.loop_success_target)
        loop_success_metric = str(self._config.loop_success_metric).lower()
        if loop_success_metric not in {"success", "medal"}:
            raise ValueError("loop_success_metric must be 'success' or 'medal'")
        loop_medal_target = (
            int(self._config.loop_medal_target)
            if self._config.loop_medal_target is not None
            else loop_success_target
        )
        rejection_policy_name = str(
            self._config.rejection_policy or loop_success_metric
        ).lower()
        rejection_policy = build_rejection_policy(
            name=rejection_policy_name,
            score_threshold=self._config.rejection_score_threshold,
            reward_threshold=self._config.rejection_reward_threshold,
            reference_scores_path=self._config.rejection_reference_scores_path,
            accepted_medals=list(self._config.rejection_accepted_medals or ()),
            apply_baseline_filters=(
                self._config.rejection_apply_baseline_filters
                and rejection_policy_name not in {"all", "accept_all", "none"}
            ),
            baseline_token_limit=self._config.rejection_baseline_token_limit,
            baseline_relative_gap_limit=(
                self._config.rejection_baseline_relative_gap_limit
            ),
            mixed_leaderboard_target=self._config.rejection_mixed_leaderboard_target,
            mixed_no_leaderboard_target=(
                self._config.rejection_mixed_no_leaderboard_target
            ),
        )
        rejection_target = int(
            self._config.rejection_target
            if self._config.rejection_target is not None
            else (
                loop_medal_target
                if rejection_policy_name in {"medal", "has_medal"}
                else loop_success_target
            )
        )
        loop_max_target = (
            int(self._config.loop_max_target)
            if self._config.loop_max_target is not None
            else None
        )
        loop_controller = GenerationLoopController(
            GenerationLoopConfig(
                batch_size=loop_increment,
                accepted_target=rejection_target,
                max_generated=loop_max_target,
            )
        )

        progress_snapshot = load_progress_snapshot(output_dir) if resume else None
        if progress_snapshot is not None and self._tree_search_enabled():
            logger.info(
                "Ignoring progress.json for tree-search resume; rebuilding from gen/eval records"
            )
            progress_snapshot = None
        if progress_snapshot is not None and self._config.resume_rebuild_task_states:
            logger.info(
                "Ignoring progress.json and rebuilding task states from gen/eval "
                "results because resume_rebuild_task_states=true"
            )
            progress_snapshot = None
        self._total_tasks = len(tasks)
        self._completed_tasks = 0

        # Load checkpoint data
        completed_eval_ids: set[str] = set()
        gen_results_data: list[dict[str, Any]] = []
        eval_results_data: list[dict[str, Any]] = []
        if resume:
            _, completed_eval_ids, gen_results_data, eval_results_data = (
                load_checkpoint(output_dir)
            )

        # Build task metadata map
        task_metadata_map: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_metadata_map[task["task_id"]] = task["metadata"]
        accepted_targets = self._build_accepted_targets(
            tasks=tasks,
            policy=rejection_policy,
            default_target=rejection_target,
        )

        # Initialize task states (progress snapshot or rebuild from results)
        task_states = build_task_states_from_progress(
            progress_snapshot=progress_snapshot,
            tasks=tasks,
            loop_increment=loop_increment,
            loop_max_target=loop_max_target,
            rejection_policy_name=rejection_policy_name,
        )
        if (
            rejection_policy_name in {"medal", "has_medal"}
            and task_states
            and not progress_snapshot_has_medal_counts(progress_snapshot)
        ):
            task_states = {}
        if not task_states:
            task_states = rebuild_task_states_from_results(
                tasks=tasks,
                gen_results_data=gen_results_data,
                eval_results_data=eval_results_data,
                loop_increment=loop_increment,
                loop_success_target=loop_success_target,
                loop_success_metric=loop_success_metric,
                loop_medal_target=loop_medal_target,
                loop_max_target=loop_max_target,
                rejection_policy=rejection_policy,
                rejection_target=rejection_target,
                accepted_targets=accepted_targets,
                count_successful_generation_as_completed=not self._config.enable_sandbox_eval,
            )
        else:
            # Ensure step indices continue from checkpointed generation
            for data in gen_results_data:
                task_id = data.get("task_id")
                if task_id not in task_states:
                    continue
                state = task_states[task_id]
                step_index = int(data.get("step_index", 0))
                if step_index >= state.next_step_index:
                    state.next_step_index = step_index + 1

        for state in task_states.values():
            if state.task_id in accepted_targets:
                state.accepted_target = accepted_targets[state.task_id]

        self._task_states = task_states
        self._completed_tasks = sum(1 for s in task_states.values() if s.done)

        # Initialize progress from checkpoint data
        gen_elapsed = (
            progress_snapshot.get("generation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        eval_elapsed = (
            progress_snapshot.get("evaluation", {}).get("elapsed_seconds")
            if progress_snapshot
            else None
        )
        gen_success = sum(1 for data in gen_results_data if data.get("success", True))
        gen_failed = len(gen_results_data) - gen_success
        eval_success = sum(
            1 for data in eval_results_data if data.get("success", False)
        )
        eval_failed = len(eval_results_data) - eval_success + gen_failed

        self._gen_progress = GenerationProgress(
            total=0,
            success=gen_success,
            failed=gen_failed,
            start_time=(
                datetime.now() - timedelta(seconds=float(gen_elapsed))
                if gen_elapsed
                else datetime.now()
            ),
        )
        self._eval_progress = EvaluationProgress(
            total=0,
            success=eval_success,
            failed=eval_failed,
            start_time=(
                datetime.now() - timedelta(seconds=float(eval_elapsed))
                if eval_elapsed
                else datetime.now()
            ),
        )
        self._update_loop_progress_totals(task_states)

        # Setup streaming output files
        self._gen_writer = StreamingJSONLWriter(
            output_dir / "gen_results.jsonl", append=resume
        )
        self._eval_writer = StreamingJSONLWriter(
            output_dir / "eval_results.jsonl", append=resume
        )

        # Pending evaluations from checkpoint (generated but not evaluated)
        pending_eval_data = [
            data
            for data in gen_results_data
            if self._config.enable_sandbox_eval
            and data.get("success", True)
            and data.get("request_id") not in completed_eval_ids
            and data.get("task_id") in task_states
        ]

        if self._config.enable_sandbox_eval:
            evaluator_ctx = EvaluatorService(
                base_url=self._config.sandbox_base_url,
                cpu_base_url=self._config.sandbox_cpu_base_url,
                resource_type_override=self._config.sandbox_resource_type_override,
                concurrency=self._config.sandbox_concurrency,
                job_timeout=self._config.job_timeout,
                wait_timeout=self._config.wait_timeout,
                poll_interval=self._config.poll_interval,
            )
        else:
            @asynccontextmanager
            async def null_evaluator_ctx():
                yield None
            evaluator_ctx = null_evaluator_ctx()

        all_gen_results: list[GenerationResult] = []
        all_eval_results: list[EvaluationResult] = []
        accept_without_eval = rejection_policy_name in {"all", "accept_all", "none"}
        acceptance_decisions: dict[str, tuple[bool, str]] = {}

        def _decision_for_eval(eval_result: EvaluationResult) -> tuple[bool, str]:
            cached = acceptance_decisions.get(eval_result.request_id)
            if cached is not None:
                return cached
            self._attach_slime_message_token_count(eval_result)
            decision = rejection_policy.accepts_result(eval_result)
            cached = (bool(decision.accepted), str(decision.reason))
            acceptance_decisions[eval_result.request_id] = cached
            return cached

        def _eval_record_search_metadata(eval_result: EvaluationResult) -> dict[str, Any]:
            gen_result = self._gen_results.get(eval_result.request_id)
            metadata = dict(gen_result.search_metadata or {}) if gen_result else {}
            if self._tree_search_enabled():
                accepted, reason = _decision_for_eval(eval_result)
                metadata.update(
                    {
                        "search_algorithm": metadata.get("search_algorithm", "greedy"),
                        "search_step": metadata.get("search_step", eval_result.step_index),
                        "program_id": metadata.get("program_id", eval_result.request_id),
                        "parent_id": metadata.get("parent_id"),
                        "secondary_parent_id": (
                            metadata.get("parent_ids", [None, None])[1]
                            if len(metadata.get("parent_ids", [])) > 1
                            else None
                        ),
                        "parent_ids": metadata.get("parent_ids", []),
                        "generation_mode": metadata.get("generation_mode", "draft"),
                        "fitness": eval_result.reward,
                        "accepted_by_rejection_policy": accepted,
                        "rejection_reason": reason,
                        "code_path": self._step_code_path(
                            eval_result.output_dir, eval_result.step_index
                        ),
                    }
                )
            return metadata

        async with evaluator_ctx as evaluator:
            self._evaluator = evaluator
            task_state_lock = asyncio.Lock()
            done_event = asyncio.Event()
            gen_queue: asyncio.Queue[GenerationRequest | None] = asyncio.Queue()

            task_info_map: dict[str, dict[str, Any]] = {}
            for task in tasks:
                task_id = task["task_id"]
                task_name = task["task_name"]
                base_task_root = (
                    output_dir / self._config.task_subdir
                    if getattr(self._config, "task_subdir", None)
                    else output_dir
                )
                base_task_root.mkdir(parents=True, exist_ok=True)
                task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                task_output_dir.mkdir(parents=True, exist_ok=True)
                messages = task["messages"]
                system_messages = [
                    message
                    for message in messages
                    if str(message.get("role", "")).lower() == "system"
                ]
                task_info_map[task_id] = {
                    "task_name": task_name,
                    "messages": messages,
                    "system_message": (
                        system_messages[0]
                        if system_messages
                        else {"role": "system", "content": ""}
                    ),
                    "description": (
                        str(messages[-1].get("content", "")) if messages else ""
                    ),
                    "metadata": task["metadata"],
                    "data_dir": task["data_dir"],
                    "output_dir": task_output_dir,
                }

            self._preload_tree_database_from_checkpoint(
                gen_results_data=gen_results_data,
                eval_results_data=eval_results_data,
                task_metadata_map=task_metadata_map,
            )

            # request_id -> number of retries already done (0 = first attempt)
            gen_retry_counts: dict[str, int] = {}
            gen_retry_max = getattr(
                self._config, "gen_retry_max", 3
            )  # None = infinite retry

            async def _refresh_completion_locked() -> None:
                self._completed_tasks = sum(1 for s in task_states.values() if s.done)
                self._update_loop_progress_totals(task_states)
                if self._completed_tasks >= self._total_tasks:
                    done_event.set()

            async def _create_requests(
                task_id: str, count: int
            ) -> list[GenerationRequest]:
                if count <= 0:
                    return []
                async with task_state_lock:
                    state = task_states[task_id]
                    if state.done:
                        return []
                    needed = max(
                        0,
                        state.target_count - state.generated_count - state.pending_gen,
                    )
                    count = min(count, needed)
                    if count <= 0:
                        return []
                    state.pending_gen += count

                    info = task_info_map[task_id]
                    requests: list[GenerationRequest] = []
                    for _ in range(count):
                        request_id = f"{task_id}_{state.next_step_index}"
                        step_index = state.next_step_index
                        state.next_step_index += 1
                        request_messages = info["messages"]
                        request_model_name = model_name
                        search_metadata: dict[str, Any] = {}
                        if self._tree_search_enabled():
                            assert self._database is not None
                            prompt, parent_program, mode, selected_model = (
                                self._search_algorithm.select(
                                    database=self._database,
                                    task_name=info["task_name"],
                                    description=info["description"],
                                    data_dir=info["data_dir"],
                                    max_steps=self._config.max_steps,
                                )
                            )
                            parent_program_id = None
                            parent_ids: list[str] = []
                            parent_db_id = None
                            parent_code = ""
                            if parent_program is not None:
                                parent_db_id = parent_program.id
                                parent_program_id = str(
                                    parent_program.metadata.get("program_id")
                                    or parent_program.id
                                )
                                parent_ids = [parent_program_id]
                                parent_code = parent_program.code
                            request_messages = [
                                dict(info["system_message"]),
                                {"role": "user", "content": str(prompt or "")},
                            ]
                            request_model_name = selected_model or model_name
                            search_metadata = {
                                "search_algorithm": str(
                                    self._config.tree_search_algorithm
                                ).lower(),
                                "search_step": step_index,
                                "program_id": request_id,
                                "parent_id": parent_program_id,
                                "parent_ids": parent_ids,
                                "parent_db_id": parent_db_id,
                                "parent_code": parent_code,
                                "generation_mode": mode,
                            }
                        requests.append(
                            GenerationRequest(
                                request_id=request_id,
                                task_id=task_id,
                                task_name=info["task_name"],
                                messages=request_messages,
                                model_name=request_model_name,
                                data_dir=info["data_dir"],
                                metadata=info["metadata"],
                                step_index=step_index,
                                output_dir=info["output_dir"],
                                search_metadata=search_metadata,
                            )
                        )
                    return requests

            async def _enqueue_requests(task_id: str, count: int) -> None:
                requests = await _create_requests(task_id, count)
                for req in requests:
                    await gen_queue.put(req)

            async def _on_gen_result(gen_result: GenerationResult) -> None:
                all_gen_results.append(gen_result)
                to_enqueue = 0
                task_state_for_event: TaskLoopState | None = None
                async with task_state_lock:
                    state = task_states.get(gen_result.task_id)
                    if state is None:
                        return
                    state.generated_count += 1
                    state.pending_gen = max(0, state.pending_gen - 1)
                    if not self._config.enable_sandbox_eval or not gen_result.success:
                        state.completed_count += 1
                        if accept_without_eval:
                            state.accepted_count += 1
                    to_enqueue = loop_controller.advance_after_completion(state)
                    task_state_for_event = state
                    await _refresh_completion_locked()

                if (
                    self._tree_search_enabled()
                    and gen_result.success
                    and task_state_for_event is not None
                    and gen_result.output_dir is not None
                ):
                    search_metadata = dict(gen_result.search_metadata or {})
                    event_log_path, _ = self._task_search_paths(gen_result.output_dir)
                    append_search_event(
                        event_log_path,
                        SearchEvent(
                            event_type=NODE_GENERATED,
                            task_id=gen_result.task_id,
                            task_name=gen_result.task_name,
                            search_algorithm=str(
                                search_metadata.get("search_algorithm", "greedy")
                            ),
                            search_step=int(
                                search_metadata.get(
                                    "search_step", gen_result.step_index
                                )
                            ),
                            program_id=str(
                                search_metadata.get(
                                    "program_id", gen_result.request_id
                                )
                            ),
                            parent_ids=[
                                str(parent_id)
                                for parent_id in search_metadata.get("parent_ids", [])
                            ],
                            generation_mode=str(
                                search_metadata.get("generation_mode", "draft")
                            ),
                            code_path=self._step_code_path(
                                gen_result.output_dir, gen_result.step_index
                            ),
                            code_sha256=self._code_sha256(gen_result.code),
                            generated_count=task_state_for_event.generated_count,
                            stop_requested=task_state_for_event.done,
                        ),
                    )

                if to_enqueue > 0:
                    await _enqueue_requests(gen_result.task_id, to_enqueue)

            async def _on_eval_result(eval_result: EvaluationResult) -> None:
                all_eval_results.append(eval_result)
                to_enqueue = 0
                task_state_for_event: TaskLoopState | None = None
                accepted, rejection_reason = _decision_for_eval(eval_result)
                gen_result_for_eval = self._gen_results.get(eval_result.request_id)
                async with task_state_lock:
                    state = task_states.get(eval_result.task_id)
                    if state is None:
                        return
                    state.completed_count += 1
                    if eval_result.success:
                        state.success_count += 1
                    if str(eval_result.submit_medal).lower() in MEDAL_LABELS:
                        state.medal_count += 1
                    if accepted:
                        state.accepted_count += 1
                    if (
                        self._tree_search_enabled()
                        and self._database is not None
                        and gen_result_for_eval is not None
                    ):
                        program = self._build_tree_program_from_eval(
                            gen_result=gen_result_for_eval,
                            eval_result=eval_result,
                            accepted=accepted,
                            rejection_reason=rejection_reason,
                        )
                        db_program_id = self._database.add(program)
                        program.id = db_program_id
                        program.metadata["db_program_id"] = db_program_id
                    to_enqueue = loop_controller.advance_after_completion(state)
                    task_state_for_event = state
                    await _refresh_completion_locked()

                if (
                    self._tree_search_enabled()
                    and gen_result_for_eval is not None
                    and eval_result.output_dir is not None
                    and task_state_for_event is not None
                ):
                    search_metadata = dict(gen_result_for_eval.search_metadata or {})
                    event_log_path, _ = self._task_search_paths(eval_result.output_dir)
                    append_search_event(
                        event_log_path,
                        SearchEvent(
                            event_type=NODE_EVALUATED,
                            task_id=eval_result.task_id,
                            task_name=eval_result.task_name,
                            search_algorithm=str(
                                search_metadata.get("search_algorithm", "greedy")
                            ),
                            search_step=int(
                                search_metadata.get(
                                    "search_step", eval_result.step_index
                                )
                            ),
                            program_id=str(
                                search_metadata.get(
                                    "program_id", eval_result.request_id
                                )
                            ),
                            parent_ids=[
                                str(parent_id)
                                for parent_id in search_metadata.get("parent_ids", [])
                            ],
                            generation_mode=str(
                                search_metadata.get("generation_mode", "draft")
                            ),
                            code_path=self._step_code_path(
                                eval_result.output_dir, eval_result.step_index
                            ),
                            code_sha256=self._code_sha256(eval_result.code),
                            score=eval_result.score,
                            reward=eval_result.reward,
                            fitness=eval_result.reward,
                            feedback_path=self._relative_to_task_dir(
                                (
                                    eval_result.output_dir
                                    / f"step_{eval_result.step_index}"
                                    / "feedback.txt"
                                ),
                                eval_result.output_dir,
                            ),
                            raw_log_path=self._relative_to_task_dir(
                                (
                                    eval_result.output_dir
                                    / f"step_{eval_result.step_index}"
                                    / "raw_run_log.txt"
                                ),
                                eval_result.output_dir,
                            ),
                            accepted_by_rejection_policy=accepted,
                            rejection_reason=rejection_reason,
                            generated_count=task_state_for_event.generated_count,
                            completed_count=task_state_for_event.completed_count,
                            accepted_count=task_state_for_event.accepted_count,
                            stop_requested=task_state_for_event.done,
                        ),
                    )
                    self._write_tree_search_state(
                        task_state=task_state_for_event,
                        task_output_dir=eval_result.output_dir,
                        accepted_target=task_state_for_event.accepted_target,
                    )

                if to_enqueue > 0:
                    await _enqueue_requests(eval_result.task_id, to_enqueue)

            self._eval_result_hook = _on_eval_result
            self._eval_record_metadata_hook = _eval_record_search_metadata

            # Queue pending evals from checkpoint
            if pending_eval_data:
                for data in pending_eval_data:
                    task_id = data["task_id"]
                    task_name = data["task_name"]
                    metadata = task_metadata_map.get(task_id, {})
                    base_task_root = (
                        output_dir / self._config.task_subdir
                        if getattr(self._config, "task_subdir", None)
                        else output_dir
                    )
                    base_task_root.mkdir(parents=True, exist_ok=True)
                    task_output_dir = base_task_root / safe_task_output_name(task_name, task_id)
                    task_output_dir.mkdir(parents=True, exist_ok=True)

                    gen_result = GenerationResult(
                        request_id=data["request_id"],
                        task_id=task_id,
                        task_name=task_name,
                        code=data.get("code", ""),
                        raw_text=data.get("raw_text", ""),
                        reasoning_content=data.get("reasoning_content"),
                        prompt_tokens=data.get("prompt_tokens", 0),
                        completion_tokens=data.get("completion_tokens", 0),
                        total_tokens=data.get("total_tokens", 0),
                        cost=data.get("cost", 0.0),
                        model_name=data.get("model_name", ""),
                        messages=[],
                        metadata=metadata,
                        step_index=data.get("step_index", 0),
                        data_dir=data.get("data_dir", metadata.get("data_dir", "")),
                        output_dir=task_output_dir,
                        search_metadata={
                            key: data[key]
                            for key in (
                                "search_algorithm",
                                "search_step",
                                "program_id",
                                "parent_id",
                                "parent_ids",
                                "parent_db_id",
                                "generation_mode",
                            )
                            if key in data
                        },
                    )
                    self._gen_results[gen_result.request_id] = gen_result
                    await self._eval_queue.put(gen_result)
                    async with self._progress_lock:
                        self._eval_progress.pending_in_queue += 1

            # Queue initial generation requests
            for task_id, state in task_states.items():
                needed = max(
                    0, state.target_count - state.generated_count - state.pending_gen
                )
                if needed > 0:
                    await _enqueue_requests(task_id, needed)

            async def _gen_worker() -> None:
                while True:
                    req = await gen_queue.get()
                    if req is None:
                        gen_queue.task_done()
                        break
                    try:
                        await asyncio.wait_for(
                            self._progress_lock.acquire(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "progress_lock timeout before generate: request_id=%s task_id=%s step=%d locked=%s",
                            req.request_id,
                            req.task_id,
                            req.step_index,
                            self._progress_lock.locked(),
                        )
                    else:
                        try:
                            self._gen_progress.in_progress += 1
                        finally:
                            self._progress_lock.release()
                    gen_result = await self._generator.generate(req)
                    logger.info(f"Generation completed: {gen_result.request_id}")

                    # Decide whether to retry on failure
                    will_retry = False
                    if not gen_result.success:
                        retries_done = gen_retry_counts.get(gen_result.request_id, 0)
                        if gen_retry_max is None or retries_done < gen_retry_max:
                            will_retry = True
                            gen_retry_counts[gen_result.request_id] = retries_done + 1
                            logger.info(
                                "Generation failed, retrying (%s/%s): request_id=%s",
                                retries_done + 1,
                                "inf" if gen_retry_max is None else gen_retry_max,
                                gen_result.request_id,
                            )

                    try:
                        await asyncio.wait_for(
                            self._progress_lock.acquire(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "progress_lock timeout after generate: request_id=%s task_id=%s step=%d locked=%s",
                            gen_result.request_id,
                            gen_result.task_id,
                            gen_result.step_index,
                            self._progress_lock.locked(),
                        )
                    else:
                        try:
                            self._gen_progress.in_progress -= 1
                            if not will_retry:
                                if gen_result.success:
                                    self._gen_progress.success += 1
                                else:
                                    self._gen_progress.failed += 1
                                    self._eval_progress.failed += 1
                        except Exception:
                            logger.exception(
                                "Progress update failed: request_id=%s task_id=%s step=%d gen_progress=%s eval_progress=%s",
                                gen_result.request_id,
                                gen_result.task_id,
                                gen_result.step_index,
                                self._gen_progress,
                                self._eval_progress,
                            )
                        finally:
                            self._progress_lock.release()

                    async with self._results_lock:
                        self._gen_results[gen_result.request_id] = gen_result

                    if will_retry:
                        # Re-enqueue same logical request for retry (no write, no _on_gen_result)
                        info = task_info_map[gen_result.task_id]
                        retry_req = GenerationRequest(
                            request_id=gen_result.request_id,
                            task_id=gen_result.task_id,
                            task_name=info["task_name"],
                            messages=gen_result.messages,
                            model_name=gen_result.model_name,
                            data_dir=info["data_dir"],
                            metadata=info["metadata"],
                            step_index=gen_result.step_index,
                            output_dir=info["output_dir"],
                            search_metadata=dict(gen_result.search_metadata or {}),
                        )
                        await gen_queue.put(retry_req)
                    else:
                        if self._gen_writer:
                            gen_record = {
                                    "request_id": gen_result.request_id,
                                    "task_id": gen_result.task_id,
                                    "task_name": gen_result.task_name,
                                    "code": gen_result.code,
                                    "raw_text": gen_result.raw_text,
                                    "reasoning_content": gen_result.reasoning_content,
                                    "prompt_tokens": gen_result.prompt_tokens,
                                    "completion_tokens": gen_result.completion_tokens,
                                    "total_tokens": gen_result.total_tokens,
                                    "cost": gen_result.cost,
                                    "model_name": gen_result.model_name,
                                    "step_index": gen_result.step_index,
                                    "data_dir": gen_result.data_dir,
                                    "success": gen_result.success,
                                    "error": gen_result.error,
                                    "timestamp": gen_result.timestamp.isoformat(),
                            }
                            gen_record.update(dict(gen_result.search_metadata or {}))
                            await self._gen_writer.write(gen_record)
                        await _on_gen_result(gen_result)
                        if self._config.enable_sandbox_eval and gen_result.success:
                            await self._eval_queue.put(gen_result)
                            async with self._progress_lock:
                                self._eval_progress.pending_in_queue += 1

                    gen_queue.task_done()

            eval_task = (
                asyncio.create_task(
                    self._run_eval_loop(output_dir, store_results=False)
                )
                if self._config.enable_sandbox_eval
                else None
            )

            interval = max(1.0, float(self._config.progress_history_interval_seconds))

            async def _progress_history_loop() -> None:
                await self._write_progress_snapshot_and_chart(output_dir)
                while not done_event.is_set():
                    try:
                        await asyncio.wait_for(
                            done_event.wait(), timeout=interval
                        )
                    except asyncio.TimeoutError:
                        pass
                    if not done_event.is_set():
                        await self._write_progress_snapshot_and_chart(output_dir)
                await self._write_progress_snapshot_and_chart(output_dir)

            progress_history_task = asyncio.create_task(_progress_history_loop())

            worker_count = max(1, int(self._config.llm_concurrency))
            gen_workers = [
                asyncio.create_task(_gen_worker()) for _ in range(worker_count)
            ]

            if self._completed_tasks >= self._total_tasks:
                done_event.set()

            await done_event.wait()
            await gen_queue.join()
            for _ in range(worker_count):
                await gen_queue.put(None)
            await asyncio.gather(*gen_workers)
            if eval_task is not None:
                await self._eval_queue.put(None)
                await eval_task
            await progress_history_task

        self._end_time = datetime.now()
        self._evaluator = None
        self._gen_progress.end_time = datetime.now()
        self._eval_progress.end_time = datetime.now()

        # Create task results
        task_results: list[TaskResult] = []
        task_gen_map: dict[str, list[GenerationResult]] = {
            task["task_id"]: [] for task in tasks
        }
        task_eval_map: dict[str, list[EvaluationResult]] = {
            task["task_id"]: [] for task in tasks
        }

        for gen_result in all_gen_results:
            if gen_result.task_id in task_gen_map:
                task_gen_map[gen_result.task_id].append(gen_result)

        for eval_result in all_eval_results:
            if eval_result.task_id in task_eval_map:
                task_eval_map[eval_result.task_id].append(eval_result)

        for task in tasks:
            task_id = task["task_id"]
            task_name = task["task_name"]
            result = self._create_task_result(
                task_id=task_id,
                task_name=task_name,
                generation_results=task_gen_map.get(task_id, []),
                evaluation_results=task_eval_map.get(task_id, []),
                metadata=task_metadata_map.get(task_id, {}),
            )
            task_results.append(result)

        self._eval_result_hook = None
        self._eval_record_metadata_hook = None
        return task_results

    def _build_accepted_targets(
        self,
        *,
        tasks: list[dict[str, Any]],
        policy: RejectionPolicy,
        default_target: int,
    ) -> dict[str, int]:
        target_for_metadata = getattr(policy, "target_for_metadata", None)
        targets: dict[str, int] = {}
        for task in tasks:
            target = default_target
            if target_for_metadata is not None:
                policy_target = target_for_metadata(task.get("metadata"))
                if policy_target is not None:
                    target = int(policy_target)
            targets[task["task_id"]] = target
        return targets

    def _tree_search_enabled(self) -> bool:
        return bool(self._config.tree_search_enabled and self._search_algorithm)

    def _task_search_paths(
        self, task_output_dir: Path
    ) -> tuple[Path, Path]:
        return task_output_dir / "search_events.jsonl", task_output_dir / "search_state.json"

    def _relative_to_task_dir(self, path: Path | None, task_output_dir: Path | None) -> str | None:
        if path is None:
            return None
        if task_output_dir is None:
            return str(path)
        try:
            return str(path.relative_to(task_output_dir))
        except ValueError:
            return str(path)

    def _step_code_path(self, output_dir: Path | None, step_index: int) -> str | None:
        if output_dir is None:
            return None
        return str(Path(f"step_{step_index}") / "valid_code.py")

    def _code_sha256(self, code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _build_sft_assistant_content(self, gen_result: GenerationResult) -> str:
        reasoning = str(gen_result.reasoning_content or "")
        code = str(gen_result.code or "")
        response = str(gen_result.raw_text or "")
        if reasoning.strip():
            if code.strip():
                return (
                    f"<think>\n{reasoning.strip()}\n</think>\n\n"
                    f"```python\n{code.rstrip()}\n```"
                )
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{response.strip()}"
        if code.strip():
            return f"```python\n{code.rstrip()}\n```"
        return response.strip()

    def _attach_slime_message_token_count(self, eval_result: EvaluationResult) -> None:
        if getattr(eval_result, "slime_message_tokens", None) is not None:
            return
        tokenizer_model = self._config.rejection_baseline_tokenizer_model
        if not tokenizer_model:
            return
        gen_result = self._gen_results.get(eval_result.request_id)
        if gen_result is None or not gen_result.messages:
            return
        if self._baseline_tokenizer is None:
            self._baseline_tokenizer = load_tokenizer(
                tokenizer_model,
                local_files_only=True,
            )
        messages = [dict(message) for message in gen_result.messages]
        messages.append(
            {
                "role": "assistant",
                "content": self._build_sft_assistant_content(gen_result),
            }
        )
        token_count = count_chat_template_tokens(messages, self._baseline_tokenizer)
        setattr(eval_result, "slime_message_tokens", int(token_count))
        setattr(eval_result, "slime_message_tokenizer_model", str(tokenizer_model))

    def _write_tree_search_state(
        self,
        *,
        task_state: TaskLoopState,
        task_output_dir: Path,
        accepted_target: int | None,
    ) -> None:
        if self._database is None:
            return
        programs = self._database.list_by_task(task_state.task_name)
        program_ids: list[str] = []
        program_scores: dict[str, float | None] = {}
        program_fitness: dict[str, float | None] = {}
        program_code_paths: dict[str, str] = {}
        parent_map: dict[str, list[str]] = {}
        generation_modes: dict[str, str] = {}
        for program in programs:
            program_id = str(program.metadata.get("program_id") or program.id)
            program_ids.append(program_id)
            program_scores[program_id] = program.score
            program_fitness[program_id] = program.fitness
            if program.metadata.get("code_path"):
                program_code_paths[program_id] = str(program.metadata["code_path"])
            parent_ids = program.metadata.get("parent_ids") or []
            if parent_ids:
                parent_map[program_id] = [str(parent_id) for parent_id in parent_ids]
            generation_modes[program_id] = str(program.generation_mode)

        _, state_path = self._task_search_paths(task_output_dir)
        atomic_write_search_state(
            state_path,
            SearchState(
                task_id=task_state.task_id,
                task_name=task_state.task_name,
                search_algorithm=str(self._config.tree_search_algorithm).lower(),
                next_search_step=task_state.next_step_index,
                generated_count=task_state.generated_count,
                completed_count=task_state.completed_count,
                accepted_count=task_state.accepted_count,
                accepted_target=accepted_target,
                max_generated=task_state.max_target,
                stop_requested=task_state.done,
                program_ids=program_ids,
                program_scores=program_scores,
                program_fitness=program_fitness,
                program_code_paths=program_code_paths,
                parent_map=parent_map,
                generation_modes=generation_modes,
            ),
        )

    def _build_tree_program_from_eval(
        self,
        *,
        gen_result: GenerationResult,
        eval_result: EvaluationResult,
        accepted: bool,
        rejection_reason: str,
    ) -> Program:
        search_metadata = dict(gen_result.search_metadata or {})
        parent_db_id = search_metadata.get("parent_db_id")
        parent_code = str(search_metadata.get("parent_code") or "")
        program_metadata = dict(gen_result.metadata)
        program_metadata.update(search_metadata)
        program_metadata.update(
            {
                "request_messages": gen_result.messages,
                "status_code": eval_result.status_code,
                "status": eval_result.status,
                "queue_time": eval_result.queue_time,
                "run_time": eval_result.run_time,
                "model_name": gen_result.model_name,
                "cost": gen_result.cost,
                "prompt_tokens": gen_result.prompt_tokens,
                "completion_tokens": gen_result.completion_tokens,
                "total_tokens": gen_result.total_tokens,
                "reasoning_content": gen_result.reasoning_content,
                "sandbox_job_id": eval_result.job_id,
                "step_index": gen_result.step_index,
                "raw_run_log": eval_result.raw_run_log,
                "clear_run_log": eval_result.clear_run_log,
                "submit_grade": eval_result.submit_grade,
                "submit_medal": eval_result.submit_medal,
                "accepted_by_rejection_policy": accepted,
                "rejection_reason": rejection_reason,
                "code_path": self._step_code_path(gen_result.output_dir, gen_result.step_index),
            }
        )
        return Program(
            task_id=eval_result.task_id,
            task_name=eval_result.task_name,
            code=gen_result.code,
            score=eval_result.score,
            reward=eval_result.reward,
            fitness=eval_result.reward,
            base_reward=eval_result.reward,
            run_log=str(eval_result.clear_run_log or ""),
            feedback=eval_result.feedback,
            parent_id=int(parent_db_id) if parent_db_id is not None else None,
            parent_code=parent_code,
            generation_mode=str(search_metadata.get("generation_mode") or "draft"),
            raw_text=gen_result.raw_text,
            metadata=program_metadata,
        )

    def _preload_tree_database_from_checkpoint(
        self,
        *,
        gen_results_data: list[dict[str, Any]],
        eval_results_data: list[dict[str, Any]],
        task_metadata_map: Mapping[str, dict[str, Any]],
    ) -> None:
        if not self._tree_search_enabled() or self._database is None:
            return
        gen_by_request = {str(row.get("request_id")): row for row in gen_results_data}
        db_id_by_program_id: dict[str, int] = {}
        for row in sorted(eval_results_data, key=lambda item: int(item.get("step_index", 0))):
            request_id = str(row.get("request_id"))
            gen_row = gen_by_request.get(request_id)
            if not gen_row:
                continue
            program_id = str(row.get("program_id") or gen_row.get("program_id") or request_id)
            if program_id in db_id_by_program_id:
                continue
            task_id = str(row.get("task_id") or gen_row.get("task_id"))
            metadata = dict(task_metadata_map.get(task_id, {}))
            parent_ids = row.get("parent_ids") or gen_row.get("parent_ids") or []
            parent_program_id = str(parent_ids[0]) if parent_ids else None
            parent_db_id = (
                db_id_by_program_id.get(parent_program_id)
                if parent_program_id is not None
                else None
            )
            metadata.update(
                {
                    "program_id": program_id,
                    "parent_ids": parent_ids,
                    "search_algorithm": row.get("search_algorithm") or "greedy",
                    "search_step": int(row.get("search_step", row.get("step_index", 0))),
                    "generation_mode": row.get("generation_mode") or "draft",
                    "accepted_by_rejection_policy": row.get(
                        "accepted_by_rejection_policy"
                    ),
                    "rejection_reason": row.get("rejection_reason"),
                    "code_path": row.get("code_path"),
                }
            )
            program = Program(
                task_id=task_id,
                task_name=str(row.get("task_name") or gen_row.get("task_name")),
                code=str(gen_row.get("code") or ""),
                score=row.get("score"),
                reward=float(row.get("reward") or 0.0),
                fitness=float(row.get("fitness", row.get("reward") or 0.0)),
                base_reward=float(row.get("reward") or 0.0),
                run_log="",
                feedback="",
                parent_id=parent_db_id,
                parent_code="",
                generation_mode=str(row.get("generation_mode") or "draft"),
                raw_text=str(gen_row.get("raw_text") or ""),
                metadata=metadata,
            )
            db_id_by_program_id[program_id] = self._database.add(program)

    def _update_loop_progress_totals(
        self, task_states: dict[str, TaskLoopState]
    ) -> None:
        total_target = sum(state.target_count for state in task_states.values())
        self._gen_progress.total = total_target
        self._eval_progress.total = total_target

    def _update_progress_trend_chart(
        self, output_dir: Path, history_path: Path
    ) -> None:
        """Read progress_history.jsonl and save progress_trend.png (generation/evaluation/completed_tasks over time)."""
        if not _HAS_MATPLOTLIB or not history_path.exists():
            print(f"No matplotlib or history_path: {history_path}")
            return
        records: list[dict[str, Any]] = []
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not records:
            return
        timestamps = []
        gen_success = []
        gen_completed = []
        gen_failed = []
        eval_success = []
        eval_completed = []
        eval_failed = []
        completed_tasks = []
        for r in records:
            try:
                ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is not None:
                    ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            except (KeyError, ValueError):
                continue
            timestamps.append(ts)
            g = r.get("generation", {})
            gen_success.append(g.get("success", 0))
            gen_completed.append(g.get("completed", 0))
            gen_failed.append(g.get("failed", 0))
            e = r.get("evaluation", {})
            eval_success.append(e.get("success", 0))
            eval_completed.append(e.get("completed", 0))
            eval_failed.append(e.get("failed", 0))
            completed_tasks.append(r.get("completed_tasks", 0))
        if not timestamps:
            return
        try:
            # Width in [w_min, w_max]. Full time span always plotted; beyond w_max the chart is compressed horizontally
            time_span_hours = (
                timestamps[-1] - timestamps[0]
            ).total_seconds() / 3600.0
            w_min = getattr(
                self._config, "progress_trend_fig_width_min", 10.0
            )
            w_max = getattr(
                self._config, "progress_trend_fig_width_max", 40.0
            )
            inches_per_hour = getattr(
                self._config, "progress_trend_inches_per_hour", 2.0
            )
            width = min(w_max, max(w_min, time_span_hours * inches_per_hour))
            fig, axes = plt.subplots(
                3, 1, figsize=(width, 8), sharex=True
            )
            ax_gen, ax_eval, ax_tasks = axes

            ax_gen.plot(timestamps, gen_success, label="success", color="C0")
            ax_gen.plot(timestamps, gen_completed, label="completed", color="C1")
            ax_gen.plot(timestamps, gen_failed, label="failed", color="C2")
            ax_gen.set_ylabel("Generation")
            ax_gen.legend(loc="upper left")
            ax_gen.grid(True, alpha=0.3)

            ax_eval.plot(timestamps, eval_success, label="success", color="C0")
            ax_eval.plot(timestamps, eval_completed, label="completed", color="C1")
            ax_eval.plot(timestamps, eval_failed, label="failed", color="C2")
            ax_eval.set_ylabel("Evaluation")
            ax_eval.legend(loc="upper left")
            ax_eval.grid(True, alpha=0.3)

            ax_tasks.plot(timestamps, completed_tasks, label="completed_tasks", color="C3")
            ax_tasks.set_ylabel("Completed tasks")
            ax_tasks.set_xlabel("Time")
            ax_tasks.legend(loc="upper left")
            ax_tasks.grid(True, alpha=0.3)

            fig.suptitle("Progress trend (gen / eval / completed_tasks)")
            fig.tight_layout()
            trend_path = output_dir / "progress_trend.png"
            fig.savefig(trend_path, dpi=120)
            plt.close(fig)
            logger.info("Updated progress trend chart: %s", trend_path)
        except Exception as exc:
            logger.warning("Failed to update progress trend chart: %s", exc)

    async def _write_progress_snapshot_and_chart(self, output_dir: Path) -> None:
        """Take a snapshot of gen/eval/completed_tasks under lock, append to progress_history.jsonl, update trend chart."""
        async with self._progress_lock:
            medal_count = 0
            accepted_count = 0
            if self._task_states is not None:
                medal_count = sum(state.medal_count for state in self._task_states.values())
                accepted_count = sum(
                    state.accepted_count for state in self._task_states.values()
                )
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "generation": {
                    "success": self._gen_progress.success,
                    "completed": self._gen_progress.success + self._gen_progress.failed,
                    "failed": self._gen_progress.failed,
                },
                "evaluation": {
                    "success": self._eval_progress.success,
                    "completed": self._eval_progress.success + self._eval_progress.failed,
                    "failed": self._eval_progress.failed,
                },
                "completed_tasks": self._completed_tasks,
                "medal_count": medal_count,
                "accepted_count": accepted_count,
            }
        history_path = output_dir / "progress_history.jsonl"
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        self._update_progress_trend_chart(output_dir, history_path)

    async def _run_eval_loop(
        self, output_dir: Path, store_results: bool = True
    ) -> list[EvaluationResult]:
        """
        Core evaluation loop using fire-and-forget pattern.

        Consumes items from self._eval_queue and evaluates them with maximum concurrency.
        Completes one eval task -> immediately starts next eval task.

        Args:
            output_dir: Output directory for evaluation results
            store_results: Whether to store results in memory (default True)

        Returns:
            List of all evaluation results
        """
        all_eval_results: list[EvaluationResult] = []
        pending_gen_results: list[GenerationResult] = []
        batch_size = self._config.sandbox_concurrency

        # Use fire-and-forget pattern to keep sandbox concurrency saturated
        pending_eval_tasks: set[asyncio.Task] = set()
        generation_done = False

        async def _drain_eval_queue(room: int) -> None:
            nonlocal generation_done
            if room <= 0:
                return
            for _ in range(room):
                try:
                    item = self._eval_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    generation_done = True
                    break
                pending_gen_results.append(item)
                async with self._progress_lock:
                    self._eval_progress.pending_in_queue -= 1

        def _submit_pending_requests() -> None:
            while len(pending_eval_tasks) < batch_size and pending_gen_results:
                batch = pending_gen_results[:1]
                del pending_gen_results[:1]
                task = asyncio.create_task(
                    self._process_eval_batch(
                        batch, output_dir, store_results=store_results
                    )
                )
                pending_eval_tasks.add(task)

        def _collect_completed_eval_tasks(
            done_tasks: set[asyncio.Task],
        ) -> None:
            for task in done_tasks:
                try:
                    batch_results = task.result()
                except Exception:
                    logger.exception("Evaluation batch task crashed unexpectedly")
                    continue
                if store_results:
                    all_eval_results.extend(batch_results)

        while not generation_done or pending_eval_tasks or pending_gen_results:
            room = batch_size - len(pending_eval_tasks) - len(pending_gen_results)
            await _drain_eval_queue(room)
            _submit_pending_requests()

            if generation_done and not pending_eval_tasks and not pending_gen_results:
                break

            if len(pending_eval_tasks) >= batch_size:
                done, pending = await asyncio.wait(
                    pending_eval_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                _collect_completed_eval_tasks(done)
                pending_eval_tasks = set(pending)
                continue

            queue_get_task = asyncio.create_task(self._eval_queue.get())
            wait_targets = set(pending_eval_tasks)
            wait_targets.add(queue_get_task)

            done, pending = await asyncio.wait(
                wait_targets,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if queue_get_task in done:
                item = queue_get_task.result()
                if item is None:
                    generation_done = True
                else:
                    pending_gen_results.append(item)
                    async with self._progress_lock:
                        self._eval_progress.pending_in_queue -= 1
            else:
                queue_get_task.cancel()
                try:
                    await queue_get_task
                except asyncio.CancelledError:
                    pass

            completed_eval_tasks = {task for task in done if task is not queue_get_task}
            if completed_eval_tasks:
                _collect_completed_eval_tasks(completed_eval_tasks)

            pending_eval_tasks = {
                task for task in pending if task is not queue_get_task
            }

        # Wait for all remaining tasks to complete
        if pending_eval_tasks:
            results = await asyncio.gather(*pending_eval_tasks)
            for batch_results in results:
                if store_results:
                    all_eval_results.extend(batch_results)

        async with self._progress_lock:
            self._eval_progress.end_time = datetime.now()

        logger.info("Evaluation pipeline completed")
        return all_eval_results

    async def _process_eval_batch(
        self,
        gen_results: list[GenerationResult],
        output_dir: Path,
        store_results: bool = True,
    ) -> list[EvaluationResult]:
        """
        Process a batch of generation results through evaluation.

        Args:
            gen_results: Generation results to evaluate
            output_dir: Output directory for evaluation artifacts
            store_results: Whether to store results in self._eval_results (for normal mode)

        Returns:
            List of evaluation results
        """
        if not gen_results or self._evaluator is None:
            return []

        # Create evaluation requests
        eval_requests = []
        for gen_result in gen_results:
            if self._config.skip_eval_for_done_tasks and self._task_states is not None:
                task_state = self._task_states.get(gen_result.task_id)
                if task_state is not None and task_state.done:
                    logger.info(
                        "Skipping eval for done task: request_id=%s task_id=%s",
                        gen_result.request_id,
                        gen_result.task_id,
                    )
                    continue

            # Determine task output directory
            base_task_root = (
                output_dir / self._config.task_subdir
                if getattr(self._config, "task_subdir", None)
                else output_dir
            )
            base_task_root.mkdir(parents=True, exist_ok=True)
            task_output_dir = base_task_root / safe_task_output_name(
                gen_result.task_name,
                gen_result.task_id,
            )
            task_output_dir.mkdir(parents=True, exist_ok=True)

            eval_request = EvaluationRequest(
                request_id=gen_result.request_id,
                task_id=gen_result.task_id,
                task_name=gen_result.task_name,
                code=gen_result.code,
                data_dir=gen_result.data_dir,
                metadata=gen_result.metadata,
                step_index=gen_result.step_index,
                output_dir=task_output_dir,
                use_score2reward=self._config.use_score2reward,
                generation_prompt_tokens=gen_result.prompt_tokens,
                generation_completion_tokens=gen_result.completion_tokens,
                generation_total_tokens=gen_result.total_tokens,
            )
            eval_requests.append(eval_request)

        if not eval_requests:
            return []

        # Update progress - track active sandbox slots
        async with self._progress_lock:
            self._eval_progress.in_progress += len(eval_requests)
            self._eval_progress.active_sandbox_slots += len(eval_requests)

        # Evaluate batch
        try:
            eval_results = await self._evaluator.evaluate_batch(eval_requests)
        finally:
            # Release sandbox slots
            async with self._progress_lock:
                self._eval_progress.active_sandbox_slots -= len(eval_requests)

        # Process results
        for eval_result in eval_results:
            # Update progress
            async with self._progress_lock:
                self._eval_progress.in_progress -= 1
                if not eval_result.success:
                    self._eval_progress.failed += 1
                else:
                    self._eval_progress.success += 1

            # Store result (optional, only in normal mode)
            if store_results:
                async with self._results_lock:
                    self._eval_results[eval_result.request_id] = eval_result

            # Write to evaluation output file
            if self._eval_writer:
                eval_record = {
                        "request_id": eval_result.request_id,
                        "task_id": eval_result.task_id,
                        "task_name": eval_result.task_name,
                        "status_code": eval_result.status_code,
                        "status": eval_result.status,
                        "score": eval_result.score,
                        "reward": eval_result.reward,
                        "submit_grade": eval_result.submit_grade,
                        "submit_medal": eval_result.submit_medal,
                        "job_id": eval_result.job_id,
                        "queue_time": eval_result.queue_time,
                        "run_time": eval_result.run_time,
                        "step_index": eval_result.step_index,
                        "success": eval_result.success,
                        "generation_prompt_tokens": (
                            eval_result.generation_prompt_tokens
                        ),
                        "generation_completion_tokens": (
                            eval_result.generation_completion_tokens
                        ),
                        "generation_total_tokens": eval_result.generation_total_tokens,
                        "slime_message_tokens": getattr(
                            eval_result, "slime_message_tokens", None
                        ),
                        "slime_message_tokenizer_model": getattr(
                            eval_result, "slime_message_tokenizer_model", None
                        ),
                        "feedback_path": (
                            str(
                                eval_result.output_dir
                                / f"step_{eval_result.step_index}"
                                / "feedback.txt"
                            )
                            if eval_result.output_dir is not None
                            else None
                        ),
                        "error": eval_result.error,
                        "timestamp": eval_result.timestamp.isoformat(),
                    }
                if self._eval_record_metadata_hook is not None:
                    eval_record.update(self._eval_record_metadata_hook(eval_result))
                await self._eval_writer.write(eval_record)
            if self._eval_result_hook is not None:
                try:
                    await self._eval_result_hook(eval_result)
                except Exception as exc:
                    logger.warning("Eval result hook failed: %s", exc)

        return eval_results

    def _create_task_result(
        self,
        task_id: str,
        task_name: str,
        generation_results: list[GenerationResult],
        evaluation_results: list[EvaluationResult],
        metadata: dict[str, Any],
    ) -> TaskResult:
        """Create TaskResult from generation and evaluation results."""
        programs: list[Program] = []
        eval_map = {r.request_id: r for r in evaluation_results}

        total_cost = 0.0
        total_tokens = 0
        status_counts: dict[str, int] = {}
        medal_counts: dict[str, int] = {medal: 0 for medal in ("gold", "silver", "bronze")}
        test_time = 0.0

        for gen_result in generation_results:
            eval_result = eval_map.get(gen_result.request_id)

            if eval_result:
                print("logging evaled program")
                program_metadata = dict(metadata)
                program_metadata["request_messages"] = gen_result.messages
                program_metadata["status_code"] = eval_result.status_code
                program_metadata["status"] = eval_result.status
                program_metadata["queue_time"] = eval_result.queue_time
                program_metadata["run_time"] = eval_result.run_time
                program_metadata["model_name"] = gen_result.model_name
                program_metadata["cost"] = gen_result.cost
                program_metadata["prompt_tokens"] = gen_result.prompt_tokens
                program_metadata["completion_tokens"] = gen_result.completion_tokens
                program_metadata["total_tokens"] = gen_result.total_tokens
                program_metadata["reasoning_content"] = gen_result.reasoning_content
                program_metadata["sandbox_job_id"] = eval_result.job_id
                program_metadata["step_index"] = gen_result.step_index
                program_metadata["raw_run_log"] = eval_result.raw_run_log
                program_metadata["clear_run_log"] = eval_result.clear_run_log
                program_metadata["submit_grade"] = eval_result.submit_grade
                program_metadata["submit_medal"] = eval_result.submit_medal

                program = Program(
                    task_id=task_id,
                    task_name=task_name,
                    code=gen_result.code,
                    score=eval_result.score,
                    reward=eval_result.reward,
                    fitness=eval_result.reward,
                    base_reward=eval_result.reward,
                    run_log=str(eval_result.clear_run_log or ""),
                    feedback=eval_result.feedback,
                    parent_id=None,
                    parent_code="",
                    generation_mode="draft",
                    raw_text=gen_result.raw_text,
                    metadata=program_metadata,
                )
                programs.append(program)

                status = eval_result.status
                status_counts[status] = status_counts.get(status, 0) + 1
                submit_medal = str(eval_result.submit_medal).lower()
                if submit_medal in medal_counts:
                    medal_counts[submit_medal] += 1
                if eval_result.run_time is not None:
                    test_time += eval_result.run_time
            elif gen_result.success and not self._config.enable_sandbox_eval:
                print("logging unevaled program")
                program_metadata = dict(metadata)
                program_metadata["request_messages"] = gen_result.messages
                program_metadata["model_name"] = gen_result.model_name
                program_metadata["cost"] = gen_result.cost
                program_metadata["prompt_tokens"] = gen_result.prompt_tokens
                program_metadata["completion_tokens"] = gen_result.completion_tokens
                program_metadata["total_tokens"] = gen_result.total_tokens
                program_metadata["reasoning_content"] = gen_result.reasoning_content
                program_metadata["step_index"] = gen_result.step_index
                program_metadata["status"] = "unevaluated"
                program_metadata["evaluated"] = False

                program = Program(
                    task_id=task_id,
                    task_name=task_name,
                    code=gen_result.code,
                    score=None,
                    reward=0.0,
                    fitness=0.0,
                    base_reward=0.0,
                    run_log="",
                    feedback="",
                    parent_id=None,
                    parent_code="",
                    generation_mode="draft",
                    raw_text=gen_result.raw_text,
                    metadata=program_metadata,
                )
                programs.append(program)
                status_counts["unevaluated"] = status_counts.get("unevaluated", 0) + 1

            total_cost += gen_result.cost
            total_tokens += gen_result.total_tokens

        best_program = max(programs, key=lambda p: p.reward) if programs else None

        if best_program is None:
            best_program = Program(
                task_id=task_id,
                task_name=task_name,
                code="",
                score=None,
                reward=0.0,
                fitness=0.0,
                base_reward=0.0,
            )

        if self._database and programs:
            for program in programs:
                program_id = self._database.add(program)
                program.id = program_id

        # if self._config.per_task_selection and programs:
        #     self._perform_task_selection(task_name, programs)

        return TaskResult(
            task_id=task_id,
            task_name=task_name,
            samples=programs,
            best_program=best_program,
            best_score=best_program.score,
            best_reward=best_program.reward,
            total_cost=total_cost,
            total_tokens=total_tokens,
            num_samples=len(programs),
            status_counts=status_counts,
            medal_count=sum(medal_counts.values()),
            medal_counts=medal_counts,
            val_time=0.0,
            test_time=test_time,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        gen_stats = self._generator.get_stats()
        eval_stats = self._evaluator.get_stats() if self._evaluator else {}

        return {
            "generator": gen_stats,
            "evaluator": eval_stats,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
            "duration_seconds": (
                (self._end_time - self._start_time).total_seconds()
                if self._start_time and self._end_time
                else None
            ),
            "progress": self.get_progress(),
        }

    def get_progress(self) -> dict[str, Any]:
        """
        Get real-time progress information for both pipelines.

        Returns:
            Dictionary with separate progress metrics for generation and evaluation
        """
        gen_elapsed = None
        if self._gen_progress.start_time:
            end = self._gen_progress.end_time or datetime.now()
            gen_elapsed = (end - self._gen_progress.start_time).total_seconds()

        eval_elapsed = None
        if self._eval_progress.start_time:
            end = self._eval_progress.end_time or datetime.now()
            eval_elapsed = (end - self._eval_progress.start_time).total_seconds()

        gen_rate = (
            (self._gen_progress.success + self._gen_progress.failed) / gen_elapsed
            if gen_elapsed and gen_elapsed > 0
            else 0
        )
        eval_rate = (
            (self._eval_progress.success + self._eval_progress.failed) / eval_elapsed
            if eval_elapsed and eval_elapsed > 0
            else 0
        )

        task_states_snapshot = None
        if self._task_states is not None:
            task_states_snapshot = {
                task_id: {
                    "task_name": state.task_name,
                    "success": state.success_count,
                    "medal": state.medal_count,
                    "accepted": state.accepted_count,
                    "completed": state.completed_count,
                    "target": state.target_count,
                    "accepted_target": state.accepted_target,
                    "max_target": state.max_target,
                    "generated": state.generated_count,
                    "done": state.done,
                }
                for task_id, state in self._task_states.items()
            }

        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "task_states": task_states_snapshot,
            "generation": {
                "total": self._gen_progress.total,
                "success": self._gen_progress.success,
                "completed": self._gen_progress.success + self._gen_progress.failed,
                "in_progress": self._gen_progress.in_progress,
                "pending": self._gen_progress.pending,
                "failed": self._gen_progress.failed,
                "elapsed_seconds": gen_elapsed,
                "rate_per_second": gen_rate,
            },
            "evaluation": {
                "total": self._eval_progress.total,
                "success": self._eval_progress.success,
                "completed": self._eval_progress.success + self._eval_progress.failed,
                "in_progress": self._eval_progress.in_progress,
                "pending_in_queue": self._eval_progress.pending_in_queue,
                "active_sandbox_slots": self._eval_progress.active_sandbox_slots,
                "sandbox_concurrency": self._config.sandbox_concurrency,
                "sandbox_utilization": (
                    self._eval_progress.active_sandbox_slots
                    / self._config.sandbox_concurrency
                    if self._config.sandbox_concurrency > 0
                    else 0
                ),
                "failed": self._eval_progress.failed,
                "elapsed_seconds": eval_elapsed,
                "rate_per_second": eval_rate,
            },
            # Legacy fields for backward compatibility
            "total_samples": self._gen_progress.total,
            "generated_samples": self._gen_progress.success,
            "evaluated_samples": self._eval_progress.success,
            "generating_samples": self._gen_progress.in_progress,
            "evaluating_samples": self._eval_progress.in_progress,
            "streaming_samples_written": 0,
        }
