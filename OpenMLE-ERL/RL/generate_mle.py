# Single-turn operators generation for MLE agent with remote sandbox

import asyncio
import json
import httpx
import logging
import math
import os
import statistics
import threading
import time
import weakref
from pathlib import Path

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from reward_func_utils import (
    get_sandbox_result,
    get_proxy_sandbox_result,
    extract_code,
    score2reward,
    score2reward_with_static_priority,
    has_static_bounds_with_priority,
    get_clear_log,
    hack_check_async,
    has_final_validation_score_print,
    extract_validation_score,
    validation_test_gap_info,
    apply_validation_test_gap_penalty,
    VALIDATION_TEST_GAP_PENALTY_ENABLED,
    VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED,
    VALIDATION_TEST_GAP_PENALTY_COEF,
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF,
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE,
)
from adaptive_reward_advantage_utils import (
    BOUND_MODE_THEORETICAL,
    SUPPORTED_ADAPTIVE_BOUND_MODES,
    score_to_group_adaptive_reward,
)

from program_database import (
    ProgramDatabase,
    Program,
    AIRAGreedySearch,
    AIRAEvoSearch,
    AIRAInferenceEvoSearch,
)
from logging_utils import init_logger, get_logger
import logging_utils as _logging_utils_module
import program_database as _program_database_module
from airaevo_experience import build_experience_card, parse_json_object
from prompt_builder import build_airaevo_rich_memory_summary_prompt

_program_database_module.statistics = statistics

# Configure logging format with timestamp at module level
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

# Get n_samples_per_prompt from environment variable or default to 8
N_SAMPLES_PER_PROMPT = int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))
GENERATION_ABORT_CODE_CATEGORIES = {"generation_abort", "generation_abort_group"}
GENERATION_ABORT_TRANSIENT_METADATA_KEYS = (
    "generation_aborted",
    "generation_abort_reason",
    "generation_abort_finish_reason",
    "group_generation_aborted",
)

GPU_BASE_URL = os.getenv("GPU_BASE_URL", "").strip()
CPU_BASE_URL = os.getenv("CPU_BASE_URL", "").strip()
try:
    SANDBOX_CONCURRENCY = int(os.getenv("SANDBOX_CONCURRENCY", "64"))
except (TypeError, ValueError):
    SANDBOX_CONCURRENCY = 64
if SANDBOX_CONCURRENCY < 0:
    SANDBOX_CONCURRENCY = 64

# Toggle score2reward transformation
USE_SCORE2REWARD = os.getenv("USE_SCORE2REWARD", "1") not in ("0")
REWARD_MAPPING_MODE = os.getenv("REWARD_MAPPING_MODE", "power_clip").strip().lower()
STATIC_REWARD_MAPPING_MODE = os.getenv("STATIC_REWARD_MAPPING_MODE", "power_clip").strip().lower()
ENABLE_SELF_VALIDATION = os.getenv("ENABLE_SELF_VALIDATION", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)

# Reward shaping experiments (configured via env vars)
# - IMPROVE_REWARD_STRATEGY:
#     - base: (default) improve reward = effective_base_reward (subject to basic failure checks)
#     - gate_parent: if effective_base_reward >= parent_reward -> effective_base_reward else 0
#     - diff_parent: max(effective_base_reward - parent_reward, 0)
IMPROVE_REWARD_STRATEGY = os.getenv("IMPROVE_REWARD_STRATEGY", "base").strip().lower()
IMPROVE_DELTA_BONUS_COEF = float(os.getenv("IMPROVE_DELTA_BONUS_COEF", "0.5"))
ENABLE_DYNAMIC_SCORE_BOUNDS = os.getenv("ENABLE_DYNAMIC_SCORE_BOUNDS", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
ADAPTIVE_REWARD_BOUND_MODE = os.getenv("ADAPTIVE_REWARD_BOUND_MODE", "top1_top8").strip().lower()
if ADAPTIVE_REWARD_BOUND_MODE not in SUPPORTED_ADAPTIVE_BOUND_MODES:
    logger.warning(
        "Unsupported ADAPTIVE_REWARD_BOUND_MODE=%s. Falling back to %s.",
        ADAPTIVE_REWARD_BOUND_MODE,
        BOUND_MODE_THEORETICAL,
    )
    ADAPTIVE_REWARD_BOUND_MODE = BOUND_MODE_THEORETICAL
DYNAMIC_SCORE_BOUND_MIN_SPAN = float(os.getenv("DYNAMIC_SCORE_BOUND_MIN_SPAN", "1e-6"))
try:
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = float(os.getenv("ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO", "0.0"))
except (TypeError, ValueError):
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = 0.0
if not math.isfinite(ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO):
    ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = 0.0
ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO = max(ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO, 0.0)
ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE = os.getenv("ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
METRIC_STATIC_BOUND_USE_CONST_FALLBACK = os.getenv("METRIC_STATIC_BOUND_USE_CONST_FALLBACK", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "",
)
METRIC_STATIC_FALLBACK_SIGNED_BEST = float(os.getenv("METRIC_STATIC_FALLBACK_SIGNED_BEST", "100.0"))
METRIC_STATIC_FALLBACK_SIGNED_WORST = float(os.getenv("METRIC_STATIC_FALLBACK_SIGNED_WORST", "-100.0"))

MLE_CONFIGS = {
    # Sandbox execution concurrency limit
    "sandbox_concurrency": SANDBOX_CONCURRENCY,
    # Program database settings
    "db_path": os.environ.get("PROGRAM_DB_PATH", "program_database.db"),
    "max_programs_per_task": int(os.environ.get("MAX_PROGRAMS_PER_TASK", "0")),
    "draft_probability": float(os.environ.get("DRAFT_PROBABILITY", "0.5")),  # 2:1:1 => draft=0.5
    "improve_probability": float(os.environ.get("IMPROVE_PROBABILITY", "0.25")),
    "debug_probability": float(os.environ.get("DEBUG_PROBABILITY", "0.25")),  # 2:1:1 => debug=0.25
    "crossover_probability": float(os.environ.get("CROSSOVER_PROBABILITY", "0.5")),
    "search_algorithm": os.environ.get("SEARCH_ALGORITHM", "evo"),  # 'greedy' | 'evo' | 'airaevo'
    "airaevo_policy": os.environ.get("AIRAEVO_POLICY", "rl_mixed"),
    "airaevo_num_islands": int(os.environ.get("AIRAEVO_NUM_ISLANDS", "1")),
    "airaevo_max_island_size": int(os.environ.get("AIRAEVO_MAX_ISLAND_SIZE", "500")),
    "airaevo_crossover_after_generation": int(os.environ.get("AIRAEVO_CROSSOVER_AFTER_GENERATION", "2")),
    "airaevo_execution_timeout": int(os.environ.get("AIRAEVO_EXECUTION_TIMEOUT", os.environ.get("JOB_TIMEOUT", "3600"))),
}



# Async primitives are bound lazily to the event loop that first waits on them.
# SLIME uses separate loops for train rollout and eval rollout in the same
# process, so keep these primitives loop-local.
_LOOP_PRIMITIVES_LOCK = threading.Lock()
_LOOP_LOCKS = weakref.WeakKeyDictionary()
_LOOP_SEMAPHORES = weakref.WeakKeyDictionary()
_DYNAMIC_SCORE_BOUND_TASK_LOCKS = weakref.WeakKeyDictionary()


def _get_loop_lock(name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _LOOP_PRIMITIVES_LOCK:
        loop_locks = _LOOP_LOCKS.get(loop)
        if loop_locks is None:
            loop_locks = {}
            _LOOP_LOCKS[loop] = loop_locks
        lock = loop_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            loop_locks[name] = lock
        return lock


def _get_loop_semaphore(name: str, value: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    with _LOOP_PRIMITIVES_LOCK:
        loop_semaphores = _LOOP_SEMAPHORES.get(loop)
        if loop_semaphores is None:
            loop_semaphores = {}
            _LOOP_SEMAPHORES[loop] = loop_semaphores
        semaphore = loop_semaphores.get(name)
        if semaphore is None:
            semaphore = asyncio.Semaphore(value)
            loop_semaphores[name] = semaphore
        return semaphore


def _get_sandbox_semaphore() -> asyncio.Semaphore:
    return _get_loop_semaphore("sandbox", MLE_CONFIGS["sandbox_concurrency"])


# Counter to track concurrent sandbox executions
SANDBOX_CONCURRENT_COUNT = 0


JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))
WAIT_TIMEOUT = int(os.getenv("WAIT_TIMEOUT", "1800"))
EVAL_JOB_TIMEOUT = int(os.getenv("EVAL_JOB_TIMEOUT", str(JOB_TIMEOUT)))
EVAL_WAIT_TIMEOUT = int(os.getenv("EVAL_WAIT_TIMEOUT", str(WAIT_TIMEOUT)))
EVAL_GENERATION_ABORT_MAX_RETRIES = int(os.getenv("EVAL_GENERATION_ABORT_MAX_RETRIES", "60"))
EVAL_GENERATION_ABORT_RETRY_SLEEP = float(os.getenv("EVAL_GENERATION_ABORT_RETRY_SLEEP", "2.0"))
USE_PROXY_SANDBOX = os.getenv("USE_PROXY_SANDBOX", "0").strip().lower() not in ("0", "false", "no", "")
ENABLE_PROMPT_DUMP = os.getenv("ENABLE_PROMPT_DUMP", "0").strip().lower() not in ("0", "false", "no", "")
PROMPT_DUMP_BASE_DIR = os.getenv(
    "PROMPT_DUMP_BASE_DIR",
    str((Path(__file__).resolve().parent / "test").resolve()),
)
PROMPT_DUMP_RUN_SUBDIR = os.getenv("PROMPT_DUMP_RUN_SUBDIR", "").strip()
AIRAEVO_RICH_MEMORY_ENABLED = os.getenv("AIRAEVO_RICH_MEMORY_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
AIRAEVO_RICH_MEMORY_MAX_TOKENS = int(os.getenv("AIRAEVO_RICH_MEMORY_MAX_TOKENS", "512"))
AIRAEVO_RICH_MEMORY_TIMEOUT = float(os.getenv("AIRAEVO_RICH_MEMORY_TIMEOUT", "120"))

# Global program database instance
_program_database = None

# Global search algorithm instance
_search_algorithm = None

# Global cache for parent selection results per group
# Key: group_index, Value: {
#   'data': (input_messages, parent_program, mode),
#   'access_count': int,  # How many times this cache has been accessed
#   'total_samples': int  # Expected total samples in this group (n_samples_per_prompt)
# }
_parent_selection_cache = {}

_history_static_base_reward_cache: dict[str, float] = {}
_prompt_dump_dir: Path | None = None
_adaptive_group_reward_cache: dict[tuple[str, str], dict[str, object]] = {}


async def _ensure_prompt_dump_dir() -> Path:
    global _prompt_dump_dir
    if _prompt_dump_dir is not None:
        return _prompt_dump_dir
    async with _get_loop_lock("prompt_dump"):
        if _prompt_dump_dir is not None:
            return _prompt_dump_dir
        run_name = PROMPT_DUMP_RUN_SUBDIR or f"prompt_dump_{int(time.time())}"
        out_dir = Path(PROMPT_DUMP_BASE_DIR) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _prompt_dump_dir = out_dir
        logger.info("[PROMPT_DUMP] output_dir=%s", str(out_dir))
        return _prompt_dump_dir


async def _dump_prompt_to_txt(
    *,
    task_name: str,
    task_id: str,
    group_index: int | None,
    mode: str,
    parent_program: Program | None,
    secondary_parent_program: Program | None,
    system_prompt: str,
    user_prompt: str,
) -> None:
    if not ENABLE_PROMPT_DUMP:
        return
    out_dir = await _ensure_prompt_dump_dir()
    ts_ms = int(time.time() * 1000)
    safe_task = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (task_name or "unknown"))[:64]
    gid = "none" if group_index is None else str(group_index)
    file_path = out_dir / f"{ts_ms}_{safe_task}_{mode}_g{gid}.txt"
    content = [
        f"task_name: {task_name}",
        f"task_id: {task_id}",
        f"group_index: {group_index}",
        f"generation_mode: {mode}",
        f"parent_id: {getattr(parent_program, 'id', None)}",
        f"secondary_parent_id: {getattr(secondary_parent_program, 'id', None)}",
        "",
        "===== SYSTEM PROMPT =====",
        system_prompt,
        "",
        "===== USER PROMPT =====",
        user_prompt,
        "",
    ]
    file_path.write_text("\n".join(content), encoding="utf-8")


def _is_finite_number(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def _dynamic_bound_task_key(metadata: dict) -> str:
    for key in ("task_id", "uuid", "data_dir", "task_name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown_task"


def _dynamic_bound_task_identity(metadata: dict) -> tuple[str, str, str]:
    task_key = _dynamic_bound_task_key(metadata)
    task_id = str(metadata.get("task_id") or metadata.get("uuid") or "")
    task_name = str(metadata.get("task_name") or "unknown")
    return task_key, task_id, task_name


async def _get_dynamic_bound_task_lock(task_key: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    async with _get_loop_lock("dynamic_score_bound_registry"):
        with _LOOP_PRIMITIVES_LOCK:
            task_locks = _DYNAMIC_SCORE_BOUND_TASK_LOCKS.get(loop)
            if task_locks is None:
                task_locks = {}
                _DYNAMIC_SCORE_BOUND_TASK_LOCKS[loop] = task_locks
            lock = task_locks.get(task_key)
        if lock is None:
            lock = asyncio.Lock()
            with _LOOP_PRIMITIVES_LOCK:
                task_locks = _DYNAMIC_SCORE_BOUND_TASK_LOCKS.get(loop)
                if task_locks is None:
                    task_locks = {}
                    _DYNAMIC_SCORE_BOUND_TASK_LOCKS[loop] = task_locks
                task_locks[task_key] = lock
        return lock


def _group_cache_key(task_key: str, group_index: int | None) -> tuple[str, str] | None:
    if group_index is None:
        return None
    return (task_key, str(group_index))


def _use_group_adaptive_reward_bounds() -> bool:
    return ENABLE_DYNAMIC_SCORE_BOUNDS and USE_SCORE2REWARD and REWARD_MAPPING_MODE == "power_clip"


def _build_adaptive_bound_metadata_context(
    *,
    task_key: str,
    group_index: int | None,
    history_count: int,
    group_count: int,
    best_signed: float | None,
    worst_signed: float | None,
    group_generation_aborted: bool = False,
) -> dict[str, float | int | str | bool | None]:
    return {
        "enabled": _is_finite_number(best_signed) and _is_finite_number(worst_signed),
        "task_key": task_key,
        "group_index": group_index,
        "bound_mode": ADAPTIVE_REWARD_BOUND_MODE,
        "lower_shift_ratio": ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
        "single_finite_reward_one": ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
        "history_score_count_before": int(history_count),
        "group_score_count": int(group_count),
        "history_score_count_after_including_group": int(history_count) + int(group_count),
        "best_signed": float(best_signed) if _is_finite_number(best_signed) else None,
        "worst_signed": float(worst_signed) if _is_finite_number(worst_signed) else None,
        "group_generation_aborted": bool(group_generation_aborted),
    }


async def _resolve_group_reward_mapping(
    metadata: dict,
    *,
    group_index: int | None,
    group_size: int,
    sample_key: str,
    score: float | None,
    base_reward: float,
) -> dict[str, object]:
    task_key, task_id, task_name = _dynamic_bound_task_identity(metadata)
    generation_aborted = bool(metadata.get("generation_aborted"))
    default_context = _build_adaptive_bound_metadata_context(
        task_key=task_key,
        group_index=group_index,
        history_count=0,
        group_count=1 if _is_finite_number(score) else 0,
        best_signed=None,
        worst_signed=None,
        group_generation_aborted=generation_aborted,
    )
    default_result: dict[str, object] = {
        "base_reward": float(base_reward),
        "context": default_context,
    }

    if not _use_group_adaptive_reward_bounds():
        if USE_SCORE2REWARD and _is_finite_number(score):
            mapped_reward = score2reward(float(score), dict(metadata), mode=REWARD_MAPPING_MODE)
            return {"base_reward": float(mapped_reward), "context": default_context}
        return default_result

    allow_mapping = _is_finite_number(score)
    db = await get_program_database()

    reward_group_id = metadata.get("reward_group_id")
    reward_group_expected_count = metadata.get("reward_group_expected_count")
    reward_group_member_index = metadata.get("reward_group_member_index")

    if group_index is None and not reward_group_id:
        historical_scores = db.get_task_scores(task_id=task_id, task_name=task_name)
        mapped_reward, context = score_to_group_adaptive_reward(
            score=score,
            metadata=metadata,
            historical_scores=historical_scores,
            current_group_scores=[score] if allow_mapping else [],
            mode=ADAPTIVE_REWARD_BOUND_MODE,
            reward_mapping_mode=REWARD_MAPPING_MODE,
            use_score2reward=allow_mapping,
            default_reward=base_reward,
            min_span=DYNAMIC_SCORE_BOUND_MIN_SPAN,
            lower_shift_ratio=ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
            single_finite_reward_one=ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
        )
        context = _build_adaptive_bound_metadata_context(
            task_key=task_key,
            group_index=group_index,
            history_count=len(historical_scores),
            group_count=1 if allow_mapping else 0,
            best_signed=context.get("best_signed"),
            worst_signed=context.get("worst_signed"),
            group_generation_aborted=generation_aborted,
        )
        return {"base_reward": float(mapped_reward), "context": context}

    if reward_group_id:
        group_key = ("reward_group", str(reward_group_id))
        group_lock_key = f"reward_group:{reward_group_id}"
        try:
            expected_count = max(int(reward_group_expected_count or group_size or 1), 1)
        except Exception:
            expected_count = max(int(group_size or 1), 1)
        if reward_group_member_index is not None:
            sample_key = f"{reward_group_id}:{reward_group_member_index}"
    else:
        group_key = _group_cache_key(task_key, group_index)
        group_lock_key = task_key
        expected_count = max(int(group_size or 1), 1)

    if group_key is None:
        return default_result

    task_lock = await _get_dynamic_bound_task_lock(group_lock_key)
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    async with task_lock:
        entry = _adaptive_group_reward_cache.get(group_key)
        if entry is None:
            entry = {
                "expected_count": expected_count,
                "items": {},
                "seen_count": 0,
            }
            _adaptive_group_reward_cache[group_key] = entry

        existing_item = entry["items"].get(sample_key)
        if existing_item is not None:
            logger.warning(
                "[GROUP REWARD BARRIER] duplicate sample ignored group_key=%s sample_key=%s member=%s seen=%s/%s",
                group_key,
                sample_key,
                reward_group_member_index,
                entry.get("seen_count", 0),
                entry.get("expected_count"),
            )
            waiter = existing_item["future"]
        else:
            entry["items"][sample_key] = {
                "score": score,
                "base_reward": float(base_reward),
                "allow_mapping": bool(allow_mapping),
                "metadata": dict(metadata),
                "generation_aborted": generation_aborted,
                "future": waiter,
            }
            entry["seen_count"] = len(entry["items"])
            logger.info(
                "[GROUP REWARD BARRIER] enter group_key=%s sample_key=%s member=%s seen=%s/%s task_key=%s group_index=%s score=%s code_category=%s",
                group_key,
                sample_key,
                reward_group_member_index,
                entry.get("seen_count", 0),
                entry.get("expected_count"),
                task_key,
                group_index,
                score,
                metadata.get("code_category"),
            )

        if int(entry["seen_count"]) >= int(entry["expected_count"]):
            try:
                historical_scores = db.get_task_scores(task_id=task_id, task_name=task_name)
                current_group_scores = [
                    float(item["score"])
                    for item in entry["items"].values()
                    if bool(item.get("allow_mapping")) and _is_finite_number(item.get("score"))
                ]
                group_generation_aborted = any(
                    bool(item.get("generation_aborted")) for item in entry["items"].values()
                )
                use_static_fallback = len(historical_scores) == 0 and len(current_group_scores) <= 1
                if ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE and len(current_group_scores) == 1:
                    use_static_fallback = False

                resolved_best_signed = None
                resolved_worst_signed = None
                resolved_any_dynamic_bound = False
                resolved_items: list[tuple[dict[str, object], float]] = []
                for item in entry["items"].values():
                    item_metadata = item.get("metadata") or metadata
                    if use_static_fallback and bool(item.get("allow_mapping")) and _is_finite_number(
                        item.get("score")
                    ):
                        mapped_reward = compute_static_score_reward(float(item["score"]), item_metadata)
                        bound_context = {"best_signed": None, "worst_signed": None}
                    else:
                        mapped_reward, bound_context = score_to_group_adaptive_reward(
                            score=item.get("score"),
                            metadata=item_metadata,
                            historical_scores=historical_scores,
                            current_group_scores=current_group_scores,
                            mode=ADAPTIVE_REWARD_BOUND_MODE,
                            reward_mapping_mode=REWARD_MAPPING_MODE,
                            use_score2reward=bool(item.get("allow_mapping")),
                            default_reward=float(item.get("base_reward", 0.0)),
                            min_span=DYNAMIC_SCORE_BOUND_MIN_SPAN,
                            lower_shift_ratio=ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO,
                            single_finite_reward_one=ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE,
                        )
                    best_signed = bound_context.get("best_signed")
                    worst_signed = bound_context.get("worst_signed")
                    if _is_finite_number(best_signed) and _is_finite_number(worst_signed):
                        resolved_best_signed = float(best_signed)
                        resolved_worst_signed = float(worst_signed)
                        resolved_any_dynamic_bound = True
                    resolved_items.append((item, float(mapped_reward)))
                context_template = _build_adaptive_bound_metadata_context(
                    task_key=task_key,
                    group_index=group_index,
                    history_count=len(historical_scores),
                    group_count=len(current_group_scores),
                    best_signed=resolved_best_signed if resolved_any_dynamic_bound else None,
                    worst_signed=resolved_worst_signed if resolved_any_dynamic_bound else None,
                    group_generation_aborted=group_generation_aborted,
                )
                logger.info(
                    "[GROUP REWARD BARRIER] release group_key=%s unique_items=%s expected=%s finite_scores=%s aborted=%s",
                    group_key,
                    len(entry["items"]),
                    entry.get("expected_count"),
                    len(current_group_scores),
                    group_generation_aborted,
                )
                for item, mapped_reward in resolved_items:
                    item["future"].set_result({"base_reward": mapped_reward, "context": dict(context_template)})
            except Exception as exc:
                logger.exception("[GROUP REWARD BARRIER] release failed group_key=%s", group_key)
                # 只处理异常路径：让同组所有等待者收到同一个错误，避免永久等待。
                for item in entry["items"].values():
                    future = item["future"]
                    if not future.done():
                        future.set_exception(exc)
            finally:
                # 正常和异常路径都必须清理组缓存，避免失败组永久残留。
                _adaptive_group_reward_cache.pop(group_key, None)

    return await waiter


def _apply_dynamic_bound_context_metadata(sample: Sample, context: dict, expected_group_count: int) -> None:
    sample.metadata["dynamic_score_bounds_enabled"] = bool(context.get("enabled"))
    sample.metadata["dynamic_score_bounds_task_key"] = context.get("task_key")
    sample.metadata["dynamic_score_bounds_group_index"] = context.get("group_index")
    sample.metadata["dynamic_score_bounds_group_expected_count"] = expected_group_count
    sample.metadata["adaptive_reward_bound_mode"] = context.get("bound_mode")
    sample.metadata["adaptive_reward_history_score_count_before"] = context.get("history_score_count_before")
    sample.metadata["adaptive_reward_group_score_count"] = context.get("group_score_count")
    sample.metadata["adaptive_reward_history_score_count_after"] = context.get("history_score_count_after_including_group")
    sample.metadata["dynamic_bound_best_signed"] = context.get("best_signed")
    sample.metadata["dynamic_bound_worst_signed"] = context.get("worst_signed")
    sample.metadata["group_generation_aborted"] = bool(context.get("group_generation_aborted"))


def _return_generation_abort_for_retry(sample: Sample, *, task_name: str, group_index: int | None) -> float:
    sample.metadata["generation_aborted"] = True
    sample.metadata.setdefault("generation_abort_reason", "group_generation_aborted")
    sample.metadata["group_generation_aborted"] = True
    sample.metadata["code_category"] = "generation_abort"
    sample.metadata["base_reward"] = -1.0
    sample.metadata["final_reward"] = -1.0
    sample.metadata["dynamic_raw_reward"] = -1.0
    sample.metadata["static_raw_reward"] = -1.0
    sample.metadata["score"] = None
    sample.metadata["running_time"] = 0.0
    sample.status = Sample.Status.ABORTED
    logger.warning(
        "[GENERATION ABORT] Requeueing group before DB/CSV logging: task=%s group_index=%s reason=%s",
        task_name,
        group_index,
        sample.metadata.get("generation_abort_reason"),
    )
    return -1.0


def compute_mode_rewards(
    *,
    generation_mode: str,
    effective_base_reward: float,
    parent_base_reward: float | None,
    parent_reward: float | None,
    parent_code: str,
    code: str,
    crossover_parent_base_reward: float | None = None,
    crossover_parent_reward: float | None = None,
    crossover_parent_code: str = "",
    improve_reward_strategy: str = "base",
    improve_delta_bonus_coef: float = 0.5,
) -> tuple[float, float, float, float]:
    """
    Compute shaped base reward and final reward for draft/improve/debug/crossover.

    Returns:
        (shaped_base_reward, final_reward, delta, improve_bonus)
    """
    reference_parent_reward = parent_base_reward if parent_base_reward is not None else parent_reward
    reference_crossover_parent_reward = (
        crossover_parent_base_reward if crossover_parent_base_reward is not None else crossover_parent_reward
    )

    if generation_mode in ("improve", "debug", "crossover") and reference_parent_reward is not None:
        if effective_base_reward == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        if parent_code == code or (generation_mode == "crossover" and crossover_parent_code == code):
            return 0.0, 0.0, 0.0, 0.0

        ref_parent = float(reference_parent_reward)
        if generation_mode == "crossover" and reference_crossover_parent_reward is not None:
            ref_parent = max(ref_parent, float(reference_crossover_parent_reward))

        strategy = (improve_reward_strategy or "base").strip().lower()
        if strategy == "gate_parent":
            shaped_base_reward = effective_base_reward if effective_base_reward >= ref_parent else 0.0
        elif strategy == "diff_parent":
            shaped_base_reward = max(effective_base_reward - ref_parent, 0.0)
        else:
            shaped_base_reward = effective_base_reward

        delta = max(effective_base_reward - ref_parent, 0.0)
        improve_bonus = float(improve_delta_bonus_coef) * delta
        final_reward = shaped_base_reward + improve_bonus
        return float(shaped_base_reward), float(final_reward), float(delta), float(improve_bonus)

    # draft mode or no parent reference
    shaped_base_reward = float(effective_base_reward)
    return shaped_base_reward, shaped_base_reward, 0.0, 0.0


def compute_static_score_reward(score: float | None, metadata: dict) -> float:
    """Map score to the original static bounds reward, independent of dynamic reward mode."""
    if score is None:
        return 0.0
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score_f):
        return 0.0

    if not USE_SCORE2REWARD:
        return score_f

    reward_metadata = dict(metadata)
    reward_metadata.pop("dynamic_bound_best_signed", None)
    reward_metadata.pop("dynamic_bound_worst_signed", None)
    fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    return float(
        score2reward_with_static_priority(
            score_f,
            reward_metadata,
            mode=STATIC_REWARD_MAPPING_MODE,
            priority="leaderboard",
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
    )


def compute_metric_static_base_reward(score: float | None, metadata: dict) -> float | None:
    """
    Metric-only static base reward:
    - no dynamic bound
    - no mode shaping
    - leaderboard bounds preferred, theoretical bounds as fallback
    - mercy semantics: penalties are handled outside and should map to 0
    """
    fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    if not has_static_bounds_with_priority(
        metadata,
        priority="leaderboard",
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    ):
        return None
    if score is None:
        return 0.0
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score_f):
        return 0.0

    if not USE_SCORE2REWARD:
        return score_f

    reward_metadata = dict(metadata)
    reward_metadata.pop("dynamic_bound_best_signed", None)
    reward_metadata.pop("dynamic_bound_worst_signed", None)
    metric_reward = score2reward_with_static_priority(
        score_f,
        reward_metadata,
        mode=STATIC_REWARD_MAPPING_MODE,
        priority="leaderboard",
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    )
    try:
        metric_reward_f = float(metric_reward)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric_reward_f):
        return None
    return metric_reward_f


def _get_program_reward_views(program: Program | None) -> dict[str, float | None]:
    if program is None:
        return {
            "dynamic_raw_reward": None,
            "dynamic_base_reward": None,
            "static_raw_reward": None,
            "static_base_reward": None,
        }

    metadata = program.metadata if isinstance(program.metadata, dict) else {}
    return {
        "dynamic_raw_reward": metadata.get("dynamic_raw_reward", program.reward),
        "dynamic_base_reward": metadata.get("dynamic_base_reward", program.base_reward),
        "static_raw_reward": metadata.get("static_raw_reward"),
        "static_base_reward": metadata.get("static_base_reward"),
    }


def _compute_task_reward_frontier(db: ProgramDatabase, *, task_name: str, metadata_fallback: dict) -> dict[str, float]:
    programs = db.get_all(task_name)
    best_dynamic = 0.0
    best_static = 0.0
    best_metric_static_base = 0.0

    for program in programs:
        program_meta = program.metadata if isinstance(program.metadata, dict) else {}
        dynamic_reward = program_meta.get("dynamic_raw_reward", program.reward)
        try:
            dynamic_reward_f = float(dynamic_reward)
        except (TypeError, ValueError):
            dynamic_reward_f = 0.0
        if math.isfinite(dynamic_reward_f):
            best_dynamic = max(best_dynamic, dynamic_reward_f)

        static_reward = program_meta.get("static_raw_reward")
        if static_reward is None and program.score is not None:
            static_reward = compute_static_score_reward(program.score, program_meta or metadata_fallback)
        try:
            static_reward_f = float(static_reward) if static_reward is not None else 0.0
        except (TypeError, ValueError):
            static_reward_f = 0.0
        if math.isfinite(static_reward_f):
            best_static = max(best_static, static_reward_f)

        metric_static_base = program_meta.get("metric_static_base_reward")
        if metric_static_base is None and program.score is not None:
            metric_static_base = compute_metric_static_base_reward(program.score, program_meta or metadata_fallback)
        try:
            metric_static_base_f = float(metric_static_base) if metric_static_base is not None else 0.0
        except (TypeError, ValueError):
            metric_static_base_f = 0.0
        if math.isfinite(metric_static_base_f):
            best_metric_static_base = max(best_metric_static_base, metric_static_base_f)

    return {
        "best_dynamic_raw_reward_before": float(best_dynamic),
        "best_static_raw_reward_before": float(best_static),
        "best_metric_static_base_reward_before": float(best_metric_static_base),
    }


async def _update_history_best_static_base_reward(task_name: str, best_before: float, current_value: float | None) -> float:
    async with _get_loop_lock("history_static_base_reward"):
        existing = _history_static_base_reward_cache.get(task_name, 0.0)
        merged = max(existing, float(best_before or 0.0))
        if current_value is not None:
            try:
                current_f = float(current_value)
            except (TypeError, ValueError):
                current_f = None
            if current_f is not None and math.isfinite(current_f):
                merged = max(merged, current_f)
        _history_static_base_reward_cache[task_name] = merged
        if not _history_static_base_reward_cache:
            return 0.0
        return float(sum(_history_static_base_reward_cache.values()) / len(_history_static_base_reward_cache))


async def get_program_database() -> ProgramDatabase:
    """Get or create the global program database instance."""
    global _program_database
    if _program_database is None:
        async with _get_loop_lock("database"):
            if _program_database is None:
                _program_database = ProgramDatabase(
                    db_path=MLE_CONFIGS["db_path"], max_per_task=MLE_CONFIGS["max_programs_per_task"]
                )
    return _program_database


def get_search_algorithm():
    """Get or create the global search algorithm instance."""
    global _search_algorithm
    if _search_algorithm is None:
        algo_type = MLE_CONFIGS["search_algorithm"]
        draft_prob = MLE_CONFIGS["draft_probability"]
        improve_prob = MLE_CONFIGS["improve_probability"]
        debug_prob = MLE_CONFIGS["debug_probability"]
        crossover_prob = MLE_CONFIGS["crossover_probability"]
        if algo_type == "greedy":
            _search_algorithm = AIRAGreedySearch(draft_probability=draft_prob, debug_probability=debug_prob)
        elif algo_type == "evo":
            _search_algorithm = AIRAEvoSearch(
                draft_probability=draft_prob,
                improve_probability=improve_prob,
                debug_probability=debug_prob,
                crossover_probability=crossover_prob,
            )
        elif algo_type == "airaevo":
            _search_algorithm = AIRAInferenceEvoSearch(
                policy=MLE_CONFIGS["airaevo_policy"],
                num_islands=MLE_CONFIGS["airaevo_num_islands"],
                max_island_size=MLE_CONFIGS["airaevo_max_island_size"],
                crossover_after_generation=MLE_CONFIGS["airaevo_crossover_after_generation"],
                crossover_probability=crossover_prob,
                execution_timeout=MLE_CONFIGS["airaevo_execution_timeout"],
            )
        else:
            raise ValueError(f"Unknown search algorithm: {algo_type}")
    return _search_algorithm



def _to_token_id_list(tokenized):
    """Normalize tokenizer output to a flat list[int] for Sample.tokens."""
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    elif isinstance(tokenized, dict) and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]

    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()

    if tokenized and isinstance(tokenized[0], (list, tuple)):
        if len(tokenized) != 1:
            raise ValueError(f"Expected a single prompt, got {len(tokenized)} tokenized prompts")
        tokenized = tokenized[0]

    return list(tokenized)

def _get_search_selection_metadata(search_algo) -> dict:
    metadata = getattr(search_algo, "last_selection_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _extract_llm_text(output: dict) -> str:
    if not isinstance(output, dict):
        return ""
    for key in ("text", "output", "response"):
        value = output.get(key)
        if isinstance(value, str):
            return value
    choices = output.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    return ""


async def _maybe_update_airaevo_program_metadata(args, db: ProgramDatabase, program: Program) -> dict:
    if MLE_CONFIGS.get("search_algorithm") != "airaevo" or program.id is None:
        return {}
    all_programs = db.get_all(program.task_name)
    previous_programs = [candidate for candidate in all_programs if candidate.id != program.id]
    card = build_experience_card(program, all_programs=previous_programs)
    updates = {"airaevo_experience_card": card}
    if AIRAEVO_RICH_MEMORY_ENABLED:
        parent_program = db.get_by_id(int(program.parent_id)) if program.parent_id is not None else None
        prompt = build_airaevo_rich_memory_summary_prompt(
            task_description=str(program.metadata.get("task_description") or ""),
            current_program=program,
            parent_program=parent_program,
            current_card=card,
        )
        url = os.getenv("AIRAEVO_MEMORY_LLM_URL", "").strip()
        if not url:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": AIRAEVO_RICH_MEMORY_MAX_TOKENS,
            },
            "return_logprob": False,
        }
        try:
            output = await asyncio.wait_for(post(url, payload), timeout=AIRAEVO_RICH_MEMORY_TIMEOUT)
            parsed = parse_json_object(_extract_llm_text(output))
            if isinstance(parsed, dict):
                method_overview = str(parsed.get("method_overview") or "").strip()
                parent_exp = str(parsed.get("parent_comparison_experience") or "").strip()
                if method_overview and parent_exp:
                    updates["airaevo_rich_summary"] = {
                        "method_overview": method_overview,
                        "parent_comparison_experience": parent_exp,
                    }
                    card["rich_summary"] = updates["airaevo_rich_summary"]
                    updates["airaevo_experience_card"] = card
                else:
                    updates["airaevo_rich_summary_error"] = "missing_required_fields"
            else:
                updates["airaevo_rich_summary_error"] = "invalid_json"
        except Exception as exc:
            logger.warning("[AIRAEVO MEMORY] rich summary failed for program_id=%s: %s", program.id, exc)
            updates["airaevo_rich_summary_error"] = type(exc).__name__
    db.merge_program_metadata(int(program.id), updates)
    return updates


def _normalize_token_ids(token_ids):
    """Normalize tokenizer/SGLang token containers to a plain list[int]."""
    # Newer transformers versions may return BatchEncoding from
    # apply_chat_template(tokenize=True). Extract input_ids before concatenation.
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]

    # torch.Tensor and numpy.ndarray both expose tolist; normalize before unwrapping.
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    # A one-sample batch may be nested; rollout Sample.tokens requires one dimension.
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError(f"Expected single prompt token sequence, got {len(token_ids)} sequences")
        token_ids = list(token_ids[0])

    return list(token_ids)



async def generate(args, sample: Sample, sampling_params, evaluation) -> Sample:
    """
    Single-turn generation with draft/improve mode support.

    Draft mode: Generate code from scratch based on task description.
    Improve mode: Improve existing code from database based on parent's score and execution.

    Optimization: Samples with the same group_index share the same parent selection result.

    Args:
        args: Arguments object
        sample: Sample to generate for
        sampling_params: Sampling parameters for generation
        evaluation: If True, force draft mode (draft_probability=1.0)
    """
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."

    assert evaluation is not None, "evaluation must be provided"
    print(f"[GENERATE] evaluation: {evaluation}")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get program database and search algorithm
    db = await get_program_database()
    search_algo = get_search_algorithm()

    # Extract task information from sample metadata
    task_name = sample.metadata.get("task_name", "unknown")
    task_id = sample.metadata.get("uuid", "")  # task_id is the uuid
    data_dir = sample.metadata.get("data_dir", "")
    group_index = getattr(sample, "group_index", None)

    # Get original prompt content from sample (messages with exactly one system and one user)
    import numpy as np

    # If sample.prompt is a numpy array, convert to list
    if isinstance(sample.prompt, np.ndarray):
        sample.prompt = sample.prompt.tolist()

    assert isinstance(sample.prompt, list), "sample.prompt must be a list of message dicts"
    input_messages: list[dict[str, str]] = []
    for message in sample.prompt:
        assert isinstance(message, dict), "each message must be a dict"
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in message.items()
        ), "each message must be dict[str, str]"
        assert "role" in message and "content" in message, "each message must contain role and content"
        input_messages.append(dict(message))
    roles = [message["role"] for message in input_messages]
    assert (
        roles.count("system") == 1 and roles.count("user") == 1
    ), "sample.prompt must contain exactly one system and one user message"
    system_index = roles.index("system")
    user_index = roles.index("user")
    system_prompt = input_messages[system_index]["content"]
    user_prompt = input_messages[user_index]["content"]
    task_description = sample.metadata.get("task_description") or ""
    data_description = sample.metadata.get("data_description") or ""

    # # Mixed draft/improve mode in a group
    # prompt, parent_program, mode = search_algo.select(
    #     database=db,
    #     task_name=task_name,
    #     description=description,
    #     data_dir=data_dir,
    #     max_steps=1
    # )

    # Check if we already have a cached parent selection for this group
    cache_key = None if evaluation else group_index
    secondary_parent_program = None
    selection_metadata = {}

    # Get n_samples_per_prompt from args
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if n_samples_per_prompt is None:
        n_samples_per_prompt = N_SAMPLES_PER_PROMPT

    # cache_key is not None and
    if cache_key in _parent_selection_cache:
        # Reuse cached result for this group
        async with _get_loop_lock("parent_selection_cache"):
            cache_entry = _parent_selection_cache.get(cache_key)
            if cache_entry is not None:
                cached_data = cache_entry["data"]
                if len(cached_data) == 4:
                    input_messages, parent_program, mode, secondary_parent_program = cached_data
                else:
                    input_messages, parent_program, mode = cached_data
                    secondary_parent_program = None
                selection_metadata = dict(cache_entry.get("selection_metadata") or {})
                cache_entry["access_count"] += 1
                access_count = cache_entry["access_count"]
                total_samples = cache_entry["total_samples"]

                # logger.info(f"[CACHE HIT] group_index={group_index}, access={access_count}/{total_samples}")

                # Auto-cleanup: Remove cache entry when all samples have accessed it
                if access_count >= total_samples:
                    del _parent_selection_cache[cache_key]
                    # logger.info(
                    #     f"[CACHE AUTO-CLEANUP] Removed group_index={group_index} after {access_count} accesses"
                    # )
            else:
                # Race condition: cache was deleted, regenerate
                (
                    (augmented_system_prompt, augmented_user_prompt),
                    parent_program,
                    mode,
                    secondary_parent_program,
                ) = search_algo.select(
                    database=db,
                    task_id=task_id,
                    task_description=task_description,
                    data_description=data_description,
                    public_system_prompt=system_prompt,
                    public_user_prompt=user_prompt,
                    data_dir=data_dir,
                    max_steps=1,
                    task_name=task_name,
                    evaluation=evaluation,
                )
                selection_metadata = _get_search_selection_metadata(search_algo)
                input_messages[system_index]["content"] = augmented_system_prompt
                input_messages[user_index]["content"] = augmented_user_prompt
                # logger.info(f"[CACHE RACE] Regenerated for group_index={group_index}")
    else:
        # Perform parent selection (first sample in the group or no group_index)
        async with _get_loop_lock("parent_selection_cache"):
            # Double-check after acquiring lock
            if cache_key is not None and cache_key in _parent_selection_cache:
                cache_entry = _parent_selection_cache[cache_key]
                cached_data = cache_entry["data"]
                if len(cached_data) == 4:
                    input_messages, parent_program, mode, secondary_parent_program = cached_data
                else:
                    input_messages, parent_program, mode = cached_data
                    secondary_parent_program = None
                selection_metadata = dict(cache_entry.get("selection_metadata") or {})
                cache_entry["access_count"] += 1
                access_count = cache_entry["access_count"]
                total_samples = cache_entry["total_samples"]

                # logger.info(f"[CACHE HIT AFTER LOCK] group_index={group_index}, access={access_count}/{total_samples}")

                # Auto-cleanup: Remove cache entry when all samples have accessed it
                if access_count >= total_samples:
                    del _parent_selection_cache[cache_key]
                    # logger.info(
                    #     f"[CACHE AUTO-CLEANUP] Removed group_index={group_index} after {access_count} accesses"
                    # )
            else:
                # Use search algorithm to decide mode, select parent, and build prompt
                (
                    (augmented_system_prompt, augmented_user_prompt),
                    parent_program,
                    mode,
                    secondary_parent_program,
                ) = search_algo.select(
                    database=db,
                    task_id=task_id,
                    task_description=task_description,
                    data_description=data_description,
                    public_system_prompt=system_prompt,
                    public_user_prompt=user_prompt,
                    data_dir=data_dir,
                    max_steps=1,
                    task_name=task_name,
                    evaluation=evaluation,
                )
                selection_metadata = _get_search_selection_metadata(search_algo)
                input_messages[system_index]["content"] = augmented_system_prompt
                input_messages[user_index]["content"] = augmented_user_prompt

                # Cache the result if we have a valid group_index
                if cache_key is not None:
                    _parent_selection_cache[cache_key] = {
                        "data": (input_messages, parent_program, mode, secondary_parent_program),
                        "access_count": 1,  # First access
                        "total_samples": n_samples_per_prompt,
                        "selection_metadata": selection_metadata,
                    }
                    # eval_tag = " [EVALUATION]" if evaluation else ""
                    # logger.info(
                    #     f"[CACHE MISS] Created cache for group_index={group_index}, mode={mode}, total_samples={n_samples_per_prompt}{eval_tag}"
                    # )

    augmented_system_prompt = input_messages[system_index]["content"]
    augmented_user_prompt = input_messages[user_index]["content"]
    input_messages[system_index]["content"] = augmented_system_prompt
    input_messages[user_index]["content"] = augmented_user_prompt

    if evaluation:
        eval_prompt_type = str(sample.metadata.get("prompt_type") or mode or "draft").strip().lower()
        if eval_prompt_type in {"draft", "improve", "debug", "crossover"}:
            mode = eval_prompt_type

    # Store mode and parent info in sample metadata for use in reward_func
    # These will be used to create the Program object in reward_func
    sample.metadata["generation_mode"] = mode
    sample.metadata["evaluation"] = bool(evaluation)
    sample.metadata["task_id"] = task_id
    if selection_metadata:
        sample.metadata.update(selection_metadata)
    secondary_parent_program = locals().get("secondary_parent_program", None)
    if parent_program:
        parent_views = _get_program_reward_views(parent_program)
        sample.metadata["parent_id"] = parent_program.id
        sample.metadata["parent_reward"] = parent_program.reward
        sample.metadata["parent_base_reward"] = parent_program.base_reward
        sample.metadata["parent_dynamic_raw_reward"] = parent_views["dynamic_raw_reward"]
        sample.metadata["parent_dynamic_base_reward"] = parent_views["dynamic_base_reward"]
        sample.metadata["parent_static_raw_reward"] = parent_views["static_raw_reward"]
        sample.metadata["parent_static_base_reward"] = parent_views["static_base_reward"]
        sample.metadata["parent_score"] = parent_program.score
        sample.metadata["parent_code"] = parent_program.code
    else:
        sample.metadata["parent_id"] = None
        sample.metadata["parent_reward"] = None
        sample.metadata["parent_base_reward"] = None
        sample.metadata["parent_dynamic_raw_reward"] = None
        sample.metadata["parent_dynamic_base_reward"] = None
        sample.metadata["parent_static_raw_reward"] = None
        sample.metadata["parent_static_base_reward"] = None
        sample.metadata["parent_score"] = None
        sample.metadata["parent_code"] = ""
    if secondary_parent_program:
        secondary_parent_views = _get_program_reward_views(secondary_parent_program)
        sample.metadata["crossover_parent_id"] = secondary_parent_program.id
        sample.metadata["crossover_parent_reward"] = secondary_parent_program.reward
        sample.metadata["crossover_parent_base_reward"] = secondary_parent_program.base_reward
        sample.metadata["crossover_parent_dynamic_raw_reward"] = secondary_parent_views["dynamic_raw_reward"]
        sample.metadata["crossover_parent_dynamic_base_reward"] = secondary_parent_views["dynamic_base_reward"]
        sample.metadata["crossover_parent_static_raw_reward"] = secondary_parent_views["static_raw_reward"]
        sample.metadata["crossover_parent_static_base_reward"] = secondary_parent_views["static_base_reward"]
        sample.metadata["crossover_parent_code"] = secondary_parent_program.code
        sample.metadata["crossover_parent_ids"] = [parent_program.id, secondary_parent_program.id]
    else:
        sample.metadata["crossover_parent_id"] = None
        sample.metadata["crossover_parent_reward"] = None
        sample.metadata["crossover_parent_base_reward"] = None
        sample.metadata["crossover_parent_dynamic_raw_reward"] = None
        sample.metadata["crossover_parent_dynamic_base_reward"] = None
        sample.metadata["crossover_parent_static_raw_reward"] = None
        sample.metadata["crossover_parent_static_base_reward"] = None
        sample.metadata["crossover_parent_code"] = ""
        sample.metadata["crossover_parent_ids"] = [parent_program.id] if parent_program else []

    input_text = state.tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    await _dump_prompt_to_txt(
        task_name=task_name,
        task_id=task_id,
        group_index=group_index,
        mode=mode,
        parent_program=parent_program,
        secondary_parent_program=secondary_parent_program,
        system_prompt=augmented_system_prompt,
        user_prompt=augmented_user_prompt,
    )
    prompt_token_ids = _to_token_id_list(
        state.tokenizer.apply_chat_template(
            input_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    prompt_token_ids = _normalize_token_ids(prompt_token_ids)

    # Single-turn generation
    # print(f"[GENERATE] prompt: {prompt}")
    payload = {
        "text": input_text,
        "sampling_params": sampling_params,
        "return_logprob": True,  # Request log probabilities for training
    }
    for key in GENERATION_ABORT_TRANSIENT_METADATA_KEYS:
        sample.metadata.pop(key, None)
    if sample.metadata.get("code_category") in GENERATION_ABORT_CODE_CATEGORIES:
        sample.metadata.pop("code_category", None)
    output = None
    eval_abort_attempts = 0
    while True:
        output = await post(url, payload)
        if output["meta_info"]["finish_reason"]["type"] != "abort":
            break
        if not evaluation:
            break
        eval_abort_attempts += 1
        sample.metadata["eval_generation_abort_retry_count"] = eval_abort_attempts
        if eval_abort_attempts > EVAL_GENERATION_ABORT_MAX_RETRIES:
            raise RuntimeError(
                "Eval generation kept returning finish_reason=abort after "
                f"{EVAL_GENERATION_ABORT_MAX_RETRIES} retries for task={task_name} group_index={group_index}"
            )
        logger.warning(
            "[EVAL GENERATION ABORT] Retrying generation: task=%s group_index=%s attempt=%s/%s",
            task_name,
            group_index,
            eval_abort_attempts,
            EVAL_GENERATION_ABORT_MAX_RETRIES,
        )
        if EVAL_GENERATION_ABORT_RETRY_SLEEP > 0:
            await asyncio.sleep(EVAL_GENERATION_ABORT_RETRY_SLEEP)

    # Handle abort: do NOT return early — set an empty response so reward_func is still
    # called.  This is necessary to unblock the group-level barrier in
    # _resolve_group_reward_mapping (ENABLE_DYNAMIC_SCORE_BOUNDS=1).  If we return early
    # here the other n_samples_per_prompt-1 coroutines in the same group will wait forever
    # on their waiter future and permanently occupy an active_tasks slot.
    if output["meta_info"]["finish_reason"]["type"] == "abort":
        finish_reason = output["meta_info"].get("finish_reason")
        sample.metadata["generation_aborted"] = True
        sample.metadata["generation_abort_reason"] = "sglang_finish_reason_abort"
        sample.metadata["generation_abort_finish_reason"] = finish_reason
        sample.metadata["code_category"] = "generation_abort"
        sample.tokens = prompt_token_ids  # no response tokens
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.status = Sample.Status.COMPLETED
        return sample

    # Process response
    response = output["text"]
    # response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

    if "output_token_logprobs" in output["meta_info"]:
        response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += cur_log_probs
    else:
        assert False
        response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]
    response_token_ids = _normalize_token_ids(response_token_ids)

    loss_mask = [1] * len(response_token_ids)

    # Set sample fields
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask

    # Set status based on finish reason
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """
    The reward function for code generation tasks.

    For draft mode: reward is calculated from score as before.
    For improve mode: reward is the difference between current reward and parent reward.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    global SANDBOX_CONCURRENT_COUNT
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    if not USE_PROXY_SANDBOX and not GPU_BASE_URL:
        raise RuntimeError("GPU_BASE_URL is required; the public release has no default sandbox endpoint")

    # Initialize logger if not already done, or re-initialize if n_samples_per_prompt changed.
    # get_logger() auto-creates with default n_samples_per_prompt=8, so we must initialize
    # explicitly here before it's first called to get the correct group size (e.g. 16 for 8x16).
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if n_samples_per_prompt is None:
        n_samples_per_prompt = int(os.environ.get("N_SAMPLES_PER_PROMPT", "8"))

    if _logging_utils_module._global_logger is None or _logging_utils_module._global_logger.n_samples_per_prompt != n_samples_per_prompt:
        init_logger(n_samples_per_prompt=n_samples_per_prompt)
    logger_instance = get_logger()

    raw_text = sample.response or ""  # Store raw response text
    data_dir = sample.metadata.get("data_dir")
    task_id = sample.metadata.get("task_id", "")
    task_name = sample.metadata.get("task_name", "unknown")
    evaluation = bool(sample.metadata.get("evaluation", False))
    sandbox_job_timeout = EVAL_JOB_TIMEOUT if evaluation else JOB_TIMEOUT
    sandbox_wait_timeout = EVAL_WAIT_TIMEOUT if evaluation else WAIT_TIMEOUT
    # Runtime safeguard: allow the sandbox job up to 3600 seconds and reserve
    # 300 additional seconds for polling and network jitter.
    sandbox_max_job_timeout = int(os.getenv("SANDBOX_MAX_JOB_TIMEOUT", "3600"))
    sandbox_max_wait_timeout = int(os.getenv("SANDBOX_MAX_WAIT_TIMEOUT", "3900"))
    if sandbox_max_job_timeout > 0:
        sandbox_job_timeout = min(int(sandbox_job_timeout), sandbox_max_job_timeout)
    if sandbox_max_wait_timeout > 0:
        sandbox_wait_timeout = min(int(sandbox_wait_timeout), sandbox_max_wait_timeout)
    group_index = getattr(sample, "group_index", None)
    generation_mode = sample.metadata.get("generation_mode", "draft")
    parent_id = sample.metadata.get("parent_id")
    parent_reward = sample.metadata.get("parent_reward")
    parent_base_reward = sample.metadata.get("parent_base_reward")
    parent_dynamic_raw_reward = sample.metadata.get("parent_dynamic_raw_reward", parent_reward)
    parent_dynamic_base_reward = sample.metadata.get("parent_dynamic_base_reward", parent_base_reward)
    parent_static_raw_reward = sample.metadata.get("parent_static_raw_reward")
    parent_static_base_reward = sample.metadata.get("parent_static_base_reward")
    parent_code = sample.metadata.get("parent_code", "")
    crossover_parent_reward = sample.metadata.get("crossover_parent_reward")
    crossover_parent_base_reward = sample.metadata.get("crossover_parent_base_reward")
    crossover_parent_dynamic_raw_reward = sample.metadata.get("crossover_parent_dynamic_raw_reward", crossover_parent_reward)
    crossover_parent_dynamic_base_reward = sample.metadata.get(
        "crossover_parent_dynamic_base_reward", crossover_parent_base_reward
    )
    crossover_parent_static_raw_reward = sample.metadata.get("crossover_parent_static_raw_reward")
    crossover_parent_static_base_reward = sample.metadata.get("crossover_parent_static_base_reward")
    crossover_parent_code = sample.metadata.get("crossover_parent_code", "")
    resource_type = sample.metadata.get("cpu_gpu", "gpu")  # default to 'file' if not specified
    # print("resource_type:", resource_type)

    # logger.info(f"data_dir: {data_dir}")
    # logger.info(f"generation_mode: {generation_mode}, parent_id: {parent_id}, parent_reward: {parent_reward}")

    if bool(sample.metadata.get("generation_aborted")):
        sample.metadata["score"] = None
        sample.metadata["running_time"] = 0.0
        sample.metadata["reward_mapping_mode"] = REWARD_MAPPING_MODE
        sample.metadata["static_reward_mapping_mode"] = STATIC_REWARD_MAPPING_MODE
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=None,
            base_reward=-1.0,
        )
        _apply_dynamic_bound_context_metadata(
            sample,
            reward_mapping_result.get("context") or {},
            n_samples_per_prompt,
        )
        return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)

    # extract code from sample.response
    code = extract_code(raw_text)

    # Determine code category for CSV logging (not stored in DB)
    code_category = "valid"  # Default to valid
    mercy_raw_reward = None  # Only set for hack/empty cases (no extra sandbox call needed)
    static_raw_reward = 0.0
    static_mercy_raw_reward = 0.0
    dynamic_raw_reward = 0.0
    dynamic_mercy_raw_reward = 0.0
    metric_static_base_reward = 0.0
    metric_fallback_best = METRIC_STATIC_FALLBACK_SIGNED_BEST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    metric_fallback_worst = METRIC_STATIC_FALLBACK_SIGNED_WORST if METRIC_STATIC_BOUND_USE_CONST_FALLBACK else None
    metric_static_base_reward_has_bounds = int(
        has_static_bounds_with_priority(
            sample.metadata,
            priority="leaderboard",
            fallback_best=metric_fallback_best,
            fallback_worst=metric_fallback_worst,
        )
    )
    hack_category = "valid"
    hack_reason = ""
    sample.metadata["self_validation_enabled"] = bool(ENABLE_SELF_VALIDATION)

    # Check if code is empty or has no valid code
    if not code or len(code.strip()) == 0:
        code_category = "empty"
        logger.info("[CODE CATEGORY] empty")
    elif ENABLE_SELF_VALIDATION and not has_final_validation_score_print(code):
        code_category = "no_verify"
        logger.info("[CODE CATEGORY] no_verify")
    else:
        require_holdout_validation = bool(ENABLE_SELF_VALIDATION)
        # Check for hacking/cheating before sandbox execution.
        logger.info("Running hack_check on generated code...")
        # Keep hack checking ahead of sandbox execution. The asynchronous path
        # yields while waiting for the judge and caps it with HACK_CHECK_CONCURRENCY.
        is_valid_ml, hack_category, hack_reason = await hack_check_async(
            code,
            require_holdout_validation=require_holdout_validation,
        )
        if not is_valid_ml:
            code_category = hack_category
            logger.info(f"[CODE CATEGORY] {code_category}, reason: {hack_reason}")
        else:
            logger.info("[CODE CATEGORY] valid")

    is_hack = code_category in {"hack", "hack_verify"}

    # Persist hack as 0/1 for downstream metric aggregation.
    sample.metadata["hack"] = 1 if is_hack else 0
    sample.metadata["code_category"] = code_category

    score = None
    base_reward = 0.0  # Raw base reward (score->reward mapping) before any shaping/baseline
    effective_base_reward = 0.0  # Base reward after optional baseline adjustment
    shaped_base_reward = 0.0  # Stored base_reward: reward before extra improvement bonus
    final_reward = 0.0  # Final reward returned for training
    mercy_raw_reward = 0.0
    result_info = None
    # run_log = ""
    status_code = None
    payload = None
    dynamic_bound_context: dict[str, float | int | str | bool | None] | None = None
    running_time = 0.0
    run_log = ""
    validation_score = None
    validation_gap_info = validation_test_gap_info(None, None, sample.metadata)
    gap_penalty = 0.0
    gap_multiplier = 1.0

    # Handle different code categories
    if code_category == "empty":
        logger.warning("[EMPTY CODE] Response contains no code")
        score = None
        base_reward = -1
        effective_base_reward = -1
        shaped_base_reward = -1
        final_reward = -1
        result_info = "empty_code"
        status_code = -2  # Special status code for empty code
        payload = {"error": "empty_code", "reason": "No code generated", "running_time": 0.0}
        # Mercy raw reward: if not penalized, would be 0
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0

    elif code_category == "no_verify":
        logger.warning("[VERIFY CHECK FAILED] Missing Final Validation Score print in generated code")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = "no_verify"
        status_code = -3
        payload = {
            "error": "no_verify",
            "reason": "Missing print(f'Final Validation Score: {score}') style validation output",
        }
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0

    elif code_category == "hack":
        # For hack, give penalty but record mercy reward (0 instead of -1)
        logger.warning(f"[HACK DETECTED] Code is cheating/guessing: {hack_reason}")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = f"hack_detected:{hack_reason}"
        status_code = -1  # Special status code for hack detection
        payload = {"error": "hack_detected", "reason": hack_reason, "running_time": 0.0}
        # Mercy raw reward: if not penalized, would be 0
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0
        logger.info(f"[MERCY MODE] mercy_raw_reward: {mercy_raw_reward} (if not penalized)")

    elif code_category == "hack_verify":
        hack_reason_detail = hack_reason
        logger.warning(f"[HACK VERIFY DETECTED] Validation score is not from a real hold-out validation: {hack_reason_detail}")
        score = None
        base_reward = -0.5
        effective_base_reward = -0.5
        shaped_base_reward = -0.5
        final_reward = -0.5
        result_info = f"hack_verify:{hack_reason_detail}"
        status_code = -4
        payload = {"error": "hack_verify", "reason": hack_reason_detail}
        mercy_raw_reward = 0.0
        static_raw_reward = final_reward
        static_mercy_raw_reward = mercy_raw_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = mercy_raw_reward
        metric_static_base_reward = 0.0
        logger.info(f"[MERCY MODE] mercy_raw_reward: {mercy_raw_reward} (if not penalized)")

    else:
        # Valid code - execute sandbox normally

        # Use semaphore to limit concurrent sandbox executions
        async with _get_sandbox_semaphore():
            # Increment counter when entering
            async with _get_loop_lock("sandbox_concurrent_count"):
                SANDBOX_CONCURRENT_COUNT += 1
                current_count = SANDBOX_CONCURRENT_COUNT

            logger.info(
                f"🚀 [SANDBOX] Entering get_sandbox_result | Current concurrent: {current_count}/{MLE_CONFIGS['sandbox_concurrency']}"
            )
            logger.info(
                "[SANDBOX TIMEOUT] evaluation=%s job_timeout=%s wait_timeout=%s",
                evaluation,
                sandbox_job_timeout,
                sandbox_wait_timeout,
            )

            try:
                # Create and reuse one AsyncClient.
                limits = httpx.Limits(max_connections=16, max_keepalive_connections=16, keepalive_expiry=30.0)
                timeout = httpx.Timeout(connect=10, read=100, write=30, pool=30)
                if resource_type == "GPU":
                    async with httpx.AsyncClient(base_url=GPU_BASE_URL, limits=limits, timeout=timeout) as client:
                        sandbox_fn = get_proxy_sandbox_result if USE_PROXY_SANDBOX else get_sandbox_result
                        status_code, payload = await sandbox_fn(
                            client=client,
                            code_str=code,
                            data_dir=data_dir,
                            job_timeout=sandbox_job_timeout,
                            wait_timeout=sandbox_wait_timeout,
                            resource_type="gpu",
                        )
                else:
                    async with httpx.AsyncClient(base_url=GPU_BASE_URL, limits=limits, timeout=timeout) as client:
                        sandbox_fn = get_proxy_sandbox_result if USE_PROXY_SANDBOX else get_sandbox_result
                        status_code, payload = await sandbox_fn(
                            client=client,
                            code_str=code,
                            data_dir=data_dir,
                            job_timeout=sandbox_job_timeout,
                            wait_timeout=sandbox_wait_timeout,
                            resource_type="gpu",
                        )
            finally:

                # Decrement counter when exiting
                async with _get_loop_lock("sandbox_concurrent_count"):
                    SANDBOX_CONCURRENT_COUNT -= 1
                    current_count = SANDBOX_CONCURRENT_COUNT

                logger.info(
                    f"✅ [SANDBOX] Exiting get_sandbox_result | Current concurrent: {current_count}/{MLE_CONFIGS['sandbox_concurrency']}"
                )

        logger.info("\n== Run Result Summary ==")
        if status_code == 200:  # Finished.
            logger.info(f"job_status: {payload.get('status')}")
            result = payload.get("result") or {}
            running_time = payload.get("running_time", 0.0) or 0.0
            run_log = get_clear_log(result.get("run_log"))
            # logger.info(f"run_log: {run_log}")
            validation_score = extract_validation_score(run_log)
            logger.info(f"run_result_status: {result.get('result')}")
            result_info = result.get("result")
            logger.info(f"running_time: {running_time}")
            logger.info(f"validation_score: {validation_score}")
            # logger.info("score:", result.get("score"))
            score = result.get("score")
            # Map score to base reward
            if score is None:
                base_reward = 0.0
                logger.info(f"score is None, base_reward: {base_reward}")
            else:
                if USE_SCORE2REWARD:
                    base_reward = 0.0
                    logger.info(
                        "score: %s, deferring score2reward until group-level adaptive bound is resolved",
                        score,
                    )
                else:
                    base_reward = score
                    logger.info(f"score: {score}, base_reward: {base_reward} (reward equals score)")

        elif status_code == 503:  # Connection failed.
            logger.info(f"NOT 200, status_code: {status_code}")
            error_type = payload.get("type", "unknown")
            error_detail = payload.get("detail", "Connection failed")
            running_time = payload.get("running_time", 0.0) or 0.0
            logger.info(f"Connection error ({error_type}): {error_detail}")
            result_info = f"connection_error:{error_type}"
            run_log = f"Connection error: {error_type}"
            base_reward = 0.0

        else:  # Still running or another non-terminal state.
            logger.info(f"NOT 200, status_code: {status_code}")
            error_msg = payload.get("error", "unknown error")
            detail = payload.get("detail", {})
            running_time = payload.get("running_time", 0.0) or 0.0
            logger.info(f"Error status code: {status_code}")
            logger.info(f"Error: {error_msg}")
            if isinstance(detail, dict):
                logger.info(f"job_id: {detail.get('job_id')}")
                logger.info(f"run_status: {detail.get('status')}")
                if detail.get("message"):
                    logger.info(f"message: {detail['message']}")
                run_log = detail.get("message", str(detail))
            else:
                logger.info(f"detail: {detail}")
                run_log = str(detail)
            result_info = detail
            base_reward = 0.0

        sample.metadata["score"] = score
        sample.metadata["running_time"] = running_time
        sample.metadata["reward_mapping_mode"] = REWARD_MAPPING_MODE
        sample.metadata["static_reward_mapping_mode"] = STATIC_REWARD_MAPPING_MODE
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=score,
            base_reward=base_reward,
        )
        base_reward = float(reward_mapping_result["base_reward"])
        adaptive_context = reward_mapping_result.get("context") or {}
        _apply_dynamic_bound_context_metadata(sample, adaptive_context, n_samples_per_prompt)
        if bool(adaptive_context.get("group_generation_aborted")):
            sample.metadata["generation_abort_reason"] = "group_generation_aborted"
            return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)
        if USE_SCORE2REWARD:
            logger.info(
                "score: %s, adaptive_bound_mode=%s, best_signed=%s, worst_signed=%s, mapped_base_reward=%s",
                score,
                adaptive_context.get("bound_mode"),
                adaptive_context.get("best_signed"),
                adaptive_context.get("worst_signed"),
                base_reward,
            )

        static_base_reward_raw = compute_static_score_reward(score, sample.metadata)
        metric_static_base_reward = compute_metric_static_base_reward(score, sample.metadata)
        metric_static_base_reward_has_bounds = int(
            has_static_bounds_with_priority(
                sample.metadata,
                priority="leaderboard",
                fallback_best=metric_fallback_best,
                fallback_worst=metric_fallback_worst,
            )
        )
        validation_gap_info = validation_test_gap_info(validation_score, score, sample.metadata)

        dynamic_base_reward_before_gap = float(base_reward)
        static_base_reward_before_gap = float(static_base_reward_raw)
        metric_static_base_reward_before_gap = metric_static_base_reward

        base_reward, gap_penalty, gap_multiplier = apply_validation_test_gap_penalty(
            base_reward,
            validation_gap_info.get("relative_gap"),
            VALIDATION_TEST_GAP_PENALTY_COEF,
        )
        static_base_reward_raw, _, _ = apply_validation_test_gap_penalty(
            static_base_reward_raw,
            validation_gap_info.get("relative_gap"),
            VALIDATION_TEST_GAP_PENALTY_COEF,
        )
        if metric_static_base_reward is not None:
            metric_static_base_reward, _, _ = apply_validation_test_gap_penalty(
                metric_static_base_reward,
                validation_gap_info.get("relative_gap"),
                VALIDATION_TEST_GAP_PENALTY_COEF,
            )
        logger.info(
            "validation/test gap: validation=%s, test=%s, relative_gap=%s, penalty=%s, multiplier=%s",
            validation_gap_info.get("validation_score"),
            validation_gap_info.get("test_score"),
            validation_gap_info.get("relative_gap"),
            gap_penalty,
            gap_multiplier,
        )

        baseline_value = None
        baseline_count = 0
        effective_base_reward = float(base_reward)
        static_effective_base_reward = float(static_base_reward_raw)
        sample.metadata["validation_score"] = validation_gap_info.get("validation_score")
        sample.metadata["validation_test_abs_gap"] = validation_gap_info.get("abs_gap")
        sample.metadata["validation_test_relative_gap"] = validation_gap_info.get("relative_gap")
        sample.metadata["validation_test_gap_denominator"] = validation_gap_info.get("gap_denominator")
        sample.metadata["validation_test_gap_denominator_source"] = validation_gap_info.get("gap_denominator_source")
        sample.metadata["validation_test_gap_penalty_enabled"] = VALIDATION_TEST_GAP_PENALTY_ENABLED
        sample.metadata["validation_test_gap_penalty_piecewise_enabled"] = VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED
        sample.metadata["validation_test_gap_penalty_coef"] = VALIDATION_TEST_GAP_PENALTY_COEF
        sample.metadata["validation_test_gap_penalty_high_coef"] = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF
        sample.metadata["validation_test_gap_penalty_tolerance"] = VALIDATION_TEST_GAP_PENALTY_TOLERANCE
        sample.metadata["validation_test_gap_penalty"] = gap_penalty
        sample.metadata["validation_test_gap_multiplier"] = gap_multiplier
        sample.metadata["dynamic_base_reward_before_gap_penalty"] = dynamic_base_reward_before_gap
        sample.metadata["static_base_reward_before_gap_penalty"] = static_base_reward_before_gap
        sample.metadata["metric_static_base_reward_before_gap_penalty"] = metric_static_base_reward_before_gap
        sample.metadata["base_reward_raw"] = base_reward
        sample.metadata["base_reward_baseline_spec"] = "none"
        sample.metadata["base_reward_baseline_value"] = baseline_value
        sample.metadata["base_reward_baseline_count"] = baseline_count
        sample.metadata["base_reward_effective"] = effective_base_reward
        sample.metadata["dynamic_base_reward_raw"] = base_reward
        sample.metadata["dynamic_base_reward_effective"] = effective_base_reward
        sample.metadata["static_base_reward_raw"] = static_base_reward_raw
        sample.metadata["static_base_reward_effective"] = static_effective_base_reward
        sample.metadata["metric_static_base_reward"] = metric_static_base_reward
        sample.metadata["metric_static_base_reward_has_bounds"] = metric_static_base_reward_has_bounds

        # Calculate final reward based on mode
        shaped_base_reward, final_reward, delta, improve_bonus = compute_mode_rewards(
            generation_mode=generation_mode,
            effective_base_reward=effective_base_reward,
            parent_base_reward=parent_dynamic_base_reward,
            parent_reward=parent_dynamic_raw_reward,
            parent_code=parent_code,
            code=code,
            crossover_parent_base_reward=crossover_parent_dynamic_base_reward,
            crossover_parent_reward=crossover_parent_dynamic_raw_reward,
            crossover_parent_code=crossover_parent_code,
            improve_reward_strategy=IMPROVE_REWARD_STRATEGY,
            improve_delta_bonus_coef=IMPROVE_DELTA_BONUS_COEF,
        )
        static_shaped_base_reward, static_final_reward, _, _ = compute_mode_rewards(
            generation_mode=generation_mode,
            effective_base_reward=static_effective_base_reward,
            parent_base_reward=parent_static_base_reward,
            parent_reward=parent_static_raw_reward,
            parent_code=parent_code,
            code=code,
            crossover_parent_base_reward=crossover_parent_static_base_reward,
            crossover_parent_reward=crossover_parent_static_raw_reward,
            crossover_parent_code=crossover_parent_code,
            improve_reward_strategy=IMPROVE_REWARD_STRATEGY,
            improve_delta_bonus_coef=IMPROVE_DELTA_BONUS_COEF,
        )
        if generation_mode in ("improve", "debug", "crossover"):
            logger.info(
                f"[{generation_mode.upper()} MODE] base_reward={shaped_base_reward:.6f} delta={delta:.6f} bonus={improve_bonus:.6f} final_reward={final_reward:.6f}"
            )
        else:
            logger.info(f"[DRAFT MODE] base_reward/final_reward: {final_reward}")
        sample.metadata["improve_delta"] = delta
        sample.metadata["improve_bonus"] = improve_bonus

        sample.metadata["base_reward"] = shaped_base_reward
        sample.metadata["final_reward"] = final_reward
        sample.metadata["dynamic_base_reward"] = shaped_base_reward
        sample.metadata["static_base_reward"] = static_shaped_base_reward
        dynamic_raw_reward = final_reward
        dynamic_mercy_raw_reward = final_reward
        static_raw_reward = static_final_reward
        static_mercy_raw_reward = static_final_reward
        mercy_raw_reward = dynamic_mercy_raw_reward
        sample.metadata["dynamic_raw_reward"] = dynamic_raw_reward
        sample.metadata["dynamic_mercy_raw_reward"] = dynamic_mercy_raw_reward
        sample.metadata["static_raw_reward"] = static_raw_reward
        sample.metadata["static_mercy_raw_reward"] = static_mercy_raw_reward
        sample.metadata["mercy_raw_reward"] = mercy_raw_reward

    sample.metadata["running_time"] = running_time
    sample.metadata["score"] = score
    sample.metadata.setdefault("validation_score", validation_gap_info.get("validation_score"))
    sample.metadata.setdefault("validation_test_abs_gap", validation_gap_info.get("abs_gap"))
    sample.metadata.setdefault("validation_test_relative_gap", validation_gap_info.get("relative_gap"))
    sample.metadata.setdefault("validation_test_gap_denominator", validation_gap_info.get("gap_denominator"))
    sample.metadata.setdefault(
        "validation_test_gap_denominator_source",
        validation_gap_info.get("gap_denominator_source"),
    )
    sample.metadata.setdefault("validation_test_gap_penalty_enabled", VALIDATION_TEST_GAP_PENALTY_ENABLED)
    sample.metadata.setdefault(
        "validation_test_gap_penalty_piecewise_enabled",
        VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED,
    )
    sample.metadata.setdefault("validation_test_gap_penalty_coef", VALIDATION_TEST_GAP_PENALTY_COEF)
    sample.metadata.setdefault("validation_test_gap_penalty_high_coef", VALIDATION_TEST_GAP_PENALTY_HIGH_COEF)
    sample.metadata.setdefault("validation_test_gap_penalty_tolerance", VALIDATION_TEST_GAP_PENALTY_TOLERANCE)
    sample.metadata.setdefault("validation_test_gap_penalty", gap_penalty)
    sample.metadata.setdefault("validation_test_gap_multiplier", gap_multiplier)
    if code_category != "valid":
        sample.metadata["base_reward_raw"] = base_reward
        sample.metadata["base_reward_effective"] = effective_base_reward
        sample.metadata["base_reward"] = shaped_base_reward
        sample.metadata["final_reward"] = final_reward
        sample.metadata["mercy_raw_reward"] = mercy_raw_reward
        sample.metadata["dynamic_base_reward_raw"] = base_reward
        sample.metadata["dynamic_base_reward_effective"] = effective_base_reward
        sample.metadata["dynamic_base_reward"] = shaped_base_reward
        sample.metadata["dynamic_raw_reward"] = dynamic_raw_reward
        sample.metadata["dynamic_mercy_raw_reward"] = dynamic_mercy_raw_reward
        sample.metadata["static_base_reward_raw"] = base_reward
        sample.metadata["static_base_reward_effective"] = effective_base_reward
        sample.metadata["static_base_reward"] = shaped_base_reward
        sample.metadata["static_raw_reward"] = static_raw_reward
        sample.metadata["static_mercy_raw_reward"] = static_mercy_raw_reward
        sample.metadata["metric_static_base_reward"] = metric_static_base_reward
        sample.metadata["metric_static_base_reward_has_bounds"] = metric_static_base_reward_has_bounds
        reward_mapping_result = await _resolve_group_reward_mapping(
            sample.metadata,
            group_index=group_index,
            group_size=n_samples_per_prompt,
            sample_key=f"{task_id}:{group_index}:{id(sample)}",
            score=score,
            base_reward=base_reward,
        )
        dynamic_bound_context = reward_mapping_result.get("context") or {}
        _apply_dynamic_bound_context_metadata(sample, dynamic_bound_context, n_samples_per_prompt)
        if bool(dynamic_bound_context.get("group_generation_aborted")):
            sample.metadata["generation_abort_reason"] = "group_generation_aborted"
            return _return_generation_abort_for_retry(sample, task_name=task_name, group_index=group_index)

    task_reward_frontier = _compute_task_reward_frontier(db=await get_program_database(), task_name=task_name, metadata_fallback=sample.metadata)
    sample.metadata["task_best_dynamic_raw_reward_before"] = task_reward_frontier["best_dynamic_raw_reward_before"]
    sample.metadata["task_best_static_raw_reward_before"] = task_reward_frontier["best_static_raw_reward_before"]
    sample.metadata["dynamic_raw_reward_gap_to_task_best_before"] = dynamic_raw_reward - task_reward_frontier[
        "best_dynamic_raw_reward_before"
    ]
    sample.metadata["static_raw_reward_gap_to_task_best_before"] = static_raw_reward - task_reward_frontier[
        "best_static_raw_reward_before"
    ]
    sample.metadata["dynamic_raw_reward_is_task_best_improvement"] = int(
        dynamic_raw_reward > task_reward_frontier["best_dynamic_raw_reward_before"] + 1e-12
    )
    sample.metadata["static_raw_reward_is_task_best_improvement"] = int(
        static_raw_reward > task_reward_frontier["best_static_raw_reward_before"] + 1e-12
    )
    sample.metadata["task_best_metric_static_base_reward_before"] = task_reward_frontier[
        "best_metric_static_base_reward_before"
    ]
    sample.metadata["metric_static_base_reward_is_task_best_improvement"] = int(
        metric_static_base_reward_has_bounds
        and metric_static_base_reward >= task_reward_frontier["best_metric_static_base_reward_before"] - 1e-12
    )
    sample.metadata["history_best_static_base_reward"] = await _update_history_best_static_base_reward(
        task_name,
        task_reward_frontier["best_metric_static_base_reward_before"],
        metric_static_base_reward,
    )

    # Store status_code in sample.metadata for logging
    if "status_codes" not in sample.metadata:
        sample.metadata["status_codes"] = []
    sample.metadata["status_codes"].append(status_code)

    # Add program to database
    db = await get_program_database()
    program_id = None
    program = Program(
        task_id=task_id,
        task_name=task_name,
        status_code=status_code,
        payload=payload,
        code=code,
        score=score,
        running_time=running_time,
        reward=final_reward,  # Training reward (includes improve bonus when applicable)
        base_reward=shaped_base_reward,  # Original reward before extra improve/debug/crossover bonus
        parent_id=parent_id,
        parent_code=parent_code,
        generation_mode=generation_mode,
        raw_text=raw_text,
        hack=sample.metadata["hack"],  # Add hack check result
        metadata=sample.metadata,  # Keep metadata for additional info
    )
    # Store successful executions plus explicitly categorized invalid code we want to keep for analysis.
    # Connection/runtime failures and empty code are still skipped.
    should_store_program = status_code == 200 or code_category in {"hack", "no_verify", "hack_verify"}
    if should_store_program:
        program_id = db.add(program)
        logger.info(f"[DATABASE] Added program id={program_id} to database for task {task_name}")
        saved_program = db.get_by_id(program_id)
        exploit_coefficient = saved_program.exploit_coefficient if saved_program is not None else None
        explore_coefficient = saved_program.explore_coefficient if saved_program is not None else None
        cooling_coefficient = saved_program.cooling_coefficient if saved_program is not None else None
        if saved_program is not None:
            airaevo_updates = await _maybe_update_airaevo_program_metadata(args, db, saved_program)
            if airaevo_updates:
                sample.metadata.update(airaevo_updates)
    else:
        logger.info(
            f"[DATABASE] Skipped program insertion for task {task_name}: status_code={status_code}, code_category={code_category}"
        )
        exploit_coefficient = None
        explore_coefficient = None
        cooling_coefficient = None

    sample.metadata["dynamic_score_bounds_group_seen_count"] = sample.metadata.get("adaptive_reward_group_score_count")
    sample.metadata["dynamic_score_bounds_group_updated"] = True
    sample.metadata["dynamic_score_bounds_best_after"] = sample.metadata.get("dynamic_bound_best_signed")
    sample.metadata["dynamic_score_bounds_worst_after"] = sample.metadata.get("dynamic_bound_worst_signed")

    # Log training data for group tracking (CSV includes code_category and mercy_raw_reward)
    logger_instance.log_sample(
        sample=sample,
        score=score,
        reward=final_reward,  # Log the final reward used for training
        static_base_reward=metric_static_base_reward,
        base_reward=base_reward,  # Log base reward separately
        result_info=result_info,
        code=code,
        raw_text=raw_text,  # Log raw model response
        parent_id=parent_id,
        parent_code=parent_code,  # Log parent code
        mode=generation_mode,
        hack=sample.metadata["hack"],
        code_category=code_category,  # Log code category (CSV only)
        static_raw_reward=static_raw_reward,
        static_mercy_raw_reward=static_mercy_raw_reward,
        dynamic_raw_reward=dynamic_raw_reward,
        exploit_coefficient=exploit_coefficient,
        explore_coefficient=explore_coefficient,
        cooling_coefficient=cooling_coefficient,
    )

    return final_reward


async def save_db_snapshot(snapshot_path: str):
    """
    Save a snapshot of the program database.

    Args:
        snapshot_path: Path where the snapshot should be saved
    """
    db = await get_program_database()
    db.save_snapshot(snapshot_path)
