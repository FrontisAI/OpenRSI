"""Collect evaluated rollouts and rebuild SFT messages."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from tts_search.data_produce.common import (
    maybe_float,
    read_jsonl,
    read_text,
    safe_task_output_name,
)
from tts_search.prompt_builder import build_pass_k_draft_system_prompt

if TYPE_CHECKING:
    from tts_search.services.rejection import RejectionPolicy

STATUS_RE = re.compile(r"^\*\*Status\*\*:\s*(.+?)\s*$", re.MULTILINE)
RESULT_RE = re.compile(r"^\*\*Result\*\*:\s*(.+?)\s*$", re.MULTILINE)


def extract_feedback_status(feedback_text: str) -> str:
    """Extract sandbox status from formatted feedback.

    Args:
        feedback_text: Contents of a feedback.txt file.

    Returns:
        Lowercase status string, or ``unknown`` when absent.
    """
    match = STATUS_RE.search(feedback_text)
    return match.group(1).strip().lower() if match else "unknown"


def extract_feedback_result(feedback_text: str) -> str:
    """Extract sandbox result label from formatted feedback.

    Args:
        feedback_text: Contents of a feedback.txt file.

    Returns:
        Lowercase result string, or ``unknown`` when absent.
    """
    match = RESULT_RE.search(feedback_text)
    return match.group(1).strip().lower() if match else "unknown"


def feedback_is_success(
    feedback_text: str,
    *,
    status_values: set[str] | None = None,
    result_values: set[str] | None = None,
) -> bool:
    """Check whether feedback represents a successful run.

    Args:
        feedback_text: Contents of a feedback.txt file.
        status_values: Accepted status labels.
        result_values: Accepted result labels.

    Returns:
        True when both status and result match accepted values.
    """
    status_values = status_values or {"completed", "success"}
    result_values = result_values or {"success"}
    return (
        extract_feedback_status(feedback_text) in status_values
        and extract_feedback_result(feedback_text) in result_values
    )


def _build_rejection_policy(name: str | None):
    """Lazily import rejection policy builder to avoid package import cycles."""
    from tts_search.services.rejection import build_rejection_policy

    return build_rejection_policy(name=name)


def build_assistant_content(reasoning: str, code: str, response: str) -> str:
    """Build the assistant message for an SFT row.

    Args:
        reasoning: Optional model reasoning text.
        code: Extracted Python code.
        response: Raw model response fallback.

    Returns:
        Assistant content with reasoning and/or a Python code block.
    """
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


def load_prompt_source(
    source_parquet: Path,
    *,
    metadata_col: str = "metadata",
    prompt_col: str = "prompt",
) -> tuple[
    dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
]:
    """Load source prompts indexed by task id and task name.

    Args:
        source_parquet: Parquet file containing metadata and prompt columns.
        metadata_col: Column containing task metadata dictionaries.
        prompt_col: Column containing chat prompt messages.

    Returns:
        Two maps: task_id -> (metadata, prompt) and task_name -> (metadata, prompt).
    """
    df = pd.read_parquet(source_parquet, columns=[metadata_col, prompt_col])
    by_task_id: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    by_task_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for _, row in df.iterrows():
        metadata = dict(row[metadata_col])
        prompt_value = row[prompt_col]
        prompt = (
            prompt_value.tolist()
            if hasattr(prompt_value, "tolist")
            else list(prompt_value)
        )
        task_id = str(metadata.get("uuid", ""))
        task_name = str(metadata.get("task_name", ""))
        if task_id:
            by_task_id[task_id] = (metadata, prompt)
        if task_name:
            by_task_name[task_name] = (metadata, prompt)
    return by_task_id, by_task_name


def _step_key(row: dict[str, Any]) -> tuple[str, int]:
    """Build the join key for gen/eval step rows.

    Args:
        row: Serialized generation or evaluation result row.

    Returns:
        Tuple of task id/name and integer step index.
    """
    task_id = row.get("task_id")
    if task_id:
        return str(task_id), int(row.get("step_index", 0))
    return str(row.get("task_name", "")), int(row.get("step_index", 0))


def _rank_eval_row(row: dict[str, Any]) -> tuple[int, float, str]:
    """Rank duplicate eval rows for deterministic collapse.

    Args:
        row: Evaluation result row.

    Returns:
        Sort key preferring successful, higher-scoring, earlier rows.
    """
    status_order = {
        "success": 0,
        "scoring_failed": 1,
        "submission_missing": 2,
        "unknown": 3,
        "timeout": 4,
        "code_execution_error": 5,
    }
    status = str(row.get("status") or "unknown")
    score = maybe_float(row.get("score"))
    score_rank = -(score if score is not None else -1e18)
    return (status_order.get(status, 99), score_rank, str(row.get("timestamp", "")))


def load_eval_results(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Load and deduplicate eval_results.jsonl.

    Args:
        path: Path to eval_results.jsonl.

    Returns:
        Mapping from step key to selected eval row, plus load statistics.
    """
    rows = read_jsonl(path)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_rows = 0
    status_counts: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        row["score"] = maybe_float(row.get("score"))
        row["reward"] = maybe_float(row.get("reward"))
        row["step_index"] = int(row.get("step_index", 0))
        key = _step_key(row)
        status_counts[str(row.get("status") or "unknown")] += 1
        if key in by_key:
            duplicate_rows += 1
            if _rank_eval_row(row) < _rank_eval_row(by_key[key]):
                by_key[key] = row
        else:
            by_key[key] = row
    return by_key, {
        "raw_rows": len(rows),
        "unique_rows": len(by_key),
        "duplicate_rows_collapsed": duplicate_rows,
        "status_counts": dict(status_counts),
    }


def load_gen_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Load latest generation row for each task step.

    Args:
        path: Path to gen_results.jsonl.

    Returns:
        Mapping from step key to generation row.
    """
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = dict(raw)
        row["step_index"] = int(row.get("step_index", 0))
        key = _step_key(row)
        previous = by_key.get(key)
        if previous is None or str(row.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            by_key[key] = row
    return by_key


def load_gen_results_by_request_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load generation rows by request id, preserving same-step rollouts.

    Args:
        path: Path to gen_results.jsonl.

    Returns:
        Mapping from request_id to the latest generation row with that id.
    """
    by_request_id: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = dict(raw)
        row["step_index"] = int(row.get("step_index", 0))
        request_id = str(row.get("request_id") or "")
        if not request_id:
            continue
        previous = by_request_id.get(request_id)
        if previous is None or str(row.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            by_request_id[request_id] = row
    return by_request_id


def find_step_dir(
    run_dir: Path,
    *,
    task_name: str,
    task_id: str | None,
    step_index: int,
) -> Path:
    """Find the artifact directory for one generated step.

    Args:
        run_dir: Root evaluation run directory.
        task_name: Task name from the result row.
        task_id: Optional task UUID from the result row.
        step_index: Step index to locate.

    Returns:
        Existing step directory when found, otherwise the expected preferred path.
    """
    tasks_dir = run_dir / "tasks"
    candidates = []
    if task_id:
        candidates.append(tasks_dir / safe_task_output_name(task_name, task_id))
    candidates.append(tasks_dir / safe_task_output_name(task_name))
    for task_dir in candidates:
        step_dir = task_dir / f"step_{step_index}"
        if step_dir.exists():
            return step_dir
    return candidates[0] / f"step_{step_index}"


def step_dir_from_eval_row(
    eval_row: dict[str, Any],
    run_dir: Path,
    *,
    task_name: str,
    task_id: str,
    step_index: int,
) -> Path:
    """Resolve the artifact directory for an eval row.

    Args:
        eval_row: Serialized eval_results row.
        run_dir: Current run directory.
        task_name: Task display name.
        task_id: Task UUID.
        step_index: Rollout step index.

    Returns:
        The directory containing feedback/code artifacts, preferring the
        explicit feedback_path in eval_results when present.
    """
    feedback_path = eval_row.get("feedback_path")
    if feedback_path:
        path = Path(str(feedback_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.parent
    return find_step_dir(
        run_dir,
        task_name=task_name,
        task_id=task_id or None,
        step_index=step_index,
    )


def _source_prompt_for_row(
    *,
    task_id: str,
    task_name: str,
    by_task_id: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    by_task_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[str, str] | None:
    """Return source system/user prompt text for a result row.

    Args:
        task_id: Task UUID.
        task_name: Human-readable task name.
        by_task_id: Prompt source map keyed by task UUID.
        by_task_name: Prompt source map keyed by task name.

    Returns:
        ``(system_content, user_content)`` or None when source prompt is unavailable.
    """
    source = by_task_id.get(task_id) or by_task_name.get(task_name)
    if source is None:
        return None
    _, prompt = source
    if len(prompt) < 2:
        return None
    return str(prompt[0].get("content") or ""), str(prompt[1].get("content") or "")


def collect_sft_rows(
    *,
    run_dir: Path,
    source_parquet: Path,
    rejection_policy: "RejectionPolicy" | str | None = "accept_scored",
    min_reward: float | None = 0.0,
    eval_statuses: set[str] | None = None,
    require_feedback_success: bool = True,
    id_prefix: str = "sft_candidate",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect evaluated rollouts as SFT training rows.

    Args:
        run_dir: Pass@k output directory containing gen/eval results and task artifacts.
        source_parquet: Source prompt parquet used to reconstruct system/user messages.
        rejection_policy: Policy name or object deciding which eval rows are accepted.
        min_reward: Minimum reward threshold; None disables this filter.
        eval_statuses: Allowed sandbox statuses.
        require_feedback_success: Whether feedback.txt must report success.
        id_prefix: Prefix for generated training row ids.

    Returns:
        Tuple of collected SFT rows and summary counters/statistics.
    """

    eval_rows = read_jsonl(run_dir / "eval_results.jsonl")
    eval_stats = {
        "raw_rows": len(eval_rows),
        "status_counts": dict(
            Counter(str(row.get("status") or "unknown") for row in eval_rows)
        ),
    }
    gen_by_key = load_gen_results(run_dir / "gen_results.jsonl")
    gen_by_request_id = load_gen_results_by_request_id(run_dir / "gen_results.jsonl")
    by_task_id, by_task_name = load_prompt_source(source_parquet)
    policy = (
        _build_rejection_policy(name=rejection_policy)
        if isinstance(rejection_policy, str) or rejection_policy is None
        else rejection_policy
    )
    eval_statuses = eval_statuses or {"success"}

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for eval_row in eval_rows:
        key = _step_key(eval_row)
        if str(eval_row.get("status")) not in eval_statuses:
            counters["dropped_eval_status"] += 1
            continue
        reward = maybe_float(eval_row.get("reward")) or 0.0
        if min_reward is not None and reward <= min_reward:
            counters["dropped_reward"] += 1
            continue
        if not policy.accepts_record(eval_row).accepted:
            counters["dropped_rejection_policy"] += 1
            continue

        request_id = str(eval_row.get("request_id") or "")
        gen_row = gen_by_request_id.get(request_id) or gen_by_key.get(key, {})
        task_name = str(eval_row.get("task_name") or gen_row.get("task_name") or key[0])
        task_id = str(eval_row.get("task_id") or gen_row.get("task_id") or "")
        step_index = int(eval_row.get("step_index", gen_row.get("step_index", key[1])))
        step_dir = step_dir_from_eval_row(
            eval_row,
            run_dir,
            task_name=task_name,
            task_id=task_id,
            step_index=step_index,
        )
        feedback = read_text(step_dir / "feedback.txt")
        if require_feedback_success and not feedback_is_success(feedback):
            counters["dropped_feedback_not_success"] += 1
            continue

        source_prompt = _source_prompt_for_row(
            task_id=task_id,
            task_name=task_name,
            by_task_id=by_task_id,
            by_task_name=by_task_name,
        )
        source_system = source_prompt[0] if source_prompt else ""
        stored_user = read_text(step_dir / "user_prompt.md")
        if not stored_user and source_prompt:
            stored_user = source_prompt[1]

        system_prompt = (
            build_pass_k_draft_system_prompt(source_system) if source_system else ""
        )
        reasoning = read_text(step_dir / "reasoning.md")
        code = read_text(step_dir / "valid_code.py")
        response = read_text(step_dir / "response.md")
        assistant = build_assistant_content(reasoning, code, response)

        rows.append(
            {
                "id": f"{id_prefix}-{len(rows)}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": stored_user},
                    {"role": "assistant", "content": assistant},
                ],
                "task_id": task_id,
                "task_name": task_name,
                "request_id": eval_row.get("request_id") or gen_row.get("request_id"),
                "step_index": step_index,
                "step_dir": str(step_dir),
                "source_status": eval_row.get("status"),
                "feedback_status": extract_feedback_status(feedback),
                "feedback_result": extract_feedback_result(feedback),
                "score": maybe_float(eval_row.get("score")),
                "reward": maybe_float(eval_row.get("reward")),
                "submit_medal": eval_row.get("submit_medal"),
                "queue_time": maybe_float(eval_row.get("queue_time")),
                "run_time": maybe_float(eval_row.get("run_time")),
                "api_prompt_tokens": maybe_float(gen_row.get("prompt_tokens")),
                "api_completion_tokens": maybe_float(gen_row.get("completion_tokens")),
                "api_total_tokens": maybe_float(gen_row.get("total_tokens")),
                "feedback_path": str(step_dir / "feedback.txt"),
                "code_path": str(step_dir / "valid_code.py"),
                "reasoning_path": str(step_dir / "reasoning.md"),
                "response_path": str(step_dir / "response.md"),
                "prompt_path": str(step_dir / "user_prompt.md"),
            }
        )

    stats = {
        **eval_stats,
        "collected_rows": len(rows),
        "collected_tasks": len({row["task_id"] or row["task_name"] for row in rows}),
        "drop_counts": dict(counters),
    }
    return rows, stats
