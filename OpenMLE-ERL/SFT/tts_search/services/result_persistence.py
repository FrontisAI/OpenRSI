"""Shared helpers for durable run result files.

The pass@k scheduler and AIRA Evo runner both expose the same top-level
artifacts: gen_results.jsonl, eval_results.jsonl, progress.json, and
progress_history.jsonl.  These helpers keep the low-level file semantics
consistent across both paths.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None  # type: ignore[assignment]


class StreamingJSONLWriter:
    """Async JSONL writer used by the service scheduler."""

    def __init__(self, output_path: Path, append: bool = False):
        self._path = output_path
        self._lock = asyncio.Lock()
        self._count = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if append and self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                self._count = sum(1 for line in f if line.strip())
        else:
            self._path.write_text("", encoding="utf-8")

    async def write(self, data: dict[str, Any]) -> None:
        async with self._lock:
            append_jsonl_record(self._path, data)
            self._count += 1

    def get_count(self) -> int:
        return self._count


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Process-safe advisory lock for writers sharing a run directory."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file and flush it to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    with _file_lock(path):
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping blank or invalid lines."""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def build_compact_progress_snapshot(output_dir: Path) -> dict[str, Any]:
    """Build the compact progress_history row used by distillation runs."""

    gen_completed = 0
    gen_success = 0
    for row in iter_jsonl_records(output_dir / "gen_results.jsonl"):
        gen_completed += 1
        if bool(row.get("success")) and not row.get("error"):
            gen_success += 1

    eval_completed = 0
    eval_success = 0
    for row in iter_jsonl_records(output_dir / "eval_results.jsonl"):
        eval_completed += 1
        if bool(row.get("success")):
            eval_success += 1

    completed_tasks = 0
    medal_count = 0
    accepted_count = 0
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
        completed_tasks = int(progress.get("completed_tasks") or 0)
        for state in dict(progress.get("task_states") or {}).values():
            if not isinstance(state, dict):
                continue
            medal_count += int(state.get("medal") or 0)
            accepted_count += int(state.get("accepted") or 0)

    return {
        "timestamp": datetime.now().isoformat(),
        "generation": {
            "success": gen_success,
            "completed": gen_completed,
            "failed": gen_completed - gen_success,
        },
        "evaluation": {
            "success": eval_success,
            "completed": eval_completed,
            "failed": eval_completed - eval_success,
        },
        "completed_tasks": completed_tasks,
        "medal_count": medal_count,
        "accepted_count": accepted_count,
    }


def append_compact_progress_snapshot(output_dir: Path) -> dict[str, Any]:
    """Append one compact progress_history row and return it."""

    snapshot = build_compact_progress_snapshot(output_dir)
    append_jsonl_record(output_dir / "progress_history.jsonl", snapshot)
    return snapshot


def update_progress_snapshot(
    path: Path,
    *,
    task_id: str,
    task_state: dict[str, Any],
    total_tasks: int,
    history_path: Path | None = None,
) -> dict[str, Any]:
    """Merge one task state into progress.json and optionally append history."""

    total_tasks = int(total_tasks)
    with _file_lock(path):
        if path.exists():
            try:
                progress = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                progress = {}
        else:
            progress = {}

        task_states = dict(progress.get("task_states") or {})
        task_states[str(task_id)] = task_state
        completed_tasks = sum(
            1 for state in task_states.values() if bool(state.get("done"))
        )
        progress = {
            **progress,
            "total_tasks": max(total_tasks, len(task_states)),
            "completed_tasks": completed_tasks,
            "task_states": task_states,
        }
        atomic_write_json(path, progress)

    if history_path is not None:
        append_jsonl_record(
            history_path,
            {"timestamp": datetime.now().isoformat(), **progress},
        )
    return progress
