from __future__ import annotations

import csv
import io
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypeVar

from tqdm import tqdm

from .common import (
    atomic_write_text,
    collect_task_dirs,
    ensure_parent,
    validate_task_name,
)
from .metric_validation import (
    evaluate_sample_submission,
    evaluate_sample_submission_process,
)
from .llm_config import eval_llm_config
from .process_runner import run_task_process

T = TypeVar("T")
R = TypeVar("R")


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


FIELDNAMES = [
    "Name",
    "Modality",
    "Task",
    "Raw Size",
    "Final Size",
    "CPU/GPU",
    "Source",
    "Metric",
    "NOTE",
    "Server Location",
]


def _require_llm_env() -> None:
    try:
        eval_llm_config().require("Evaluation")
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}. Use --skip-llm for offline metadata.") from exc


def _chunked_map(
    items: list[T],
    worker_count: int,
    func: Callable[[T], R],
    desc: str,
) -> list[R]:
    """Run work in deterministic chunks and return results in input order."""
    if not items:
        return []
    worker_count = max(1, min(worker_count, len(items)))
    if worker_count == 1:
        return [func(item) for item in tqdm(items, desc=desc)]

    chunk_size = max(1, math.ceil(len(items) / worker_count))
    indexed_items = list(enumerate(items))
    chunks = [
        indexed_items[index:index + chunk_size]
        for index in range(0, len(indexed_items), chunk_size)
    ]
    results: list[R | None] = [None] * len(items)

    def process_chunk(chunk: list[tuple[int, T]]) -> list[tuple[int, R]]:
        return [(index, func(item)) for index, item in chunk]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            for index, result in future.result():
                results[index] = result

    if any(result is None for result in results):
        raise RuntimeError("Internal error: missing chunked metadata result")
    return [result for result in results if result is not None]


def _read_overview_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_overview_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    atomic_write_text(path, "\ufeff" + output.getvalue())


def _run_metadata_pipeline(
    task_dirs: list[Path],
    output_csv: Path,
    workers: int,
    execution_mode: str = "process",
    task_timeout: float = 600,
) -> None:
    """Update the base CSV with metadata pipeline LLM and metric steps."""
    _require_llm_env()
    worker_count = max(1, min(workers, len(task_dirs)))
    rows = _read_overview_rows(output_csv)
    row_by_name = {row.get("Name", ""): row for row in rows}
    task_names = [task_dir.name for task_dir in task_dirs]
    if len(row_by_name) != len(rows):
        raise ValueError("Overview CSV contains duplicate task names")
    if set(row_by_name) != set(task_names):
        raise ValueError("Overview CSV task names do not match task directories")
    def analyze_in_process(task_dir: Path) -> dict[str, Any]:
        outcome = run_task_process(
            "overview",
            {
                "task_dir": str(task_dir),
                "skip_llm": False,
                "metric_execution_mode": execution_mode,
                "metric_timeout": task_timeout,
            },
            timeout=task_timeout,
            execution_mode="process",
        )
        if not outcome.ok or not isinstance(outcome.result, dict):
            raise RuntimeError(outcome.error or "Metadata worker returned an invalid result")
        return outcome.result

    updated = _chunked_map(task_dirs, worker_count, analyze_in_process, "metadata")
    updated_by_name = {row["Name"]: row for row in updated}
    if set(updated_by_name) != set(task_names):
        raise RuntimeError("Metadata results do not match task directories")
    _write_overview_rows(output_csv, [updated_by_name[name] for name in task_names])


def analyze_task(
    task_dir: Path,
    skip_llm: bool = False,
    metric_execution_mode: str = "process",
    metric_timeout: float = 120,
) -> dict[str, Any]:
    public_dir = task_dir / "data" / "public"
    raw_path = task_dir / "raw"
    if not raw_path.exists():
        raw_path = task_dir / "raw.txt"

    row: dict[str, Any] = {
        "Name": task_dir.name,
        "Modality": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        "Task": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        "Raw Size": _format_size(_directory_size(raw_path)) if raw_path.is_dir() else (
            _format_size(raw_path.stat().st_size) if raw_path.exists() else "0.00 B"
        ),
        "Final Size": _format_size(_directory_size(task_dir / "data")),
        "CPU/GPU": "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM",
        "Source": "OpenMLE",
        "Metric": "",
        "NOTE": "",
        "Server Location": str(task_dir),
    }

    desc = public_dir / "description.txt"
    if not desc.exists():
        row["NOTE"] = "#DESCRIPTION_NOT_FOUND"

    try:
        if metric_execution_mode == "isolated":
            result = evaluate_sample_submission_process(
                task_dir,
                execution_mode="isolated",
                task_timeout=metric_timeout,
            )
        else:
            result = evaluate_sample_submission(task_dir)
        row["Metric"] = f"@Evaluation result: {{'score': {result['score']}}}"
    except Exception as exc:
        if metric_execution_mode == "isolated":
            raise
        row["Metric"] = f"#MEVA_RUNTIME_ERROR: {type(exc).__name__}: {exc}"
    return row


def _analyze_task_full(
    task_dir: Path,
    skip_llm: bool = False,
    metric_execution_mode: str = "process",
    metric_timeout: float = 120,
) -> dict[str, Any]:
    """Analyze one task entirely inside its task worker process."""
    row = analyze_task(
        task_dir,
        skip_llm=skip_llm,
        metric_execution_mode=metric_execution_mode,
        metric_timeout=metric_timeout,
    )
    if skip_llm:
        return row
    _require_llm_env()
    from metadata_pipeline import (
        compute_requirement_classifier,
        sample_submission_validator,
        task_categorizer,
    )

    category = task_categorizer.extract_choice_with_llm(str(task_dir))
    parts = category.split("@")
    row["Modality"] = parts[0].strip() if parts else category
    row["Task"] = parts[1].strip() if len(parts) > 1 else category
    row["CPU/GPU"] = compute_requirement_classifier.extract_compute_requirement_with_llm(
        str(task_dir),
        row.get("Final Size", "Unknown"),
    )
    metric_result, _ = sample_submission_validator.validate_sample_submission(task_dir)
    row["Metric"] = metric_result
    return row


def _failed_overview_row(task_dir: Path, reason: str, skip_llm: bool) -> dict[str, Any]:
    marker = "#SKIPPED_LLM" if skip_llm else "#PENDING_LLM"
    return {
        "Name": task_dir.name,
        "Modality": marker,
        "Task": marker,
        "Raw Size": "0.00 B",
        "Final Size": "0.00 B",
        "CPU/GPU": marker,
        "Source": "OpenMLE",
        "Metric": f"#MEVA_RUNTIME_ERROR: {reason}",
        "NOTE": f"#TASK_RUNTIME_ERROR: {reason}",
        "Server Location": str(task_dir),
    }


def overview_has_task_failures(path: str | Path) -> bool:
    return any(
        row.get("NOTE", "").startswith("#TASK_RUNTIME_ERROR:")
        for row in _read_overview_rows(Path(path))
    )


def generate_overview(
    tasks_root: str | Path,
    output_csv: str | Path,
    workers: int = 1,
    skip_llm: bool = False,
    execution_mode: str = "process",
    task_timeout: float = 600,
) -> Path:
    tasks_root_path = Path(tasks_root).resolve()
    task_dirs = collect_task_dirs(tasks_root_path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories containing data/ found under {tasks_root}")

    worker_count = max(1, min(workers, len(task_dirs)))
    def run_one_impl(task_dir: Path) -> dict[str, Any]:
        try:
            validate_task_name(task_dir.name)
            task_dir.resolve().relative_to(tasks_root_path)
        except ValueError as exc:
            return _failed_overview_row(task_dir, str(exc), skip_llm)
        outcome = run_task_process(
            "overview",
            {
                "task_dir": str(task_dir),
                "skip_llm": skip_llm,
                "metric_execution_mode": execution_mode,
                "metric_timeout": task_timeout,
            },
            timeout=task_timeout,
            execution_mode="process",
            readonly_paths=(task_dir,),
        )
        if outcome.stdout:
            print(outcome.stdout, end="")
        if outcome.stderr:
            print(outcome.stderr, end="", file=sys.stderr)
        if (
            not outcome.ok
            or not isinstance(outcome.result, dict)
            or outcome.result.get("Name") != task_dir.name
        ):
            return _failed_overview_row(
                task_dir,
                outcome.error or "Task worker returned an invalid task result",
                skip_llm,
            )
        return outcome.result

    def run_one(task_dir: Path) -> dict[str, Any]:
        try:
            return run_one_impl(task_dir)
        except Exception as exc:
            return _failed_overview_row(
                task_dir,
                f"{type(exc).__name__}: {exc}",
                skip_llm,
            )

    rows = _chunked_map(task_dirs, worker_count, run_one, "overview")

    target = ensure_parent(output_csv)
    _write_overview_rows(target, sorted(rows, key=lambda item: item["Name"]))
    return target
