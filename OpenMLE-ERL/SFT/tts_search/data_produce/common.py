"""Common helpers for data production modules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def maybe_float(value: Any) -> float | None:
    """Convert a value to a finite float.

    Args:
        value: Raw scalar value from JSON/CSV/pandas.

    Returns:
        Finite float, or None when conversion fails or the value is NaN/Inf.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-serializable Python objects.

    Args:
        value: Arbitrary nested value that may contain numpy scalars or arrays.

    Returns:
        JSON-safe value with non-finite floats converted to None.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def read_text(path: Path) -> str:
    """Read text from disk if present.

    Args:
        path: File path to read.

    Returns:
        File contents, or an empty string when the file is missing.
    """
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records.

    Args:
        path: JSONL file path.

    Returns:
        List of decoded JSON object rows.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as newline-delimited JSON.

    Args:
        path: Destination JSONL path.
        rows: Iterable of dictionary records.

    Returns:
        None. Creates parent directories and overwrites the target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def safe_task_output_name(task_name: str, task_id: str | None = None) -> str:
    """Build a filesystem-safe task directory name.

    Args:
        task_name: Human-readable task name.
        task_id: Optional task UUID to make duplicate names unique.

    Returns:
        Sanitized task directory name.
    """
    safe_task_name = str(task_name).replace("/", "_").replace("\\", "_")
    if task_id is None:
        return safe_task_name
    safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
    return f"{safe_task_name}_{safe_task_id}"


def summary_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize numeric values for reports.

    Args:
        values: Iterable of raw numeric values.

    Returns:
        Count, mean, std, quantiles, min, and max; empty stats when no values are finite.
    """
    clean_values = []
    for value in values:
        number = maybe_float(value)
        if number is not None:
            clean_values.append(number)
    clean = np.array(clean_values, dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "p05": float(np.quantile(clean, 0.05)),
        "p25": float(np.quantile(clean, 0.25)),
        "p50": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "p95": float(np.quantile(clean, 0.95)),
        "max": float(clean.max()),
    }
