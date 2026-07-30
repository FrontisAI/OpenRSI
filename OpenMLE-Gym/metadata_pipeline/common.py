from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def merge_task_values(
    frame: pd.DataFrame,
    task_dirs: list[str | Path],
    values: dict[str, list[Any]],
) -> pd.DataFrame:
    """Assign metadata by task name rather than directory iteration order."""
    if "Name" not in frame.columns:
        raise ValueError("Metadata CSV is missing the Name column")
    names = [Path(task_dir).name for task_dir in task_dirs]
    if len(set(names)) != len(names):
        raise ValueError("Task directory list contains duplicate task names")
    csv_names = frame["Name"].astype(str).tolist()
    if len(set(csv_names)) != len(csv_names):
        raise ValueError("Metadata CSV contains duplicate task names")
    if set(names) != set(csv_names):
        raise ValueError("Metadata CSV task names do not match task directories")
    for column, column_values in values.items():
        if len(column_values) != len(names):
            raise ValueError(
                f"Metadata result count for {column} does not match task count"
            )
        by_name = dict(zip(names, column_values, strict=True))
        frame[column] = frame["Name"].map(by_name)
    return frame
