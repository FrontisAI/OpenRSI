from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __package__:
    from .common import merge_task_values
else:
    from common import merge_task_values

def format_size(size_bytes: int) -> str:
    """Return a human-readable size string."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def annotate_dataset_sizes(
    all_subdirs: list[Path] | None = None,
    csv_name: str = "overview.csv",
    timeout: int = 5,
) -> None:
    """Update Raw Size and Final Size columns in an overview CSV.

    The timeout argument is accepted for compatibility with the legacy API.
    """
    _ = timeout
    task_dirs = all_subdirs or []
    raw_sizes = []
    final_sizes = []

    for task_dir in tqdm(task_dirs, desc="Measuring directories"):
        task_path = Path(task_dir)
        raw_path = task_path / "raw"
        if not raw_path.exists():
            raw_path = task_path / "raw.txt"
        data_path = task_path / "data"

        raw_size = _directory_size(raw_path) if raw_path.is_dir() else raw_path.stat().st_size if raw_path.exists() else 0
        raw_sizes.append(format_size(raw_size))
        final_sizes.append(format_size(_directory_size(data_path)))

    csv_path = Path(csv_name)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        frame = pd.DataFrame({"Name": [Path(path).name for path in task_dirs]})

    frame = merge_task_values(
        frame,
        task_dirs,
        {"Raw Size": raw_sizes, "Final Size": final_sizes},
    )
    frame.to_csv(csv_path, index=False)
