from __future__ import annotations

from pathlib import Path

import pandas as pd


def initialize_overview_table(
    all_subdirs: list[Path] | None = None,
    csv_name: str = "overview.csv",
) -> None:
    """Create the base overview CSV expected by the legacy overview steps."""
    task_dirs = all_subdirs or []
    frame = pd.DataFrame(
        {
            "Name": [Path(task_dir).name for task_dir in task_dirs],
            "Modality": ["" for _ in task_dirs],
            "Task": ["" for _ in task_dirs],
            "Raw Size": ["" for _ in task_dirs],
            "Final Size": ["" for _ in task_dirs],
            "CPU/GPU": ["" for _ in task_dirs],
            "Source": ["OpenMLE" for _ in task_dirs],
            "Metric": ["" for _ in task_dirs],
            "NOTE": ["" for _ in task_dirs],
            "Server Location": [str(task_dir) for task_dir in task_dirs],
        }
    )
    frame.to_csv(csv_name, index=False)
    print(f"Processed {len(task_dirs)} task directories")
    print(f"Overview CSV written to: {csv_name}")


if __name__ == "__main__":
    pass
