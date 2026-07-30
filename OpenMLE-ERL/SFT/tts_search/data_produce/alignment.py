"""Writers for SFT alignment and training datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tts_search.data_produce.common import json_safe, write_jsonl


def to_training_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the minimal training format expected by SFT jobs.

    Args:
        rows: SFT rows containing ``id`` and ``messages`` fields.

    Returns:
        DataFrame with one row per training conversation.
    """

    return pd.DataFrame(
        [{"id": row["id"], "messages": row["messages"]} for row in rows]
    )


def write_alignment_dataset(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    name: str,
    write_manifest: bool = True,
) -> dict[str, str]:
    """Write JSONL, parquet, and optional manifest for SFT rows.

    Args:
        rows: Selected SFT rows to write.
        output_dir: Directory for dataset files.
        name: Base filename for written artifacts.
        write_manifest: Whether to write metadata CSV excluding messages.

    Returns:
        Mapping of artifact type to written path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = to_training_frame(rows)
    parquet_path = output_dir / f"{name}.parquet"
    jsonl_path = output_dir / f"{name}.jsonl"
    train_df.to_parquet(parquet_path, index=False)
    write_jsonl(jsonl_path, train_df.to_dict("records"))

    paths = {
        "parquet": str(parquet_path),
        "jsonl": str(jsonl_path),
    }
    if write_manifest:
        manifest_rows = [
            {key: value for key, value in row.items() if key != "messages"}
            for row in rows
        ]
        manifest_path = output_dir / f"{name}_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        paths["manifest"] = str(manifest_path)
    return paths


def sample_tasks(
    rows: list[dict[str, Any]],
    *,
    fractions: tuple[float, ...] = (0.5, 0.25),
    seed: int = 20260502,
    task_key: str = "task_id",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Sample task ids and keep all rows for selected tasks.

    Args:
        rows: Candidate SFT rows.
        fractions: Fractions of unique tasks to sample.
        seed: Random seed for deterministic task sampling.
        task_key: Row field identifying the task.

    Returns:
        Mapping from sample label to sampled rows, plus sampling summary.
    """

    df = pd.DataFrame([{task_key: row[task_key], "id": row["id"]} for row in rows])
    task_ids = (
        df[task_key]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    shuffled = task_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    samples: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "seed": seed,
        "sampling_unit": task_key,
        "full": {
            "tasks": int(task_ids.size),
            "rows": int(len(rows)),
        },
        "samples": {},
    }
    for fraction in fractions:
        count = int(task_ids.size * fraction)
        selected_tasks = set(shuffled.iloc[:count].astype(str))
        label = f"{int(fraction * 100)}pct"
        selected_rows = [
            row for row in rows if str(row.get(task_key)) in selected_tasks
        ]
        samples[label] = selected_rows
        summary["samples"][label] = {
            "fraction": fraction,
            "tasks": len(selected_tasks),
            "rows": len(selected_rows),
        }
    return samples, summary


def write_sampling_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a JSON sampling summary.

    Args:
        path: Destination report path.
        summary: Sampling summary dictionary.

    Returns:
        None. Creates parent directories and overwrites the report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
