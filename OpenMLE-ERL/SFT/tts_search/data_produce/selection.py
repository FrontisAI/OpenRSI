"""Reward deduplication, top-k selection, and old/new dataset merge helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from tts_search.services.rejection import RejectionPolicy


def _build_rejection_policy(**kwargs):
    """Lazily import rejection policy builder to avoid package import cycles."""
    from tts_search.services.rejection import build_rejection_policy

    return build_rejection_policy(**kwargs)


def _sort_rows_for_selection(
    df: pd.DataFrame,
    *,
    task_key: str,
    rank_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Sort rows deterministically before per-task selection.

    Args:
        df: Candidate row DataFrame.
        task_key: Column grouping rows by task.
        rank_fields: Descending score/reward fields used for ranking.

    Returns:
        Stable sorted DataFrame.
    """
    sort_cols = [task_key]
    ascending = [True]
    for field in rank_fields:
        if field in df.columns:
            sort_cols.append(field)
            ascending.append(False)
    for field in ("relative_gap", "token_length", "step_index", "id"):
        if field in df.columns and field not in sort_cols:
            sort_cols.append(field)
            ascending.append(True)
    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def select_top_per_task(
    rows: list[dict[str, Any]],
    *,
    top_k: int = 4,
    task_key: str = "task_id",
    dedupe_field: str | None = "reward",
    rank_fields: tuple[str, ...] = ("reward", "score"),
    rejection_policy: "RejectionPolicy" | str | None = "accept_scored",
    rejection_score_threshold: float | None = None,
    rejection_reward_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply rejection policy, dedupe, then keep top-k rows per task.

    Args:
        rows: Candidate SFT row dictionaries.
        top_k: Maximum number of rows retained per task.
        task_key: Field used to group rows by task.
        dedupe_field: Optional field used to drop duplicate per-task values.
        rank_fields: Descending fields used to rank rows before selection.
        rejection_policy: Policy name or object for accepting rows.
        rejection_score_threshold: Optional score threshold for score policies.
        rejection_reward_threshold: Optional reward threshold for reward policies.

    Returns:
        Selected rows and selection statistics.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    policy = (
        _build_rejection_policy(
            name=rejection_policy,
            score_threshold=rejection_score_threshold,
            reward_threshold=rejection_reward_threshold,
        )
        if isinstance(rejection_policy, str) or rejection_policy is None
        else rejection_policy
    )
    policy_rows = [row for row in rows if policy.accepts_record(row).accepted]
    if not policy_rows:
        return [], {
            "input_rows": len(rows),
            "policy_rows": 0,
            "selected_rows": 0,
            "selected_tasks": 0,
        }

    df = pd.DataFrame(policy_rows)
    if task_key not in df.columns:
        raise KeyError(f"Missing task key column: {task_key}")
    sorted_df = _sort_rows_for_selection(
        df,
        task_key=task_key,
        rank_fields=rank_fields,
    )

    if dedupe_field is not None and dedupe_field in sorted_df.columns:
        deduped = sorted_df.drop_duplicates(
            subset=[task_key, dedupe_field],
            keep="first",
        )
    else:
        deduped = sorted_df
    selected = deduped.groupby(task_key, group_keys=False).head(top_k)
    selected = selected.reset_index(drop=True)

    stats = {
        "input_rows": len(rows),
        "policy_rows": len(policy_rows),
        "rows_after_dedupe": int(len(deduped)),
        "selected_rows": int(len(selected)),
        "input_tasks": int(df[task_key].nunique()),
        "selected_tasks": int(selected[task_key].nunique()),
        "dedupe_field": dedupe_field,
        "top_k": top_k,
        "per_task_selected_count_distribution": {
            str(k): int(v)
            for k, v in selected.groupby(task_key)
            .size()
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return selected.to_dict("records"), stats


def merge_new_with_old_data(
    *,
    new_train: pd.DataFrame,
    new_manifest: pd.DataFrame,
    old_train: pd.DataFrame,
    old_manifest: pd.DataFrame,
    task_key: str = "task_id",
    drop_new_when_new_le: int | None = 1,
    drop_new_when_old_between: tuple[int, int] | None = (3, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Merge new SFT rows with old rows while handling task overlap.

    Args:
        new_train: New training DataFrame with ``id`` and messages.
        new_manifest: New row manifest containing task ids.
        old_train: Existing training DataFrame.
        old_manifest: Existing manifest containing task ids.
        task_key: Manifest field identifying tasks.
        drop_new_when_new_le: Drop weak new overlap when new count is at most this.
        drop_new_when_old_between: Old-count interval for dropping weak new overlap.

    Returns:
        Merged training DataFrame, merged manifest, and merge statistics.
    """

    new_manifest = new_manifest.copy()
    old_manifest = old_manifest.copy()
    new_train = new_train.copy()
    old_train = old_train.copy()
    original_new_rows = len(new_train)
    original_old_rows = len(old_train)
    new_manifest[task_key] = new_manifest[task_key].astype(str)
    old_manifest[task_key] = old_manifest[task_key].astype(str)

    new_counts = new_manifest.groupby(task_key).size()
    old_counts = old_manifest.groupby(task_key).size()
    overlap_task_ids = set(new_counts.index) & set(old_counts.index)

    drop_new_task_ids: set[str] = set()
    if drop_new_when_new_le is not None and drop_new_when_old_between is not None:
        low, high = drop_new_when_old_between
        drop_new_task_ids = {
            task_id
            for task_id in overlap_task_ids
            if new_counts[task_id] <= drop_new_when_new_le
            and low <= old_counts[task_id] <= high
        }

    if drop_new_task_ids:
        new_manifest = new_manifest.loc[~new_manifest[task_key].isin(drop_new_task_ids)]
        new_train = new_train.loc[new_train["id"].isin(set(new_manifest["id"]))]

    final_new_task_ids = set(new_manifest[task_key].astype(str))
    old_keep_mask = ~old_manifest[task_key].isin(final_new_task_ids)
    old_manifest_kept = old_manifest.loc[old_keep_mask].copy()
    old_train_kept = old_train.loc[old_train["id"].isin(set(old_manifest_kept["id"]))]

    merged_train = pd.concat(
        [
            new_train.assign(dataset_source="new"),
            old_train_kept.assign(dataset_source="old"),
        ],
        ignore_index=True,
    )
    merged_manifest = pd.concat(
        [
            new_manifest.assign(dataset_source="new"),
            old_manifest_kept.assign(dataset_source="old"),
        ],
        ignore_index=True,
        sort=False,
    )

    stats = {
        "new_input_rows": int(original_new_rows),
        "new_input_tasks": int(new_counts.size),
        "old_input_rows": int(original_old_rows),
        "old_input_tasks": int(old_counts.size),
        "overlap_tasks_before_rule": int(len(overlap_task_ids)),
        "new_tasks_dropped_by_overlap_rule": int(len(drop_new_task_ids)),
        "new_rows_after_overlap_rule": int(len(new_train)),
        "old_rows_removed_by_new_task_id": int(original_old_rows - len(old_train_kept)),
        "old_rows_remaining": int(len(old_train_kept)),
        "merged_rows": int(len(merged_train)),
        "merged_tasks": int(merged_manifest[task_key].nunique()),
    }
    return merged_train, merged_manifest, stats
