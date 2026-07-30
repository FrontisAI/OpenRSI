"""Composable SFT data production pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tts_search.data_produce.alignment import write_alignment_dataset
from tts_search.data_produce.collect import collect_sft_rows
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    annotate_gap_rows,
    filter_by_relative_gap,
    load_task_meta,
    load_task_meta_from_parquet,
)
from tts_search.data_produce.selection import select_top_per_task
from tts_search.data_produce.token_filter import (
    TokenFilterConfig,
    filter_by_token_length,
)


@dataclass(frozen=True)
class DataProductionConfig:
    """Configuration for the end-to-end SFT data production pipeline.

    Args:
        run_dir: Pass@k evaluation output directory.
        source_parquet: Source prompt parquet for reconstructing conversations.
        task_manifest: Optional legacy CSV with task score-range metadata.
        task_metadata_parquet: Optional parquet whose metadata column is used for
            valid-test gap ranges. Defaults to source_parquet.
        metric_metadata_parquet: Optional metricprompt parquet used to attach
            ``self_valid_protocol`` for metric-aware gap calculation.
        tokenizer_model: Tokenizer path or model id for length filtering.
        output_dir: Directory where selected datasets are written.
        token_limit: Maximum total chat-template tokens.
        relative_gap_limit: Maximum validation/test relative gap.
        top_k_per_task: Number of selected rows per task.
        token_workers: Number of tokenization workers.

    Returns:
        Immutable pipeline configuration.
    """

    run_dir: Path
    source_parquet: Path
    tokenizer_model: Path | str
    output_dir: Path
    task_manifest: Path | None = None
    task_metadata_parquet: Path | None = None
    metric_metadata_parquet: Path | None = None
    token_limit: int = 32768
    relative_gap_limit: float = 0.12
    top_k_per_task: int = 4
    token_workers: int = 1


def build_sft_dataset(config: DataProductionConfig) -> dict[str, Any]:
    """Run collect -> token filter -> gap filter -> top-k -> write dataset.

    Args:
        config: Data production configuration.

    Returns:
        Summary statistics and output paths for each pipeline stage.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)

    collected, collect_stats = collect_sft_rows(
        run_dir=config.run_dir,
        source_parquet=config.source_parquet,
        rejection_policy="accept_scored",
        min_reward=0.0,
        require_feedback_success=True,
    )
    token_kept, token_dropped, token_stats = filter_by_token_length(
        collected,
        TokenFilterConfig(
            tokenizer_model=config.tokenizer_model,
            max_total_tokens=config.token_limit,
            workers=config.token_workers,
            keep_equal=True,
        ),
    )
    task_meta = (
        load_task_meta_from_parquet(
            config.task_metadata_parquet or config.source_parquet,
            metric_metadata_parquet=config.metric_metadata_parquet,
        )
        if config.task_manifest is None
        else load_task_meta(config.task_manifest)
    )
    gap_rows = annotate_gap_rows(
        token_kept,
        task_meta,
        GapFilterConfig(max_relative_gap=config.relative_gap_limit),
    )
    gap_kept, gap_dropped, gap_stats = filter_by_relative_gap(
        gap_rows,
        GapFilterConfig(max_relative_gap=config.relative_gap_limit),
    )
    selected, selection_stats = select_top_per_task(
        gap_kept,
        top_k=config.top_k_per_task,
        dedupe_field="reward",
        rejection_policy="accept_scored",
    )
    output_paths = write_alignment_dataset(
        selected,
        output_dir=config.output_dir,
        name="sft_selected",
    )
    return {
        "collect": collect_stats,
        "token_filter": token_stats,
        "token_dropped": len(token_dropped),
        "gap_filter": gap_stats,
        "gap_dropped": len(gap_dropped),
        "selection": selection_stats,
        "paths": output_paths,
    }
