"""Reusable data production utilities for SFT dataset construction."""

from tts_search.data_produce.alignment import (
    sample_tasks,
    to_training_frame,
    write_alignment_dataset,
)
from tts_search.data_produce.baseline_filter import (
    BaselineTokenGapConfig,
    BaselineTokenGapDecision,
    evaluate_baseline_token_gap,
)
from tts_search.data_produce.collect import collect_sft_rows
from tts_search.data_produce.gap_filter import (
    GapFilterConfig,
    annotate_gap_rows,
    choose_gap_denominator,
    choose_legacy_gap_denominator,
    classify_metric,
    enrich_tasks_with_metric_protocol,
    filter_by_relative_gap,
    load_task_meta_from_parquet,
    relative_gap_from_feedback,
)
from tts_search.data_produce.pipeline import DataProductionConfig, build_sft_dataset
from tts_search.data_produce.selection import (
    merge_new_with_old_data,
    select_top_per_task,
)
from tts_search.data_produce.token_filter import (
    TokenFilterConfig,
    count_chat_template_tokens,
    count_message_tokens,
    filter_by_token_length,
    load_tokenizer,
    token_count_within_limit,
)

__all__ = [
    "DataProductionConfig",
    "GapFilterConfig",
    "TokenFilterConfig",
    "BaselineTokenGapConfig",
    "BaselineTokenGapDecision",
    "annotate_gap_rows",
    "build_sft_dataset",
    "choose_gap_denominator",
    "choose_legacy_gap_denominator",
    "classify_metric",
    "collect_sft_rows",
    "count_chat_template_tokens",
    "count_message_tokens",
    "enrich_tasks_with_metric_protocol",
    "evaluate_baseline_token_gap",
    "filter_by_relative_gap",
    "filter_by_token_length",
    "load_task_meta_from_parquet",
    "load_tokenizer",
    "merge_new_with_old_data",
    "relative_gap_from_feedback",
    "sample_tasks",
    "select_top_per_task",
    "token_count_within_limit",
    "to_training_frame",
    "write_alignment_dataset",
]
