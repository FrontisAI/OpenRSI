"""Validation-test gap extraction and filtering."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tts_search.data_produce.common import maybe_float, read_text, summary_stats

SCORE_RE = re.compile(r"^\*\*Score\*\*:\s*([-+eE0-9.]+)\s*$", re.MULTILINE)
FINAL_SCORE_RE = re.compile(r"^Final Score:\s*([-+eE0-9.]+)\s*$", re.MULTILINE)
PREFIX_SCORE_RE = re.compile(r"##SCORE##([-+eE0-9.]+)")
VAL_SCORE_RE = re.compile(r"Final Validation Score:\s*([-+eE0-9.]+)")
STATUS_RE = re.compile(r"^\*\*Status\*\*:\s*(.+?)\s*$", re.MULTILINE)
RESULT_RE = re.compile(r"^\*\*Result\*\*:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TaskMeta:
    """Task score-range metadata used by validation/test gap filtering.

    Args:
        task_name: Task name used for lookup.
        higher_is_better: Metric direction when known.
        theoretical_min: Theoretical minimum metric value.
        theoretical_max: Theoretical maximum metric value.
        leaderboard_min: Observed leaderboard minimum.
        leaderboard_max: Observed leaderboard maximum.
        task_id: Optional task UUID used for exact parquet metadata lookup.
        metric_*: Optional fields usually populated from the sibling
            ``*_metricprompt`` parquet.

    Returns:
        Immutable metadata object.
    """

    task_name: str
    higher_is_better: bool | None
    theoretical_min: float | None
    theoretical_max: float | None
    leaderboard_min: float | None
    leaderboard_max: float | None
    task_id: str | None = None
    metric_label: str | None = None
    metric_name: str | None = None
    metric_direction: str | None = None
    metric_class: str | None = None
    validation_strategy: str | None = None
    metric_source: str | None = None
    matched_text: str | None = None


@dataclass(frozen=True)
class GapFilterConfig:
    """Configuration for relative validation/test gap filtering.

    Args:
        max_relative_gap: Maximum accepted relative gap.
        theoretical_small_range_max: Threshold for trusting small theoretical ranges.
        big_range_threshold: Threshold where ranges are treated as too large.
        no_range_is_big: Use score magnitudes when no range metadata exists.
        require_comparable: Drop rows where relative gap cannot be computed.
        require_feedback_success: Drop rows without successful feedback.
        unitless_loss_floor: Minimum denominator for unitless lower-is-better
            losses such as RMSLE, log loss, and NLL.
        use_metric_aware_denominator: Use the new metric-aware denominator rules.

    Returns:
        Immutable gap filtering configuration.
    """

    max_relative_gap: float = 0.12
    theoretical_small_range_max: float = 2.0
    big_range_threshold: float = 100.0
    no_range_is_big: bool = True
    require_comparable: bool = True
    require_feedback_success: bool = True
    unitless_loss_floor: float = 1.0
    use_metric_aware_denominator: bool = True


def load_task_meta(path: Path) -> dict[str, TaskMeta]:
    """Load task metadata from a CSV manifest.

    Args:
        path: CSV path containing task names and score range columns.

    Returns:
        Mapping from task name to TaskMeta.
    """
    meta: dict[str, TaskMeta] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            higher = None
            if row.get("higher_is_better"):
                higher = row["higher_is_better"].strip().upper() == "TRUE"
            item = TaskMeta(
                task_name=row["task_name"],
                higher_is_better=higher,
                theoretical_min=maybe_float(row.get("theoretical_min")),
                theoretical_max=maybe_float(row.get("theoretical_max")),
                leaderboard_min=maybe_float(row.get("leaderboard_min")),
                leaderboard_max=maybe_float(row.get("leaderboard_max")),
                task_id=str(row.get("task_id") or row.get("uuid") or "") or None,
                metric_label=row.get("metric_label") or None,
                metric_name=row.get("metric_name") or None,
                metric_direction=row.get("metric_direction") or None,
                metric_class=classify_metric(
                    row.get("metric_label"), row.get("metric_name")
                ),
                validation_strategy=row.get("validation_strategy") or None,
                metric_source=row.get("metric_source") or None,
                matched_text=row.get("matched_text") or None,
            )
            meta[item.task_name] = item
            if item.task_id:
                meta[item.task_id] = item
    return meta


def _parse_self_valid_protocol(value: Any) -> dict[str, Any]:
    """Return a normalized self-validation protocol dictionary."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def classify_metric(metric_label: Any, metric_name: Any = None) -> str:
    """Classify a metric into a gap-normalization family.

    Args:
        metric_label: Short metric label, for example ``rmsle``.
        metric_name: Human-readable metric name.

    Returns:
        Metric class string used by denominator selection.
    """
    text = f"{metric_label or ''} {metric_name or ''}".lower()
    text = text.replace("_", " ").replace("-", " ")
    if "smape" in text:
        return "bounded_percentage_error_lower_better"
    if any(key in text for key in ("roc auc", "auroc", "auc")):
        return "bounded_probabilistic_score_0_1"
    if any(
        key in text
        for key in (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "jaccard",
            "dice",
            "iou",
            "map@",
            " map ",
            "ndcg",
            "mrr",
        )
    ):
        return "bounded_score_0_1_or_percent"
    if any(
        key in text
        for key in ("qwk", "quadratic weighted kappa", "cohen", "kappa", "mcc")
    ):
        return "bounded_agreement_minus1_1"
    if any(key in text for key in ("spearman", "pearson", "correlation", "r2")):
        return "bounded_correlation_or_r2"
    if any(key in text for key in ("rmsle", "msle")):
        return "log_target_error_lower_better"
    if any(key in text for key in ("log loss", "logloss", "nll", "cross entropy")):
        return "probabilistic_loss_lower_better"
    if "mape" in text:
        return "percentage_error_lower_better"
    if any(key in text for key in ("pinball", "quantile", "crps")):
        return "target_scale_distribution_loss_lower_better"
    if any(
        key in text
        for key in ("rmse", "mae", "mse", "rmspe", "rmsse", "error", "loss")
    ):
        return "target_scale_error_lower_better"
    return "unknown_or_unbounded"


def _metric_fields_from_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Extract metric classification fields from a parquet metadata row."""
    protocol = _parse_self_valid_protocol(row.get("self_valid_protocol"))
    label = protocol.get("metric_label") or protocol.get("metric_name")
    name = protocol.get("metric_name")
    return {
        "metric_label": str(label).lower() if label else None,
        "metric_name": str(name) if name else None,
        "metric_direction": (
            str(protocol.get("metric_direction"))
            if protocol.get("metric_direction") is not None
            else None
        ),
        "metric_class": classify_metric(label, name) if label or name else None,
        "validation_strategy": (
            str(protocol.get("validation_strategy"))
            if protocol.get("validation_strategy") is not None
            else None
        ),
        "metric_source": (
            str(protocol.get("source")) if protocol.get("source") is not None else None
        ),
        "matched_text": (
            str(protocol.get("matched_text"))
            if protocol.get("matched_text") is not None
            else None
        ),
    }


def infer_metricprompt_parquet(path: Path) -> Path | None:
    """Infer a sibling ``*_metricprompt/train.parquet`` path when it exists."""
    if path.name != "train.parquet":
        return None
    parent = path.parent
    if parent.name.endswith("_metricprompt"):
        return path if path.exists() else None
    candidate = parent.with_name(f"{parent.name}_metricprompt") / path.name
    return candidate if candidate.exists() else None


def _load_metric_fields_from_parquet(path: Path) -> dict[str, dict[str, Any]]:
    """Load metric protocol fields keyed by task UUID and task name."""
    metric_fields: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return metric_fields
    df = pd.read_parquet(path, columns=["metadata"])
    for value in df["metadata"]:
        row = dict(value)
        fields = _metric_fields_from_metadata(row)
        if not any(fields.values()):
            continue
        task_id = str(row.get("uuid") or row.get("task_id") or "") or None
        task_name = str(row.get("task_name") or "") or None
        if task_id:
            metric_fields[task_id] = fields
        if task_name:
            metric_fields.setdefault(task_name, fields)
    return metric_fields


def enrich_metadata_with_metric_protocol(
    metadata: dict[str, Any],
    metric_fields: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach metric protocol fields to one task metadata dictionary."""
    task_id = str(metadata.get("uuid") or metadata.get("task_id") or "")
    task_name = str(metadata.get("task_name") or "")
    fields = metric_fields.get(task_id) or metric_fields.get(task_name)
    if fields is None:
        return metadata
    metadata.setdefault("metric_label", fields.get("metric_label"))
    metadata.setdefault("metric_name", fields.get("metric_name"))
    metadata.setdefault("metric_direction", fields.get("metric_direction"))
    metadata.setdefault("metric_class", fields.get("metric_class"))
    metadata.setdefault("validation_strategy", fields.get("validation_strategy"))
    metadata.setdefault("metric_source", fields.get("metric_source"))
    metadata.setdefault("matched_text", fields.get("matched_text"))
    if "self_valid_protocol" not in metadata:
        metadata["self_valid_protocol"] = {
            "metric_label": fields.get("metric_label"),
            "metric_name": fields.get("metric_name"),
            "metric_direction": fields.get("metric_direction"),
            "validation_strategy": fields.get("validation_strategy"),
            "source": fields.get("metric_source"),
            "matched_text": fields.get("matched_text"),
        }
    return metadata


def enrich_tasks_with_metric_protocol(
    tasks: list[dict[str, Any]],
    *,
    source_parquet: Path | None = None,
    metric_metadata_parquet: Path | None = None,
    metadata_key: str = "metadata",
) -> int:
    """Enrich loaded task records with metric protocol from metricprompt parquet.

    Args:
        tasks: Task records loaded from the generation parquet.
        source_parquet: Official source parquet path. Used to infer sibling
            metricprompt path when ``metric_metadata_parquet`` is not provided.
        metric_metadata_parquet: Explicit metricprompt parquet path.
        metadata_key: Metadata key in each task record.

    Returns:
        Number of task records that received metric fields.
    """
    metric_path = metric_metadata_parquet
    if metric_path is None and source_parquet is not None:
        metric_path = infer_metricprompt_parquet(source_parquet)
    if metric_path is None:
        return 0
    metric_fields = _load_metric_fields_from_parquet(metric_path)
    enriched = 0
    for task in tasks:
        metadata = task.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        before = metadata.get("metric_class")
        enrich_metadata_with_metric_protocol(metadata, metric_fields)
        after = metadata.get("metric_class")
        if before is None and after is not None:
            enriched += 1
    return enriched


def load_task_meta_from_parquet(
    path: Path,
    *,
    metadata_col: str = "metadata",
    metric_metadata_parquet: Path | None = None,
    enrich_metric_protocol: bool = True,
) -> dict[str, TaskMeta]:
    """Load task score-range metadata from a source parquet.

    Args:
        path: Parquet path containing a metadata column with task UUID and ranges.
        metadata_col: Name of the metadata column.
        metric_metadata_parquet: Optional metricprompt parquet containing
            ``self_valid_protocol``. When omitted, a sibling ``*_metricprompt``
            parquet is used if it exists.
        enrich_metric_protocol: Attach metric fields from the metricprompt
            parquet when available.

    Returns:
        Mapping keyed primarily by task UUID, with task-name fallback keys.
    """
    meta: dict[str, TaskMeta] = {}
    metric_fields: dict[str, dict[str, Any]] = {}
    if enrich_metric_protocol:
        metric_path = metric_metadata_parquet or infer_metricprompt_parquet(path)
        if metric_path is not None:
            metric_fields = _load_metric_fields_from_parquet(metric_path)
    df = pd.read_parquet(path, columns=[metadata_col])
    for value in df[metadata_col]:
        row = dict(value)
        task_id = str(row.get("uuid") or row.get("task_id") or "") or None
        task_name = str(row.get("task_name") or "")
        fields = _metric_fields_from_metadata(row)
        if not any(fields.values()):
            fields = metric_fields.get(task_id or "") or metric_fields.get(task_name) or {}
        higher = row.get("higher_is_better")
        if isinstance(higher, str):
            higher_bool = higher.strip().lower() in {"true", "1", "yes"}
        elif higher is None:
            higher_bool = None
        else:
            higher_bool = bool(higher)
        item = TaskMeta(
            task_name=task_name,
            higher_is_better=higher_bool,
            theoretical_min=maybe_float(row.get("theoretical_min")),
            theoretical_max=maybe_float(row.get("theoretical_max")),
            leaderboard_min=maybe_float(row.get("leaderboard_min")),
            leaderboard_max=maybe_float(row.get("leaderboard_max")),
            task_id=task_id,
            metric_label=fields.get("metric_label"),
            metric_name=fields.get("metric_name"),
            metric_direction=fields.get("metric_direction"),
            metric_class=fields.get("metric_class"),
            validation_strategy=fields.get("validation_strategy"),
            metric_source=fields.get("metric_source"),
            matched_text=fields.get("matched_text"),
        )
        if task_id:
            meta[task_id] = item
        if task_name and task_name not in meta:
            meta[task_name] = item
    return meta


def lookup_task_meta(
    task_name: str,
    meta_map: dict[str, TaskMeta],
    *,
    task_id: str | None = None,
) -> TaskMeta | None:
    """Look up task metadata with optional version suffix stripping.

    Args:
        task_name: Task name from a rollout row.
        meta_map: Metadata mapping produced by ``load_task_meta``.
        task_id: Optional task UUID. This is preferred when present.

    Returns:
        Matching TaskMeta or None.
    """
    if task_id and task_id in meta_map:
        return meta_map[task_id]
    if task_name in meta_map:
        return meta_map[task_name]
    stripped = re.sub(r"@\d+$", "", task_name)
    return meta_map.get(stripped)


def extract_submission_score(feedback_text: str) -> float | None:
    """Extract final submission/test score from feedback text.

    Args:
        feedback_text: Feedback log text.

    Returns:
        Parsed score or None when no score is found.
    """
    for pattern in (SCORE_RE, FINAL_SCORE_RE, PREFIX_SCORE_RE):
        match = pattern.search(feedback_text)
        if match:
            return maybe_float(match.group(1))
    return None


def extract_validation_score(feedback_text: str) -> float | None:
    """Extract validation score from feedback text.

    Args:
        feedback_text: Feedback log text.

    Returns:
        Parsed validation score or None.
    """
    match = VAL_SCORE_RE.search(feedback_text)
    return maybe_float(match.group(1)) if match else None


def extract_feedback_status(feedback_text: str) -> str:
    """Extract sandbox status from formatted feedback."""
    match = STATUS_RE.search(feedback_text)
    return match.group(1).strip().lower() if match else "unknown"


def extract_feedback_result(feedback_text: str) -> str:
    """Extract sandbox result label from formatted feedback."""
    match = RESULT_RE.search(feedback_text)
    return match.group(1).strip().lower() if match else "unknown"


def feedback_is_success(
    feedback_text: str,
    *,
    status_values: set[str] | None = None,
    result_values: set[str] | None = None,
) -> bool:
    """Check whether feedback represents a successful run."""
    status_values = status_values or {"completed", "success"}
    result_values = result_values or {"success"}
    return (
        extract_feedback_status(feedback_text) in status_values
        and extract_feedback_result(feedback_text) in result_values
    )


def _range_size(low: float | None, high: float | None) -> float | None:
    """Compute a positive score range width.

    Args:
        low: Lower bound.
        high: Upper bound.

    Returns:
        Positive absolute range size, or None when unavailable.
    """
    if low is None or high is None:
        return None
    value = abs(high - low)
    return value if value > 0 else None


def _metadata_range_denominator(
    *,
    metadata: TaskMeta | None,
    config: GapFilterConfig,
) -> tuple[float | None, str]:
    """Return the legacy range-based denominator candidate."""
    theoretical = (
        None
        if metadata is None
        else _range_size(metadata.theoretical_min, metadata.theoretical_max)
    )
    leaderboard = (
        None
        if metadata is None
        else _range_size(metadata.leaderboard_min, metadata.leaderboard_max)
    )
    if theoretical is not None and theoretical <= config.theoretical_small_range_max:
        return theoretical, "theoretical_range_le_small_threshold"
    candidate = leaderboard or theoretical
    if candidate is None:
        return None, "missing_range"
    if candidate > config.big_range_threshold:
        return None, "range_gt_big_threshold"
    if leaderboard is not None:
        return leaderboard, "leaderboard_range"
    return theoretical, "theoretical_range_gt_small_threshold"


def _score_scale(
    validation_score: float | None,
    test_score: float | None,
    *,
    use_abs: bool = True,
) -> float | None:
    """Return a positive per-row score scale."""
    values = [value for value in (validation_score, test_score) if value is not None]
    if not values:
        return None
    if use_abs:
        scale = max(abs(value) for value in values)
    else:
        scale = max(values)
    return scale if scale > 0 else None


def _score_looks_percent_scale(
    validation_score: float | None,
    test_score: float | None,
) -> bool:
    """Heuristically detect 0-100 metric scores."""
    scale = _score_scale(validation_score, test_score, use_abs=True)
    return bool(scale is not None and scale > 2.0)


def _metric_class(metadata: TaskMeta | None) -> str:
    if metadata is None:
        return "unknown_or_unbounded"
    if metadata.metric_class:
        return metadata.metric_class
    return classify_metric(metadata.metric_label, metadata.metric_name)


def metric_gap_threshold(
    metadata: TaskMeta | None,
    config: GapFilterConfig,
) -> float:
    """Return the relative-gap threshold for a task metric."""
    metric_class = _metric_class(metadata)
    if metric_class in {"bounded_agreement_minus1_1", "bounded_correlation_or_r2"}:
        # With a [-1, 1] denominator of 2, this keeps the raw difference bound
        # at roughly 0.12 instead of allowing a 0.24 score swing.
        return min(config.max_relative_gap, config.max_relative_gap / 2.0)
    return config.max_relative_gap


def choose_legacy_gap_denominator(
    *,
    metadata: TaskMeta | None,
    validation_score: float | None,
    test_score: float | None,
    config: GapFilterConfig,
) -> tuple[float | None, str]:
    """Choose denominator using the previous final-run range rules.

    Args:
        metadata: Optional score-range metadata for the task.
        validation_score: Parsed validation score.
        test_score: Parsed submission/test score.
        config: Gap filtering configuration.

    Returns:
        Denominator value and a string explaining the selected source.
    """
    denom, source = _metadata_range_denominator(metadata=metadata, config=config)
    if denom is not None:
        return denom, source
    score_range = max(validation_score or 0.0, test_score or 0.0)
    if source == "range_gt_big_threshold":
        if score_range > 0:
            return score_range, "max_abs_valid_test_for_big_range"
        return None, "nonpositive_score_range_for_big_range"
    if config.no_range_is_big and score_range > 0:
        return score_range, "max_abs_valid_test_for_missing_range"
    return None, "missing_range"


def choose_gap_denominator(
    *,
    metadata: TaskMeta | None,
    validation_score: float | None,
    test_score: float | None,
    config: GapFilterConfig,
    task_score_scale: float | None = None,
) -> tuple[float | None, str]:
    """Choose a metric-aware denominator for validation/test gap.

    The previous range-based method is preserved in
    ``choose_legacy_gap_denominator``. This function is the default for both
    data production and generation-time rejection.
    """
    if not config.use_metric_aware_denominator:
        return choose_legacy_gap_denominator(
            metadata=metadata,
            validation_score=validation_score,
            test_score=test_score,
            config=config,
        )

    metric_class = _metric_class(metadata)
    row_scale = _score_scale(validation_score, test_score, use_abs=True)
    metadata_denom, metadata_source = _metadata_range_denominator(
        metadata=metadata, config=config
    )

    if metric_class in {
        "bounded_probabilistic_score_0_1",
        "bounded_score_0_1_or_percent",
    }:
        if _score_looks_percent_scale(validation_score, test_score):
            return 100.0, f"metric_aware:{metric_class}:percent_range_100"
        return 1.0, f"metric_aware:{metric_class}:unit_range_1"

    if metric_class in {"bounded_agreement_minus1_1", "bounded_correlation_or_r2"}:
        return 2.0, f"metric_aware:{metric_class}:minus1_1_range_2"

    if metric_class == "bounded_percentage_error_lower_better":
        if _score_looks_percent_scale(validation_score, test_score):
            return 100.0, "metric_aware:smape:percent_range_100"
        return 1.0, "metric_aware:smape:fraction_range_1"

    if metric_class in {
        "log_target_error_lower_better",
        "probabilistic_loss_lower_better",
    }:
        scale = max(row_scale or 0.0, float(config.unitless_loss_floor))
        return scale, f"metric_aware:{metric_class}:unitless_loss_floor"

    if metric_class == "percentage_error_lower_better":
        if _score_looks_percent_scale(validation_score, test_score):
            return 100.0, "metric_aware:mape:percent_range_100"
        return 1.0, "metric_aware:mape:fraction_range_1"

    if metric_class in {
        "target_scale_error_lower_better",
        "target_scale_distribution_loss_lower_better",
    }:
        if metadata_denom is not None:
            return metadata_denom, f"metric_aware:{metric_class}:{metadata_source}"
        if task_score_scale is not None and task_score_scale > 0:
            return task_score_scale, f"metric_aware:{metric_class}:task_score_scale"
        if row_scale is not None:
            return row_scale, f"metric_aware:{metric_class}:row_score_scale"
        return None, f"metric_aware:{metric_class}:missing_scale"

    if metadata_denom is not None:
        return metadata_denom, f"metric_aware:unknown:{metadata_source}"
    if task_score_scale is not None and task_score_scale > 0:
        return task_score_scale, "metric_aware:unknown:task_score_scale"
    if row_scale is not None and config.no_range_is_big:
        return row_scale, "metric_aware:unknown:row_score_scale"
    return None, "metric_aware:unknown:missing_scale"


def relative_gap_from_scores(
    *,
    validation_score: float | None,
    test_score: float | None,
    metadata: TaskMeta | None,
    config: GapFilterConfig | None = None,
    task_score_scale: float | None = None,
) -> tuple[float | None, float | None, float | None, str, float]:
    """Compute abs gap, relative gap, denominator source, and threshold."""
    config = config or GapFilterConfig()
    abs_gap = (
        abs(test_score - validation_score)
        if test_score is not None and validation_score is not None
        else None
    )
    denom, denom_source = choose_gap_denominator(
        metadata=metadata,
        validation_score=validation_score,
        test_score=test_score,
        config=config,
        task_score_scale=task_score_scale,
    )
    relative_gap = (
        abs_gap / denom
        if abs_gap is not None and denom is not None and denom > 0
        else None
    )
    return abs_gap, relative_gap, denom, denom_source, metric_gap_threshold(metadata, config)


def relative_gap_from_feedback(
    feedback_text: str,
    metadata: TaskMeta | dict[str, Any] | None,
    config: GapFilterConfig | None = None,
    *,
    task_score_scale: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Compute metric-aware relative gap from feedback text and metadata."""
    config = config or GapFilterConfig()
    task_meta = metadata_to_task_meta(metadata)
    test_score = extract_submission_score(feedback_text)
    validation_score = extract_validation_score(feedback_text)
    abs_gap, relative_gap, denom, source, threshold = relative_gap_from_scores(
        validation_score=validation_score,
        test_score=test_score,
        metadata=task_meta,
        config=config,
        task_score_scale=task_score_scale,
    )
    return relative_gap, {
        "validation_score": validation_score,
        "test_score": test_score,
        "abs_gap": abs_gap,
        "relative_gap": relative_gap,
        "gap_denominator": denom,
        "gap_denominator_source": source,
        "gap_threshold": threshold,
        "metric_class": None if task_meta is None else task_meta.metric_class,
        "metric_label": None if task_meta is None else task_meta.metric_label,
        "metric_name": None if task_meta is None else task_meta.metric_name,
    }


def metadata_to_task_meta(metadata: TaskMeta | dict[str, Any] | None) -> TaskMeta | None:
    """Convert generation-time metadata dicts to ``TaskMeta``."""
    if metadata is None or isinstance(metadata, TaskMeta):
        return metadata
    fields = _metric_fields_from_metadata(metadata)
    metric_label = metadata.get("metric_label") or fields.get("metric_label")
    metric_name = metadata.get("metric_name") or fields.get("metric_name")
    metric_class = (
        metadata.get("metric_class")
        or fields.get("metric_class")
        or classify_metric(metric_label, metric_name)
    )
    higher = metadata.get("higher_is_better")
    if isinstance(higher, str):
        higher_bool = higher.strip().lower() in {"true", "1", "yes"}
    elif higher is None:
        higher_bool = None
    else:
        higher_bool = bool(higher)
    return TaskMeta(
        task_name=str(metadata.get("task_name") or ""),
        higher_is_better=higher_bool,
        theoretical_min=maybe_float(metadata.get("theoretical_min")),
        theoretical_max=maybe_float(metadata.get("theoretical_max")),
        leaderboard_min=maybe_float(metadata.get("leaderboard_min")),
        leaderboard_max=maybe_float(metadata.get("leaderboard_max")),
        task_id=str(metadata.get("uuid") or metadata.get("task_id") or "") or None,
        metric_label=str(metric_label).lower() if metric_label else None,
        metric_name=str(metric_name) if metric_name else None,
        metric_direction=metadata.get("metric_direction") or fields.get("metric_direction"),
        metric_class=metric_class,
        validation_strategy=metadata.get("validation_strategy")
        or fields.get("validation_strategy"),
        metric_source=metadata.get("metric_source") or fields.get("metric_source"),
        matched_text=metadata.get("matched_text") or fields.get("matched_text"),
    )


def annotate_gap_rows(
    rows: list[dict[str, Any]],
    task_meta: dict[str, TaskMeta],
    config: GapFilterConfig | None = None,
) -> list[dict[str, Any]]:
    """Annotate rows with validation/test gap fields.

    Args:
        rows: Candidate SFT rows with feedback paths.
        task_meta: Task metadata mapping for score-range normalization.
        config: Optional gap filtering configuration.

    Returns:
        New rows with score, gap, denominator, and feedback annotations.
    """
    config = config or GapFilterConfig()
    parsed_scores: list[tuple[float | None, float | None]] = []
    task_scales: dict[str, float] = {}
    scale_values_by_task: dict[str, list[float]] = {}
    for row in rows:
        feedback_text = read_text(Path(str(row.get("feedback_path", ""))))
        test_score = extract_submission_score(feedback_text)
        validation_score = extract_validation_score(feedback_text)
        parsed_scores.append((test_score, validation_score))
        row_scale = _score_scale(validation_score, test_score, use_abs=True)
        if row_scale is None:
            continue
        task_key = str(row.get("task_id") or row.get("task_name") or "")
        if task_key:
            scale_values_by_task.setdefault(task_key, []).append(row_scale)
    for task_key, values in scale_values_by_task.items():
        sorted_values = sorted(values)
        idx = min(len(sorted_values) - 1, int(math.ceil(0.75 * len(sorted_values))) - 1)
        task_scales[task_key] = sorted_values[max(0, idx)]

    annotated_rows: list[dict[str, Any]] = []
    for row, (test_score, validation_score) in zip(rows, parsed_scores, strict=True):
        feedback_path = Path(str(row.get("feedback_path", "")))
        feedback_text = read_text(feedback_path)
        meta = lookup_task_meta(
            str(row.get("task_name", "")),
            task_meta,
            task_id=str(row.get("task_id") or "") or None,
        )
        task_key = str(row.get("task_id") or row.get("task_name") or "")
        abs_gap, relative_gap, denom, denom_source, threshold = relative_gap_from_scores(
            metadata=meta,
            validation_score=validation_score,
            test_score=test_score,
            config=config,
            task_score_scale=task_scales.get(task_key),
        )
        annotated = dict(row)
        annotated.update(
            {
                "feedback_status": extract_feedback_status(feedback_text),
                "feedback_result": extract_feedback_result(feedback_text),
                "feedback_success": feedback_is_success(feedback_text),
                "validation_score": validation_score,
                "test_score": test_score,
                "abs_gap": abs_gap,
                "relative_gap": relative_gap,
                "gap_denominator": denom,
                "gap_denominator_source": denom_source,
                "gap_threshold": threshold,
                "task_score_scale": task_scales.get(task_key),
                "gap_metadata_task_id": None if meta is None else meta.task_id,
                "gap_metadata_task_name": None if meta is None else meta.task_name,
                "higher_is_better": None if meta is None else meta.higher_is_better,
                "theoretical_min": None if meta is None else meta.theoretical_min,
                "theoretical_max": None if meta is None else meta.theoretical_max,
                "leaderboard_min": None if meta is None else meta.leaderboard_min,
                "leaderboard_max": None if meta is None else meta.leaderboard_max,
                "metric_label": None if meta is None else meta.metric_label,
                "metric_name": None if meta is None else meta.metric_name,
                "metric_class": None if meta is None else meta.metric_class,
                "metric_direction": None if meta is None else meta.metric_direction,
                "validation_strategy": (
                    None if meta is None else meta.validation_strategy
                ),
                "metric_source": None if meta is None else meta.metric_source,
            }
        )
        annotated_rows.append(annotated)
    return annotated_rows


def filter_by_relative_gap(
    rows: list[dict[str, Any]],
    config: GapFilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split annotated rows by relative gap rules.

    Args:
        rows: Rows produced by ``annotate_gap_rows``.
        config: Gap filtering configuration.

    Returns:
        Kept rows, dropped rows, and summary statistics.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        comparable = row.get("relative_gap") is not None
        threshold = float(row.get("gap_threshold") or config.max_relative_gap)
        keep = True
        if config.require_comparable and not comparable:
            keep = False
        if config.require_feedback_success and not row.get("feedback_success", False):
            keep = False
        if comparable and float(row["relative_gap"]) > threshold:
            keep = False
        annotated = dict(row)
        annotated["kept_gap_filter"] = bool(keep)
        if keep:
            kept.append(annotated)
        else:
            dropped.append(annotated)

    stats = {
        "max_relative_gap": config.max_relative_gap,
        "before": len(rows),
        "after": len(kept),
        "dropped": len(dropped),
        "comparable_rows": sum(
            1 for row in rows if row.get("relative_gap") is not None
        ),
        "relative_gap": summary_stats(
            [
                float(row["relative_gap"])
                for row in rows
                if row.get("relative_gap") is not None
            ]
        ),
        "abs_gap": summary_stats(
            [float(row["abs_gap"]) for row in rows if row.get("abs_gap") is not None]
        ),
    }
    return kept, dropped, stats
