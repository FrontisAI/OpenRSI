#!/usr/bin/env python3
"""Recover and select useful steps from evolutionary rollout trajectories."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


ROOT_OPERATORS = {"draft", "improve", "crossover"}
DEBUG_OPERATOR = "debug"
MEDAL_RANK = {
    "": 0,
    "none": 0,
    "null": 0,
    "nan": 0,
    "n/a": 0,
    "na": 0,
    "bronze": 1,
    "silver": 2,
    "gold": 3,
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def normalize_operator(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return {"closeover": "crossover", "cross_over": "crossover"}.get(text, text)


def numeric_field(obj: dict[str, Any] | None, key: str) -> float | None:
    if not obj:
        return None
    value = obj.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def strict_score(obj: dict[str, Any] | None) -> float | None:
    return numeric_field(obj, "score")


def medal_rank(value: Any) -> int:
    return MEDAL_RANK.get(str(value or "").strip().lower(), 0)


def load_task_stat(
    stat_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[int]]]:
    stat = json.loads(stat_path.read_text(encoding="utf-8", errors="replace"))
    steps = stat.get("steps") or []
    steps_by_index = {
        int(step["step"]): step
        for step in steps
        if isinstance(step, dict) and "step" in step
    }
    children: dict[int, list[int]] = defaultdict(list)
    for step in steps:
        if not isinstance(step, dict) or "step" not in step:
            continue
        child = int(step["step"])
        for parent in step.get("parent_steps") or []:
            try:
                children[int(parent)].append(child)
            except (TypeError, ValueError):
                continue
    return steps_by_index, children


def load_higher_is_better(task_dir: Path) -> bool:
    config_path = task_dir / "aira_evo" / "dojo_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
    value = config.get("task", {}).get("higher_is_better")
    if not isinstance(value, bool):
        raise RuntimeError(f"missing boolean higher_is_better in {config_path}")
    return value


def debug_descendants(
    root_step: int,
    children: dict[int, list[int]],
    steps_by_index: dict[int, dict[str, Any]],
) -> list[int]:
    result: list[int] = []
    queue: deque[int] = deque(children.get(root_step, []))
    while queue:
        step_index = queue.popleft()
        step = steps_by_index.get(step_index)
        if not step:
            continue
        if normalize_operator(step.get("operator") or step.get("mode")) != DEBUG_OPERATOR:
            continue
        result.append(step_index)
        queue.extend(children.get(step_index, []))
    return sorted(set(result))


def parent_scores_from_root(
    root: dict[str, Any],
    steps_by_index: dict[int, dict[str, Any]],
) -> tuple[list[int], list[float], list[int]]:
    parent_indices: list[int] = []
    parent_scores: list[float] = []
    missing_scores: list[int] = []
    for raw_index in root.get("parent_steps") or []:
        try:
            parent_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        parent_indices.append(parent_index)
        score = strict_score(steps_by_index.get(parent_index))
        if score is None:
            missing_scores.append(parent_index)
        else:
            parent_scores.append(score)
    return parent_indices, parent_scores, missing_scores


def score_better_than_parents(
    *,
    endpoint_score: float | None,
    parent_scores: list[float],
    higher_is_better: bool,
    allow_missing_parent_by_positive_endpoint: bool,
) -> bool:
    if endpoint_score is None:
        return False
    if not parent_scores:
        return bool(allow_missing_parent_by_positive_endpoint and endpoint_score > 0)
    baseline = max(parent_scores) if higher_is_better else min(parent_scores)
    return endpoint_score > baseline if higher_is_better else endpoint_score < baseline


def class_flags(
    *,
    segment_type: str,
    endpoint_score: float | None,
    endpoint_medal_rank: int,
    parent_scores: list[float],
    missing_parent_score_steps: list[int],
    higher_is_better: bool,
) -> tuple[bool, bool, str]:
    if segment_type == "draft":
        if endpoint_score is None:
            return False, False, "draft_endpoint_score_missing"
        score_gate = endpoint_score > 0
        reason = "draft_score_gt0" if score_gate else "draft_score_not_gt0"
    elif segment_type in {"improve", "crossover"}:
        if endpoint_score is None:
            return False, False, f"{segment_type}_endpoint_score_missing"
        score_gate = score_better_than_parents(
            endpoint_score=endpoint_score,
            parent_scores=parent_scores,
            higher_is_better=higher_is_better,
            allow_missing_parent_by_positive_endpoint=bool(missing_parent_score_steps),
        )
        reason = (
            f"{segment_type}_endpoint_score_better_than_parent"
            if score_gate
            else f"{segment_type}_endpoint_score_not_better_than_parent"
        )
    else:
        return False, False, "unsupported_segment_type"
    medal_gate = score_gate and endpoint_medal_rank > 0
    return medal_gate, score_gate, reason


def build_segments(stat_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stat_path in sorted(stat_root.glob("*/stat.json")):
        task_dir = stat_path.parent
        task_name = task_dir.name
        try:
            higher_is_better = load_higher_is_better(task_dir)
            missing_metric_direction = False
        except Exception:
            higher_is_better = True
            missing_metric_direction = True

        steps_by_index, children = load_task_stat(stat_path)
        for root_index, root in sorted(steps_by_index.items()):
            segment_type = normalize_operator(root.get("operator") or root.get("mode"))
            if segment_type not in ROOT_OPERATORS:
                continue
            segment_steps = [root_index] + debug_descendants(
                root_index, children, steps_by_index
            )
            endpoint_index = max(segment_steps)
            endpoint = steps_by_index.get(endpoint_index)
            if endpoint is None:
                continue
            endpoint_score = strict_score(endpoint)
            endpoint_medal = endpoint.get("submit_medal") or endpoint.get("medal")
            parent_indices, parent_scores, missing_parent_scores = (
                parent_scores_from_root(root, steps_by_index)
            )
            medal_gate, score_gate, reason = class_flags(
                segment_type=segment_type,
                endpoint_score=endpoint_score,
                endpoint_medal_rank=medal_rank(endpoint_medal),
                parent_scores=parent_scores,
                missing_parent_score_steps=missing_parent_scores,
                higher_is_better=higher_is_better,
            )
            if missing_metric_direction and segment_type in {"improve", "crossover"}:
                score_gate = False
                medal_gate = False
                reason = "missing_metric_direction"
            records.append(
                {
                    "segment_id": f"aira::{task_name}::root_{root_index}",
                    "task_name": task_name,
                    "segment_type": segment_type,
                    "segment_root_step": root_index,
                    "segment_endpoint_step": endpoint_index,
                    "segment_step_indices": segment_steps,
                    "segment_length": len(segment_steps),
                    "higher_is_better": higher_is_better,
                    "parent_step_indices": parent_indices,
                    "parent_scores": parent_scores,
                    "missing_parent_score_steps": missing_parent_scores,
                    "endpoint_score": endpoint_score,
                    "endpoint_submit_medal": endpoint_medal,
                    "endpoint_medal_rank": medal_rank(endpoint_medal),
                    "passes_score_gate": score_gate,
                    "passes_medal_gate": medal_gate,
                    "score_gate_reason": reason,
                    "stat_path": str(stat_path.relative_to(stat_root)),
                    "endpoint_step_dir": str(
                        (task_dir / f"step_{endpoint_index}").relative_to(stat_root)
                    ),
                }
            )
    return records


def parse_index_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return [int(item) for item in parsed] if isinstance(parsed, list) else []


def load_annotations(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    result: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        segment_id = (
            row.get("segment_id")
            or metadata.get("segment_id")
            or row.get("custom_id")
        )
        annotation = row.get("annotation")
        if isinstance(annotation, dict):
            row = {**row, **annotation}
        indices = parse_index_list(row.get("keep_indices"))
        if not indices and isinstance(row.get("steps"), list):
            indices = [
                int(step["step_index"])
                for step in row["steps"]
                if isinstance(step, dict)
                and step.get("keep_for_sft") is True
                and isinstance(step.get("step_index"), int)
            ]
        if isinstance(segment_id, str):
            result[segment_id] = indices
    return result


def select_steps(
    segments: list[dict[str, Any]],
    annotations: dict[str, list[int]],
    stat_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_step_dirs: set[str] = set()
    missing_annotations = 0
    for segment in segments:
        if not segment["passes_medal_gate"]:
            continue
        valid_steps = set(int(index) for index in segment["segment_step_indices"])
        if segment["segment_length"] == 1:
            keep_steps = list(valid_steps)
            source = "single_step_medal_segment"
        else:
            if segment["segment_id"] not in annotations:
                missing_annotations += 1
                continue
            keep_steps = [
                index
                for index in annotations[segment["segment_id"]]
                if index in valid_steps
            ]
            source = "causal_inheritance_annotation"
        task_dir = (stat_root / str(segment["stat_path"])).parent
        for step_index in keep_steps:
            step_dir = task_dir / f"step_{step_index}"
            relative_step_dir = str(step_dir.relative_to(stat_root))
            if relative_step_dir in seen_step_dirs:
                continue
            seen_step_dirs.add(relative_step_dir)
            selected.append(
                {
                    "id": f"aira::{segment['task_name']}::step_{step_index}",
                    "task_name": segment["task_name"],
                    "step_index": step_index,
                    "step_dir": relative_step_dir,
                    "segment_id": segment["segment_id"],
                    "segment_type": segment["segment_type"],
                    "segment_length": segment["segment_length"],
                    "selection_source": source,
                }
            )
    selected.sort(key=lambda row: (row["task_name"], row["step_index"]))
    return selected, {
        "selected_steps": len(selected),
        "unique_step_dirs": len(seen_step_dirs),
        "missing_multi_step_annotations": missing_annotations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stat-root",
        type=Path,
        required=True,
        help="Directory containing <task>/stat.json and <task>/aira_evo/dojo_config.json.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional causal-inheritance annotation JSONL for multi-step segments.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete-annotations",
        action="store_true",
        help="Fail when a selected multi-step segment has no annotation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stat_root.exists():
        raise FileNotFoundError(f"stat root does not exist: {args.stat_root}")
    segments = build_segments(args.stat_root)
    annotations = load_annotations(args.annotations)
    selected_steps, selection_summary = select_steps(
        segments, annotations, args.stat_root
    )
    if (
        args.require_complete_annotations
        and selection_summary["missing_multi_step_annotations"]
    ):
        raise RuntimeError(
            "selected multi-step segments are missing causal-inheritance annotations: "
            f"{selection_summary['missing_multi_step_annotations']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "segments.jsonl", segments)
    write_jsonl(args.output_dir / "selected_steps.jsonl", selected_steps)
    summary = {
        "input_stat_root": str(args.stat_root),
        "annotations": str(args.annotations) if args.annotations else None,
        "segment_count": len(segments),
        "segment_type_counts": dict(
            Counter(segment["segment_type"] for segment in segments)
        ),
        "score_gate_segments": sum(
            bool(segment["passes_score_gate"]) for segment in segments
        ),
        "medal_gate_segments": sum(
            bool(segment["passes_medal_gate"]) for segment in segments
        ),
        **selection_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
