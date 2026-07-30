#!/usr/bin/env python3
"""Select high-scoring candidates from parallel rollout results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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


def numeric_score(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"candidate has no numeric {field!r}: {row.get('request_id')}") from exc
    if not math.isfinite(score):
        raise ValueError(f"candidate has a non-finite {field!r}: {row.get('request_id')}")
    return score


def task_value(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "")
    if not value:
        raise ValueError(f"candidate has no {field!r}: {row.get('request_id')}")
    return value


def stable_id(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "")
    if not value:
        raise ValueError(f"candidate has no {field!r}")
    return value


def generator_family(row: dict[str, Any], field: str) -> str:
    explicit = str(row.get(field) or "").strip().lower()
    if explicit in {"glm", "qwen"}:
        return explicit
    source_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("model", "model_name", "run_key", "source_name")
    )
    if "glm" in source_text:
        return "glm"
    if "qwen" in source_text:
        return "qwen"
    return "unknown"


def select_glm_top4_unique_score(
    rows: list[dict[str, Any]],
    *,
    task_field: str,
    score_field: str,
    id_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate equal scores per task, then retain the highest four.

    This preserves the historical first-batch order of operations: duplicate
    reward values are removed before the per-task Top-4 is taken.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[task_value(row, task_field)].append(row)

    selected: list[dict[str, Any]] = []
    duplicate_score_rows = 0
    for task in sorted(grouped):
        ordered = sorted(
            grouped[task],
            key=lambda row: (-numeric_score(row, score_field), stable_id(row, id_field)),
        )
        unique_scores: list[dict[str, Any]] = []
        seen_scores: set[float] = set()
        for row in ordered:
            score = numeric_score(row, score_field)
            if score in seen_scores:
                duplicate_score_rows += 1
                continue
            seen_scores.add(score)
            unique_scores.append(row)
        for rank, row in enumerate(unique_scores[:4], 1):
            output = dict(row)
            output["selection_rank"] = rank
            output["selection_policy"] = "per_task_score_dedup_then_top4"
            selected.append(output)

    summary = {
        "policy": "glm-top4-unique-score",
        "input_rows": len(rows),
        "task_count": len(grouped),
        "duplicate_score_rows_removed": duplicate_score_rows,
        "rows_after_score_dedup": len(rows) - duplicate_score_rows,
        "selected_rows": len(selected),
        "selected_count_per_task": dict(
            Counter(
                str(count)
                for count in Counter(task_value(row, task_field) for row in selected).values()
            )
        ),
    }
    return selected, summary


def select_joint_top4(
    rows: list[dict[str, Any]],
    *,
    task_field: str,
    score_field: str,
    id_field: str,
    family_field: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the historical mixed-family joint Top-4 policy."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_rows = 0
    input_families: Counter[str] = Counter()
    for original in rows:
        row = dict(original)
        family = generator_family(row, family_field)
        input_families[family] += 1
        if family == "unknown":
            unknown_rows += 1
            continue
        row[family_field] = family
        grouped[task_value(row, task_field)].append(row)

    selected: list[dict[str, Any]] = []
    selected_families: Counter[str] = Counter()
    qwen_dropped_by_rank: Counter[str] = Counter()
    for task in sorted(grouped):
        ordered = sorted(
            grouped[task],
            key=lambda row: (-numeric_score(row, score_field), stable_id(row, id_field)),
        )
        for rank, row in enumerate(ordered[:4], 1):
            family = str(row[family_field])
            keep = family == "glm" or (family == "qwen" and rank == 1)
            if family == "qwen" and rank > 1:
                qwen_dropped_by_rank[f"rank_{rank}"] += 1
            if not keep:
                continue
            output = dict(row)
            output["selection_rank"] = rank
            output["selection_observed_topk"] = 4
            output["selection_policy"] = "joint_glm_qwen_top4_qwen_only_rank1"
            selected.append(output)
            selected_families[family] += 1

    summary = {
        "policy": "joint-top4",
        "input_rows": len(rows),
        "input_family_counts": dict(input_families),
        "unknown_family_rows_dropped": unknown_rows,
        "task_count": len(grouped),
        "selected_rows": len(selected),
        "selected_family_counts": dict(selected_families),
        "qwen_dropped_by_joint_rank": dict(qwen_dropped_by_rank),
        "maximum_rows_per_task": max(
            Counter(task_value(row, task_field) for row in selected).values(),
            default=0,
        ),
    }
    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Candidate JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Selected JSONL file.")
    parser.add_argument(
        "--policy",
        choices=("glm-top4-unique-score", "joint-top4"),
        required=True,
        help="Selection policy to apply.",
    )
    parser.add_argument("--task-field", default="task_name")
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--id-field", default="request_id")
    parser.add_argument("--family-field", default="generator_family")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if args.policy == "glm-top4-unique-score":
        selected, summary = select_glm_top4_unique_score(
            rows,
            task_field=args.task_field,
            score_field=args.score_field,
            id_field=args.id_field,
        )
    else:
        selected, summary = select_joint_top4(
            rows,
            task_field=args.task_field,
            score_field=args.score_field,
            id_field=args.id_field,
            family_field=args.family_field,
        )

    write_jsonl(args.output, selected)
    summary.update({"input": str(args.input), "output": str(args.output)})
    summary_path = args.summary or Path(str(args.output) + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
