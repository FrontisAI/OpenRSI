#!/usr/bin/env python3
"""Exclude training rows whose task description matches a reserved task."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if hasattr(value, "tolist"):
        return normalize_value(value.tolist())
    return value


def canonicalize_task_description(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_user_message(messages: Any) -> str:
    normalized = normalize_value(messages)
    if not isinstance(normalized, list):
        return ""
    for item in normalized:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def extract_task_description_from_user(user_text: str) -> tuple[str, str]:
    marker = "Task Description:"
    marker_index = user_text.find(marker)
    if marker_index < 0:
        return "", "missing_task_description_marker"
    task_body = user_text[marker_index + len(marker) :]
    end_patterns = [
        "\n\nData Description:\n\n```",
        "\nData Description:\n\n```",
        "\n\nData Preview:\n```",
        "\nData Preview:\n```",
        "\n\nOutput Instructions:",
        "\nOutput Instructions:",
    ]
    end_positions = [
        task_body.find(pattern)
        for pattern in end_patterns
        if task_body.find(pattern) >= 0
    ]
    if end_positions:
        return task_body[: min(end_positions)].strip(), "ok_cut_before_appended_context"
    return task_body.strip(), "ok_no_cut_marker"


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


def read_rows(path: Path) -> tuple[list[dict[str, Any]], Any | None]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pylist(), table
    return list(read_jsonl(path)), None


def reserved_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = normalize_value(row.get("metadata"))
    if isinstance(metadata, dict):
        return metadata
    return row


def build_reserved_hash_index(
    prompt_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    ordered_metadata: list[dict[str, str]] = []
    for position, row in enumerate(prompt_rows):
        metadata = reserved_metadata(row)
        description = canonicalize_task_description(
            str(metadata.get("task_description") or "")
        )
        if not description:
            raise ValueError(f"reserved prompt row {position} has no task_description")
        record = {
            "uuid": str(metadata.get("uuid") or ""),
            "task_name": str(metadata.get("task_name") or ""),
        }
        ordered_metadata.append(record)
        index[sha256_text(description)].append(record)
    return index, ordered_metadata


def validate_manifest_order(
    manifest_path: Path,
    ordered_metadata: list[dict[str, str]],
) -> None:
    import csv

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(ordered_metadata):
        raise ValueError(
            "reserved manifest and prompt file have different row counts: "
            f"{len(rows)} != {len(ordered_metadata)}"
        )
    for position, (manifest, metadata) in enumerate(zip(rows, ordered_metadata)):
        for key in ("uuid", "task_name"):
            if key in manifest and str(manifest.get(key) or "") != metadata[key]:
                raise ValueError(
                    f"reserved manifest {key} mismatch at row {position}: "
                    f"{manifest.get(key)!r} != {metadata[key]!r}"
                )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_value(row), ensure_ascii=False) + "\n")


def write_kept_data(
    input_path: Path,
    output_path: Path,
    kept_rows: list[dict[str, Any]],
    kept_indices: list[int],
    source_table: Any | None,
) -> None:
    if output_path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        if source_table is not None:
            table = source_table.take(pa.array(kept_indices, type=pa.int64()))
        else:
            table = pa.Table.from_pylist(kept_rows)
        pq.write_table(table, output_path, row_group_size=max(1, len(kept_rows)))
        return
    write_jsonl(output_path, kept_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Training JSONL or Parquet.")
    parser.add_argument(
        "--reserved-prompts",
        type=Path,
        required=True,
        help="Reserved prompt JSONL or Parquet.",
    )
    parser.add_argument(
        "--reserved-manifest",
        type=Path,
        help="Optional CSV used to validate UUID and task-name order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--allow-missing-task-description",
        action="store_true",
        help=(
            "Keep rows without a Task Description marker. By default they are "
            "rejected because overlap cannot be determined safely."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"output directory is not empty: {args.output_dir}; use --overwrite to proceed"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows, _ = read_rows(args.reserved_prompts)
    reserved_hashes, ordered_metadata = build_reserved_hash_index(prompt_rows)
    if args.reserved_manifest:
        validate_manifest_order(args.reserved_manifest, ordered_metadata)

    training_rows, source_table = read_rows(args.input)
    kept_rows: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    kept_manifest: list[dict[str, Any]] = []
    removed_manifest: list[dict[str, Any]] = []
    extraction_reasons: Counter[str] = Counter()
    matched_hashes: set[str] = set()

    for index, row in enumerate(training_rows):
        if "messages" not in row:
            raise ValueError(f"training row {index} has no messages field")
        raw_description, reason = extract_task_description_from_user(
            get_user_message(row["messages"])
        )
        if (
            reason == "missing_task_description_marker"
            and not args.allow_missing_task_description
        ):
            raise ValueError(
                "cannot safely check reserved-task overlap because a training row "
                f"has no Task Description marker: row_index={index}, id={row.get('id')!r}"
            )
        description = canonicalize_task_description(raw_description)
        digest = sha256_text(description) if description else ""
        matches = reserved_hashes.get(digest, [])
        extraction_reasons[reason] += 1
        audit = {
            "row_index": index,
            "id": row.get("id"),
            "task_description_sha256": digest,
            "task_description_chars": len(description),
            "extract_reason": reason,
        }
        if matches:
            matched_hashes.add(digest)
            removed_manifest.append(
                {
                    **audit,
                    "remove_reason": "task_description_sha256_exact_match",
                }
            )
            continue
        kept_indices.append(index)
        kept_rows.append(row)
        kept_manifest.append(audit)

    output_suffix = ".parquet" if args.input.suffix.lower() == ".parquet" else ".jsonl"
    output_data = args.output_dir / f"train{output_suffix}"
    write_kept_data(
        args.input,
        output_data,
        kept_rows,
        kept_indices,
        source_table,
    )
    write_jsonl(args.output_dir / "removed_overlap.jsonl", removed_manifest)
    write_jsonl(args.output_dir / "kept_manifest.jsonl", kept_manifest)
    summary = {
        "input": str(args.input),
        "reserved_prompts": str(args.reserved_prompts),
        "reserved_manifest": (
            str(args.reserved_manifest) if args.reserved_manifest else None
        ),
        "output": str(output_data),
        "input_rows": len(training_rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(removed_manifest),
        "reserved_rows": len(prompt_rows),
        "reserved_unique_description_hashes": len(reserved_hashes),
        "matched_reserved_unique_description_hashes": len(matched_hashes),
        "extraction_reason_counts": dict(extraction_reasons),
        "rule": (
            "exact SHA256 match after minimal newline and trailing-space normalization; "
            "UUID and task name are audit fields only"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
