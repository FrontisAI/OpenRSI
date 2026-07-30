#!/usr/bin/env python3
"""Deduplicate SFT messages and apply a full-message token-length gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if hasattr(value, "tolist"):
        return normalize_for_json(value.tolist())
    return value


def normalized_message_text(messages: Any) -> str:
    return json.dumps(
        normalize_for_json(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def message_hash(messages: Any) -> str:
    return hashlib.sha1(normalized_message_text(messages).encode("utf-8")).hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, Any]], Any | None]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pylist(), table
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows, None


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    source_table: Any | None,
    indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        if source_table is not None:
            output_table = source_table.take(pa.array(indices, type=pa.int64()))
        else:
            output_table = pa.Table.from_pylist(rows)
        pq.write_table(output_table, path, row_group_size=max(1, len(rows)))
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_for_json(row), ensure_ascii=False) + "\n")


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    seen: dict[str, int] = {}
    keep_indices: list[int] = []
    dropped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "messages" not in row:
            raise ValueError(f"row {index} has no messages field")
        digest = message_hash(row["messages"])
        if digest in seen:
            dropped.append(
                {
                    "row_index": index,
                    "id": row.get("id"),
                    "first_row_index": seen[digest],
                    "message_hash": digest,
                    "drop_reason": "duplicate_normalized_messages",
                }
            )
            continue
        seen[digest] = index
        keep_indices.append(index)
    return keep_indices, dropped


def full_message_tokens(tokenizer: Any, messages: Any) -> tuple[int, bool]:
    normalized = normalize_for_json(messages)
    try:
        token_ids = tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=False,
        )
        return len(token_ids), False
    except Exception:
        text = normalized_message_text(normalized)
        return len(tokenizer.encode(text, add_special_tokens=False)), True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_for_json(row), ensure_ascii=False) + "\n")


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dedup_parser = subparsers.add_parser(
        "dedup", help="Keep the first occurrence of each normalized messages value."
    )
    add_shared_arguments(dedup_parser)
    dedup_parser.add_argument(
        "--dropped",
        type=Path,
        help="Dropped-row JSONL path. Defaults to <output>.dropped.jsonl.",
    )

    token_parser = subparsers.add_parser(
        "token-filter", help="Filter rows by full chat-template token length."
    )
    add_shared_arguments(token_parser)
    token_parser.add_argument("--model-path", required=True)
    token_parser.add_argument("--max-tokens", type=int, default=32768)
    token_parser.add_argument(
        "--dropped",
        type=Path,
        help="Dropped-row JSONL path. Defaults to <output>.dropped.jsonl.",
    )
    token_parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, source_table = read_rows(args.input)
    dropped_path = args.dropped or Path(str(args.output) + ".dropped.jsonl")
    summary_path = args.summary or Path(str(args.output) + ".summary.json")

    if args.command == "dedup":
        keep_indices, dropped = deduplicate(rows)
        summary = {
            "operation": "exact_message_deduplication",
            "input_rows": len(rows),
            "kept_rows": len(keep_indices),
            "dropped_rows": len(dropped),
            "policy": (
                "SHA1 of normalized messages JSON; keep first occurrence in input order"
            ),
        }
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
        )
        keep_indices = []
        dropped = []
        fallback_count = 0
        for index, row in enumerate(rows):
            if "messages" not in row:
                raise ValueError(f"row {index} has no messages field")
            token_count, fallback = full_message_tokens(tokenizer, row["messages"])
            fallback_count += int(fallback)
            if token_count <= args.max_tokens:
                keep_indices.append(index)
            else:
                dropped.append(
                    {
                        "row_index": index,
                        "id": row.get("id"),
                        "full_messages_tokens": token_count,
                        "drop_reason": "full_messages_token_limit",
                    }
                )
        summary = {
            "operation": "full_message_token_filter",
            "input_rows": len(rows),
            "kept_rows": len(keep_indices),
            "dropped_rows": len(dropped),
            "max_tokens": args.max_tokens,
            "model_path": args.model_path,
            "tokenizer_fallback_rows": fallback_count,
        }

    kept_rows = [rows[index] for index in keep_indices]
    write_rows(
        args.output,
        kept_rows,
        source_table=source_table,
        indices=keep_indices,
    )
    write_jsonl(dropped_path, dropped)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "dropped_manifest": str(dropped_path),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
