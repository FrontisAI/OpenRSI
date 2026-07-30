from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GUIDANCE_HEADING = "Validation Split Guidance:"
DEFAULT_INSERT_MARKER = "**FINAL OUTPUT**"


@dataclass(frozen=True)
class GuidancePatchStats:
    input_path: Path
    output_path: Path
    total_rows: int
    patched_rows: int
    missing_tasks: tuple[str, ...]
    skipped_existing_rows: int


def load_guidance_map(instructions_path: Path) -> dict[str, str]:
    """Parse task-specific validation guidance blocks from the markdown document."""
    text = instructions_path.read_text(encoding="utf-8")
    sections = list(re.finditer(r"^##\s+\d+\.\s+(.+?)\s*$", text, flags=re.M))
    guidance: dict[str, str] = {}
    for idx, match in enumerate(sections):
        task_name = match.group(1).strip()
        start = match.end()
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(text)
        section = text[start:end]
        block = re.search(
            r"Recommended prompt instruction:\s*```(?:text)?\s*(.*?)\s*```",
            section,
            flags=re.S,
        )
        if block:
            guidance[task_name] = block.group(1).strip()
    return guidance


def patch_eval_data_with_guidance(
    *,
    eval_data_path: Path,
    output_path: Path,
    instructions_path: Path,
    input_key: str,
    metadata_key: str,
    insert_marker: str = DEFAULT_INSERT_MARKER,
    heading: str = DEFAULT_GUIDANCE_HEADING,
    strict: bool = False,
) -> GuidancePatchStats:
    guidance_map = load_guidance_map(instructions_path)
    if not guidance_map:
        raise ValueError(f"No validation split guidance found in {instructions_path}")

    suffix = eval_data_path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(eval_data_path).to_dict(orient="records")
        patched_rows, missing_tasks, skipped_existing = _patch_rows(
            rows=rows,
            guidance_map=guidance_map,
            input_key=input_key,
            metadata_key=metadata_key,
            insert_marker=insert_marker,
            heading=heading,
            strict=strict,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output_path, index=False)
    elif suffix == ".json":
        rows = json.loads(eval_data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON eval data to be a list: {eval_data_path}")
        patched_rows, missing_tasks, skipped_existing = _patch_rows(
            rows=rows,
            guidance_map=guidance_map,
            input_key=input_key,
            metadata_key=metadata_key,
            insert_marker=insert_marker,
            heading=heading,
            strict=strict,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported eval_data format: {eval_data_path}")

    return GuidancePatchStats(
        input_path=eval_data_path,
        output_path=output_path,
        total_rows=len(rows),
        patched_rows=patched_rows,
        missing_tasks=tuple(sorted(missing_tasks)),
        skipped_existing_rows=skipped_existing,
    )


def _patch_rows(
    *,
    rows: list[dict[str, Any]],
    guidance_map: dict[str, str],
    input_key: str,
    metadata_key: str,
    insert_marker: str,
    heading: str,
    strict: bool,
) -> tuple[int, set[str], int]:
    patched_rows = 0
    skipped_existing = 0
    missing_tasks: set[str] = set()

    for row in rows:
        task_name = _task_name_from_metadata(row.get(metadata_key))
        if not task_name:
            task_name = str(row.get("task_name") or row.get("name") or "").strip()
        guidance = guidance_map.get(task_name)
        if not guidance:
            missing_tasks.add(task_name or "<unknown>")
            continue
        patched_prompt, changed, skipped = _inject_guidance(
            row.get(input_key),
            guidance=guidance,
            insert_marker=insert_marker,
            heading=heading,
        )
        if changed:
            row[input_key] = patched_prompt
            patched_rows += 1
        if skipped:
            skipped_existing += 1

    if strict and missing_tasks:
        missing = ", ".join(sorted(missing_tasks))
        raise ValueError(f"Missing validation split guidance for tasks: {missing}")
    return patched_rows, missing_tasks, skipped_existing


def _task_name_from_metadata(metadata: Any) -> str:
    if hasattr(metadata, "item"):
        try:
            metadata = metadata.item()
        except Exception:
            pass
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return ""
    if not isinstance(metadata, dict):
        return ""
    for key in ("task_name", "competition_id", "competition", "name"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return ""


def _inject_guidance(
    prompt: Any,
    *,
    guidance: str,
    insert_marker: str,
    heading: str,
) -> tuple[Any, bool, bool]:
    if isinstance(prompt, str):
        return _inject_into_text(prompt, guidance, insert_marker, heading)

    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    if isinstance(prompt, tuple):
        prompt = list(prompt)

    if isinstance(prompt, list):
        for idx in range(len(prompt) - 1, -1, -1):
            message = prompt[idx]
            if isinstance(message, dict) and message.get("role") == "user":
                patched, changed, skipped = _inject_into_text(
                    str(message.get("content") or ""),
                    guidance,
                    insert_marker,
                    heading,
                )
                if changed:
                    new_message = dict(message)
                    new_message["content"] = patched
                    prompt[idx] = new_message
                return prompt, changed, skipped
        return prompt, False, False

    return prompt, False, False


def _inject_into_text(
    text: str,
    guidance: str,
    insert_marker: str,
    heading: str,
) -> tuple[str, bool, bool]:
    if heading in text:
        return text, False, True

    block = f"\n\n{heading}\n{guidance.strip()}\n\n"
    marker_index = text.find(insert_marker) if insert_marker else -1
    if marker_index >= 0:
        patched = f"{text[:marker_index].rstrip()}{block}{text[marker_index:]}"
    else:
        patched = f"{text.rstrip()}{block}"
    return patched, True, False
