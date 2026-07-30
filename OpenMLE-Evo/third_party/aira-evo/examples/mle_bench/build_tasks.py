#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tts_search.prompt_builder import split_legacy_public_prompts


def _sanitize_task_name(task_name: str) -> str:
    return task_name.replace("/", "_").replace("\\", "_")


def _extract_prompt_fields(
    record: dict[str, Any],
    input_key: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    messages_value = record[input_key]
    if hasattr(messages_value, "tolist"):
        messages = list(messages_value.tolist() or [])
    else:
        messages = list(messages_value or [])

    public_system_prompt = ""
    public_user_prompt = ""
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", ""))
        if role == "system" and not public_system_prompt:
            public_system_prompt = content
        elif role == "user" and not public_user_prompt:
            public_user_prompt = content
        if public_system_prompt and public_user_prompt:
            break

    task_description = str(metadata.get("task_description") or metadata.get("task_desc") or "")
    data_description = str(metadata.get("data_description") or metadata.get("data_desc") or "")

    if task_description.strip() == "" and data_description.strip() == "":
        (
            task_description,
            data_description,
            public_system_prompt,
            public_user_prompt,
        ) = split_legacy_public_prompts(public_system_prompt, public_user_prompt)

    return {
        "task_description": task_description,
        "data_description": data_description,
        "public_system_prompt": public_system_prompt,
        "public_user_prompt": public_user_prompt,
    }


def _sandbox_visible_path(path_value: Any) -> Any:
    if path_value is None:
        return None
    return path_value


def build_tasks(config_path: Path, output_root: Path | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    base_dir = (
        output_root.resolve()
        if output_root is not None
        else Path(__file__).resolve().parent
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_parquet(cfg.data.eval_data).to_dict(orient="records")
    base_url_map = {item.resource: item.base_url for item in cfg.sandbox.base_url_map}

    for record in records:
        metadata = dict(record[cfg.data.metadata_key])
        task_name = str(metadata["task_name"])
        task_dir = base_dir / _sanitize_task_name(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)

        prompt_fields = _extract_prompt_fields(
            record=record,
            input_key=str(cfg.data.input_key),
            metadata=metadata,
        )
        resource = str(metadata["cpu_gpu"]).lower()

        task_payload = {
            "uuid": metadata["uuid"],
            "task_name": task_name,
            "task": metadata["task"],
            "source": str(metadata.get("source") or "MLE-Bench"),
            "modality": metadata.get("modality"),
            "data_dir": str(_sandbox_visible_path(metadata["data_dir"])),
            "higher_is_better": bool(metadata["higher_is_better"]),
            "theoretical_max": metadata.get("theoretical_max"),
            "theoretical_min": metadata.get("theoretical_min"),
            "leaderboard_max": metadata.get("leaderboard_max"),
            "leaderboard_min": metadata.get("leaderboard_min"),
            "leaderboard_dir": cfg.data.leaderboard_dir,
            "submit_dir": str(cfg.data.submit_dir),
            "submit_data_dir_root": _sandbox_visible_path(
                cfg.data.get("submit_data_dir_root")
            ),
            "evaluation_protocol": str(cfg.data.get("evaluation_protocol", "legacy")),
            "validation_eval_split": cfg.data.get("validation_eval_split"),
            "test_eval_split": cfg.data.get("test_eval_split"),
            "selection_score_source": cfg.data.get("selection_score_source"),
            "submit_job_timeout": cfg.data.get("submit_job_timeout"),
            "submit_wait_timeout": cfg.data.get("submit_wait_timeout"),
            "submit_repeats": int(cfg.data.get("submit_repeats", 1) or 1),
            "sandbox": {
                "resource": resource,
                "base_url": str(base_url_map[resource]),
                "job_timeout": int(cfg.sandbox.job_timeout),
                "wait_timeout": int(cfg.sandbox.wait_timeout),
                "poll_interval": int(cfg.sandbox.poll_interval),
                "use_score2reward": bool(cfg.sandbox.use_score2reward),
                "use_clear_run_log_score": bool(
                    cfg.sandbox.get("use_clear_run_log_score", False)
                ),
            },
            **prompt_fields,
        }

        (task_dir / "config.yaml").write_text(
            yaml.safe_dump(task_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (task_dir / "task_metadata.json").write_text(
            json.dumps(task_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIRA Dojo MLE-Bench tasks")
    parser.add_argument("--config", required=True, help="Path to build config yaml")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where task config directories should be written.",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    build_tasks(Path(args.config), output_root=output_root)


if __name__ == "__main__":
    main()
