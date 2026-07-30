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


def build_tasks(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    base_dir = Path(__file__).resolve().parent
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
            "data_dir": str(metadata["data_dir"]),
            "higher_is_better": bool(metadata["higher_is_better"]),
            "theoretical_max": metadata.get("theoretical_max"),
            "theoretical_min": metadata.get("theoretical_min"),
            "leaderboard_max": metadata.get("leaderboard_max"),
            "leaderboard_min": metadata.get("leaderboard_min"),
            "leaderboard_dir": cfg.data.leaderboard_dir,
            "submit_dir": str(cfg.data.submit_dir),
            "submit_data_dir_root": cfg.data.get("submit_data_dir_root"),
            "sandbox": {
                "resource": resource,
                "base_url": str(base_url_map[resource]),
                "job_timeout": int(cfg.sandbox.job_timeout),
                "wait_timeout": int(cfg.sandbox.wait_timeout),
                "poll_interval": int(cfg.sandbox.poll_interval),
                "use_score2reward": bool(cfg.sandbox.use_score2reward),
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
    args = parser.parse_args()
    build_tasks(Path(args.config))


if __name__ == "__main__":
    main()
