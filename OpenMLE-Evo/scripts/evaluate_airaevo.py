from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
logger = logging.getLogger(__name__)

from tts_search.eval_network import disable_proxy_env  # noqa: E402
from tts_search.task_skill_guidance import (  # noqa: E402
    TaskSkillGuidanceInjector,
    load_task_skill_map,
)
from tts_search.validation_split_guidance import (  # noqa: E402
    patch_eval_data_with_guidance,
)


def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _first_model_payload(cfg: DictConfig) -> dict[str, Any]:
    model_list = list(_to_plain(cfg.litellm.model_list) or [])
    if not model_list:
        raise ValueError("cfg.litellm.model_list is empty")
    return dict(model_list[0])


def _generation_kwargs_from_litellm_params(
    litellm_params: dict[str, Any],
) -> dict[str, Any]:
    litellm_params = dict(_to_plain(litellm_params) or {})
    generation_keys = {
        "frequency_penalty",
        "include_usage",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "min_p",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_logprobs",
        "top_p",
    }
    generation_kwargs = {
        key: value
        for key, value in litellm_params.items()
        if key in generation_keys and value is not None
    }
    if litellm_params.get("extra_body") is not None:
        generation_kwargs["extra_body"] = dict(_to_plain(litellm_params["extra_body"]))
    return generation_kwargs


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _maybe_patch_eval_data_with_validation_guidance(
    cfg: DictConfig,
    *,
    hydra_dir: Path,
) -> Path:
    eval_data_path = Path(str(cfg.data.eval_data)).resolve()
    guidance_cfg = OmegaConf.select(cfg, "data.validation_split_guidance") or {}
    if not bool(guidance_cfg.get("enabled", False)):
        return eval_data_path

    instructions_path = _resolve_repo_path(
        str(
            guidance_cfg.get(
                "instructions_path",
                "docs/mlebench_validation_split_instructions_22.md",
            )
        )
    )
    if instructions_path is None:
        raise ValueError("data.validation_split_guidance.instructions_path is empty")

    output_filename = str(
        guidance_cfg.get(
            "output_filename",
            f"{eval_data_path.stem}_validation_split_guidance{eval_data_path.suffix}",
        )
    )
    output_path = hydra_dir / output_filename
    stats = patch_eval_data_with_guidance(
        eval_data_path=eval_data_path,
        output_path=output_path,
        instructions_path=instructions_path,
        input_key=str(cfg.data.input_key),
        metadata_key=str(cfg.data.metadata_key),
        insert_marker=str(guidance_cfg.get("insert_marker", "**FINAL OUTPUT**")),
        heading=str(guidance_cfg.get("heading", "Validation Split Guidance:")),
        strict=bool(guidance_cfg.get("strict", False)),
    )
    logger.info(
        "Validation split guidance patched eval_data: %s -> %s "
        "(patched_rows=%s/%s, skipped_existing=%s, missing_tasks=%s)",
        stats.input_path,
        stats.output_path,
        stats.patched_rows,
        stats.total_rows,
        stats.skipped_existing_rows,
        list(stats.missing_tasks),
    )
    return output_path


def _metadata_task_name(metadata: Any) -> str | None:
    metadata = _to_plain(metadata)
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if not isinstance(metadata, dict):
        return None
    for key in ("task_name", "competition_id", "name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _inject_task_skill_into_prompt_value(
    prompt_value: Any,
    *,
    task_name: str,
    injector: TaskSkillGuidanceInjector,
) -> tuple[Any, Any]:
    prompt_value = _to_plain(prompt_value)
    messages: list[Any] | None = None

    if hasattr(prompt_value, "tolist"):
        messages = list(prompt_value.tolist() or [])
    elif isinstance(prompt_value, (list, tuple)):
        messages = list(prompt_value)

    if messages is None:
        patched_prompt, result = injector.inject(str(prompt_value), task_name)
        return patched_prompt, result

    patched_messages = [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]
    for index, message in enumerate(patched_messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        patched_content, result = injector.inject(
            str(message.get("content", "")),
            task_name,
        )
        patched_messages[index] = {
            **message,
            "content": patched_content,
        }
        return patched_messages, result

    patched_prompt, result = injector.inject(str(prompt_value), task_name)
    return patched_prompt, result


def _maybe_patch_eval_data_with_task_skill_guidance(
    cfg: DictConfig,
    *,
    eval_data_path: Path,
    hydra_dir: Path,
) -> Path:
    guidance_cfg = OmegaConf.select(cfg, "data.task_skill_guidance") or {}
    if not bool(guidance_cfg.get("enabled", False)):
        return eval_data_path

    skills_dir = _resolve_repo_path(str(guidance_cfg.get("skills_dir") or "").strip())
    if skills_dir is None:
        raise ValueError("data.task_skill_guidance.skills_dir is empty")

    skill_map = load_task_skill_map(
        skills_dir,
        recursive=bool(guidance_cfg.get("recursive", True)),
    )
    if not skill_map:
        raise ValueError(f"No task skill Markdown files found in {skills_dir}")

    injector = TaskSkillGuidanceInjector(
        skill_map=skill_map,
        heading=str(
            guidance_cfg.get(
                "heading",
                "Task-Specific Skill Reference:",
            )
        ),
        intro=str(guidance_cfg.get("intro", "")),
        strict=bool(guidance_cfg.get("strict", False)),
    )
    input_key = str(cfg.data.input_key)
    metadata_key = str(cfg.data.metadata_key)
    output_filename = str(
        guidance_cfg.get(
            "output_filename",
            f"{eval_data_path.stem}_task_skill_guidance{eval_data_path.suffix}",
        )
    )
    output_path = hydra_dir / output_filename

    dataframe = pd.read_parquet(eval_data_path)
    changed = 0
    skipped_existing = 0
    missing: set[str] = set()
    patched_prompts: list[Any] = []
    for index, row in dataframe.iterrows():
        task_name = _metadata_task_name(row.get(metadata_key))
        if not task_name:
            task_name = f"<row:{index}>"
            missing.add(task_name)
            if injector.strict:
                raise ValueError(f"Missing task name in metadata row: {index}")
            patched_prompts.append(row.get(input_key))
            continue

        patched_prompt, result = _inject_task_skill_into_prompt_value(
            row.get(input_key),
            task_name=task_name,
            injector=injector,
        )
        patched_prompts.append(patched_prompt)
        if result.changed:
            changed += 1
        if result.skipped_existing:
            skipped_existing += 1
        if result.missing:
            missing.add(result.task_name)

    patched_dataframe = dataframe.copy()
    patched_dataframe[input_key] = patched_prompts
    patched_dataframe.to_parquet(output_path, index=False)
    logger.info(
        "Task skill guidance patched eval_data: %s -> %s "
        "(patched_rows=%s/%s, skipped_existing=%s, missing_tasks=%s)",
        eval_data_path,
        output_path,
        changed,
        len(dataframe),
        skipped_existing,
        sorted(missing),
    )
    return output_path


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/openmle_evo",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    disable_proxy_env()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(str(cfg.output_dir)).resolve()
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    task_root = hydra_dir / "airaevo_tasks"
    eval_data_path = _maybe_patch_eval_data_with_validation_guidance(
        cfg,
        hydra_dir=hydra_dir,
    )
    # Patch once before task preparation so all workers share one prompt.
    eval_data_path = _maybe_patch_eval_data_with_task_skill_guidance(
        cfg,
        eval_data_path=eval_data_path,
        hydra_dir=hydra_dir,
    )

    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    first_model = _first_model_payload(cfg)
    litellm_params = dict(first_model.get("litellm_params") or {})
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    default_task_root = str(
        search_runner_cfg.get("task_root", "third_party/aira-evo/examples/mle_bench")
    )
    package_root_value = str(
        search_runner_cfg.get(
            "package_root",
            default_task_root.removesuffix("/examples/mle_bench"),
        )
    )
    package_root = Path(package_root_value)
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value

    prepare_payload = {
        "data": {
            "eval_data": str(eval_data_path),
            "leaderboard_dir": (
                str(cfg.data.leaderboard_dir)
                if cfg.data.leaderboard_dir is not None
                else None
            ),
            "submit_dir": str(
                search_runner_cfg.get("submit_dir", "MLTasks/MLE-Bench-Lite-Modified")
            ),
            "submit_data_dir_root": (
                str(search_runner_cfg["submit_data_dir_root"])
                if search_runner_cfg.get("submit_data_dir_root") is not None
                else None
            ),
            "evaluation_protocol": str(
                search_runner_cfg.get("evaluation_protocol", "legacy")
            ),
            "validation_eval_split": search_runner_cfg.get("validation_eval_split"),
            "test_eval_split": search_runner_cfg.get("test_eval_split"),
            "selection_score_source": search_runner_cfg.get("selection_score_source"),
            "submit_job_timeout": search_runner_cfg.get("submit_job_timeout"),
            "submit_wait_timeout": search_runner_cfg.get("submit_wait_timeout"),
            "submit_repeats": int(search_runner_cfg.get("submit_repeats", 1) or 1),
            "input_key": str(cfg.data.input_key),
            "metadata_key": str(cfg.data.metadata_key),
        },
        "sandbox": {
            "base_url_map": [
                {
                    "resource": "cpu",
                    "base_url": str(
                        search_runner_cfg.get("cpu_base_url") or cfg.sandbox.base_url
                    ),
                },
                {
                    "resource": "gpu",
                    "base_url": str(
                        search_runner_cfg.get("gpu_base_url") or cfg.sandbox.base_url
                    ),
                },
            ],
            "job_timeout": int(cfg.sandbox.job_timeout),
            "wait_timeout": int(cfg.sandbox.wait_timeout),
            "poll_interval": int(cfg.sandbox.poll_interval),
            "use_score2reward": bool(cfg.sandbox.use_score2reward),
            "use_clear_run_log_score": bool(
                getattr(cfg.sandbox, "use_clear_run_log_score", False)
            ),
        },
    }

    runner_payload = {
        "experiment_name": str(cfg.experiment_name),
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "time_budget": cfg.time_budget,
        "model_plus_sandbox_time_budget": getattr(
            cfg, "model_plus_sandbox_time_budget", None
        ),
        "n_samples_per_task": int(cfg.n_samples_per_task),
        "candidates_per_step": candidates_per_step,
        "task_concurrency": (
            int(search_runner_cfg["task_concurrency"])
            if search_runner_cfg.get("task_concurrency") is not None
            else None
        ),
        "llm_concurrency": int(cfg.llm_concurrency),
        "sandbox_concurrency": int(cfg.sandbox.concurrency),
        "strict_resume": bool(search_runner_cfg.get("strict_resume", False)),
        "task_root": str(task_root),
        "task_list": list(search_runner_cfg.get("task_list", [])),
        "leaderboard_dir": (
            str(cfg.data.leaderboard_dir)
            if cfg.data.leaderboard_dir is not None
            else None
        ),
        "llm": {
            "api": str(search_runner_cfg.get("llm", {}).get("api", "litellm")),
            "model_id": str(first_model.get("model_name")),
            "base_url": str(litellm_params.get("base_url")),
            "generation_kwargs": llm_generation_kwargs,
            "use_azure_client": bool(
                search_runner_cfg.get("llm", {}).get("use_azure_client", False)
            ),
            "provider": str(
                search_runner_cfg.get("llm", {}).get("provider", "selfhosted")
            ),
        },
        "solver": dict(search_runner_cfg.get("solver") or {}),
        "interpreter": dict(search_runner_cfg.get("interpreter") or {}),
        "logger": dict(search_runner_cfg.get("logger") or {}),
        "hydra": {
            "run": {"dir": f"{output_dir}/.hydra"},
            "output_subdir": ".",
            "job": {"chdir": False},
        },
    }
    if cfg.max_steps is not None:
        runner_payload["solver"]["step_limit"] = int(cfg.max_steps)
    if candidates_per_step > 1:
        runner_payload["solver"]["individuals_per_generation"] = candidates_per_step

    prepare_config_path = hydra_dir / "airaevo_prepare_config.yaml"
    runner_config_path = hydra_dir / "airaevo_runner_config.yaml"
    _write_yaml(prepare_config_path, prepare_payload)
    _write_yaml(runner_config_path, runner_payload)

    build_tasks_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
        "--output-root",
        str(task_root),
    ]
    logger.info("Building AIRA Dojo tasks with %s", prepare_config_path)
    subprocess.run(build_tasks_cmd, check=True, cwd=REPO_ROOT)

    runner_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    logger.info("Running AIRA Dojo evaluation with %s", runner_config_path)
    subprocess.run(runner_cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
