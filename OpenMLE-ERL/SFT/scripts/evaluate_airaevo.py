from __future__ import annotations

import logging
import os
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

from tts_search.eval_network import disable_proxy_env


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


def _configure_airaevo_api_key_env(litellm_params: dict[str, Any]) -> None:
    """Bridge Hydra LiteLLM keys to the AIRA Evo backend env convention."""
    api_key = str(litellm_params.get("api_key") or "").strip()
    if not api_key or api_key.upper() == "EMPTY":
        return
    os.environ.setdefault("PRIMARY_KEY", api_key)


def _sanitize_task_name(task_name: str) -> str:
    return task_name.replace("/", "_").replace("\\", "_")


def _task_list_from_eval_data(cfg: DictConfig) -> list[str]:
    configured = list(_to_plain(cfg.search.runner.get("task_list", [])) or [])
    if configured:
        return [str(item) for item in configured]

    records = pd.read_parquet(
        str(cfg.data.eval_data),
        columns=[str(cfg.data.metadata_key)],
    ).to_dict(orient="records")
    task_names: list[str] = []
    for record in records:
        metadata = dict(record[str(cfg.data.metadata_key)])
        task_names.append(_sanitize_task_name(str(metadata["task_name"])))
    if cfg.max_tasks is not None:
        task_names = task_names[: int(cfg.max_tasks)]
    return task_names


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/evolutionary_glm47",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    load_dotenv(REPO_ROOT / ".env")
    disable_proxy_env()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(str(cfg.output_dir)).resolve()
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    first_model = _first_model_payload(cfg)
    litellm_params = dict(first_model.get("litellm_params") or {})
    _configure_airaevo_api_key_env(litellm_params)
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    task_list = _task_list_from_eval_data(cfg)
    default_task_root = str(
        search_runner_cfg.get("task_root", "third_party/aira-evo/examples/mle_bench")
    )
    package_root_value = str(
        search_runner_cfg.get(
            "package_root",
            default_task_root.removesuffix("/examples/mle_bench"),
        )
    )

    prepare_payload = {
        "data": {
            "eval_data": str(cfg.data.eval_data),
            "leaderboard_dir": (
                str(cfg.data.leaderboard_dir)
                if cfg.data.leaderboard_dir is not None
                else None
            ),
            "submit_dir": str(
                search_runner_cfg.get("submit_dir", "MLTasks/MLE-Bench-Lite-Modified")
            ),
            "submit_data_dir_root": str(
                search_runner_cfg.get(
                    "submit_data_dir_root",
                    "",
                )
            ),
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
        "accepted_target": int(
            getattr(cfg, "rejection_target", None) or cfg.n_samples_per_task
        ),
        "rejection_policy": getattr(cfg, "rejection_policy", None),
        "rejection_target": getattr(cfg, "rejection_target", None),
        "rejection_score_threshold": getattr(cfg, "rejection_score_threshold", None),
        "rejection_reward_threshold": getattr(cfg, "rejection_reward_threshold", None),
        "rejection_reference_scores_path": getattr(
            cfg, "rejection_reference_scores_path", None
        ),
        "rejection_accepted_medals": _to_plain(
            getattr(cfg, "rejection_accepted_medals", [])
        ),
        "rejection_apply_baseline_filters": bool(
            getattr(cfg, "rejection_apply_baseline_filters", True)
        ),
        "rejection_baseline_token_limit": int(
            getattr(cfg, "rejection_baseline_token_limit", 32768)
        ),
        "rejection_baseline_tokenizer_model": getattr(
            cfg, "rejection_baseline_tokenizer_model", None
        ),
        "rejection_baseline_relative_gap_limit": float(
            getattr(cfg, "rejection_baseline_relative_gap_limit", 0.12)
        ),
        "rejection_mixed_leaderboard_target": int(
            getattr(cfg, "rejection_mixed_leaderboard_target", 2)
        ),
        "rejection_mixed_no_leaderboard_target": int(
            getattr(cfg, "rejection_mixed_no_leaderboard_target", 4)
        ),
        "candidates_per_step": candidates_per_step,
        "task_concurrency": (
            int(search_runner_cfg["task_concurrency"])
            if search_runner_cfg.get("task_concurrency") is not None
            else None
        ),
        "llm_concurrency": int(cfg.llm_concurrency),
        "sandbox_concurrency": int(cfg.sandbox.concurrency),
        "strict_resume": bool(search_runner_cfg.get("strict_resume", False)),
        "progress_history_interval_seconds": float(
            getattr(cfg, "progress_history_interval_seconds", 600.0) or 600.0
        ),
        "task_root": default_task_root,
        "task_list": task_list,
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

    package_root = Path(package_root_value)
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value
    mle_bench_dir = package_root / "examples" / "mle_bench"
    if not mle_bench_dir.is_dir():
        raise FileNotFoundError(
            f"AIRA Evo MLE bridge directory does not exist: {mle_bench_dir}"
        )

    subprocess_env = os.environ.copy()
    python_path_entries = [str(REPO_ROOT), str(package_root / "src")]
    existing_python_path = subprocess_env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    subprocess_env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    build_tasks_cmd = [
        sys.executable,
        str(mle_bench_dir / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
    ]
    logger.info("Building AIRA Evo tasks with %s", prepare_config_path)
    subprocess.run(
        build_tasks_cmd,
        check=True,
        cwd=REPO_ROOT,
        env=subprocess_env,
    )

    runner_cmd = [
        sys.executable,
        str(mle_bench_dir / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    logger.info("Running AIRA Evo evaluation with %s", runner_config_path)
    subprocess.run(
        runner_cmd,
        check=True,
        cwd=REPO_ROOT,
        env=subprocess_env,
    )


if __name__ == "__main__":
    main()
