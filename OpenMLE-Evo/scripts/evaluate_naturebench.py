from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TTS_SEARCH_DIR = REPO_ROOT / "tts_search"
logger = logging.getLogger(__name__)

from tts_search.config_security import redact_sensitive_yaml_file  # noqa: E402
from tts_search.eval_network import disable_proxy_env  # noqa: E402


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


def _subprocess_env(*extra_paths: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(REPO_ROOT)]
    python_paths.extend(str(path) for path in extra_paths if path.exists())
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _data_value(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return _to_plain(OmegaConf.select(cfg, f"data.{key}", default=default))


def _build_payloads(
    cfg: DictConfig,
    *,
    output_dir: Path,
    task_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    solver_payload = dict(search_runner_cfg.get("solver") or {})
    first_model = _first_model_payload(cfg)
    litellm_params = dict(first_model.get("litellm_params") or {})
    llm_generation_kwargs = _generation_kwargs_from_litellm_params(litellm_params)
    candidates_per_step = int(getattr(cfg, "candidates_per_step", 1) or 1)
    execution_timeout = _data_value(cfg, "execution_timeout")
    if execution_timeout is None:
        execution_timeout = OmegaConf.select(cfg, "sandbox.job_timeout")
    if execution_timeout is None:
        execution_timeout = solver_payload.get("execution_timeout")

    prepare_payload = {
        "data": {
            "naturebench_root": str(_data_value(cfg, "naturebench_root")),
            "tasks_root": _data_value(cfg, "tasks_root"),
            "task_list": list(_data_value(cfg, "task_list", []) or []),
            "task_set_path": _data_value(cfg, "task_set_path"),
            "eval_service_url": str(
                _data_value(cfg, "eval_service_url", "http://127.0.0.1:8321")
            ),
            "batch_name": str(_data_value(cfg, "batch_name", str(cfg.experiment_name))),
            "execution_mode": str(_data_value(cfg, "execution_mode", "docker")),
            "workspace_root": _data_value(cfg, "workspace_root"),
            "docker_image": _data_value(cfg, "docker_image"),
            "gpu_devices": _data_value(cfg, "gpu_devices"),
            "local_python": _data_value(cfg, "local_python"),
            "local_conda_env": _data_value(cfg, "local_conda_env"),
            "local_conda_executable": _data_value(
                cfg, "local_conda_executable", "conda"
            ),
            "local_terminate_grace_seconds": _data_value(
                cfg, "local_terminate_grace_seconds", 5
            ),
            "candidate_env_allowlist": _data_value(cfg, "candidate_env_allowlist", []),
            "scm_host": _data_value(cfg, "scm_host"),
            "scm_workspace_root": _data_value(cfg, "scm_workspace_root"),
            "scm_task_roots": _data_value(cfg, "scm_task_roots"),
            "scm_task_set_host": _data_value(cfg, "scm_task_set_host"),
            "scm_task_set_root": _data_value(cfg, "scm_task_set_root"),
            "scm_resource_lines": _data_value(cfg, "scm_resource_lines"),
            "scm_eval_service_url": _data_value(cfg, "scm_eval_service_url"),
            "scm_container_eval_service_url": _data_value(
                cfg, "scm_container_eval_service_url"
            ),
            "candidate_preflight": _data_value(cfg, "candidate_preflight", True),
            "candidate_preflight_imports": _data_value(
                cfg, "candidate_preflight_imports", True
            ),
            "candidate_preflight_timeout": _data_value(
                cfg, "candidate_preflight_timeout", 45
            ),
            "scm_gpu_wait_timeout": _data_value(cfg, "scm_gpu_wait_timeout"),
            "visible_data_analysis_root": _data_value(
                cfg, "visible_data_analysis_root"
            ),
            "submit_repeats": int(
                1
                if _data_value(cfg, "submit_repeats", None) is None
                else _data_value(cfg, "submit_repeats")
            ),
            "eval_timeout": int(_data_value(cfg, "eval_timeout", 3600) or 3600),
            "public_system_prompt": str(_data_value(cfg, "public_system_prompt", "")),
            "public_user_prompt": str(_data_value(cfg, "public_user_prompt", "")),
        }
    }
    if execution_timeout is not None:
        prepare_payload["data"]["execution_timeout"] = int(execution_timeout)

    runner_payload = {
        "experiment_name": str(cfg.experiment_name),
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "time_budget": cfg.time_budget,
        "model_plus_sandbox_time_budget": getattr(
            cfg,
            "model_plus_sandbox_time_budget",
            None,
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
        "leaderboard_dir": None,
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
        "solver": solver_payload,
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
    return prepare_payload, runner_payload


@hydra.main(
    config_path=f"{TTS_SEARCH_DIR}/configs",
    config_name="experiment/naturebench_smoke",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    disable_proxy_env()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(str(cfg.output_dir)).resolve()
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    task_root = hydra_dir / "naturebench_tasks"

    search_runner_cfg = dict(_to_plain(cfg.search.runner) or {})
    package_root_value = str(
        search_runner_cfg.get("package_root", "third_party/aira-evo")
    )
    package_root = Path(package_root_value)
    if not package_root.is_absolute():
        package_root = REPO_ROOT / package_root_value

    prepare_payload, runner_payload = _build_payloads(
        cfg,
        output_dir=output_dir,
        task_root=task_root,
    )
    first_model = _first_model_payload(cfg)
    llm_api_key = dict(first_model.get("litellm_params") or {}).get("api_key")
    prepare_config_path = hydra_dir / "naturebench_prepare_config.yaml"
    runner_config_path = hydra_dir / "naturebench_runner_config.yaml"
    _write_yaml(prepare_config_path, prepare_payload)
    _write_yaml(runner_config_path, runner_payload)
    for hydra_config_path in sorted(hydra_dir.glob("*.yaml")):
        redact_sensitive_yaml_file(hydra_config_path)

    build_tasks_cmd = [
        sys.executable,
        str(package_root / "examples" / "nature_bench" / "build_tasks.py"),
        "--config",
        str(prepare_config_path),
        "--output-root",
        str(task_root),
    ]
    logger.info("Building NatureBench AIRA-Evo tasks with %s", prepare_config_path)
    env = _subprocess_env(package_root / "src")
    subprocess.run(build_tasks_cmd, check=True, cwd=REPO_ROOT, env=env)
    if llm_api_key is not None:
        env["OPENMLE_LLM_API_KEY"] = str(llm_api_key)

    runner_cmd = [
        sys.executable,
        str(package_root / "examples" / "mle_bench" / "runner.py"),
        "--config-path",
        str(runner_config_path.parent),
        "--config-name",
        runner_config_path.stem,
    ]
    logger.info("Running NatureBench AIRA-Evo evaluation with %s", runner_config_path)
    subprocess.run(runner_cmd, check=True, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    main()
