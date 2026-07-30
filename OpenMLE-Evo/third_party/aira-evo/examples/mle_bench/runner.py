from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tts_search.airaevo_concurrency import (
    SharedConcurrencyServer,
    resolve_task_concurrency_per_epoch,
)
from tts_search.airaevo_async_resources import (
    resolve_async_worker_count,
    resolve_execution_mode,
)
from tts_search.config_security import (
    redact_sensitive_config as _redact_sensitive_config,
)
from tts_search import eval_utils
from tts_search.eval_network import disable_proxy_env


def _load_task_name(task_dir: Path) -> str:
    config_path = task_dir / "config.yaml"
    if not config_path.exists():
        return task_dir.name

    task_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return str(task_cfg.get("task_name") or task_dir.name)


def _write_failed_task_outputs(
    output_dir: Path,
    task_name: str,
    error: str,
    return_code: int | None,
) -> None:
    (output_dir / "runner_error.txt").write_text(error, encoding="utf-8")

    stat_path = output_dir / "stat.json"
    if stat_path.exists():
        return

    failure_stat = {
        "task_name": task_name,
        "steps": [],
        "num_steps": 0,
        "step_limit": 0,
        "num_generations": 0,
        "individuals_per_generation": 0,
        "final_score": None,
        "final_reward": 0.0,
        "final_step": None,
        "submit_score": None,
        "submit_reward": 0.0,
        "submit_grade": 1.0,
        "submit_medal": "N/A",
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "val_time": 0.0,
        "test_time": 0.0,
        "total_time": 0.0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
        "sandbox_12h_score": 0.0,
        "model_plus_sandbox_12h_score": 0.0,
        "status_count": {"unknown": 1},
        "error": error,
        "return_code": return_code,
    }
    stat_path.write_text(json.dumps(failure_stat, indent=2), encoding="utf-8")


def _has_completed_task_outputs(output_dir: Path) -> bool:
    base_required_paths = [
        output_dir / "stat.json",
        output_dir / "valid_code_final.py",
        output_dir / "submit_code.py",
    ]
    if not all(path.exists() for path in base_required_paths):
        return False

    try:
        task_stat = json.loads(
            (output_dir / "stat.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning(
            "Ignoring unreadable resume stat %s: %s",
            output_dir / "stat.json",
            exc,
        )
        return False
    if (
        str(task_stat.get("benchmark") or "").strip().lower() == "naturebench"
        and task_stat.get("final_submit_skipped") is True
    ):
        return task_stat.get("submit_score") is not None

    submit_required_paths = [
        output_dir / "submit_raw_run_log.txt",
        output_dir / "submit_clear_run_log.txt",
        output_dir / "submit_feedback.txt",
    ]
    if not all(path.exists() for path in submit_required_paths):
        return False
    return task_stat.get("submit_score") is not None


async def run_task(
    *,
    task_dir: Path,
    epoch_index: int,
    output_root: Path,
    resolved_runner_config_path: Path,
    task_concurrency: asyncio.Semaphore,
    subprocess_env: dict[str, str],
    strict_resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / f"program_ep_{epoch_index}" / task_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    task_name = _load_task_name(task_dir)

    if strict_resume and _has_completed_task_outputs(output_dir):
        logging.info("Skipping completed task sample: %s ep=%d", task_name, epoch_index)
        runner_error_path = output_dir / "runner_error.txt"
        if runner_error_path.exists():
            runner_error_path.unlink()
        return {
            "success": True,
            "skipped": True,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": 0,
            "error": None,
        }

    runner_script = Path(__file__).resolve().with_name("single_task_runner.py")
    cmd = [
        sys.executable,
        str(runner_script),
        "--task-dir",
        str(task_dir),
        "--output-dir",
        str(output_dir),
        "--runner-config",
        str(resolved_runner_config_path),
        "--sample-index",
        str(epoch_index),
    ]

    try:
        async with task_concurrency:
            process = await asyncio.create_subprocess_exec(*cmd, env=subprocess_env)
            return_code = await process.wait()
    except Exception as exc:
        error = f"AIRA Dojo task crashed before completion: {task_name}\n{exc}"
        logging.exception(error)
        _write_failed_task_outputs(
            output_dir=output_dir,
            task_name=task_name,
            error=error,
            return_code=None,
        )
        return {
            "success": False,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": None,
            "error": error,
        }

    if return_code != 0:
        error = f"AIRA Dojo task failed: {task_name} (return_code={return_code})"
        logging.error(error)
        _write_failed_task_outputs(
            output_dir=output_dir,
            task_name=task_name,
            error=error,
            return_code=return_code,
        )
        return {
            "success": False,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": return_code,
            "error": error,
        }

    runner_error_path = output_dir / "runner_error.txt"
    if runner_error_path.exists():
        runner_error_path.unlink()

    return {
        "success": True,
        "skipped": False,
        "task_name": task_name,
        "epoch_index": epoch_index,
        "output_dir": str(output_dir),
        "return_code": return_code,
        "error": None,
    }


@hydra.main(config_path=".", config_name="runner_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    disable_proxy_env()

    output_root = Path(cfg.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_runner_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_runner_config, dict):
        raise TypeError("Resolved runner configuration must be a mapping")
    llm_cfg = resolved_runner_config.get("llm")
    if isinstance(llm_cfg, dict) and llm_cfg.get("api_key") is not None:
        os.environ["OPENMLE_LLM_API_KEY"] = str(llm_cfg["api_key"])

    resolved_runner_config_path = output_root / "runner_resolved.yaml"
    resolved_runner_config_path.write_text(
        yaml.safe_dump(
            _redact_sensitive_config(resolved_runner_config),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    task_root = Path(cfg.task_root).resolve()
    task_names = list(cfg.task_list) if cfg.task_list else None
    if task_names is None:
        task_names = sorted(
            [
                path.name
                for path in task_root.iterdir()
                if path.is_dir() and (path / "config.yaml").exists()
            ]
        )

    num_epochs = int(cfg.n_samples_per_task)
    llm_concurrency = int(cfg.llm_concurrency)
    sandbox_concurrency = int(cfg.sandbox_concurrency)
    async_workers = OmegaConf.select(cfg, "solver.async_workers", default=1)
    execution_mode = resolve_execution_mode(
        OmegaConf.select(cfg, "solver.execution_mode", default="generation")
    )
    resolved_async_workers = resolve_async_worker_count(async_workers)
    if execution_mode == "generation" and resolved_async_workers != 1:
        raise ValueError(
            "generation mode requires exactly one worker; "
            "use execution=multi_gpu for async steady-state search"
        )
    strict_resume = bool(getattr(cfg, "strict_resume", False))
    task_concurrency = resolve_task_concurrency_per_epoch(
        configured_task_concurrency=(
            int(cfg.task_concurrency) if cfg.task_concurrency is not None else None
        ),
        llm_concurrency=llm_concurrency,
        sandbox_concurrency=sandbox_concurrency,
        num_epochs=num_epochs,
        async_workers=resolved_async_workers,
    )
    task_dirs = [task_root / name for name in task_names]
    epoch_semaphores = {
        epoch_index: asyncio.Semaphore(task_concurrency)
        for epoch_index in range(num_epochs)
    }
    concurrency_server = SharedConcurrencyServer(
        llm_concurrency=llm_concurrency,
        sandbox_concurrency=sandbox_concurrency,
    )
    subprocess_env = os.environ.copy()
    subprocess_env.update(concurrency_server.env)

    async def run_all() -> list[dict[str, Any]]:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for task_dir in task_dirs:
            for epoch_index in range(num_epochs):
                tasks.append(
                    asyncio.create_task(
                        run_task(
                            task_dir=task_dir,
                            epoch_index=epoch_index,
                            output_root=output_root,
                            resolved_runner_config_path=resolved_runner_config_path,
                            task_concurrency=epoch_semaphores[epoch_index],
                            subprocess_env=subprocess_env,
                            strict_resume=strict_resume,
                        )
                    )
                )
        return await asyncio.gather(*tasks)

    try:
        task_results = asyncio.run(run_all())
    finally:
        concurrency_server.shutdown()

    failures = [result for result in task_results if not result["success"]]
    skipped = [result for result in task_results if result.get("skipped")]
    (output_root / "runner_failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    if failures:
        logging.warning(
            "AIRA Dojo runner completed with %d failed tasks.", len(failures)
        )

    task_metadata_map: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        task_cfg = yaml.safe_load(
            (task_dir / "config.yaml").read_text(encoding="utf-8")
        )
        task_metadata_map[str(task_cfg["task_name"])] = task_cfg
    expected_task_names = list(task_metadata_map)

    for epoch_index in range(num_epochs):
        eval_utils.write_epoch_stat(
            output_root / f"program_ep_{epoch_index}",
            expected_task_names=expected_task_names,
        )

    eval_utils.write_summary_csv(
        output_root,
        task_metadata_map,
        expected_task_names=expected_task_names,
    )
    eval_utils.write_global_stat(output_root)

    (output_root / "runner_manifest.json").write_text(
        json.dumps(
            {
                "experiment_name": cfg.experiment_name,
                "task_root": str(task_root),
                "task_names": task_names,
                "n_samples_per_task": num_epochs,
                "llm_concurrency": llm_concurrency,
                "sandbox_concurrency": sandbox_concurrency,
                "execution_mode": execution_mode,
                "async_workers": resolved_async_workers,
                "requested_async_workers": async_workers,
                "resolved_async_workers": resolved_async_workers,
                "strict_resume": strict_resume,
                "skipped_completed_tasks": len(skipped),
                "launched_tasks": len(task_results) - len(skipped),
                "task_concurrency_per_epoch": task_concurrency,
                "max_concurrent_tasks": num_epochs * task_concurrency,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
