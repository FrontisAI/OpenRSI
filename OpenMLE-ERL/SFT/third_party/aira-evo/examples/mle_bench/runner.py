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

from tts_search.airaevo_concurrency import (
    SharedConcurrencyServer,
    resolve_task_concurrency_per_epoch,
)
from tts_search import eval_utils
from tts_search.eval_network import disable_proxy_env
from tts_search.services.result_persistence import append_compact_progress_snapshot


TERMINAL_SKIP_DONE_REASONS = {
    "accepted_target_reached",
    "step_limit_reached",
    "aira_time_limit_reached",
    "aira_step_limit_reached",
    "aira_num_generations_exhausted",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_task_config(task_dir: Path) -> dict[str, Any]:
    config_path = task_dir / "config.yaml"
    if not config_path.exists():
        return {}
    return dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})


def _load_task_name(task_dir: Path) -> str:
    task_cfg = _load_task_config(task_dir)
    return str(task_cfg.get("task_name") or task_dir.name)


def _infer_legacy_done_reason(
    *,
    task_state: dict[str, Any],
    output_dir: Path,
    runner_cfg: dict[str, Any],
) -> str | None:
    if not bool(task_state.get("done")):
        return None

    done_reason = task_state.get("done_reason")
    if done_reason:
        return str(done_reason)

    solver_cfg = dict(runner_cfg.get("solver") or {})
    accepted_target = _safe_int(
        task_state.get("accepted_target")
        or task_state.get("target")
        or runner_cfg.get("rejection_target")
        or runner_cfg.get("accepted_target")
        or runner_cfg.get("n_samples_per_task"),
        1,
    )
    if _safe_int(task_state.get("accepted")) >= accepted_target:
        return "accepted_target_reached"

    requested_step_limit = _safe_int(
        task_state.get("max_target") or solver_cfg.get("step_limit"),
        0,
    )
    if requested_step_limit and _safe_int(task_state.get("completed")) >= requested_step_limit:
        return "step_limit_reached"

    checkpoint_path = output_dir / "aira_evo" / "checkpoint" / "state.json"
    if checkpoint_path.exists():
        try:
            checkpoint_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            checkpoint_state = {}

        time_limit_secs = _safe_float(
            runner_cfg.get("model_plus_sandbox_time_budget")
            or solver_cfg.get("time_limit_secs")
            or 86400.0,
            86400.0,
        )
        if _safe_float(checkpoint_state.get("running_time")) >= time_limit_secs:
            return "aira_time_limit_reached"

        # single_task_runner passes requested_step_limit + 1 to AIRA because
        # the AIRA journal contains a root node.
        aira_step_limit = _safe_int(solver_cfg.get("step_limit"), requested_step_limit)
        if aira_step_limit:
            aira_step_limit += 1
        if aira_step_limit and _safe_int(checkpoint_state.get("current_step")) >= aira_step_limit:
            return "aira_step_limit_reached"

        num_generations = _safe_int(solver_cfg.get("num_generations"), 0)
        if (
            num_generations
            and "current_generation" in checkpoint_state
            and _safe_int(checkpoint_state.get("current_generation")) >= num_generations
        ):
            return "aira_num_generations_exhausted"

    return "legacy_done_without_reason"


def _terminal_done_reason_for_task(
    *,
    output_root: Path,
    task_dir: Path,
    epoch_index: int,
    runner_cfg: dict[str, Any],
) -> str | None:
    progress_path = output_root / "progress.json"
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    task_cfg = _load_task_config(task_dir)
    task_id = str(task_cfg.get("uuid") or task_dir.name)
    task_states = dict(progress.get("task_states") or {})
    task_state = task_states.get(task_id)
    if not isinstance(task_state, dict):
        return None

    output_dir = output_root / f"program_ep_{epoch_index}" / task_dir.name
    done_reason = _infer_legacy_done_reason(
        task_state=task_state,
        output_dir=output_dir,
        runner_cfg=runner_cfg,
    )
    if done_reason in TERMINAL_SKIP_DONE_REASONS:
        return done_reason
    return None


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
        "submit_grade": None,
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


async def run_task(
    *,
    task_dir: Path,
    epoch_index: int,
    output_root: Path,
    resolved_runner_config_path: Path,
    runner_cfg: dict[str, Any],
    task_concurrency: asyncio.Semaphore,
    subprocess_env: dict[str, str],
) -> dict[str, Any]:
    output_dir = output_root / f"program_ep_{epoch_index}" / task_dir.name
    task_name = _load_task_name(task_dir)
    skip_reason = _terminal_done_reason_for_task(
        output_root=output_root,
        task_dir=task_dir,
        epoch_index=epoch_index,
        runner_cfg=runner_cfg,
    )
    if skip_reason is not None:
        logging.info("Skipping completed AIRA Dojo task %s (%s).", task_name, skip_reason)
        return {
            "success": True,
            "skipped": True,
            "skip_reason": skip_reason,
            "task_name": task_name,
            "epoch_index": epoch_index,
            "output_dir": str(output_dir),
            "return_code": 0,
            "error": None,
        }

    output_dir.mkdir(parents=True, exist_ok=True)

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

    return {
        "success": True,
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

    resolved_runner_config_path = output_root / "runner_resolved.yaml"
    resolved_runner_config_path.write_text(
        yaml.safe_dump(
            OmegaConf.to_container(cfg, resolve=True),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner_cfg = dict(yaml.safe_load(resolved_runner_config_path.read_text(encoding="utf-8")) or {})

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
    task_concurrency = resolve_task_concurrency_per_epoch(
        configured_task_concurrency=(
            int(cfg.task_concurrency) if cfg.task_concurrency is not None else None
        ),
        llm_concurrency=llm_concurrency,
        sandbox_concurrency=sandbox_concurrency,
        num_epochs=num_epochs,
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
        progress_history_interval = max(
            1.0,
            float(getattr(cfg, "progress_history_interval_seconds", 600.0) or 600.0),
        )
        done_event = asyncio.Event()

        async def progress_history_loop() -> None:
            append_compact_progress_snapshot(output_root)
            while not done_event.is_set():
                try:
                    await asyncio.wait_for(
                        done_event.wait(),
                        timeout=progress_history_interval,
                    )
                except asyncio.TimeoutError:
                    pass
                if not done_event.is_set():
                    append_compact_progress_snapshot(output_root)
            append_compact_progress_snapshot(output_root)

        progress_task = asyncio.create_task(progress_history_loop())
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        try:
            for task_dir in task_dirs:
                for epoch_index in range(num_epochs):
                    tasks.append(
                        asyncio.create_task(
                            run_task(
                                task_dir=task_dir,
                                epoch_index=epoch_index,
                                output_root=output_root,
                                resolved_runner_config_path=resolved_runner_config_path,
                                runner_cfg=runner_cfg,
                                task_concurrency=epoch_semaphores[epoch_index],
                                subprocess_env=subprocess_env,
                            )
                        )
                    )
            return await asyncio.gather(*tasks)
        finally:
            done_event.set()
            await progress_task

    try:
        task_results = asyncio.run(run_all())
    finally:
        concurrency_server.shutdown()

    failures = [result for result in task_results if not result["success"]]
    (output_root / "runner_failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    if failures:
        logging.warning("AIRA Dojo runner completed with %d failed tasks.", len(failures))

    task_metadata_map: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        task_cfg = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8"))
        task_metadata_map[str(task_cfg["task_name"])] = task_cfg

    for epoch_index in range(num_epochs):
        eval_utils.write_epoch_stat(output_root / f"program_ep_{epoch_index}")

    eval_utils.write_summary_csv(output_root, task_metadata_map)
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
                "task_concurrency_per_epoch": task_concurrency,
                "max_concurrent_tasks": num_epochs * task_concurrency,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
