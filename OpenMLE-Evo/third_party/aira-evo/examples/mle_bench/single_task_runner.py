#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv


class StopSearch(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _benchmark_name(task_cfg: dict[str, Any]) -> str:
    return str(task_cfg.get("benchmark") or "mlebench").strip().lower()


def _solver_defaults_relative_path(task_cfg: dict[str, Any]) -> Path:
    configured = task_cfg.get("solver_defaults")
    if configured:
        return Path(str(configured))
    if _benchmark_name(task_cfg) == "naturebench":
        return Path("naturebench/evo.yaml")
    return Path("mlebench/evo.yaml")


def _operator_paths_for_benchmark(
    benchmark: str,
    *,
    experience_enabled: bool,
) -> dict[str, str]:
    family = "naturebench" if str(benchmark).strip().lower() == "naturebench" else "mlebench"
    prefix = f"{family}/aira_operators"
    paths = {
        "draft": f"{prefix}/draft.yaml",
        "improve": (
            f"{prefix}/improve_experience.yaml"
            if experience_enabled
            else f"{prefix}/improve.yaml"
        ),
        "debug": (
            f"{prefix}/debug_experience.yaml"
            if experience_enabled
            else f"{prefix}/debug.yaml"
        ),
        "crossover": (
            f"{prefix}/crossover_experience.yaml"
            if experience_enabled
            else f"{prefix}/crossover.yaml"
        ),
        "analyze": (
            "naturebench/aira_operators/analyze.yaml"
            if family == "naturebench"
            else "mlebench/aide_operators/analyze.yaml"
        ),
    }
    if experience_enabled:
        paths["rich_memory_summary"] = f"{prefix}/rich_memory_summary.yaml"
    return paths


def _load_naturebench_task_class(example_dir: Path) -> type[Any]:
    module_path = example_dir.parent / "nature_bench" / "base_task.py"
    spec = importlib.util.spec_from_file_location(
        "naturebench_base_task_runtime",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load NatureBenchTask from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NatureBenchTask


def _strip_target(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "_target_"}


def _merge_generation_kwargs(
    base_kwargs: dict[str, Any],
    override_kwargs: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_kwargs)
    override_payload = dict(override_kwargs)
    if "extra_body" in base_kwargs and "extra_body" in override_kwargs:
        override_payload["extra_body"] = {
            **dict(base_kwargs["extra_body"] or {}),
            **dict(override_kwargs["extra_body"] or {}),
        }
    merged.update(override_payload)
    return merged


def _prompt_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content or "")


def _extract_prompt_messages(
    operator_metric: dict[str, Any] | None,
) -> tuple[str, str, str]:
    if not operator_metric:
        return "", "", ""

    prompt_messages = list(operator_metric.get("prompt_messages") or [])
    completion_text = str(operator_metric.get("completion_text") or "")
    system_prompt = ""
    user_prompt = ""
    for message in prompt_messages:
        if not system_prompt and str(message.get("role")) == "system":
            system_prompt = _prompt_content(message)
        if str(message.get("role")) == "user":
            user_prompt = _prompt_content(message)
    return system_prompt, user_prompt, completion_text


def _sum_operator_usage(operator_metrics: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "cost": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "latency": 0.0,
    }
    for operator_metric in operator_metrics:
        usage = dict(operator_metric.get("usage") or {})
        totals["cost"] += float(usage.get("cost") or 0.0)
        totals["prompt_tokens"] += float(usage.get("prompt_tokens") or 0.0)
        totals["completion_tokens"] += float(usage.get("completion_tokens") or 0.0)
        totals["total_tokens"] += float(usage.get("total_tokens") or 0.0)
        totals["latency"] += float(usage.get("latency") or 0.0)
    return totals


def _node_sort_key(node: Any) -> tuple[int, str]:
    step = getattr(node, "step", None)
    try:
        step_value = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_value = -1
    return step_value, str(getattr(node, "id", ""))


def _node_aux_info(node: Any) -> dict[str, Any]:
    if getattr(node, "metric", None) is None:
        return {}
    return dict(getattr(node.metric, "info", None) or {})


def _node_selection_score(node: Any) -> Any | None:
    if getattr(node, "metric", None) is None:
        return None
    aux_info = _node_aux_info(node)
    if "selection_score" in aux_info:
        return aux_info.get("selection_score")
    return aux_info.get("validation_score")


def _naturebench_successful_aux_info(aux_info: dict[str, Any]) -> bool:
    status_code = _coerce_float_value(aux_info.get("status_code"))
    if status_code != 200:
        return False
    status = str(aux_info.get("status") or "").strip().lower()
    return status not in {"buggy", "error", "failed", "failure", "timeout"}


def _naturebench_score_from_aux_info(aux_info: dict[str, Any]) -> float | None:
    for key in (
        "aggregate_improvement",
        "raw_aggregate_improvement",
        "raw_score",
        "score",
        "selection_score",
        "validation_score",
    ):
        score = _coerce_float_value(aux_info.get(key))
        if score is not None:
            return score
    return None


def _naturebench_score_from_node(node: Any) -> float | None:
    return _naturebench_score_from_aux_info(_node_aux_info(node))


def _naturebench_valid_score_from_node(node: Any) -> float | None:
    aux_info = _node_aux_info(node)
    if not _naturebench_successful_aux_info(aux_info):
        return None
    return _naturebench_score_from_aux_info(aux_info)


def _is_naturebench_successful_node(node: Any) -> bool:
    return _naturebench_successful_aux_info(_node_aux_info(node))


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _nested_cfg_get(payload: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _should_run_final_submit(
    *,
    benchmark: str,
    submit_repeats: int,
    task_cfg: dict[str, Any],
    runner_cfg: dict[str, Any],
) -> bool:
    if int(submit_repeats) <= 0:
        return False
    if str(benchmark).strip().lower() != "naturebench":
        return True

    configured = _nested_cfg_get(
        runner_cfg,
        ("solver", "final_submit"),
        task_cfg.get("final_submit", False),
    )
    return _coerce_bool(configured, default=False)


def _effective_submit_score(
    *,
    benchmark: str,
    submit_score: Any,
    final_selection_score: Any,
    final_submit_enabled: bool,
) -> float | None:
    numeric_submit_score = _coerce_float_value(submit_score)
    if (
        str(benchmark).strip().lower() == "naturebench"
        and not final_submit_enabled
        and numeric_submit_score is None
    ):
        return _coerce_float_value(final_selection_score)
    return numeric_submit_score


def _final_score_source(selected_source: str | None, aux_source: Any | None) -> str | None:
    if selected_source == "random_no_valid":
        return "random_no_valid"
    return str(aux_source) if aux_source is not None else selected_source


def _coerce_float_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


NATUREBENCH_SURPASS_SOTA_THRESHOLD = 0.1


def _naturebench_leaderboard_flags(score: Any) -> dict[str, bool]:
    """Return the official Lite/leaderboard threshold flags for a valid g score."""
    numeric_score = _coerce_float_value(score)
    return {
        "match_sota": numeric_score is not None and numeric_score >= 0.0,
        "surpass_sota": (
            numeric_score is not None
            and numeric_score > NATUREBENCH_SURPASS_SOTA_THRESHOLD
        ),
    }


def _score_is_better_value(
    candidate: Any,
    incumbent: Any,
    *,
    higher_is_better: bool,
) -> bool:
    candidate_score = _coerce_float_value(candidate)
    incumbent_score = _coerce_float_value(incumbent)
    if candidate_score is None:
        return False
    if incumbent_score is None:
        return True
    if higher_is_better:
        return candidate_score > incumbent_score
    return candidate_score < incumbent_score


def _submit_score_from_result(
    result: dict[str, Any] | None,
    *,
    evaluation_protocol: str,
) -> float | None:
    if not result:
        return None
    submit_score = _coerce_float_value(result.get("submit_score"))
    if submit_score is not None:
        return submit_score
    if str(evaluation_protocol).strip().lower() == "method2_0616":
        test_score = _coerce_float_value(result.get("test_score"))
        if test_score is not None:
            return test_score
        raw_scores = dict(result.get("raw_scores") or {})
        return _coerce_float_value(raw_scores.get("sandbox_test_score"))
    return _coerce_float_value(result.get("score"))


def _select_method2_oracle_score(
    *,
    final_test_score: Any,
    submit_test_score: Any,
    higher_is_better: bool,
) -> tuple[float | None, str | None]:
    # Method2 oracle is intentionally scoped to the validation-selected final node.
    # It compares that selected node's test score observed during search with the
    # fresh submit rerun test score for the same selected node. It must not scan
    # all search nodes and choose the global best test score.
    best_score = None
    best_source = None
    for source, score in (
        ("final_test_score", final_test_score),
        ("submit_test_score", submit_test_score),
    ):
        if _score_is_better_value(
            score,
            best_score,
            higher_is_better=higher_is_better,
        ):
            best_score = _coerce_float_value(score)
            best_source = source
    return best_score, best_source


def _select_best_available_node(
    solver: Any,
    *,
    seed: int,
    task_name: str,
    sample_index: int,
    benchmark: str = "mlebench",
) -> Any | None:
    candidate_nodes = [
        node
        for node in solver.journal.nodes
        if not solver.journal.is_root_node(node) and str(getattr(node, "code", "") or "")
    ]
    is_naturebench = str(benchmark).strip().lower() == "naturebench"
    if is_naturebench:
        valid_nodes = [
            node
            for node in candidate_nodes
            if _is_naturebench_successful_node(node)
            and _naturebench_score_from_node(node) is not None
        ]
    else:
        valid_nodes = [
            node
            for node in candidate_nodes
            if _node_selection_score(node) is not None
        ]
    if valid_nodes:
        if is_naturebench:
            selected = max(
                valid_nodes,
                key=lambda node: (_naturebench_score_from_node(node), _node_sort_key(node)),
            )
        else:
            selected = max(valid_nodes, key=lambda node: node.metric)
        setattr(selected, "_final_selection_source", "validation")
        return selected

    if candidate_nodes:
        stable_candidates = sorted(candidate_nodes, key=_node_sort_key)
        rng = random.Random(f"{seed}:{task_name}:{sample_index}:final_select")
        selected = rng.choice(stable_candidates)
        setattr(selected, "_final_selection_source", "random_no_valid")
        return selected
    return None


def _step_index_from_payload(step_payload: dict[str, Any]) -> int | None:
    try:
        return int(step_payload["step"])
    except (KeyError, TypeError, ValueError):
        return None


def _step_index_from_experience_card(card: dict[str, Any]) -> int | None:
    try:
        return int(card["step_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_step_schedule_metadata(node: Any, solver: Any) -> dict[str, Any]:
    work_metadata = dict(getattr(node, "async_work_metadata", {}) or {})
    generation_id = int(
        work_metadata.get(
            "generation_id",
            solver.state.current_generation,
        )
    )
    temperature = work_metadata.get("temperature")
    if temperature is None:
        temperature = solver.linear_decay(generation_id)
    return {
        "execution_mode": work_metadata.get("execution_mode", "generation"),
        "attempt_id": work_metadata.get("attempt_id"),
        "generation_id": generation_id,
        "temperature": float(temperature),
        "worker_id": work_metadata.get("worker_id"),
        "gpu_index": work_metadata.get("gpu_index"),
        "sandbox_url": work_metadata.get("sandbox_url"),
        "debug_attempt_index": work_metadata.get("debug_attempt_index"),
    }


def _load_checkpoint_current_step(checkpoint_dir: Path) -> int:
    state_path = checkpoint_dir / "state.json"
    if not state_path.exists():
        return 0

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    current_step = max(0, int(state_payload.get("current_step") or 0))
    if current_step > 0 and not (checkpoint_dir / "journal.jsonl").exists():
        raise FileNotFoundError(
            f"Checkpoint state exists at {state_path}, but journal.jsonl is missing."
        )
    return current_step


def _step_validation_time(step_payload: dict[str, Any]) -> float:
    try:
        is_successful_validation = int(step_payload.get("status_code")) == 200
    except (TypeError, ValueError):
        is_successful_validation = False

    if not is_successful_validation:
        return 0.0

    for key in ("sandbox_time_used", "val_time_used", "run_time"):
        value = step_payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _next_archive_path(archive_root: Path, step_dir_name: str) -> Path:
    archive_path = archive_root / step_dir_name
    suffix = 1
    while archive_path.exists():
        archive_path = archive_root / f"{step_dir_name}.{suffix}"
        suffix += 1
    return archive_path


def _archive_orphan_step_dirs(
    output_dir: Path,
    orphan_step_indices: set[int],
) -> None:
    if not orphan_step_indices:
        return

    archive_root = output_dir / "orphaned_steps_after_checkpoint"
    archive_root.mkdir(parents=True, exist_ok=True)
    for step_index in sorted(orphan_step_indices):
        step_dir = output_dir / f"step_{step_index}"
        if not step_dir.exists():
            continue
        shutil.move(str(step_dir), str(_next_archive_path(archive_root, step_dir.name)))


def _reconcile_resume_steps_with_checkpoint(
    output_dir: Path,
    checkpoint_dir: Path,
    existing_task_stat: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[int], float, bool]:
    step_stats = list(existing_task_stat.get("steps") or [])
    if existing_task_stat.get("submit_score") is not None:
        written_steps = {
            step_index
            for step in step_stats
            if (step_index := _step_index_from_payload(step)) is not None
        }
        validation_time_used = float(existing_task_stat.get("val_time") or 0.0)
        return step_stats, written_steps, validation_time_used, False

    checkpoint_current_step = _load_checkpoint_current_step(checkpoint_dir)
    checkpoint_step_count = max(0, checkpoint_current_step - 1)
    kept_step_stats: list[dict[str, Any]] = []
    orphan_step_indices: set[int] = set()

    for step in step_stats:
        step_index = _step_index_from_payload(step)
        if step_index is None:
            kept_step_stats.append(step)
            continue
        if step_index < checkpoint_step_count:
            kept_step_stats.append(step)
        else:
            orphan_step_indices.add(step_index)

    for step_dir in output_dir.glob("step_*"):
        if not step_dir.is_dir():
            continue
        try:
            step_index = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if step_index >= checkpoint_step_count:
            orphan_step_indices.add(step_index)

    _archive_orphan_step_dirs(output_dir, orphan_step_indices)
    written_steps = {
        step_index
        for step in kept_step_stats
        if (step_index := _step_index_from_payload(step)) is not None
    }
    validation_time_used = sum(_step_validation_time(step) for step in kept_step_stats)
    return kept_step_stats, written_steps, validation_time_used, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single AIRA Dojo task")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    example_dir = Path(__file__).resolve().parent
    task_dir = Path(args.task_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    runner_config_path = Path(args.runner_config).resolve()
    aira_evo_dir = output_dir / "aira_evo"
    aira_evo_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("LOGGING_DIR", str(aira_evo_dir))

    dojo_src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(dojo_src_dir) not in sys.path:
        sys.path.insert(0, str(dojo_src_dir))
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))

    from base_task import SandboxMLEBenchTask
    from dojo.config_dataclasses.client.base import ClientConfig
    from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig
    from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig
    from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig
    from dojo.config_dataclasses.logger import LoggerConfig
    from dojo.config_dataclasses.operators.base import OperatorConfig
    from dojo.config_dataclasses.operators.memory import MemoryOpConfig
    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig
    from dojo.core.interpreters.python import PythonInterpreter
    from dojo.core.solvers.utils.response import parse_thinking_tags
    from dojo.core.solvers.utils.search_exporter import export_search_results
    from dojo.solvers.evo.experience import (
        build_experience_card,
        build_strategy_board,
        load_experience_cards,
    )
    from dojo.solvers.evo.evo import Evolutionary
    from dojo.utils.logger import config_logger
    from tts_search import eval_utils

    runner_cfg = _load_yaml(runner_config_path)
    task_cfg = _load_yaml(task_dir / "config.yaml")
    benchmark = _benchmark_name(task_cfg)

    if runner_cfg.get("leaderboard_dir"):
        task_cfg["leaderboard_dir"] = runner_cfg["leaderboard_dir"]

    seed = int(runner_cfg.get("seed", 42)) + int(args.sample_index)
    random.seed(seed)
    np.random.seed(seed)

    logger_cfg = LoggerConfig(
        output_dir=str(aira_evo_dir),
        use_console=bool(runner_cfg.get("logger", {}).get("use_console", True)),
        use_wandb=bool(runner_cfg.get("logger", {}).get("use_wandb", False)),
        use_json=bool(runner_cfg.get("logger", {}).get("use_json", True)),
        print_config=False,
        write_env_vars=False,
    )
    logger = config_logger(SimpleNamespace(logger=logger_cfg))

    task_cls = (
        _load_naturebench_task_class(example_dir)
        if benchmark == "naturebench"
        else SandboxMLEBenchTask
    )
    task = task_cls(
        task_cfg,
        time_budget=runner_cfg.get("time_budget"),
    )
    higher_is_better = bool(task_cfg["higher_is_better"])
    final_submit_enabled = _should_run_final_submit(
        benchmark=benchmark,
        submit_repeats=int(task.submit_repeats),
        task_cfg=task_cfg,
        runner_cfg=runner_cfg,
    )

    def coerce_float(value: Any) -> float | None:
        return _coerce_float_value(value)

    def score_is_better(candidate: Any, incumbent: Any) -> bool:
        return _score_is_better_value(
            candidate,
            incumbent,
            higher_is_better=higher_is_better,
        )

    def submit_score_from_result(result: dict[str, Any] | None) -> float | None:
        return _submit_score_from_result(
            result,
            evaluation_protocol=str(task.evaluation_protocol),
        )

    def select_best_submit_result(
        submit_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best_result = submit_results[0] if submit_results else None
        best_score = submit_score_from_result(best_result)
        for result in submit_results[1:]:
            score = submit_score_from_result(result)
            if score_is_better(score, best_score):
                best_result = result
                best_score = score
        return best_result

    def build_submit_attempt_payload(
        result: dict[str, Any],
        *,
        attempt_index: int,
        leaderboard: Any | None,
    ) -> dict[str, Any]:
        score = submit_score_from_result(result)
        grade, medal = eval_utils.build_submit_grade_and_medal(score, leaderboard)
        return {
            "attempt_index": attempt_index,
            "status_code": result.get("status_code"),
            "status": result.get("status"),
            "submit_score": score,
            "score": result.get("score"),
            "submit_valid_score": result.get("valid_score"),
            "submit_test_score": result.get("test_score"),
            "submit_raw_scores": dict(result.get("raw_scores") or {}) or None,
            "submit_score_source": result.get("submit_score_source"),
            "submit_reward": result.get("submit_reward", result.get("reward")),
            "submit_grade": grade,
            "submit_medal": medal,
            "run_time": result.get("run_time"),
            "queue_time": result.get("queue_time"),
            "job_id": result.get("job_id"),
            "eval_split": result.get("eval_split"),
            "requested_eval_split": result.get("requested_eval_split"),
            "data_dir": result.get("data_dir"),
        }

    def mean_or_none(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    step_stats: list[dict[str, Any]] = []
    written_steps: set[int] = set()
    resume_stat_reconciled = False
    existing_task_stat_path = output_dir / "stat.json"
    if existing_task_stat_path.exists():
        existing_task_stat = json.loads(
            existing_task_stat_path.read_text(encoding="utf-8")
        )
        step_stats, written_steps, validation_time_used, resume_stat_reconciled = (
            _reconcile_resume_steps_with_checkpoint(
                output_dir,
                aira_evo_dir / "checkpoint",
                existing_task_stat,
            )
        )
        task.validation_time_used = validation_time_used

    interpreter_cfg = PythonInterpreterConfig(
        working_dir=str(aira_evo_dir / "workspace"),
        timeout=int(runner_cfg["solver"]["execution_timeout"]),
        use_symlinks=bool(runner_cfg.get("interpreter", {}).get("use_symlinks", True)),
        format_tb_ipython=bool(
            runner_cfg.get("interpreter", {}).get("format_tb_ipython", False)
        ),
    )
    solver_interpreter = PythonInterpreter(
        interpreter_cfg,
        data_dir=Path(str(task_cfg.get("interpreter_data_dir") or task_cfg["data_dir"])),
    )

    client_cfg = ClientConfig(
        api=str(runner_cfg["llm"].get("api", "litellm")),
        model_id=str(runner_cfg["llm"]["model_id"]),
        base_url=str(runner_cfg["llm"]["base_url"]),
        api_key=(
            str(runner_cfg["llm"]["api_key"])
            if runner_cfg["llm"].get("api_key") is not None
            else os.environ.get("OPENMLE_LLM_API_KEY")
        ),
        use_azure_client=bool(runner_cfg["llm"].get("use_azure_client", False)),
        provider=str(runner_cfg["llm"].get("provider", "selfhosted")),
    )

    configs_root = dojo_src_dir / "dojo" / "configs"
    solver_root = configs_root / "solver"

    def build_prompt_config(prompt_payload: dict[str, Any]) -> JinjaPromptConfig:
        return JinjaPromptConfig(**_strip_target(dict(prompt_payload)))

    def maybe_build_prompt_config(prompt_payload: dict[str, Any] | None) -> JinjaPromptConfig:
        if prompt_payload is None:
            return JinjaPromptConfig()
        return build_prompt_config(prompt_payload)

    def build_operator_config(name: str, relative_path: str) -> OperatorConfig:
        operator_yaml = _load_yaml(solver_root / "operators" / relative_path)
        operator_payload = dict(operator_yaml[name])
        llm_payload = _strip_target(dict(operator_payload["llm"]))
        generation_kwargs = dict(runner_cfg["llm"].get("generation_kwargs") or {})
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(llm_payload.get("generation_kwargs") or {}),
        )

        solver_operator_overrides = (
            runner_cfg.get("solver", {}).get("operators", {}).get(name, {})
        )
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(
                solver_defaults["operators"].get(name, {})
                .get("llm", {})
                .get("generation_kwargs", {})
            ),
        )
        generation_kwargs = _merge_generation_kwargs(
            generation_kwargs,
            dict(
                solver_operator_overrides.get("llm", {}).get("generation_kwargs", {})
            ),
        )

        return OperatorConfig(
            llm=GenericLLMConfig(
                client=client_cfg,
                generation_kwargs=generation_kwargs,
            ),
            system_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("system_message_prompt_template")
            ),
            init_user_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("init_user_message_prompt_template")
            ),
            user_message_prompt_template=maybe_build_prompt_config(
                operator_payload.get("user_message_prompt_template")
            ),
        )

    base_solver_defaults = _load_yaml(solver_root / "evo.yaml")
    solver_defaults = _load_yaml(solver_root / _solver_defaults_relative_path(task_cfg))
    solver_cfg_overrides = dict(runner_cfg.get("solver") or {})

    experience_cfg = dict(
        solver_cfg_overrides.get(
            "experience",
            solver_defaults.get("experience", {}),
        )
    )
    experience_cfg_enabled = bool(experience_cfg.get("enabled", False))

    operator_paths = _operator_paths_for_benchmark(
        benchmark,
        experience_enabled=experience_cfg_enabled,
    )

    operators = {
        "draft": build_operator_config("draft", operator_paths["draft"]),
        "improve": build_operator_config("improve", operator_paths["improve"]),
        "debug": build_operator_config("debug", operator_paths["debug"]),
        "analyze": build_operator_config("analyze", operator_paths["analyze"]),
        "crossover": build_operator_config("crossover", operator_paths["crossover"]),
    }
    if experience_cfg_enabled and "rich_memory_summary" in operator_paths:
        operators["rich_memory_summary"] = build_operator_config(
            "rich_memory_summary",
            operator_paths["rich_memory_summary"],
        )

    memory_cfg = MemoryOpConfig(
        **_strip_target(_load_yaml(solver_root / "memory" / "simple_memory.yaml"))
    )
    debug_memory_cfg = MemoryOpConfig(
        **_strip_target(_load_yaml(solver_root / "memory" / "debug_memory.yaml"))
    )

    time_limit_secs = int(
        runner_cfg.get("model_plus_sandbox_time_budget")
        or solver_cfg_overrides.get("time_limit_secs")
        or base_solver_defaults.get("time_limit_secs")
    )
    solver_cfg = EvolutionarySolverConfig(
        step_limit=int(solver_cfg_overrides.get("step_limit", solver_defaults["step_limit"])),
        available_packages=list(
            solver_cfg_overrides.get(
                "available_packages",
                solver_defaults.get("available_packages", base_solver_defaults["available_packages"]),
            )
        ),
        operators=operators,
        memory=memory_cfg,
        debug_memory=debug_memory_cfg,
        exp_name=str(task_cfg["task_name"]).replace("/", "_").replace("\\", "_"),
        execution_timeout=int(
            solver_cfg_overrides.get(
                "execution_timeout",
                base_solver_defaults["execution_timeout"],
            )
        ),
        time_limit_secs=time_limit_secs,
        export_search_results=bool(
            solver_cfg_overrides.get(
                "export_search_results",
                base_solver_defaults.get("export_search_results", True),
            )
        ),
        checkpoint_path=str(aira_evo_dir / "checkpoint"),
        use_test_score=bool(
            solver_cfg_overrides.get(
                "use_test_score",
                solver_defaults.get("use_test_score", False),
            )
        ),
        use_complexity=bool(
            solver_cfg_overrides.get(
                "use_complexity",
                solver_defaults.get("use_complexity", False),
            )
        ),
        max_llm_call_retries=int(
            solver_cfg_overrides.get(
                "max_llm_call_retries",
                base_solver_defaults.get("max_llm_call_retries", 3),
            )
        ),
        num_islands=int(solver_cfg_overrides.get("num_islands", solver_defaults["num_islands"])),
        max_island_size=int(
            solver_cfg_overrides.get(
                "max_island_size",
                solver_defaults["max_island_size"],
            )
        ),
        crossover_prob=float(
            solver_cfg_overrides.get(
                "crossover_prob",
                solver_defaults["crossover_prob"],
            )
        ),
        migration_prob=float(
            solver_cfg_overrides.get(
                "migration_prob",
                solver_defaults["migration_prob"],
            )
        ),
        initial_temp=float(
            solver_cfg_overrides.get(
                "initial_temp",
                solver_defaults["initial_temp"],
            )
        ),
        final_temp=float(
            solver_cfg_overrides.get(
                "final_temp",
                solver_defaults["final_temp"],
            )
        ),
        num_generations_till_migration=int(
            solver_cfg_overrides.get(
                "num_generations_till_migration",
                solver_defaults["num_generations_till_migration"],
            )
        ),
        num_generations_till_crossover=int(
            solver_cfg_overrides.get(
                "num_generations_till_crossover",
                solver_defaults["num_generations_till_crossover"],
            )
        ),
        few_shot=dict(
            solver_cfg_overrides.get(
                "few_shot",
                base_solver_defaults["few_shot"],
            )
        ),
        num_generations=int(
            solver_cfg_overrides.get(
                "num_generations",
                solver_defaults["num_generations"],
            )
        ),
        individuals_per_generation=int(
            solver_cfg_overrides.get(
                "individuals_per_generation",
                solver_defaults["individuals_per_generation"],
            )
        ),
        max_debug_time=float(
            solver_cfg_overrides.get(
                "max_debug_time",
                solver_defaults["max_debug_time"],
            )
        ),
        max_debug_depth=int(
            solver_cfg_overrides.get(
                "max_debug_depth",
                solver_defaults["max_debug_depth"],
            )
        ),
        fresh_draft_prob=float(
            solver_cfg_overrides.get(
                "fresh_draft_prob",
                solver_defaults.get("fresh_draft_prob", 0.0),
            )
        ),
        max_wall_time_secs=float(
            solver_cfg_overrides.get(
                "max_wall_time_secs",
                solver_defaults.get("max_wall_time_secs", 0.0),
            )
            or 0.0
        ),
        data_preview=bool(
            solver_cfg_overrides.get(
                "data_preview",
                solver_defaults["data_preview"],
            )
        ),
        experience=experience_cfg,
        execution_mode=str(
            solver_cfg_overrides.get(
                "execution_mode",
                solver_defaults.get("execution_mode", "generation"),
            )
        ),
        async_workers=solver_cfg_overrides.get(
            "async_workers",
            solver_defaults.get("async_workers", 1),
        ),
        async_sandbox_urls=list(
            solver_cfg_overrides.get(
                "async_sandbox_urls",
                solver_defaults.get("async_sandbox_urls", []),
            )
        ),
        async_sandbox_assignment=str(
            solver_cfg_overrides.get(
                "async_sandbox_assignment",
                solver_defaults.get("async_sandbox_assignment", "round_robin"),
            )
        ),
        async_checkpoint_every_commits=int(
            solver_cfg_overrides.get(
                "async_checkpoint_every_commits",
                solver_defaults.get("async_checkpoint_every_commits", 1),
            )
        ),
    )

    dojo_config_payload = {
        "seed": seed,
        "sample_index": int(args.sample_index),
        "task": task_cfg,
        "interpreter": asdict(interpreter_cfg),
        "logger": asdict(logger_cfg),
        "solver": asdict(solver_cfg),
    }
    (aira_evo_dir / "dojo_config.json").write_text(
        json.dumps(dojo_config_payload, indent=2),
        encoding="utf-8",
    )

    state, task_info = task.prepare(
        solver_interpreter=solver_interpreter,
        eval_interpreter=None,
    )
    solver = Evolutionary(solver_cfg, task_info=task_info)
    solver.load_checkpoint()
    experience_enabled = bool(
        dict(getattr(solver_cfg, "experience", {}) or {}).get("enabled", False)
    )
    experience_cards: list[dict[str, Any]] = []
    if experience_enabled:
        experience_cards = [
            card
            for card in load_experience_cards(output_dir)
            if (step_index := _step_index_from_experience_card(card)) is None or step_index in written_steps
        ]
        cards_by_node_id = {
            str(card.get("node_id")): card
            for card in experience_cards
            if card.get("node_id") is not None
        }
        for journal_node in solver.journal.nodes:
            card = cards_by_node_id.get(str(getattr(journal_node, "id", "")))
            if card is not None:
                journal_node.experience_card = card

    def write_task_stat(
        *,
        best_node: Any | None = None,
        submit_result: dict[str, Any] | None = None,
        submit_results: list[dict[str, Any]] | None = None,
        test_time: float = 0.0,
    ) -> None:
        is_naturebench = benchmark == "naturebench"
        final_score = None
        final_reward = None
        final_selection_score = None
        final_validation_score = None
        final_sandbox_score = None
        final_raw_scores = None
        final_aggregate_improvement = None
        final_per_instance_improvement = None
        final_model_final_validation_score = None
        final_valid_score = None
        final_test_score = None
        final_score_source = None
        final_step = None
        if best_node is not None:
            selected_source = getattr(best_node, "_final_selection_source", None)
            if getattr(best_node, "metric", None) is not None:
                aux_info = dict(best_node.metric.info or {})
                raw_scores = dict(aux_info.get("raw_scores") or {})
                final_raw_scores = raw_scores or None
                final_selection_score = aux_info.get("selection_score")
                if final_selection_score is None and "selection_score" not in aux_info:
                    final_selection_score = _node_selection_score(best_node)
                final_validation_score = aux_info.get("validation_score")
                final_aggregate_improvement = (
                    _naturebench_valid_score_from_node(best_node)
                    if is_naturebench
                    else _naturebench_score_from_aux_info(aux_info)
                )
                final_per_instance_improvement = (
                    aux_info.get("per_instance_improvement")
                    if not is_naturebench or final_aggregate_improvement is not None
                    else None
                )
                final_model_final_validation_score = raw_scores.get(
                    "model_final_validation_score",
                    aux_info.get("model_final_validation_score"),
                )
                final_sandbox_score = raw_scores.get(
                    "sandbox_score",
                    aux_info.get("sandbox_score", aux_info.get("raw_score", aux_info.get("score"))),
                )
                final_valid_score = raw_scores.get(
                    "sandbox_valid_score",
                    aux_info.get("sandbox_valid_score", aux_info.get("valid_score")),
                )
                final_test_score = raw_scores.get(
                    "sandbox_test_score",
                    aux_info.get("sandbox_test_score", aux_info.get("test_score")),
                )
                final_score = final_selection_score
                final_reward = aux_info.get("selection_reward", aux_info.get("validation_reward"))
                final_score_source = _final_score_source(
                    selected_source,
                    aux_info.get("selection_score_source"),
                )
                if is_naturebench:
                    final_score = final_aggregate_improvement
                    final_score_source = _final_score_source(
                        selected_source,
                        "aggregate_improvement",
                    )
            final_step = int(best_node.step) - 1 if best_node.step is not None else None

        all_submit_results = list(submit_results or [])
        if submit_result is not None and not all_submit_results:
            all_submit_results = [submit_result]
        submit_result = select_best_submit_result(all_submit_results)

        submit_raw_scores = dict(submit_result.get("raw_scores") or {}) if submit_result else {}
        submit_valid_score = submit_result.get("valid_score") if submit_result else None
        submit_test_score = submit_result.get("test_score") if submit_result else None
        submit_score = submit_score_from_result(submit_result)
        submit_reward = (
            submit_result.get("submit_reward", submit_result.get("reward"))
            if submit_result
            else None
        )
        submit_score = _effective_submit_score(
            benchmark=benchmark,
            submit_score=submit_score,
            final_selection_score=(
                final_aggregate_improvement if is_naturebench else final_selection_score
            ),
            final_submit_enabled=final_submit_enabled,
        )
        submit_score_source = (
            submit_result.get("submit_score_source") if submit_result else None
        )
        if is_naturebench and not final_submit_enabled and submit_result is None:
            submit_score_source = "search_aggregate_improvement"
            submit_reward = final_reward

        total_cost = sum(float(step["cost"]) for step in step_stats)
        total_prompt_tokens = sum(int(step["prompt_tokens"]) for step in step_stats)
        total_completion_tokens = sum(
            int(step["completion_tokens"]) for step in step_stats
        )
        total_tokens = sum(int(step["total_tokens"]) for step in step_stats)
        total_model_time = sum(float(step["model_time_used"]) for step in step_stats)
        total_sandbox_time = float(task.validation_time_used) + test_time
        total_model_plus_sandbox_time = total_model_time + total_sandbox_time

        status_count: dict[str, int] = {}
        for step in step_stats:
            status = str(step["status"])
            status_count[status] = status_count.get(status, 0) + 1

        leaderboard_needed = (
            not is_naturebench
            and (
                submit_score is not None
                or final_sandbox_score is not None
                or bool(all_submit_results)
            )
        )
        leaderboard = eval_utils.load_leaderboard(task_cfg) if leaderboard_needed else None
        submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
            submit_score,
            leaderboard,
        )
        submit_attempts = [
            build_submit_attempt_payload(
                result,
                attempt_index=attempt_index,
                leaderboard=leaderboard,
            )
            for attempt_index, result in enumerate(all_submit_results, start=1)
        ]
        attempt_scores = [
            score
            for attempt in submit_attempts
            if (score := coerce_float(attempt.get("submit_score"))) is not None
        ]
        attempt_grades = [
            float(attempt["submit_grade"])
            for attempt in submit_attempts
            if attempt.get("submit_grade") is not None
        ]
        submit_score_avg = mean_or_none(attempt_scores)
        submit_grade_avg = mean_or_none(attempt_grades)
        submit_medal_avg = (
            eval_utils.get_medal_for_grade(submit_grade_avg, leaderboard)
            if submit_grade_avg is not None and leaderboard is not None
            else "N/A"
        )
        submit_grade_best = min(attempt_grades) if attempt_grades else submit_grade
        submit_medal_best = (
            eval_utils.get_medal_for_grade(submit_grade_best, leaderboard)
            if leaderboard is not None
            else submit_medal
        )

        self_valid_oracle_score = None
        self_valid_oracle_source = None
        is_self_valid = (
            str(task.evaluation_protocol).strip().lower() not in {"method1_0616", "method2_0616"}
            and bool(task_cfg.get("sandbox", {}).get("use_clear_run_log_score", False))
        )
        if is_self_valid:
            for source, score in (
                ("final_sandbox_score", final_sandbox_score),
                ("submit_score", submit_score),
            ):
                if score_is_better(score, self_valid_oracle_score):
                    self_valid_oracle_score = coerce_float(score)
                    self_valid_oracle_source = source
        self_valid_oracle_grade, self_valid_oracle_medal = (
            eval_utils.build_submit_grade_and_medal(
                self_valid_oracle_score,
                leaderboard,
            )
            if self_valid_oracle_source is not None
            else ((1.0, "N/A") if is_self_valid else (None, None))
        )
        is_method2 = str(task.evaluation_protocol).strip().lower() == "method2_0616"
        method2_oracle_score = None
        method2_oracle_source = None
        method2_oracle_grade = None
        method2_oracle_medal = None
        if is_method2:
            method2_oracle_score, method2_oracle_source = _select_method2_oracle_score(
                final_test_score=final_test_score,
                submit_test_score=submit_test_score,
                higher_is_better=higher_is_better,
            )
            method2_oracle_grade, method2_oracle_medal = (
                eval_utils.build_submit_grade_and_medal(
                    method2_oracle_score,
                    leaderboard,
                )
                if method2_oracle_source is not None
                else (1.0, "N/A")
            )

        task_stat_payload = {
            "benchmark": benchmark,
            "task_name": task_cfg["task_name"],
            "val_data_dir": task.validation_data_dir,
            "submit_data_dir": task.submit_data_dir,
            "steps": step_stats,
            "num_steps": len(step_stats),
            "step_limit": int(solver_cfg.step_limit),
            "num_generations": int(solver_cfg.num_generations),
            "individuals_per_generation": int(solver_cfg.individuals_per_generation),
            "final_score": final_score,
            "final_reward": float(final_reward) if final_reward is not None else None,
            "final_selection_score": final_selection_score,
            "final_validation_score": final_validation_score,
            "final_model_final_validation_score": final_model_final_validation_score,
            "final_sandbox_score": final_sandbox_score,
            "final_valid_score": final_valid_score,
            "final_test_score": final_test_score,
            "final_raw_scores": final_raw_scores,
            "final_score_source": final_score_source,
            "final_step": final_step,
            "submit_score": submit_score,
            "submit_valid_score": submit_valid_score,
            "submit_test_score": submit_test_score,
            "submit_raw_scores": submit_raw_scores or None,
            "submit_score_source": submit_score_source,
            "submit_reward": float(submit_reward) if submit_reward is not None else None,
            "submit_grade": submit_grade,
            "submit_medal": submit_medal,
            "submit_attempts": submit_attempts if len(submit_attempts) > 1 else None,
            "submit_repeats": len(submit_attempts) if submit_attempts else 0,
            "submit_score_best": submit_score,
            "submit_grade_best": submit_grade_best,
            "submit_medal_best": submit_medal_best,
            "submit_score_avg": submit_score_avg,
            "submit_grade_avg": submit_grade_avg,
            "submit_medal_avg": submit_medal_avg,
            "self_valid_oracle_best_score": self_valid_oracle_score,
            "self_valid_oracle_best_score_source": self_valid_oracle_source,
            "self_valid_oracle_best_grade": self_valid_oracle_grade,
            "self_valid_oracle_best_medal": self_valid_oracle_medal,
            "total_cost": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "val_time": float(task.validation_time_used),
            "naturebench_gpu_wait_time": float(
                getattr(task, "validation_gpu_wait_time", 0.0)
            )
            if is_naturebench
            else None,
            "naturebench_preflight_time": float(
                getattr(task, "validation_preflight_time_used", 0.0)
            )
            if is_naturebench
            else None,
            "test_time": test_time,
            "total_time": float(task.validation_time_used) + test_time,
            "total_model_time": total_model_time,
            "total_sandbox_time": total_sandbox_time,
            "total_model_plus_sandbox_time": total_model_plus_sandbox_time,
            "sandbox_12h_score": total_sandbox_time
            / eval_utils.TIME_SCALING_SCORE_SECONDS,
            "model_plus_sandbox_12h_score": total_model_plus_sandbox_time
            / eval_utils.TIME_SCALING_SCORE_SECONDS,
            "status_count": status_count,
            "final_submit_enabled": final_submit_enabled,
            "final_submit_skipped": is_naturebench and not final_submit_enabled,
            "configured_submit_repeats": int(task.submit_repeats),
        }
        if is_naturebench:
            naturebench_best_candidates = [
                coerce_float(step.get("aggregate_improvement"))
                for step in step_stats
                if _naturebench_successful_aux_info(step)
            ]
            naturebench_best_candidates.extend(
                coerce_float(result.get("best_aggregate_improvement"))
                for result in all_submit_results
            )
            naturebench_best_candidates.append(coerce_float(final_aggregate_improvement))
            naturebench_best_scores = [
                score for score in naturebench_best_candidates if score is not None
            ]
            best_aggregate_improvement = (
                max(naturebench_best_scores) if naturebench_best_scores else None
            )
            task_stat_payload.update(
                {
                    "final_aggregate_improvement": final_aggregate_improvement,
                    "submit_aggregate_improvement": submit_score,
                    "best_aggregate_improvement": best_aggregate_improvement,
                    **_naturebench_leaderboard_flags(best_aggregate_improvement),
                    "naturebench_raw_scores": submit_raw_scores or final_raw_scores,
                    "naturebench_per_instance_improvement": (
                        submit_result.get("per_instance_improvement")
                        if submit_result
                        else final_per_instance_improvement
                    ),
                }
            )
        if is_method2:
            task_stat_payload.update(
                {
                    # Backward-compatible Method2 oracle fields. Despite the
                    # "best" name, this is only the better of two test scores
                    # for the validation-selected final node:
                    #   1. final_test_score from the search-stage evaluation.
                    #   2. submit_test_score from the final submit rerun.
                    "method2_oracle_best_test_score": method2_oracle_score,
                    "method2_oracle_best_test_score_source": method2_oracle_source,
                    "method2_oracle_best_grade": method2_oracle_grade,
                    "method2_oracle_best_medal": method2_oracle_medal,
                    "method2_oracle_search_test_score": final_test_score,
                    "method2_oracle_submit_test_score": submit_test_score,
                }
            )
        (output_dir / "stat.json").write_text(
            json.dumps(task_stat_payload, indent=2),
            encoding="utf-8",
        )

    if resume_stat_reconciled:
        write_task_stat(
            best_node=_select_best_available_node(
                solver,
                seed=seed,
                task_name=str(task_cfg["task_name"]),
                sample_index=int(args.sample_index),
                benchmark=benchmark,
            )
        )

    def mirror_latest_node() -> None:
        current_step = int(solver.state.current_step)
        node = solver.journal[current_step]
        if solver.journal.is_root_node(node):
            return

        step_index = int(node.step) - 1
        if step_index in written_steps:
            return

        written_steps.add(step_index)
        step_dir = output_dir / f"step_{step_index}"
        step_dir.mkdir(parents=True, exist_ok=True)

        creation_metric = node.operators_metrics[0] if node.operators_metrics else None
        system_prompt, user_prompt, completion_text = _extract_prompt_messages(
            creation_metric
        )
        reasoning_content, _ = parse_thinking_tags(completion_text)
        aux_info = dict(node.metric.info or {}) if node.metric is not None else {}
        usage = _sum_operator_usage(node.operators_metrics)
        sandbox_time_used = float(aux_info.get("run_time") or node.exec_time or 0.0)
        raw_scores = dict(aux_info.get("raw_scores") or {})
        model_final_validation_score = raw_scores.get(
            "model_final_validation_score",
            aux_info.get("model_final_validation_score"),
        )
        sandbox_score = raw_scores.get(
            "sandbox_score",
            aux_info.get("sandbox_score", aux_info.get("raw_score", aux_info.get("score"))),
        )
        valid_score = raw_scores.get(
            "sandbox_valid_score",
            aux_info.get("sandbox_valid_score", aux_info.get("valid_score")),
        )
        test_score = raw_scores.get(
            "sandbox_test_score",
            aux_info.get("sandbox_test_score", aux_info.get("test_score")),
        )
        validation_score = aux_info.get("validation_score")
        scores_by_split = aux_info.get("scores_by_split")
        aggregate_improvement = aux_info.get("aggregate_improvement")
        best_aggregate_improvement = aux_info.get("best_aggregate_improvement")
        selection_score = aux_info.get("selection_score")
        if selection_score is None and "selection_score" not in aux_info:
            selection_score = node.metric.value if node.metric is not None else None
        reported_score = selection_score
        validation_reward = aux_info.get("validation_reward")
        selection_reward = aux_info.get("selection_reward")
        reported_reward = (
            selection_reward if selection_reward is not None else validation_reward
        )
        operator = (
            str(node.operators_used[0])
            if node.operators_used
            else ("debug" if node.is_buggy else "unknown")
        )
        raw_run_log = str(aux_info.get("raw_run_log") or "")
        clear_run_log = str(aux_info.get("clear_run_log") or node.term_out or "")
        feedback = str(aux_info.get("feedback") or node.analysis or node.term_out or "")
        schedule_metadata = _resolve_step_schedule_metadata(node, solver)
        generation_id = int(schedule_metadata["generation_id"])

        (step_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
        (step_dir / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
        (step_dir / "response.md").write_text(completion_text, encoding="utf-8")
        (step_dir / "reasoning_content.md").write_text(
            reasoning_content,
            encoding="utf-8",
        )
        (step_dir / "valid_code.py").write_text(node.code, encoding="utf-8")
        (step_dir / "raw_run_log.txt").write_text(raw_run_log, encoding="utf-8")
        (step_dir / "clear_run_log.txt").write_text(clear_run_log, encoding="utf-8")
        (step_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

        step_stat_payload = {
            "mode": operator,
            "operator": operator,
            "execution_mode": schedule_metadata["execution_mode"],
            "attempt_id": schedule_metadata["attempt_id"],
            "generation_id": generation_id,
            "temperature": schedule_metadata["temperature"],
            "worker_id": schedule_metadata["worker_id"],
            "gpu_index": schedule_metadata["gpu_index"],
            "sandbox_url": schedule_metadata["sandbox_url"],
            "debug_attempt_index": schedule_metadata["debug_attempt_index"],
            "score": reported_score,
            "raw_score": sandbox_score,
            "raw_scores": raw_scores or None,
            "model_final_validation_score": model_final_validation_score,
            "sandbox_score": sandbox_score,
            "sandbox_valid_score": valid_score,
            "sandbox_test_score": test_score,
            "validation_score": validation_score,
            "valid_score": valid_score,
            "test_score": test_score,
            "scores_by_split": scores_by_split,
            "aggregate_improvement": aggregate_improvement,
            "best_aggregate_improvement": best_aggregate_improvement,
            "per_instance_improvement": aux_info.get("per_instance_improvement"),
            "eval_split": aux_info.get("eval_split"),
            "requested_eval_split": aux_info.get("requested_eval_split"),
            "selection_score": selection_score,
            "selection_score_source": aux_info.get("selection_score_source"),
            "submit_score": aux_info.get("submit_score"),
            "submit_score_source": aux_info.get("submit_score_source"),
            "score_source": aux_info.get("selection_score_source"),
            "reward": float(reported_reward or 0.0),
            "sandbox_reward": float(
                aux_info.get("sandbox_reward", aux_info.get("reward")) or 0.0
            ),
            "validation_reward": float(validation_reward or 0.0),
            "selection_reward": float(selection_reward or 0.0),
            "fitness": selection_score,
            "cost": float(usage["cost"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "status": str(aux_info.get("status") or ("buggy" if node.is_buggy else "unknown")),
            "status_code": aux_info.get("status_code"),
            "model_time_used": float(usage["latency"]),
            "sandbox_time_used": sandbox_time_used,
            "model_plus_sandbox_time_used": float(usage["latency"]) + sandbox_time_used,
            "is_buggy": bool(node.is_buggy),
            "parent_steps": [
                int(parent.step) - 1 for parent in list(node.parents or []) if parent.step
            ],
            "node_id": node.id,
        }
        if experience_enabled:
            card = build_experience_card(
                node=node,
                step_index=step_index,
                generation_id=generation_id,
                lower_is_better=bool(solver.lower_is_better),
                previous_cards=experience_cards,
                step_stat=step_stat_payload,
                usage=usage,
                clear_run_log=clear_run_log,
                raw_run_log=raw_run_log,
            )
            parent_selection_trace = getattr(
                node,
                "experience_parent_selection",
                None,
            )
            if isinstance(parent_selection_trace, dict):
                card["parent_selection_trace"] = parent_selection_trace
                selected_node_ids = {
                    str(node_id) for node_id in parent_selection_trace.get("selected_node_ids", [])
                }
                selected_utilities = [
                    candidate
                    for candidate in parent_selection_trace.get("candidates", [])
                    if str(candidate.get("node_id")) in selected_node_ids
                ]
                if selected_utilities:
                    card["selection_utility"] = selected_utilities

            experience_cards.append(card)
            board = build_strategy_board(
                experience_cards,
                lower_is_better=bool(solver.lower_is_better),
            )
            card["rank"] = board["rank_by_node"].get(card["node_id"])
            card["current_best"] = board["current_best_by_node"].get(card["node_id"])
            experience_cards[-1] = card
            board = build_strategy_board(
                experience_cards,
                lower_is_better=bool(solver.lower_is_better),
            )
            node.experience_card = card
            step_stat_payload["experience_card_path"] = str(
                step_dir / "experience_card.json"
            )
            step_stat_payload["strategy_board_path"] = str(
                output_dir / "strategy_board.json"
            )
            (step_dir / "experience_card.json").write_text(
                json.dumps(card, indent=2),
                encoding="utf-8",
            )
            (output_dir / "experience_cards.jsonl").write_text(
                "\n".join(json.dumps(item) for item in experience_cards) + "\n",
                encoding="utf-8",
            )
            (output_dir / "strategy_board.json").write_text(
                json.dumps(board, indent=2),
                encoding="utf-8",
            )
        (step_dir / "stat.json").write_text(
            json.dumps(step_stat_payload, indent=2),
            encoding="utf-8",
        )
        step_stats.append({"step": step_index, **step_stat_payload})
        write_task_stat(
            best_node=_select_best_available_node(
                solver,
                seed=seed,
                task_name=str(task_cfg["task_name"]),
                sample_index=int(args.sample_index),
                benchmark=benchmark,
            )
        )

    original_log_journal = solver.log_journal

    def patched_log_journal() -> None:
        original_log_journal()
        mirror_latest_node()

    solver.log_journal = patched_log_journal

    def should_stop_search() -> bool:
        if task.stop_requested:
            return True
        wall_elapsed = max(0.0, time.monotonic() - solver.start_time)
        get_exempt_wait = getattr(task, "get_budget_exempt_wait_seconds", None)
        try:
            exempt_wait = float(get_exempt_wait()) if callable(get_exempt_wait) else 0.0
        except (TypeError, ValueError):
            exempt_wait = 0.0
        effective_elapsed = max(0.0, wall_elapsed - max(0.0, exempt_wait))
        if effective_elapsed >= float(solver_cfg.time_limit_secs):
            return True
        max_wall_time = float(getattr(solver_cfg, "max_wall_time_secs", 0.0) or 0.0)
        if max_wall_time > 0 and wall_elapsed >= max_wall_time:
            return True
        return int(solver.state.current_step) >= int(solver_cfg.step_limit)

    def guard_operator_call(fn: Any) -> Any:
        def wrapped(*operator_args: Any, **operator_kwargs: Any) -> Any:
            if should_stop_search():
                raise StopSearch("Search budget reached.")
            return fn(*operator_args, **operator_kwargs)

        return wrapped

    solver._draft = guard_operator_call(solver._draft)
    solver._improve = guard_operator_call(solver._improve)
    solver._debug = guard_operator_call(solver._debug)
    solver._crossover = guard_operator_call(solver._crossover)

    best_node = None
    submit_results: list[dict[str, Any]] = []
    test_time = 0.0

    try:
        try:
            _, _, best_node = solver(task, state)
        except StopSearch:
            solver.save_checkpoint()
            if bool(getattr(solver.cfg, "export_search_results", False)):
                export_search_results(solver.cfg, solver.journal, logger, "EVO")
            best_node = _select_best_available_node(
                solver,
                seed=seed,
                task_name=str(task_cfg["task_name"]),
                sample_index=int(args.sample_index),
                benchmark=benchmark,
            )

        best_node = _select_best_available_node(
            solver,
            seed=seed,
            task_name=str(task_cfg["task_name"]),
            sample_index=int(args.sample_index),
            benchmark=benchmark,
        )

        best_code = best_node.code if best_node is not None else ""
        (output_dir / "valid_code_final.py").write_text(best_code, encoding="utf-8")
        submit_code = task.build_submit_code(best_code)
        (output_dir / "submit_code.py").write_text(submit_code, encoding="utf-8")

        if not final_submit_enabled:
            skip_message = (
                "Final submit skipped; using the best search-stage NatureBench "
                "aggregate_improvement recorded during exploration."
                if benchmark == "naturebench"
                else "Final submit skipped by configuration."
            )
            (output_dir / "submit_raw_run_log.txt").write_text(
                skip_message,
                encoding="utf-8",
            )
            (output_dir / "submit_clear_run_log.txt").write_text(
                skip_message,
                encoding="utf-8",
            )
            (output_dir / "submit_feedback.txt").write_text(
                skip_message,
                encoding="utf-8",
            )

        for attempt_index in range(
            1,
            (int(task.submit_repeats) + 1) if final_submit_enabled else 1,
        ):
            submit_result = task.evaluate_code(best_code, phase="test")
            submit_results.append(submit_result)
            if submit_result["status_code"] == 200 and submit_result.get("run_time") is not None:
                test_time += float(submit_result["run_time"])

            raw_log = str(submit_result.get("raw_run_log") or "")
            clear_log = str(submit_result.get("clear_run_log") or "")
            feedback = str(submit_result.get("feedback") or "")
            (output_dir / f"submit_attempt_{attempt_index}_raw_run_log.txt").write_text(
                raw_log,
                encoding="utf-8",
            )
            (
                output_dir / f"submit_attempt_{attempt_index}_clear_run_log.txt"
            ).write_text(clear_log, encoding="utf-8")
            (output_dir / f"submit_attempt_{attempt_index}_feedback.txt").write_text(
                feedback,
                encoding="utf-8",
            )
            if attempt_index == 1:
                (output_dir / "submit_raw_run_log.txt").write_text(
                    raw_log,
                    encoding="utf-8",
                )
                (output_dir / "submit_clear_run_log.txt").write_text(
                    clear_log,
                    encoding="utf-8",
                )
                (output_dir / "submit_feedback.txt").write_text(
                    feedback,
                    encoding="utf-8",
                )

        write_task_stat(
            best_node=best_node,
            submit_results=submit_results,
            test_time=test_time,
        )
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            task.close(state)
        except Exception:
            if not active_exception:
                raise
            logging.exception("Task cleanup failed while preserving an earlier error")
        try:
            logger.stop()
        except Exception:
            if not active_exception:
                raise
            logging.exception("Logger cleanup failed while preserving an earlier error")


if __name__ == "__main__":
    main()
