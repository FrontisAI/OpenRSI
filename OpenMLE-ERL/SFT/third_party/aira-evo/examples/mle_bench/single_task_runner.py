#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
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


def _extract_completion_artifacts(
    operator_metric: dict[str, Any] | None,
    parse_thinking_tags_fn: Any,
) -> tuple[str, str, dict[str, Any], str]:
    if not operator_metric:
        return "", "", {}, ""

    completion_text = str(operator_metric.get("completion_text") or "")
    usage = dict(operator_metric.get("usage") or {})
    parsed_reasoning, text_without_thinking = parse_thinking_tags_fn(completion_text)
    field_reasoning = (
        usage.get("reasoning_content")
        or usage.get("reasoning")
        or usage.get("thinking")
    )
    reasoning_content = str(field_reasoning or parsed_reasoning or "")
    response_text = text_without_thinking if parsed_reasoning else completion_text
    return response_text, reasoning_content, usage, completion_text


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


def _select_best_available_node(solver: Any) -> Any | None:
    best_node = solver.journal.get_best_node()
    if best_node is not None:
        return best_node

    for node in reversed(solver.journal.nodes):
        if not solver.journal.is_root_node(node):
            return node
    return None


def _safe_task_output_name(task_name: str, task_id: str) -> str:
    clean = task_name.replace("/", "_").replace("\\", "_")
    return f"{clean}_{task_id}" if task_id else clean


def _sync_task_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        import shutil

        shutil.copytree(src, dst, dirs_exist_ok=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single AIRA Dojo task")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    repo_root = Path(__file__).resolve().parents[4]
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
    from dojo.solvers.evo.evo import Evolutionary
    from dojo.utils.logger import config_logger
    from tts_search import eval_utils
    from tts_search.data_produce.token_filter import (
        count_chat_template_tokens,
        load_tokenizer,
    )
    from tts_search.services.result_persistence import (
        append_jsonl_record,
        update_progress_snapshot,
    )
    from tts_search.services.rejection import build_rejection_policy
    from tts_search.services.tree_search_state import (
        NODE_EVALUATED,
        NODE_GENERATED,
        SearchEvent,
        SearchState,
        append_search_event,
        atomic_write_search_state,
        read_search_events,
        replay_search_events,
    )

    runner_cfg = _load_yaml(runner_config_path)
    task_cfg = _load_yaml(task_dir / "config.yaml")
    task_name = str(task_cfg.get("task_name") or task_dir.name)
    task_id = str(task_cfg.get("uuid") or task_dir.name)
    run_dir = output_dir.parent.parent if output_dir.parent.name == "program_ep_0" else output_dir.parent
    compat_task_dir = run_dir / "tasks" / _safe_task_output_name(task_name, task_id)
    _sync_task_link(output_dir, compat_task_dir)
    total_runner_tasks = len(runner_cfg.get("task_list") or [])
    if total_runner_tasks <= 0:
        try:
            total_runner_tasks = sum(1 for p in output_dir.parent.iterdir() if p.is_dir())
        except OSError:
            total_runner_tasks = 1

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

    task = SandboxMLEBenchTask(
        task_cfg,
        time_budget=runner_cfg.get("time_budget"),
    )

    step_stats: list[dict[str, Any]] = []
    written_steps: set[int] = set()
    search_event_log_path = output_dir / "search_events.jsonl"
    search_state_path = output_dir / "search_state.json"
    replay_summary = replay_search_events(read_search_events(search_event_log_path))
    accepted_count = int(replay_summary.accepted_count)
    generated_count = int(replay_summary.generated_count)
    completed_count = int(replay_summary.completed_count)
    accepted_target = int(
        runner_cfg.get("rejection_target")
        or runner_cfg.get("accepted_target")
        or runner_cfg.get("n_samples_per_task", 1)
    )
    rejection_policy_name = str(runner_cfg.get("rejection_policy") or "accept_scored")
    rejection_policy = build_rejection_policy(
        name=rejection_policy_name,
        score_threshold=runner_cfg.get("rejection_score_threshold"),
        reward_threshold=runner_cfg.get("rejection_reward_threshold"),
        reference_scores_path=runner_cfg.get("rejection_reference_scores_path"),
        accepted_medals=runner_cfg.get("rejection_accepted_medals") or [],
        apply_baseline_filters=bool(
            runner_cfg.get("rejection_apply_baseline_filters", True)
        ),
        baseline_token_limit=int(runner_cfg.get("rejection_baseline_token_limit", 32768)),
        baseline_relative_gap_limit=float(
            runner_cfg.get("rejection_baseline_relative_gap_limit", 0.12)
        ),
        mixed_leaderboard_target=int(
            runner_cfg.get("rejection_mixed_leaderboard_target", 2)
        ),
        mixed_no_leaderboard_target=int(
            runner_cfg.get("rejection_mixed_no_leaderboard_target", 4)
        ),
    )
    baseline_tokenizer: Any | None = None

    def build_sft_assistant_content(
        *,
        reasoning_content: str,
        code: str,
        response_text: str,
    ) -> str:
        reasoning = str(reasoning_content or "")
        if reasoning.strip():
            if code.strip():
                return (
                    f"<think>\n{reasoning.strip()}\n</think>\n\n"
                    f"```python\n{code.rstrip()}\n```"
                )
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{response_text.strip()}"
        if code.strip():
            return f"```python\n{code.rstrip()}\n```"
        return response_text.strip()

    def count_slime_message_tokens(
        *,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        reasoning_content: str,
        code: str,
    ) -> tuple[int | None, str | None]:
        nonlocal baseline_tokenizer
        tokenizer_model = runner_cfg.get("rejection_baseline_tokenizer_model")
        if not tokenizer_model:
            return None, None
        if baseline_tokenizer is None:
            baseline_tokenizer = load_tokenizer(
                str(tokenizer_model),
                local_files_only=True,
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": build_sft_assistant_content(
                    reasoning_content=reasoning_content,
                    code=code,
                    response_text=response_text,
                ),
            },
        ]
        token_count = count_chat_template_tokens(messages, baseline_tokenizer)
        return int(token_count), str(tokenizer_model)
    existing_task_stat_path = output_dir / "stat.json"
    if existing_task_stat_path.exists():
        existing_task_stat = json.loads(
            existing_task_stat_path.read_text(encoding="utf-8")
        )
        step_stats = list(existing_task_stat.get("steps") or [])
        written_steps = {int(step["step"]) for step in step_stats}
        task.validation_time_used = float(existing_task_stat.get("val_time") or 0.0)

    interpreter_cfg = PythonInterpreterConfig(
        working_dir=str(aira_evo_dir / "workspace"),
        timeout=int(runner_cfg["solver"]["execution_timeout"]),
        use_symlinks=bool(runner_cfg.get("interpreter", {}).get("use_symlinks", True)),
    )
    solver_interpreter = PythonInterpreter(
        interpreter_cfg,
        data_dir=Path(task_cfg["data_dir"]),
    )

    client_cfg = ClientConfig(
        api=str(runner_cfg["llm"].get("api", "litellm")),
        model_id=str(runner_cfg["llm"]["model_id"]),
        base_url=str(runner_cfg["llm"]["base_url"]),
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
    solver_defaults = _load_yaml(solver_root / "mlebench" / "evo.yaml")
    solver_cfg_overrides = dict(runner_cfg.get("solver") or {})

    operators = {
        "draft": build_operator_config("draft", "mlebench/aira_operators/draft.yaml"),
        "improve": build_operator_config("improve", "mlebench/aira_operators/improve.yaml"),
        "debug": build_operator_config("debug", "mlebench/aira_operators/debug.yaml"),
        "analyze": build_operator_config("analyze", "mlebench/aide_operators/analyze.yaml"),
        "crossover": build_operator_config(
            "crossover",
            "mlebench/aira_operators/crossover.yaml",
        ),
    }

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
    requested_step_limit = int(
        solver_cfg_overrides.get("step_limit", solver_defaults["step_limit"])
    )
    solver_cfg = EvolutionarySolverConfig(
        step_limit=requested_step_limit + 1,
        available_packages=list(
            solver_cfg_overrides.get(
                "available_packages",
                base_solver_defaults["available_packages"],
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
        data_preview=bool(
            solver_cfg_overrides.get(
                "data_preview",
                solver_defaults["data_preview"],
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
    completed_generation_ids = [
        int(step["generation_id"])
        for step in step_stats
        if step.get("generation_id") is not None
    ]
    if completed_generation_ids and solver.state.current_generation == 0:
        solver.state.current_generation = max(completed_generation_ids) + 1

    def write_task_stat(
        *,
        best_node: Any | None = None,
        submit_result: dict[str, Any] | None = None,
        test_time: float = 0.0,
    ) -> None:
        final_score = None
        final_reward = None
        final_step = None
        if best_node is not None and getattr(best_node, "metric", None) is not None:
            aux_info = dict(best_node.metric.info or {})
            final_score = aux_info.get("score", best_node.metric.value)
            final_reward = aux_info.get("reward")
            final_step = int(best_node.step) - 1 if best_node.step is not None else None

        submit_score = submit_result.get("score") if submit_result else None
        submit_reward = submit_result.get("reward") if submit_result else None

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

        submit_grade = None
        submit_medal = "N/A"
        if submit_score is not None:
            leaderboard = eval_utils.load_leaderboard(task_cfg)
            submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
                submit_score,
                leaderboard,
            )

        task_stat_payload = {
            "task_name": task_cfg["task_name"],
            "val_data_dir": task.validation_data_dir,
            "submit_data_dir": task.submit_data_dir,
            "steps": step_stats,
            "num_steps": len(step_stats),
            "step_limit": requested_step_limit,
            "num_generations": int(solver_cfg.num_generations),
            "individuals_per_generation": int(solver_cfg.individuals_per_generation),
            "final_score": final_score,
            "final_reward": float(final_reward) if final_reward is not None else None,
            "final_step": final_step,
            "submit_score": submit_score,
            "submit_reward": float(submit_reward) if submit_reward is not None else None,
            "submit_grade": submit_grade,
            "submit_medal": submit_medal,
            "total_cost": total_cost,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "val_time": float(task.validation_time_used),
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
        }
        (output_dir / "stat.json").write_text(
            json.dumps(task_stat_payload, indent=2),
            encoding="utf-8",
        )

    def resolve_done_reason(*, final: bool = False) -> str | None:
        if accepted_count >= accepted_target:
            return "accepted_target_reached"
        if len(written_steps) >= requested_step_limit:
            return "step_limit_reached"
        if task.stop_requested:
            return "task_stop_requested"
        if not final:
            return None
        if _safe_float(getattr(solver.state, "running_time", 0.0)) >= _safe_float(
            solver_cfg.time_limit_secs
        ):
            return "aira_time_limit_reached"
        if _safe_int(getattr(solver.state, "current_step", 0)) >= _safe_int(
            solver_cfg.step_limit
        ):
            return "aira_step_limit_reached"
        if _safe_int(getattr(solver.state, "current_generation", 0)) >= _safe_int(
            solver_cfg.num_generations
        ):
            return "aira_num_generations_exhausted"
        return "aira_search_returned"

    def build_task_progress_state(*, final: bool = False) -> dict[str, Any]:
        done_reason = resolve_done_reason(final=final)
        return {
            "task_name": task_name,
            "success": sum(
                1
                for stat in step_stats
                if not bool(stat.get("is_buggy")) and stat.get("score") is not None
            ),
            "medal": sum(
                1
                for stat in step_stats
                if str(stat.get("submit_medal", "")).lower()
                in {"gold", "silver", "bronze"}
            ),
            "accepted": accepted_count,
            "completed": len(written_steps),
            "target": accepted_target,
            "accepted_target": accepted_target,
            "max_target": requested_step_limit,
            "generated": generated_count,
            "done": done_reason is not None,
            "done_reason": done_reason,
        }

    if accepted_count >= accepted_target or len(written_steps) >= requested_step_limit:
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(),
            total_tasks=total_runner_tasks,
        )
        task.close(state)
        logger.stop()
        return

    def mirror_latest_node() -> None:
        nonlocal accepted_count, generated_count, completed_count
        node = None
        step_index = None
        for candidate in reversed(list(solver.journal.nodes)):
            if candidate.step is None or solver.journal.is_root_node(candidate):
                continue
            candidate_step_index = int(candidate.step) - 1
            if candidate_step_index in written_steps:
                continue
            node = candidate
            step_index = candidate_step_index
            break
        if node is None or step_index is None:
            return

        written_steps.add(step_index)
        step_dir = output_dir / f"step_{step_index}"
        step_dir.mkdir(parents=True, exist_ok=True)

        creation_metric = node.operators_metrics[0] if node.operators_metrics else None
        system_prompt, user_prompt, _ = _extract_prompt_messages(
            creation_metric
        )
        (
            response_text,
            reasoning_content,
            creation_usage,
            completion_text,
        ) = _extract_completion_artifacts(creation_metric, parse_thinking_tags)
        aux_info = dict(node.metric.info or {}) if node.metric is not None else {}
        usage = _sum_operator_usage(node.operators_metrics)
        sandbox_time_used = float(aux_info.get("run_time") or node.exec_time or 0.0)
        operator = (
            str(node.operators_used[0])
            if node.operators_used
            else ("debug" if node.is_buggy else "unknown")
        )
        leaderboard = eval_utils.load_leaderboard(task_cfg)
        submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
            aux_info.get("score"),
            leaderboard,
        )
        slime_message_tokens, slime_message_tokenizer_model = count_slime_message_tokens(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            reasoning_content=reasoning_content,
            code=node.code,
        )

        (step_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")
        (step_dir / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
        (step_dir / "response.md").write_text(response_text, encoding="utf-8")
        (step_dir / "response.json").write_text(
            json.dumps(
                {
                    "raw_text": response_text,
                    "reasoning_content": reasoning_content,
                    "usage": creation_usage,
                    "completion_text": completion_text,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (step_dir / "reasoning.md").write_text(
            reasoning_content,
            encoding="utf-8",
        )
        (step_dir / "valid_code.py").write_text(node.code, encoding="utf-8")
        (step_dir / "raw_run_log.txt").write_text(
            str(aux_info.get("raw_run_log") or ""),
            encoding="utf-8",
        )
        (step_dir / "clear_run_log.txt").write_text(
            str(aux_info.get("clear_run_log") or node.term_out or ""),
            encoding="utf-8",
        )
        (step_dir / "feedback.txt").write_text(
            str(aux_info.get("feedback") or node.analysis or node.term_out or ""),
            encoding="utf-8",
        )

        step_stat_payload = {
            "mode": operator,
            "operator": operator,
            "generation_id": int(solver.state.current_generation),
            "temperature": float(solver.linear_decay(int(solver.state.current_generation))),
            "score": aux_info.get("score"),
            "reward": float(aux_info.get("reward") or 0.0),
            "submit_grade": submit_grade,
            "submit_medal": submit_medal,
            "fitness": node.metric.value if node.metric is not None else None,
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
        (step_dir / "stat.json").write_text(
            json.dumps(step_stat_payload, indent=2),
            encoding="utf-8",
        )
        step_stats.append({"step": step_index, **step_stat_payload})
        program_id = str(node.id or f"step_{step_index}")
        parent_ids = [
            str(parent.id)
            for parent in list(node.parents or [])
            if parent is not None and not solver.journal.is_root_node(parent)
        ]
        search_metadata = {
            "search_algorithm": "airaevo",
            "search_step": step_index,
            "program_id": program_id,
            "parent_ids": parent_ids,
            "generation_mode": operator,
            "fitness": node.metric.value if node.metric is not None else None,
            "code_path": str(Path(f"step_{step_index}") / "valid_code.py"),
        }
        accepted_decision = rejection_policy.accepts_record(
            {
                "score": aux_info.get("score"),
                "reward": float(aux_info.get("reward") or 0.0),
                "submit_medal": submit_medal,
                "status": str(aux_info.get("status") or ""),
                "success": not bool(node.is_buggy),
                "feedback": str(aux_info.get("feedback") or node.analysis or node.term_out or ""),
                "slime_message_tokens": slime_message_tokens,
                "generation_prompt_tokens": int(usage["prompt_tokens"]),
                "generation_completion_tokens": int(usage["completion_tokens"]),
                "generation_total_tokens": int(usage["total_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
            },
            task_cfg,
        )
        generated_count += 1
        completed_count += 1
        if accepted_decision.accepted:
            accepted_count += 1
        request_id = f"{task_id}_{step_index}"
        append_jsonl_record(
            run_dir / "gen_results.jsonl",
            {
                "request_id": request_id,
                "task_id": task_id,
                "task_name": task_name,
                "code": node.code,
                "raw_text": response_text,
                "reasoning_content": reasoning_content,
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
                "cost": float(usage["cost"]),
                "model_name": str(runner_cfg["llm"]["model_id"]).split("/")[-1],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "metadata": task_cfg,
                "step_index": step_index,
                "data_dir": str(task_cfg.get("data_dir") or ""),
                "output_dir": str(compat_task_dir),
                "success": bool(node.code),
                "error": None,
                "timestamp": datetime.now().isoformat(),
                **search_metadata,
            },
        )
        append_jsonl_record(
            run_dir / "eval_results.jsonl",
            {
                "request_id": request_id,
                "task_id": task_id,
                "task_name": task_name,
                "status_code": aux_info.get("status_code"),
                "status": str(aux_info.get("status") or ("buggy" if node.is_buggy else "unknown")),
                "score": aux_info.get("score"),
                "reward": float(aux_info.get("reward") or 0.0),
                "submit_grade": submit_grade,
                "submit_medal": submit_medal,
                "job_id": aux_info.get("job_id"),
                "queue_time": aux_info.get("queue_time"),
                "run_time": sandbox_time_used,
                "step_index": step_index,
                "success": not bool(node.is_buggy) and aux_info.get("score") is not None,
                "generation_prompt_tokens": int(usage["prompt_tokens"]),
                "generation_completion_tokens": int(usage["completion_tokens"]),
                "generation_total_tokens": int(usage["total_tokens"]),
                "slime_message_tokens": slime_message_tokens,
                "slime_message_tokenizer_model": slime_message_tokenizer_model,
                "feedback_path": str(step_dir / "feedback.txt"),
                "error": None,
                "timestamp": datetime.now().isoformat(),
                "accepted_by_rejection_policy": accepted_decision.accepted,
                "rejection_reason": accepted_decision.reason,
                **search_metadata,
            },
        )
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(),
            total_tasks=total_runner_tasks,
        )
        append_search_event(
            search_event_log_path,
            SearchEvent(
                event_type=NODE_GENERATED,
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                search_step=step_index,
                program_id=program_id,
                parent_ids=parent_ids,
                generation_mode=operator,
                code_path=str(Path(f"step_{step_index}") / "valid_code.py"),
                generated_count=generated_count,
            ),
        )
        append_search_event(
            search_event_log_path,
            SearchEvent(
                event_type=NODE_EVALUATED,
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                search_step=step_index,
                program_id=program_id,
                parent_ids=parent_ids,
                generation_mode=operator,
                code_path=str(Path(f"step_{step_index}") / "valid_code.py"),
                score=aux_info.get("score"),
                reward=float(aux_info.get("reward") or 0.0),
                fitness=node.metric.value if node.metric is not None else None,
                feedback_path=str(Path(f"step_{step_index}") / "feedback.txt"),
                raw_log_path=str(Path(f"step_{step_index}") / "raw_run_log.txt"),
                accepted_by_rejection_policy=accepted_decision.accepted,
                rejection_reason=accepted_decision.reason,
                generated_count=generated_count,
                completed_count=completed_count,
                accepted_count=accepted_count,
                stop_requested=accepted_count >= accepted_target,
                extra={
                    "generation_prompt_tokens": int(usage["prompt_tokens"]),
                    "generation_completion_tokens": int(usage["completion_tokens"]),
                    "generation_total_tokens": int(usage["total_tokens"]),
                },
            ),
        )
        solution_database_state = (
            solver.solution_database.state_dict()
            if hasattr(solver, "solution_database")
            else {}
        )
        replay_summary = replay_search_events(read_search_events(search_event_log_path))
        atomic_write_search_state(
            search_state_path,
            SearchState(
                task_id=task_id,
                task_name=task_name,
                search_algorithm="airaevo",
                next_search_step=step_index + 1,
                generated_count=generated_count,
                completed_count=completed_count,
                accepted_count=accepted_count,
                accepted_target=accepted_target,
                max_generated=requested_step_limit,
                stop_requested=accepted_count >= accepted_target,
                program_ids=list(replay_summary.program_ids),
                program_scores=dict(replay_summary.program_scores),
                program_fitness=dict(replay_summary.program_fitness),
                program_code_paths=dict(replay_summary.program_code_paths),
                parent_map=dict(replay_summary.parent_map),
                generation_modes=dict(replay_summary.generation_modes),
                island_populations=solution_database_state.get("island_populations"),
                generation_buffer={
                    "current_generation": int(solver.state.current_generation),
                    "next_counter_id": None,
                },
                journal_path="aira_evo/checkpoint/journal.jsonl",
            ),
        )
        if accepted_count >= accepted_target:
            task.stop_requested = True
        write_task_stat(best_node=_select_best_available_node(solver))

    original_log_journal = solver.log_journal

    def patched_log_journal() -> None:
        original_log_journal()
        mirror_latest_node()

    solver.log_journal = patched_log_journal

    def should_stop_search() -> bool:
        if task.stop_requested:
            return True
        if time.monotonic() - solver.start_time >= float(solver_cfg.time_limit_secs):
            return True
        return len(written_steps) >= requested_step_limit

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
    submit_result: dict[str, Any] | None = None
    test_time = 0.0

    try:
        _, _, best_node = solver(task, state)
    except StopSearch:
        solver.save_checkpoint()
        export_search_results(solver.cfg, solver.journal, logger, "EVO")
        best_node = _select_best_available_node(solver)
    finally:
        best_node = _select_best_available_node(solver) if best_node is None else best_node
        while True:
            before = len(written_steps)
            mirror_latest_node()
            if len(written_steps) == before:
                break

        best_code = best_node.code if best_node is not None else ""
        (output_dir / "valid_code_final.py").write_text(best_code, encoding="utf-8")
        submit_code = task.build_submit_code(best_code)
        (output_dir / "submit_code.py").write_text(submit_code, encoding="utf-8")

        submit_result = task.evaluate_code(best_code, phase="test")
        if submit_result["status_code"] == 200 and submit_result.get("run_time") is not None:
            test_time = float(submit_result["run_time"])

        (output_dir / "submit_raw_run_log.txt").write_text(
            str(submit_result.get("raw_run_log") or ""),
            encoding="utf-8",
        )
        (output_dir / "submit_clear_run_log.txt").write_text(
            str(submit_result.get("clear_run_log") or ""),
            encoding="utf-8",
        )
        (output_dir / "submit_feedback.txt").write_text(
            str(submit_result.get("feedback") or ""),
            encoding="utf-8",
        )

        write_task_stat(
            best_node=best_node,
            submit_result=submit_result,
            test_time=test_time,
        )
        update_progress_snapshot(
            run_dir / "progress.json",
            task_id=task_id,
            task_state=build_task_progress_state(final=True),
            total_tasks=total_runner_tasks,
        )
        task.close(state)
        logger.stop()


if __name__ == "__main__":
    main()
