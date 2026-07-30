from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
from tts_search import reward_func_utils
from tts_search.reward_func_utils import (
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    parse_final_validation_score,
    score2reward,
)
from tts_search.airaevo_concurrency import acquire_sandbox_slot
from tts_search.airaevo_concurrency import acquire_sandbox_slot_async


class SandboxMLEBenchTask(Task):
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        time_budget: float | None = None,
    ) -> None:
        super().__init__(cfg)
        self.cfg = dict(cfg)
        self.validation_data_dir = str(self.cfg["data_dir"])
        self.submit_dir = str(self.cfg["submit_dir"])
        self.submit_data_dir_root = self.cfg.get("submit_data_dir_root")
        self.evaluation_protocol = str(
            self.cfg.get("evaluation_protocol") or "legacy"
        ).strip().lower()
        self.validation_eval_split = self._normalize_eval_split(
            self.cfg.get("validation_eval_split")
        )
        self.test_eval_split = self._normalize_eval_split(self.cfg.get("test_eval_split"))
        if self.evaluation_protocol == "method2_0616":
            self.validation_eval_split = self.validation_eval_split or "both"
            self.test_eval_split = self.test_eval_split or "both"
        self.time_budget = float(time_budget) if time_budget is not None else None
        self.validation_time_used = 0.0
        self.stop_requested = False
        self._validation_time_lock = asyncio.Lock()

        validation_path = Path(self.validation_data_dir)
        self.validation_dir_name = validation_path.parent.name
        if self.submit_data_dir_root:
            # Keep submit evaluation aligned with the ordinary runner semantics.
            self.submit_data_dir = str(
                Path(str(self.submit_data_dir_root)) / validation_path.name
            )
        else:
            self.submit_data_dir = str(
                validation_path.parent.parent / self.submit_dir / validation_path.name
            )
        self.public_system_prompt = str(self.cfg.get("public_system_prompt") or "")
        self.public_user_prompt = str(self.cfg.get("public_user_prompt") or "")
        self.raw_task_description = str(self.cfg.get("task_description") or "")
        self.data_description = str(self.cfg.get("data_description") or "")
        self.task_description = self._build_task_description()
        self.lower_is_better = not bool(self.cfg["higher_is_better"])
        self.use_clear_run_log_score = bool(
            self.cfg.get("sandbox", {}).get("use_clear_run_log_score", False)
        )
        self.submit_job_timeout = self.cfg.get("submit_job_timeout")
        self.submit_wait_timeout = self.cfg.get("submit_wait_timeout")
        self.submit_repeats = int(self.cfg.get("submit_repeats", 1) or 1)

    @staticmethod
    def _normalize_eval_split(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric

    def _build_task_description(self) -> str:
        sections = [
            ("Task Description", self.raw_task_description),
            ("Data Description", self.data_description),
        ]
        return "\n\n".join(
            f"{title}:\n{content.strip()}"
            for title, content in sections
            if content.strip()
        )

    def build_submit_code(self, code: str) -> str:
        if self.submit_data_dir_root:
            return code
        return code.replace(self.validation_dir_name, self.submit_dir)

    def _score_to_reward(self, score: float | None) -> float:
        if score is None:
            return 0.0
        sandbox_cfg = dict(self.cfg["sandbox"])
        return (
            score2reward(score, self.cfg, mode="power_sigmoid")
            if bool(sandbox_cfg["use_score2reward"])
            else float(score)
        )

    def _score_protocol(self) -> str:
        if self.evaluation_protocol == "self_valid":
            return "self_valid"
        if self.evaluation_protocol == "method1_0616":
            return "method1"
        if self.evaluation_protocol == "method2_0616":
            return "method2"
        if self.use_clear_run_log_score:
            return "self_valid"
        return "legacy"

    def _annotate_eval_scores(self, eval_payload: dict[str, Any], *, phase: str) -> None:
        protocol = self._score_protocol()
        sandbox_score = self._coerce_score(eval_payload.get("sandbox_score"))
        if sandbox_score is None:
            sandbox_score = self._coerce_score(eval_payload.get("raw_score"))
        if sandbox_score is None:
            sandbox_score = self._coerce_score(eval_payload.get("score"))

        sandbox_valid_score = self._coerce_score(eval_payload.get("sandbox_valid_score"))
        if sandbox_valid_score is None:
            sandbox_valid_score = self._coerce_score(eval_payload.get("valid_score"))

        sandbox_test_score = self._coerce_score(eval_payload.get("sandbox_test_score"))
        if sandbox_test_score is None:
            sandbox_test_score = self._coerce_score(eval_payload.get("test_score"))

        model_final_validation_score = None
        if protocol == "self_valid" and phase == "validation":
            model_final_validation_score = parse_final_validation_score(
                eval_payload.get("clear_run_log")
            )

        raw_scores = {
            "model_final_validation_score": model_final_validation_score,
            "sandbox_score": sandbox_score,
            "sandbox_valid_score": sandbox_valid_score,
            "sandbox_test_score": sandbox_test_score,
        }

        selection_score = None
        selection_score_source = None
        submit_score = None
        submit_score_source = None

        if protocol == "self_valid":
            if phase == "validation":
                trust_model_score = bool(
                    self.cfg["sandbox"].get(
                        "trust_model_validation_score",
                        False,
                    )
                )
                if trust_model_score:
                    selection_score = model_final_validation_score
                    selection_score_source = (
                        "model_final_validation_score"
                        if model_final_validation_score is not None
                        else "model_final_validation_score_missing"
                    )
                else:
                    selection_score = (
                        sandbox_valid_score
                        if sandbox_valid_score is not None
                        else sandbox_score
                    )
                    selection_score_source = (
                        "sandbox_valid_score"
                        if sandbox_valid_score is not None
                        else (
                            "sandbox_score"
                            if sandbox_score is not None
                            else "sandbox_score_missing"
                        )
                    )
            elif phase == "test":
                submit_score = sandbox_score
                submit_score_source = (
                    "sandbox_score" if sandbox_score is not None else "sandbox_score_missing"
                )
        elif protocol == "legacy":
            if phase == "validation":
                selection_score = sandbox_score
                selection_score_source = (
                    "sandbox_score" if sandbox_score is not None else "sandbox_score_missing"
                )
            elif phase == "test":
                submit_score = sandbox_score
                submit_score_source = (
                    "sandbox_score" if sandbox_score is not None else "sandbox_score_missing"
                )
        elif protocol == "method1":
            if phase == "validation":
                selection_score = sandbox_score
                selection_score_source = (
                    "sandbox_score" if sandbox_score is not None else "sandbox_score_missing"
                )
            elif phase == "test":
                submit_score = sandbox_score
                submit_score_source = (
                    "sandbox_score" if sandbox_score is not None else "sandbox_score_missing"
                )
        elif protocol == "method2":
            selection_score = sandbox_valid_score
            selection_score_source = (
                "sandbox_valid_score"
                if sandbox_valid_score is not None
                else "sandbox_valid_score_missing"
            )
            if phase == "test":
                submit_score = sandbox_test_score
                submit_score_source = (
                    "sandbox_test_score"
                    if sandbox_test_score is not None
                    else "sandbox_test_score_missing"
                )

        selection_reward = self._score_to_reward(selection_score)
        submit_reward = self._score_to_reward(submit_score)
        sandbox_reward = self._score_to_reward(sandbox_score)

        eval_payload["score_protocol"] = protocol
        eval_payload["raw_scores"] = raw_scores
        eval_payload["model_final_validation_score"] = model_final_validation_score
        eval_payload["sandbox_score"] = sandbox_score
        eval_payload["sandbox_valid_score"] = sandbox_valid_score
        eval_payload["sandbox_test_score"] = sandbox_test_score
        eval_payload["score"] = sandbox_score
        eval_payload["raw_score"] = sandbox_score
        eval_payload["valid_score"] = sandbox_valid_score
        eval_payload["test_score"] = sandbox_test_score
        eval_payload["sandbox_reward"] = sandbox_reward
        eval_payload["selection_score"] = selection_score
        eval_payload["selection_reward"] = selection_reward
        eval_payload["selection_score_source"] = selection_score_source
        eval_payload["submit_score"] = submit_score
        eval_payload["submit_reward"] = submit_reward
        eval_payload["submit_score_source"] = submit_score_source
        eval_payload["validation_score"] = selection_score if phase == "validation" else None
        eval_payload["validation_reward"] = selection_reward if phase == "validation" else 0.0
        eval_payload["reward"] = (
            selection_reward
            if phase == "validation"
            else submit_reward if submit_score is not None else sandbox_reward
        )

    def prepare(self, **task_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(task_args)
        state["init_obs"] = {}
        state["validation_time_used"] = self.validation_time_used
        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": self.lower_is_better,
            "task_description": self.raw_task_description,
            "data_description": self.data_description,
            "public_system_prompt": self.public_system_prompt,
            "public_user_prompt": self.public_user_prompt,
        }
        return state, task_info

    def _eval_split_for_phase(self, phase: str) -> str | None:
        return self.test_eval_split if phase == "test" else self.validation_eval_split

    async def _run_eval_async(
        self,
        code: str,
        *,
        data_dir: str,
        sandbox_base_url: str | None = None,
        eval_split: str | None = None,
        job_timeout: int | None = None,
        wait_timeout: int | None = None,
    ) -> dict[str, Any]:
        sandbox_cfg = dict(self.cfg["sandbox"])
        resource = str(sandbox_cfg["resource"])
        reward_func_utils.API_KEY = os.environ[
            "SANDBOX_CPU_API_KEY" if resource == "cpu" else "SANDBOX_GPU_API_KEY"
        ]

        timeout = httpx.Timeout(connect=10, read=100, write=30, pool=60)
        limits = httpx.Limits(
            max_connections=16,
            max_keepalive_connections=16,
            keepalive_expiry=30.0,
        )
        async with httpx.AsyncClient(
            base_url=str(sandbox_base_url or sandbox_cfg["base_url"]),
            limits=limits,
            timeout=timeout,
            trust_env=False,
            verify=bool(sandbox_cfg.get("verify_tls", True)),
        ) as client:
            status_code, payload = await get_sandbox_result(
                client=client,
                code_str=code,
                data_dir=data_dir,
                resource_type=resource,
                eval_split=eval_split,
                priority=1,
                job_timeout=int(job_timeout or sandbox_cfg["job_timeout"]),
                wait_timeout=int(wait_timeout or sandbox_cfg["wait_timeout"]),
                poll_interval=int(sandbox_cfg["poll_interval"]),
            )

        result_payload = payload.get("result") or {}
        status = str(result_payload.get("result") or "unknown")
        sandbox_score = (
            self._coerce_score(result_payload.get("score"))
            if status_code == 200
            else None
        )
        sandbox_valid_score = (
            self._coerce_score(
                result_payload.get("valid_score", result_payload.get("validation_score"))
            )
            if status_code == 200
            else None
        )
        sandbox_test_score = (
            self._coerce_score(result_payload.get("test_score"))
            if status_code == 200
            else None
        )
        scores_by_split = result_payload.get("scores_by_split")
        if not isinstance(scores_by_split, dict):
            scores_by_split = None

        queue_time = None
        run_time = None
        if status_code == 200:
            created_at = payload.get("created_at")
            started_at = payload.get("started_at")
            completed_at = payload.get("completed_at")
            if started_at and completed_at:
                started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                completed_ts = datetime.fromisoformat(
                    completed_at.replace("Z", "+00:00")
                )
                run_time = (completed_ts - started_ts).total_seconds()
                if created_at:
                    created_ts = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                    queue_time = (started_ts - created_ts).total_seconds()

        raw_run_log = result_payload.get("run_log")
        clear_run_log = get_clear_log(raw_run_log)
        feedback = format_sandbox_feedback(status_code, payload)

        return {
            "status_code": status_code,
            "status": status,
            "score": sandbox_score,
            "raw_score": sandbox_score,
            "sandbox_score": sandbox_score,
            "valid_score": sandbox_valid_score,
            "test_score": sandbox_test_score,
            "sandbox_valid_score": sandbox_valid_score,
            "sandbox_test_score": sandbox_test_score,
            "scores_by_split": scores_by_split,
            "eval_split": result_payload.get("eval_split"),
            "reward": self._score_to_reward(sandbox_score),
            "job_id": payload.get("job_id"),
            "queue_time": queue_time,
            "run_time": run_time,
            "raw_run_log": raw_run_log,
            "clear_run_log": clear_run_log,
            "feedback": feedback,
            "data_dir": data_dir,
            "sandbox_base_url": str(sandbox_base_url or sandbox_cfg["base_url"]),
        }

    def evaluate_code(self, code: str, *, phase: str) -> dict[str, Any]:
        sandbox_cfg = dict(self.cfg["sandbox"])
        data_dir = self.validation_data_dir
        code_to_run = code
        job_timeout = int(sandbox_cfg["job_timeout"])
        wait_timeout = int(sandbox_cfg["wait_timeout"])
        if phase == "test":
            data_dir = self.submit_data_dir
            code_to_run = self.build_submit_code(code)
            job_timeout = int(self.submit_job_timeout or job_timeout)
            wait_timeout = int(self.submit_wait_timeout or wait_timeout)

        eval_split = self._eval_split_for_phase(phase)
        with acquire_sandbox_slot():
            eval_payload = asyncio.run(
                self._run_eval_async(
                    code_to_run,
                    data_dir=data_dir,
                    eval_split=eval_split,
                    job_timeout=job_timeout,
                    wait_timeout=wait_timeout,
                )
            )
        eval_payload["phase"] = phase
        eval_payload["requested_eval_split"] = eval_split
        self._annotate_eval_scores(eval_payload, phase=phase)
        if (
            phase == "validation"
            and eval_payload["status_code"] == 200
            and eval_payload.get("run_time") is not None
        ):
            self.validation_time_used += float(eval_payload["run_time"])
            if (
                self.time_budget is not None
                and self.validation_time_used >= self.time_budget
            ):
                self.stop_requested = True
        return eval_payload

    async def evaluate_code_async(
        self,
        code: str,
        *,
        phase: str,
        sandbox_base_url: str | None = None,
    ) -> dict[str, Any]:
        sandbox_cfg = dict(self.cfg["sandbox"])
        data_dir = self.validation_data_dir
        code_to_run = code
        job_timeout = int(sandbox_cfg["job_timeout"])
        wait_timeout = int(sandbox_cfg["wait_timeout"])
        if phase == "test":
            data_dir = self.submit_data_dir
            code_to_run = self.build_submit_code(code)
            job_timeout = int(self.submit_job_timeout or job_timeout)
            wait_timeout = int(self.submit_wait_timeout or wait_timeout)

        eval_split = self._eval_split_for_phase(phase)
        async with acquire_sandbox_slot_async():
            eval_payload = await self._run_eval_async(
                code_to_run,
                data_dir=data_dir,
                sandbox_base_url=sandbox_base_url,
                eval_split=eval_split,
                job_timeout=job_timeout,
                wait_timeout=wait_timeout,
            )
        eval_payload["phase"] = phase
        eval_payload["requested_eval_split"] = eval_split
        self._annotate_eval_scores(eval_payload, phase=phase)
        if (
            phase == "validation"
            and eval_payload["status_code"] == 200
            and eval_payload.get("run_time") is not None
        ):
            async with self._validation_time_lock:
                self.validation_time_used += float(eval_payload["run_time"])
                if (
                    self.time_budget is not None
                    and self.validation_time_used >= self.time_budget
                ):
                    self.stop_requested = True
        return eval_payload

    def _build_execution_result(self, eval_payload: dict[str, Any]) -> ExecutionResult:
        clear_run_log = str(eval_payload.get("clear_run_log") or "")
        feedback = str(eval_payload.get("feedback") or "")
        terminal_output = clear_run_log if clear_run_log else feedback
        status = str(eval_payload.get("status") or "")
        return ExecutionResult(
            term_out=[terminal_output],
            exec_time=float(eval_payload.get("run_time") or 0.0),
            exit_code=0 if eval_payload.get("status_code") == 200 else 1,
            timed_out=bool(
                eval_payload.get("status_code") == 504 or status.lower() == "timeout"
            ),
        )

    def step_task(
        self,
        state: dict[str, Any],
        action: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        eval_payload = self.evaluate_code(str(action), phase="validation")
        state["validation_time_used"] = self.validation_time_used
        valid_solution = eval_payload["status_code"] == 200
        validation_fitness = eval_payload.get("selection_score")
        if validation_fitness is None:
            validation_fitness = eval_payload.get("validation_score")
        if validation_fitness is None and self._score_protocol() in {"legacy", "method1"}:
            validation_fitness = eval_payload.get("sandbox_score")
            if validation_fitness is None:
                validation_fitness = eval_payload.get("score")
        validation_fitness = validation_fitness if valid_solution else None
        eval_result = {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: validation_fitness,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }
        return state, eval_result

    async def step_task_async(
        self,
        state: dict[str, Any],
        action: Any,
        *,
        sandbox_base_url: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        eval_payload = await self.evaluate_code_async(
            str(action),
            phase="validation",
            sandbox_base_url=sandbox_base_url,
        )
        state["validation_time_used"] = self.validation_time_used
        valid_solution = eval_payload["status_code"] == 200
        validation_fitness = eval_payload.get("selection_score")
        if validation_fitness is None:
            validation_fitness = eval_payload.get("validation_score")
        if validation_fitness is None and self._score_protocol() in {"legacy", "method1"}:
            validation_fitness = eval_payload.get("sandbox_score")
            if validation_fitness is None:
                validation_fitness = eval_payload.get("score")
        validation_fitness = validation_fitness if valid_solution else None
        eval_result = {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: validation_fitness,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }
        return state, eval_result

    def evaluate_fitness(
        self,
        solution: Any | None = None,
        state: dict[str, Any] | None = None,
        interpreter: Interpreter | None = None,
        aux_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if solution is None:
            return {}

        eval_payload = self.evaluate_code(str(solution), phase="test")
        valid_solution = eval_payload["status_code"] == 200
        test_fitness = eval_payload.get("submit_score")
        if test_fitness is None:
            if self._score_protocol() == "method2":
                test_fitness = eval_payload.get("sandbox_test_score")
                if test_fitness is None:
                    test_fitness = eval_payload.get("test_score")
            else:
                test_fitness = eval_payload.get("sandbox_score")
                if test_fitness is None:
                    test_fitness = eval_payload.get("score")
        return {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            TEST_FITNESS: test_fitness if valid_solution else None,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }

    def close(self, state: dict[str, Any]) -> None:
        for interp_key in ("solver_interpreter", "eval_interpreter"):
            interpreter = state.get(interp_key)
            if interpreter is None:
                continue
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()
            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
