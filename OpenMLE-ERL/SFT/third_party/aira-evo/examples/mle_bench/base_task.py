from __future__ import annotations

import asyncio
import os
import random
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
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)
from tts_search import reward_func_utils
from tts_search.reward_func_utils import (
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    score2reward,
)
from tts_search.airaevo_concurrency import acquire_sandbox_slot


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
        self.time_budget = float(time_budget) if time_budget is not None else None
        self.validation_time_used = 0.0
        self.stop_requested = False

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

    def _build_task_description(self) -> str:
        sections = [
            ("Task Description", self.raw_task_description),
            ("Data Description", self.data_description),
        ]
        return "\n\n".join(
            f"{title}:\n{content.strip()}" for title, content in sections if content.strip()
        )

    def build_submit_code(self, code: str) -> str:
        if self.submit_data_dir_root:
            return code
        return code.replace(self.validation_dir_name, self.submit_dir)

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

    async def _run_eval_async(self, code: str, *, data_dir: str) -> dict[str, Any]:
        if os.getenv("AIRAEVO_FAKE_SANDBOX") == "1":
            code_text = str(code or "")
            if not code_text.strip():
                return self._fake_sandbox_failure(
                    reason="empty code",
                    data_dir=data_dir,
                )
            try:
                compile(code_text, "<airaevo_fake_sandbox>", "exec")
            except SyntaxError as exc:
                return self._fake_sandbox_failure(
                    reason=f"syntax error: {exc.msg}",
                    data_dir=data_dir,
                )

            score = self._next_fake_score()
            reward = (
                score2reward(score, self.cfg, mode="power_sigmoid")
                if bool(self.cfg["sandbox"]["use_score2reward"])
                else score
            )
            return {
                "status_code": 200,
                "status": "completed",
                "score": score,
                "reward": reward,
                "job_id": f"fake-sandbox-{score:.4f}",
                "queue_time": 0.0,
                "run_time": 1.0,
                "raw_run_log": f"FAKE_SANDBOX score={score:.4f}",
                "clear_run_log": f"FAKE_SANDBOX score={score:.4f}",
                "feedback": f"FAKE_SANDBOX completed with score={score:.4f}",
                "data_dir": data_dir,
            }

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
            base_url=str(sandbox_cfg["base_url"]),
            limits=limits,
            timeout=timeout,
            trust_env=False,
            verify=False,
        ) as client:
            status_code, payload = await get_sandbox_result(
                client=client,
                code_str=code,
                data_dir=data_dir,
                resource_type=resource,
                priority=1,
                job_timeout=int(sandbox_cfg["job_timeout"]),
                wait_timeout=int(sandbox_cfg["wait_timeout"]),
                poll_interval=int(sandbox_cfg["poll_interval"]),
            )

        result_payload = payload.get("result") or {}
        status = str(result_payload.get("result") or "unknown")
        score_value = result_payload.get("score")
        score = float(score_value) if status_code == 200 and score_value is not None else None

        reward = 0.0
        if score is not None:
            reward = (
                score2reward(score, self.cfg, mode="power_sigmoid")
                if bool(sandbox_cfg["use_score2reward"])
                else score
            )

        queue_time = None
        run_time = None
        if status_code == 200:
            created_at = payload.get("created_at")
            started_at = payload.get("started_at")
            completed_at = payload.get("completed_at")
            if started_at and completed_at:
                started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                completed_ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                run_time = (completed_ts - started_ts).total_seconds()
                if created_at:
                    created_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    queue_time = (started_ts - created_ts).total_seconds()

        raw_run_log = result_payload.get("run_log")
        clear_run_log = get_clear_log(raw_run_log)
        feedback = format_sandbox_feedback(status_code, payload)

        return {
            "status_code": status_code,
            "status": status,
            "score": score,
            "reward": reward,
            "job_id": payload.get("job_id"),
            "queue_time": queue_time,
            "run_time": run_time,
            "raw_run_log": raw_run_log,
            "clear_run_log": clear_run_log,
            "feedback": feedback,
            "data_dir": data_dir,
        }

    def _next_fake_score(self) -> float:
        scores = [
            float(item)
            for item in os.getenv("AIRAEVO_FAKE_SANDBOX_SCORES", "").split(",")
            if item.strip()
        ]
        counter_path_value = os.getenv("AIRAEVO_FAKE_SANDBOX_COUNTER_PATH")
        counter_path = Path(counter_path_value) if counter_path_value else None
        index = 0
        if counter_path is not None and counter_path.exists():
            index = int(counter_path.read_text(encoding="utf-8").strip() or "0")
        if scores:
            score = scores[index % len(scores)]
        else:
            rng = random.Random(int(os.getenv("AIRAEVO_FAKE_SANDBOX_SEED", "42")) + index)
            score = rng.random()
        if counter_path is not None:
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            counter_path.write_text(str(index + 1), encoding="utf-8")
        return float(score)

    def _fake_sandbox_failure(self, *, reason: str, data_dir: str) -> dict[str, Any]:
        feedback = f"FAKE_SANDBOX failed: {reason}"
        return {
            "status_code": 400,
            "status": "failed",
            "score": None,
            "reward": 0.0,
            "job_id": "fake-sandbox-failed",
            "queue_time": 0.0,
            "run_time": 0.0,
            "raw_run_log": feedback,
            "clear_run_log": feedback,
            "feedback": feedback,
            "data_dir": data_dir,
        }

    def evaluate_code(self, code: str, *, phase: str) -> dict[str, Any]:
        data_dir = self.validation_data_dir
        code_to_run = code
        if phase == "test":
            data_dir = self.submit_data_dir
            code_to_run = self.build_submit_code(code)

        with acquire_sandbox_slot():
            eval_payload = asyncio.run(self._run_eval_async(code_to_run, data_dir=data_dir))
        eval_payload["phase"] = phase
        if (
            phase == "validation"
            and eval_payload["status_code"] == 200
            and eval_payload.get("run_time") is not None
        ):
            self.validation_time_used += float(eval_payload["run_time"])
            if self.time_budget is not None and self.validation_time_used >= self.time_budget:
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
        eval_result = {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: eval_payload.get("score") if valid_solution else None,
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
        return {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: eval_payload.get("score") if valid_solution else None,
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
