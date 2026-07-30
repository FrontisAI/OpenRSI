"""
Evaluator Service - Handles sandbox code evaluation.

This service is responsible for evaluating generated code solutions
using the sandbox environment. It operates independently from the
generation service and communicates through async queues.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from tts_search import eval_utils
from tts_search.reward_func_utils import (
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    normalize_sandbox_resource_type,
    resolve_sandbox_resource_type,
    score2reward,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRequest:
    """Request for code evaluation."""

    request_id: str
    task_id: str
    task_name: str
    code: str
    data_dir: str
    metadata: dict[str, Any]
    step_index: int
    output_dir: Path | None = None
    use_score2reward: bool = True
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    generation_total_tokens: int | None = None


@dataclass
class EvaluationResult:
    """Result from code evaluation."""

    request_id: str
    task_id: str
    task_name: str
    status_code: int
    status: str
    score: float | None
    reward: float
    job_id: str | None
    queue_time: float | None
    run_time: float | None
    raw_run_log: str | None
    clear_run_log: str
    feedback: str
    submit_grade: float | None
    submit_medal: str
    metadata: dict[str, Any]
    step_index: int
    data_dir: str
    code: str
    output_dir: Path | None = None
    generation_prompt_tokens: int | None = None
    generation_completion_tokens: int | None = None
    generation_total_tokens: int | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.score is not None


class EvaluatorService:
    """
    Service for sandbox code evaluation.

    This service handles:
    - Async sandbox evaluation with retry logic
    - Score and reward calculation
    - Test set evaluation (no validation set)
    - Output artifact persistence
    """

    def __init__(
        self,
        base_url: str,
        concurrency: int = 32,
        job_timeout: int = 3600,
        wait_timeout: int = 86400,
        poll_interval: int = 5,
        cpu_base_url: str | None = None,
        resource_type_override: str | None = None,
    ):
        """
        Initialize the evaluator service.

        Args:
            base_url: Sandbox API base URL for GPU resources
            concurrency: Maximum concurrent evaluation requests
            job_timeout: Job execution timeout in seconds
            wait_timeout: Maximum wait time for job completion
            poll_interval: Polling interval for job status
            cpu_base_url: Optional separate base URL for CPU resources (defaults to base_url)
            resource_type_override: Force all sandbox jobs to use this resource type
        """
        self._base_url = base_url
        self._cpu_base_url = cpu_base_url or base_url
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._job_timeout = job_timeout
        self._wait_timeout = wait_timeout
        self._poll_interval = poll_interval
        self._resource_type_override = (
            normalize_sandbox_resource_type(resource_type_override, default="gpu")
            if resource_type_override is not None
            else None
        )

        # HTTP clients will be created in context manager
        self._gpu_client: httpx.AsyncClient | None = None
        self._cpu_client: httpx.AsyncClient | None = None

        # Statistics
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        self._total_score = 0.0
        self._status_counts: dict[str, int] = {}

    def _build_internal_error_result(
        self, request: EvaluationRequest, exc: Exception
    ) -> EvaluationResult:
        """Convert unexpected evaluator exceptions into a normal failed result."""
        status = "internal_error"
        self._total_evaluations += 1
        self._failed_evaluations += 1
        self._status_counts[status] = self._status_counts.get(status, 0) + 1

        error_text = f"{type(exc).__name__}: {exc}"
        feedback = f"Internal evaluation error: {error_text}"
        return EvaluationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            status_code=500,
            status=status,
            score=None,
            reward=0.0,
            job_id=None,
            queue_time=None,
            run_time=None,
            raw_run_log=None,
            clear_run_log="",
            feedback=feedback,
            submit_grade=None,
            submit_medal="N/A",
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            code=request.code,
            output_dir=request.output_dir,
            generation_prompt_tokens=request.generation_prompt_tokens,
            generation_completion_tokens=request.generation_completion_tokens,
            generation_total_tokens=request.generation_total_tokens,
            error=error_text,
        )

    async def __aenter__(self) -> "EvaluatorService":
        """Create HTTP clients on context entry."""
        if self._resource_type_override is not None:
            logger.info(
                "Sandbox resource override enabled: forcing all evaluations to %s",
                self._resource_type_override,
            )

        timeout = httpx.Timeout(connect=10, read=100, write=30, pool=60)
        max_connections = min(max(int(1.5 * self._concurrency), 16), 128)
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=30.0,
        )

        # Create GPU client
        self._gpu_client = httpx.AsyncClient(
            base_url=self._base_url, limits=limits, timeout=timeout
        )

        # Create CPU client (reuse GPU client if URLs are the same)
        if self._cpu_base_url == self._base_url:
            self._cpu_client = self._gpu_client
        else:
            self._cpu_client = httpx.AsyncClient(
                base_url=self._cpu_base_url, limits=limits, timeout=timeout
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP clients on context exit."""
        # Close GPU client
        if self._gpu_client:
            await self._gpu_client.aclose()
            self._gpu_client = None

        # Close CPU client only if it's different from GPU client
        if self._cpu_client and self._cpu_client != self._gpu_client:
            await self._cpu_client.aclose()
        self._cpu_client = None

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate code in sandbox.

        Args:
            request: Evaluation request with code and task details

        Returns:
            EvaluationResult with score and execution details
        """
        if self._gpu_client is None or self._cpu_client is None:
            raise RuntimeError("EvaluatorService must be used as async context manager")

        # Determine the actual data directory based on eval mode
        data_dir = request.data_dir
        code = request.code
        print(f"data_dir: {data_dir}")

        # Resolve effective resource_type from config override or task metadata.
        resource_type = resolve_sandbox_resource_type(
            request.metadata,
            override=self._resource_type_override,
            default="gpu",
        )

        # Select client based on resource_type
        client = (
            self._cpu_client if resource_type.lower() == "cpu" else self._gpu_client
        )

        async with self._semaphore:
            status_code, payload = await get_sandbox_result(
                client=client,
                code_str=code,
                data_dir=data_dir,
                job_timeout=self._job_timeout,
                wait_timeout=self._wait_timeout,
                resource_type=resource_type,
                priority=2,
                poll_interval=self._poll_interval,
            )

        sandbox_job_id = payload.get("job_id")
        result_payload = payload.get("result") or {}
        status = result_payload.get("result") or "unknown"
        status = str(status)

        score_value = result_payload.get("score")
        if status_code != 200 or score_value is None:
            score = None
        else:
            score = float(score_value)

        reward = 0.0
        if score is not None:
            reward = (
                score2reward(score, request.metadata, mode="power_sigmoid")
                if request.use_score2reward
                else score
            )

        submit_grade: float | None = None
        submit_medal = "N/A"
        if score is not None:
            try:
                leaderboard = eval_utils.load_leaderboard(request.metadata)
                if leaderboard is not None:
                    submit_grade, submit_medal = (
                        eval_utils.build_submit_grade_and_medal(score, leaderboard)
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to compute submit medal for task=%s score=%s: %s",
                    request.task_name,
                    score,
                    exc,
                )

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

        # Save artifacts if output directory is specified
        if request.output_dir:
            print(f"request.output_dir: {request.output_dir}")
            step_dir = request.output_dir / f"step_{request.step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "raw_run_log.txt").write_text(
                str(raw_run_log or ""), encoding="utf-8"
            )
            (step_dir / "clear_run_log.txt").write_text(
                str(clear_run_log or ""), encoding="utf-8"
            )
            (step_dir / "feedback.txt").write_text(
                str(feedback or ""), encoding="utf-8"
            )

        # Update statistics
        self._total_evaluations += 1
        self._status_counts[status] = self._status_counts.get(status, 0) + 1
        if score is not None:
            self._successful_evaluations += 1
            self._total_score += score
        else:
            self._failed_evaluations += 1

        return EvaluationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            status_code=status_code,
            status=status,
            score=score,
            reward=reward,
            job_id=sandbox_job_id,
            queue_time=queue_time,
            run_time=run_time,
            raw_run_log=raw_run_log,
            clear_run_log=clear_run_log,
            feedback=feedback,
            submit_grade=submit_grade,
            submit_medal=submit_medal,
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            code=request.code,
            output_dir=request.output_dir,
            generation_prompt_tokens=request.generation_prompt_tokens,
            generation_completion_tokens=request.generation_completion_tokens,
            generation_total_tokens=request.generation_total_tokens,
        )

    async def evaluate_batch(
        self, requests: list[EvaluationRequest]
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple code samples concurrently.

        Args:
            requests: List of evaluation requests

        Returns:
            List of evaluation results
        """
        tasks = [self.evaluate(req) for req in requests]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[EvaluationResult] = []
        for request, result in zip(requests, raw_results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Evaluation crashed for request_id=%s task_id=%s step_index=%s",
                    request.request_id,
                    request.task_id,
                    request.step_index,
                    exc_info=(type(result), result, result.__traceback__),
                )
                results.append(self._build_internal_error_result(request, result))
            else:
                results.append(result)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "successful_evaluations": self._successful_evaluations,
            "failed_evaluations": self._failed_evaluations,
            "total_score": self._total_score,
            "average_score": (
                self._total_score / self._successful_evaluations
                if self._successful_evaluations > 0
                else 0.0
            ),
            "status_counts": dict(self._status_counts),
        }
