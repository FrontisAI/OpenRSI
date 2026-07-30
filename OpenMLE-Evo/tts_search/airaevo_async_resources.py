from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

EXECUTION_MODES = frozenset({"generation", "async_steady_state"})
SANDBOX_ASSIGNMENTS = frozenset({"round_robin"})


def resolve_execution_mode(value: str | None) -> str:
    mode = str(value or "generation").strip().lower()
    if mode not in EXECUTION_MODES:
        supported = ", ".join(sorted(EXECUTION_MODES))
        raise ValueError(
            f"Unsupported execution mode: {mode}. Supported modes: {supported}"
        )
    return mode


def detect_gpu_indices() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def resolve_async_worker_count(
    value: int | str | None, *, gpu_indices: list[int] | None = None
) -> int:
    if value is None:
        return 1
    if isinstance(value, str) and value.lower() == "auto":
        resolved_gpu_indices = (
            detect_gpu_indices() if gpu_indices is None else gpu_indices
        )
        return max(1, len(resolved_gpu_indices))
    return max(1, int(value))


def resolve_async_sandbox_urls(
    value: Any, *, fallback_url: str | None = None
) -> list[str]:
    if value is None or value == "":
        return [str(fallback_url).strip()] if str(fallback_url or "").strip() else []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    urls = [str(item).strip() for item in list(value) if str(item).strip()]
    if urls:
        return urls
    return [str(fallback_url).strip()] if str(fallback_url or "").strip() else []


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    gpu_index: int | None
    sandbox_url: str


class RoundRobinSandboxPool:
    def __init__(self, urls: list[str]) -> None:
        self.urls = list(urls)
        self._index = 0

    def next_url(self) -> str:
        url = self.urls[self._index % len(self.urls)]
        self._index += 1
        return url


def build_worker_specs(
    *,
    worker_count: int,
    sandbox_urls: list[str],
    assignment: str = "round_robin",
) -> list[WorkerSpec]:
    if assignment not in SANDBOX_ASSIGNMENTS:
        supported = ", ".join(sorted(SANDBOX_ASSIGNMENTS))
        raise ValueError(
            f"Unsupported sandbox assignment: {assignment}. "
            f"Supported assignments: {supported}"
        )
    if not sandbox_urls:
        raise ValueError("At least one async sandbox URL is required")
    gpu_indices = detect_gpu_indices()
    pool = RoundRobinSandboxPool(sandbox_urls)
    return [
        WorkerSpec(
            worker_id=worker_id,
            gpu_index=gpu_indices[worker_id % len(gpu_indices)]
            if gpu_indices
            else None,
            sandbox_url=pool.next_url(),
        )
        for worker_id in range(worker_count)
    ]


async def fetch_router_idle_worker_count(
    router_url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 5.0,
) -> int | None:
    import httpx

    headers = {"X-API-Key": api_key or os.environ.get("SANDBOX_GPU_API_KEY", "")}
    try:
        async with httpx.AsyncClient(
            base_url=router_url, timeout=timeout_seconds, trust_env=False
        ) as client:
            response = await client.get("/api/v1/workers/status", headers=headers)
        if response.status_code != 200:
            return None
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    workers = payload.get("workers_show") if isinstance(payload, dict) else None
    if not isinstance(workers, list):
        return None
    return sum(
        1 for worker in workers if str(worker.get("status", "")).lower() == "idle"
    )
