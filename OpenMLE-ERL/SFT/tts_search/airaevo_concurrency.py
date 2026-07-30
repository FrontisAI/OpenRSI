from __future__ import annotations

import base64
import math
import os
import time
from contextlib import contextmanager
from multiprocessing import Semaphore
from multiprocessing.managers import SyncManager
from secrets import token_bytes
from typing import Any, Iterator


ENV_ENABLED = "AIRAEVO_SHARED_CONCURRENCY"
ENV_HOST = "AIRAEVO_SHARED_CONCURRENCY_HOST"
ENV_PORT = "AIRAEVO_SHARED_CONCURRENCY_PORT"
ENV_AUTHKEY = "AIRAEVO_SHARED_CONCURRENCY_AUTHKEY"


class _ClientSemaphoreManager(SyncManager):
    pass


_ClientSemaphoreManager.register("get_llm_semaphore")
_ClientSemaphoreManager.register("get_sandbox_semaphore")


class SharedConcurrencyServer:
    def __init__(self, *, llm_concurrency: int, sandbox_concurrency: int) -> None:
        self._llm_semaphore = Semaphore(llm_concurrency)
        self._sandbox_semaphore = Semaphore(sandbox_concurrency)
        authkey = token_bytes(16)

        class _ServerSemaphoreManager(SyncManager):
            pass

        _ServerSemaphoreManager.register(
            "get_llm_semaphore",
            callable=self._get_llm_semaphore,
        )
        _ServerSemaphoreManager.register(
            "get_sandbox_semaphore",
            callable=self._get_sandbox_semaphore,
        )

        self._manager = _ServerSemaphoreManager(
            address=("127.0.0.1", 0),
            authkey=authkey,
        )
        self._manager.start()
        host, port = self._manager.address
        self.env = {
            ENV_ENABLED: "1",
            ENV_HOST: str(host),
            ENV_PORT: str(port),
            ENV_AUTHKEY: base64.urlsafe_b64encode(authkey).decode("ascii"),
        }

    def _get_llm_semaphore(self) -> Semaphore:
        return self._llm_semaphore

    def _get_sandbox_semaphore(self) -> Semaphore:
        return self._sandbox_semaphore

    def shutdown(self) -> None:
        self._manager.shutdown()


_client_manager: _ClientSemaphoreManager | None = None
_llm_semaphore_proxy: Any | None = None
_sandbox_semaphore_proxy: Any | None = None


def resolve_task_concurrency_per_epoch(
    *,
    configured_task_concurrency: int | None,
    llm_concurrency: int,
    sandbox_concurrency: int,
    num_epochs: int,
) -> int:
    if configured_task_concurrency is not None:
        return int(configured_task_concurrency)

    target_total_concurrency = max(int(llm_concurrency), int(sandbox_concurrency))
    return max(1, math.ceil(target_total_concurrency / int(num_epochs)))


def _connect_client_manager() -> _ClientSemaphoreManager | None:
    global _client_manager
    if os.getenv(ENV_ENABLED) != "1":
        return None

    if _client_manager is None:
        authkey = base64.urlsafe_b64decode(os.environ[ENV_AUTHKEY].encode("ascii"))
        _client_manager = _ClientSemaphoreManager(
            address=(os.environ[ENV_HOST], int(os.environ[ENV_PORT])),
            authkey=authkey,
        )
        _client_manager.connect()
    return _client_manager


def _get_llm_semaphore_proxy() -> Any | None:
    global _llm_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    if _llm_semaphore_proxy is None:
        _llm_semaphore_proxy = manager.get_llm_semaphore()
    return _llm_semaphore_proxy


def _get_sandbox_semaphore_proxy() -> Any | None:
    global _sandbox_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    if _sandbox_semaphore_proxy is None:
        _sandbox_semaphore_proxy = manager.get_sandbox_semaphore()
    return _sandbox_semaphore_proxy


@contextmanager
def acquire_llm_slot() -> Iterator[float]:
    semaphore = _get_llm_semaphore_proxy()
    if semaphore is None:
        yield 0.0
        return

    wait_start = time.perf_counter()
    semaphore.acquire()
    queue_time = time.perf_counter() - wait_start
    try:
        yield queue_time
    finally:
        semaphore.release()


@contextmanager
def acquire_sandbox_slot() -> Iterator[float]:
    semaphore = _get_sandbox_semaphore_proxy()
    if semaphore is None:
        yield 0.0
        return

    wait_start = time.perf_counter()
    semaphore.acquire()
    queue_time = time.perf_counter() - wait_start
    try:
        yield queue_time
    finally:
        semaphore.release()
