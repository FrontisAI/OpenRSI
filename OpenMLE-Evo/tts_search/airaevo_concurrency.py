from __future__ import annotations

import asyncio
import base64
import math
import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager
from secrets import token_bytes
from typing import Any, AsyncIterator, Iterator

ENV_ENABLED = "AIRAEVO_SHARED_CONCURRENCY"
ENV_HOST = "AIRAEVO_SHARED_CONCURRENCY_HOST"
ENV_PORT = "AIRAEVO_SHARED_CONCURRENCY_PORT"
ENV_AUTHKEY = "AIRAEVO_SHARED_CONCURRENCY_AUTHKEY"
ASYNC_SEMAPHORE_POLL_INTERVAL_SECONDS = 0.25


class _ClientSemaphoreManager(SyncManager):
    pass


class _ServerSemaphoreManager(SyncManager):
    pass


_server_llm_semaphore: Any | None = None
_server_sandbox_semaphore: Any | None = None


def _initialize_server_semaphores(llm_semaphore: Any, sandbox_semaphore: Any) -> None:
    global _server_llm_semaphore, _server_sandbox_semaphore
    _server_llm_semaphore = llm_semaphore
    _server_sandbox_semaphore = sandbox_semaphore


def _get_server_llm_semaphore() -> Any:
    if _server_llm_semaphore is None:
        raise RuntimeError("Shared LLM semaphore is not initialized")
    return _server_llm_semaphore


def _get_server_sandbox_semaphore() -> Any:
    if _server_sandbox_semaphore is None:
        raise RuntimeError("Shared sandbox semaphore is not initialized")
    return _server_sandbox_semaphore


_ClientSemaphoreManager.register("get_llm_semaphore")
_ClientSemaphoreManager.register("get_sandbox_semaphore")
_ServerSemaphoreManager.register(
    "get_llm_semaphore",
    callable=_get_server_llm_semaphore,
)
_ServerSemaphoreManager.register(
    "get_sandbox_semaphore",
    callable=_get_server_sandbox_semaphore,
)


class SharedConcurrencyServer:
    def __init__(
        self,
        *,
        llm_concurrency: int,
        sandbox_concurrency: int,
        mp_context: BaseContext | None = None,
    ) -> None:
        context = mp_context or multiprocessing.get_context()
        self._llm_semaphore = context.Semaphore(llm_concurrency)
        self._sandbox_semaphore = context.Semaphore(sandbox_concurrency)
        authkey = token_bytes(16)

        self._manager = _ServerSemaphoreManager(
            address=("127.0.0.1", 0),
            authkey=authkey,
            ctx=context,
        )
        self._manager.start(
            initializer=_initialize_server_semaphores,
            initargs=(self._llm_semaphore, self._sandbox_semaphore),
        )
        host, port = self._manager.address
        self.env = {
            ENV_ENABLED: "1",
            ENV_HOST: str(host),
            ENV_PORT: str(port),
            ENV_AUTHKEY: base64.urlsafe_b64encode(authkey).decode("ascii"),
        }

    def shutdown(self) -> None:
        self._manager.shutdown()


_client_manager: _ClientSemaphoreManager | None = None
_llm_semaphore_proxy: Any | None = None
_sandbox_semaphore_proxy: Any | None = None
_client_lock = threading.Lock()


def resolve_task_concurrency_per_epoch(
    *,
    configured_task_concurrency: int | None,
    llm_concurrency: int,
    sandbox_concurrency: int,
    num_epochs: int,
    async_workers: int | str | None = 1,
) -> int:
    if configured_task_concurrency is not None:
        return int(configured_task_concurrency)

    from tts_search.airaevo_async_resources import resolve_async_worker_count

    target_total_concurrency = max(int(llm_concurrency), int(sandbox_concurrency))
    task_worker_count = int(num_epochs) * resolve_async_worker_count(async_workers)
    return max(1, math.ceil(target_total_concurrency / task_worker_count))


def _connect_client_manager() -> _ClientSemaphoreManager | None:
    global _client_manager
    if os.getenv(ENV_ENABLED) != "1":
        return None

    with _client_lock:
        if _client_manager is None:
            authkey = base64.urlsafe_b64decode(os.environ[ENV_AUTHKEY].encode("ascii"))
            manager = _ClientSemaphoreManager(
                address=(os.environ[ENV_HOST], int(os.environ[ENV_PORT])),
                authkey=authkey,
            )
            manager.connect()
            _client_manager = manager
    return _client_manager


def _get_llm_semaphore_proxy() -> Any | None:
    global _llm_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    with _client_lock:
        if _llm_semaphore_proxy is None:
            _llm_semaphore_proxy = manager.get_llm_semaphore()
    return _llm_semaphore_proxy


def _get_sandbox_semaphore_proxy() -> Any | None:
    global _sandbox_semaphore_proxy
    manager = _connect_client_manager()
    if manager is None:
        return None
    with _client_lock:
        if _sandbox_semaphore_proxy is None:
            _sandbox_semaphore_proxy = manager.get_sandbox_semaphore()
    return _sandbox_semaphore_proxy


async def _acquire_semaphore_async(semaphore: Any) -> float:
    wait_start = time.perf_counter()
    while not await asyncio.to_thread(semaphore.acquire, False):
        await asyncio.sleep(ASYNC_SEMAPHORE_POLL_INTERVAL_SECONDS)
    return time.perf_counter() - wait_start


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


@asynccontextmanager
async def acquire_llm_slot_async() -> AsyncIterator[float]:
    semaphore = await asyncio.to_thread(_get_llm_semaphore_proxy)
    if semaphore is None:
        yield 0.0
        return

    queue_time = await _acquire_semaphore_async(semaphore)
    try:
        yield queue_time
    finally:
        await asyncio.to_thread(semaphore.release)


@asynccontextmanager
async def acquire_sandbox_slot_async() -> AsyncIterator[float]:
    semaphore = await asyncio.to_thread(_get_sandbox_semaphore_proxy)
    if semaphore is None:
        yield 0.0
        return

    queue_time = await _acquire_semaphore_async(semaphore)
    try:
        yield queue_time
    finally:
        await asyncio.to_thread(semaphore.release)
