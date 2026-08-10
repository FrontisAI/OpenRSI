from __future__ import annotations

import base64
import multiprocessing

from tts_search.airaevo_concurrency import (
    ENV_AUTHKEY,
    ENV_HOST,
    ENV_PORT,
    SharedConcurrencyServer,
    _ClientSemaphoreManager,
)


def test_shared_concurrency_server_supports_spawn_context():
    server = SharedConcurrencyServer(
        llm_concurrency=1,
        sandbox_concurrency=1,
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        manager = _ClientSemaphoreManager(
            address=(server.env[ENV_HOST], int(server.env[ENV_PORT])),
            authkey=base64.urlsafe_b64decode(server.env[ENV_AUTHKEY].encode("ascii")),
        )
        manager.connect()

        llm_semaphore = manager.get_llm_semaphore()
        assert llm_semaphore.acquire(False) is True
        assert llm_semaphore.acquire(False) is False
        llm_semaphore.release()

        sandbox_semaphore = manager.get_sandbox_semaphore()
        assert sandbox_semaphore.acquire(False) is True
        assert sandbox_semaphore.acquire(False) is False
        sandbox_semaphore.release()
    finally:
        server.shutdown()
