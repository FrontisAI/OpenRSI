import asyncio
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("LOGGING_DIR", "/tmp")

if "dataclasses_json" not in sys.modules:
    dataclasses_json_stub = types.ModuleType("dataclasses_json")
    dataclasses_json_stub.DataClassJsonMixin = object
    sys.modules["dataclasses_json"] = dataclasses_json_stub

for module_name in ["jsonlines", "wandb"]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

if "colorama" not in sys.modules:
    colorama_stub = types.ModuleType("colorama")
    colorama_stub.Fore = types.SimpleNamespace(
        MAGENTA="",
        GREEN="",
        WHITE="",
        CYAN="",
        YELLOW="",
        RED="",
        BLUE="",
    )
    colorama_stub.Style = types.SimpleNamespace(BRIGHT="", RESET_ALL="")
    sys.modules["colorama"] = colorama_stub

if "igraph" not in sys.modules:
    igraph_stub = types.ModuleType("igraph")
    igraph_stub.Graph = type("Graph", (), {})
    sys.modules["igraph"] = igraph_stub

REPO_DIR = Path(__file__).resolve().parents[3]
AIRA_EVO_DIR = REPO_DIR / "third_party" / "aira-evo"
SRC_DIR = AIRA_EVO_DIR / "src"
EXAMPLES_MLE_BENCH_DIR = AIRA_EVO_DIR / "examples" / "mle_bench"
for path in (REPO_DIR, SRC_DIR, EXAMPLES_MLE_BENCH_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from base_task import SandboxMLEBenchTask
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.metric import MetricValue
from dojo.core.tasks.constants import AUX_EVAL_INFO
from dojo.core.tasks.constants import VALIDATION_FITNESS
from dojo.solvers.evo.evo import AsyncWorkItem
from dojo.solvers.evo.evo import Evolutionary
from dojo.solvers.evo.evo import SolutionsDatabase
from tts_search.airaevo_async_resources import WorkerSpec
from tts_search.airaevo_async_resources import build_worker_specs
from tts_search.airaevo_async_resources import detect_gpu_indices
from tts_search.airaevo_async_resources import RoundRobinSandboxPool
from tts_search.airaevo_async_resources import resolve_execution_mode
from tts_search.airaevo_async_resources import resolve_async_sandbox_urls
from tts_search.airaevo_async_resources import resolve_async_worker_count
from tts_search import airaevo_concurrency
from tts_search.airaevo_concurrency import acquire_sandbox_slot_async
from tts_search.airaevo_concurrency import resolve_task_concurrency_per_epoch


class Logger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_node(score: float) -> Node:
    node = Node(code="print('ok')")
    node.metric = MetricValue(
        score,
        maximize=True,
        info={
            "score": score,
            "raw_score": score,
            "validation_score": score,
            "status": "success",
            "status_code": 200,
        },
    )
    node.experience_card = {
        "node_id": node.id,
        "method_family_auto": "lightgbm",
        "fitness": score,
        "status": "success",
        "is_buggy": False,
    }
    return node


def test_async_worker_count_auto_uses_visible_gpu_count():
    assert resolve_async_worker_count("auto", gpu_indices=[0, 1, 2, 3]) == 4
    assert resolve_async_worker_count("auto", gpu_indices=[]) == 1
    assert resolve_async_worker_count(8) == 8


def test_execution_mode_is_explicit_and_validated():
    assert resolve_execution_mode("generation") == "generation"
    assert resolve_execution_mode("async_steady_state") == "async_steady_state"

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        resolve_execution_mode("automatic")


def test_worker_specs_validate_assignment(monkeypatch):
    monkeypatch.setattr(
        "tts_search.airaevo_async_resources.detect_gpu_indices",
        lambda: [0, 1],
    )

    specs = build_worker_specs(
        worker_count=2,
        sandbox_urls=["http://router-a", "http://router-b"],
        assignment="round_robin",
    )

    assert specs == [
        WorkerSpec(worker_id=0, gpu_index=0, sandbox_url="http://router-a"),
        WorkerSpec(worker_id=1, gpu_index=1, sandbox_url="http://router-b"),
    ]
    with pytest.raises(ValueError, match="Unsupported sandbox assignment"):
        build_worker_specs(
            worker_count=1,
            sandbox_urls=["http://router-a"],
            assignment="least_loaded",
        )


def test_gpu_detection_treats_missing_nvidia_smi_as_cpu_only(monkeypatch):
    def missing_binary(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(
        "tts_search.airaevo_async_resources.subprocess.run",
        missing_binary,
    )

    assert detect_gpu_indices() == []


def test_async_sandbox_urls_and_round_robin_assignment():
    urls = resolve_async_sandbox_urls(
        "http://router-a:6591, http://router-b:6591",
        fallback_url="http://fallback:6591",
    )
    pool = RoundRobinSandboxPool(urls)

    assert urls == ["http://router-a:6591", "http://router-b:6591"]
    assert [pool.next_url() for _ in range(4)] == [
        "http://router-a:6591",
        "http://router-b:6591",
        "http://router-a:6591",
        "http://router-b:6591",
    ]


class EmptySolutionDatabase:
    num_islands = 1
    has_nodes = False


def _make_async_solver(*, execution_mode: str, workers: int = 1) -> Evolutionary:
    solver = Evolutionary.__new__(Evolutionary)
    solver.cfg = types.SimpleNamespace(
        execution_mode=execution_mode,
        async_workers=workers,
        async_sandbox_assignment="round_robin",
        num_generations=3,
        individuals_per_generation=2,
        num_generations_till_crossover=1,
        initial_temp=1.0,
        final_temp=0.0,
        time_limit_secs=3600,
        step_limit=100,
    )
    solver.state = types.SimpleNamespace(current_generation=0, current_step=0)
    solver.start_time = 0.0
    return solver


def test_async_attempts_map_to_virtual_generations_and_budget(monkeypatch):
    solver = _make_async_solver(execution_mode="async_steady_state")
    worker = WorkerSpec(worker_id=0, gpu_index=0, sandbox_url="http://router")
    monkeypatch.setattr("dojo.solvers.evo.evo.time.monotonic", lambda: 1.0)

    items = [
        solver._next_async_work_item(
            solution_database=EmptySolutionDatabase(),
            worker=worker,
        )
        for _ in range(6)
    ]

    assert [item.attempt_id for item in items] == list(range(6))
    assert [item.generation_id for item in items] == [0, 0, 1, 1, 2, 2]
    assert solver._async_attempt_limit() == 6
    assert solver.linear_decay(99) == pytest.approx(0.0)
    assert solver._should_stop_async_search(task=None) is True


def test_async_time_budget_excludes_task_reported_scheduler_wait(monkeypatch):
    solver = _make_async_solver(execution_mode="async_steady_state")
    solver.cfg.time_limit_secs = 100
    solver.cfg.max_wall_time_secs = 200
    solver.start_time = 0.0
    task = types.SimpleNamespace(
        stop_requested=False,
        get_budget_exempt_wait_seconds=lambda: 10.0,
    )
    monkeypatch.setattr("dojo.solvers.evo.evo.time.monotonic", lambda: 105.0)

    assert solver._should_stop_async_search(task) is False

    solver.cfg.max_wall_time_secs = 104
    assert solver._should_stop_async_search(task) is True


def test_solution_database_restore_is_shared_by_sync_and_async_paths():
    solver = _make_async_solver(execution_mode="generation")
    database = object()
    calls = []
    solver._new_solution_database = lambda: database
    solver._seed_solution_database_from_journal = lambda value: calls.append(value)

    restored = solver._restore_solution_database_from_journal()

    assert restored is database
    assert calls == [database]
    assert solver.solution_database is database


def test_async_sampling_honors_fresh_draft_probability():
    solver = _make_async_solver(execution_mode="async_steady_state")
    solver.cfg.crossover_prob = 0.0
    solver.cfg.few_shot = {"improve": 1, "crossover": 2}
    solver.cfg.fresh_draft_prob = 1.0
    database = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=False,
        logger=Logger(),
    )
    database.seed_islands_with_nodes([make_node(0.5)], [0])

    item = solver._next_async_work_item(
        solution_database=database,
        worker=WorkerSpec(
            worker_id=0,
            gpu_index=None,
            sandbox_url="http://router",
        ),
    )

    assert item.operator == "draft"
    assert item.parent_nodes == []
    assert item.parent_selection_trace["reason"] == "fresh_draft_probability"


def test_async_worker_retries_same_attempt_after_transient_failure(monkeypatch):
    solver = _make_async_solver(execution_mode="async_steady_state")
    solver.start_time = time.monotonic()
    solver.cfg.async_worker_max_retries = 2
    solver.cfg.async_worker_retry_backoff_secs = 0.01
    solver.logger = Logger()
    database = EmptySolutionDatabase()
    worker = WorkerSpec(
        worker_id=0,
        gpu_index=None,
        sandbox_url="http://router",
    )
    stop_event = asyncio.Event()
    allocated_attempts = []
    create_calls = []
    original_next = solver._next_async_work_item

    def tracked_next(**kwargs):
        item = original_next(**kwargs)
        allocated_attempts.append(item.attempt_id)
        return item

    def flaky_create(work_item):
        create_calls.append(work_item.attempt_id)
        if len(create_calls) == 1:
            raise RuntimeError("temporary upstream failure")
        return make_node(0.5)

    async def fake_step_task_async(task, state, code, sandbox_url):
        return state, {}

    async def fake_commit_async_node(**kwargs):
        stop_event.set()
        return True

    async def no_sleep(delay):
        return None

    solver._next_async_work_item = tracked_next
    solver._create_node_from_work_item = flaky_create
    solver._step_task_async = fake_step_task_async
    solver.parse_eval_result = lambda node, eval_result: None
    solver._commit_async_node = fake_commit_async_node
    monkeypatch.setattr("dojo.solvers.evo.evo.asyncio.sleep", no_sleep)

    asyncio.run(
        solver._async_worker_loop(
            worker=worker,
            state={},
            task=types.SimpleNamespace(stop_requested=False),
            solution_database=database,
            sample_lock=asyncio.Lock(),
            commit_lock=asyncio.Lock(),
            stop_event=stop_event,
            commit_count=[0],
        )
    )

    assert allocated_attempts == [0]
    assert create_calls == [0, 0]


def test_async_node_keeps_its_work_item_metadata():
    solver = _make_async_solver(execution_mode="async_steady_state")
    solver._draft = lambda: make_node(0.5)
    work_item = AsyncWorkItem(
        attempt_id=3,
        generation_id=1,
        temperature=0.5,
        island_id=0,
        operator="draft",
        parent_nodes=[],
        parent_selection_trace=None,
        worker=WorkerSpec(
            worker_id=2,
            gpu_index=4,
            sandbox_url="http://router",
        ),
    )

    node = solver._create_node_from_work_item(work_item)

    assert node.async_work_metadata == {
        "execution_mode": "async_steady_state",
        "attempt_id": 3,
        "generation_id": 1,
        "temperature": 0.5,
        "worker_id": 2,
        "gpu_index": 4,
        "sandbox_url": "http://router",
    }


def test_async_mode_with_one_worker_does_not_fall_back_to_generation():
    solver = _make_async_solver(
        execution_mode="async_steady_state",
        workers=1,
    )

    async def fake_async_search(task, state, *, worker_count):
        assert worker_count == 1
        return state, "async-result"

    solver.search_async_steady_state = fake_async_search

    assert solver.search(task=None, state={"value": 1}) == (
        {"value": 1},
        "async-result",
    )


def test_generation_mode_rejects_multiple_workers():
    solver = _make_async_solver(execution_mode="generation", workers=2)

    with pytest.raises(ValueError, match="generation mode requires exactly one worker"):
        solver.search(task=None, state={})


def test_task_concurrency_default_accounts_for_async_workers():
    assert (
        resolve_task_concurrency_per_epoch(
            configured_task_concurrency=None,
            llm_concurrency=66,
            sandbox_concurrency=66,
            num_epochs=3,
            async_workers=4,
        )
        == 6
    )

    assert (
        resolve_task_concurrency_per_epoch(
            configured_task_concurrency=11,
            llm_concurrency=66,
            sandbox_concurrency=66,
            num_epochs=3,
            async_workers=4,
        )
        == 11
    )


def test_async_sandbox_slot_wait_does_not_block_event_loop(monkeypatch):
    semaphore = threading.Semaphore(1)
    monkeypatch.setattr(
        airaevo_concurrency,
        "_get_sandbox_semaphore_proxy",
        lambda: semaphore,
    )

    async def holder(release: asyncio.Event):
        async with acquire_sandbox_slot_async():
            await release.wait()

    async def waiter(started: asyncio.Event):
        started.set()
        async with acquire_sandbox_slot_async():
            return "acquired"

    async def probe():
        release = asyncio.Event()
        started = asyncio.Event()
        first = asyncio.create_task(holder(release))
        await asyncio.sleep(0)
        second = asyncio.create_task(waiter(started))
        await started.wait()

        ticker = asyncio.create_task(asyncio.sleep(0.05))
        await asyncio.wait_for(ticker, timeout=0.5)
        assert not second.done()

        release.set()
        assert await asyncio.wait_for(second, timeout=1.0) == "acquired"
        await first

    asyncio.run(probe())


def test_solution_database_returns_parent_selection_trace_for_async_sampling():
    nodes = [make_node(0.9), make_node(0.7)]
    database = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=False,
        logger=Logger(),
        experience_config={
            "enabled": True,
            "parent_selection": {
                "enabled": True,
                "weights": {"score": 1.0, "delta": 0.0, "novelty": 0.0},
            },
        },
    )
    database.seed_islands_with_nodes(nodes, [0, 0])

    selected_nodes, island_id, operator, trace = database.sample_in_context_with_trace(
        {"improve": 1},
        temperature=1.0,
        crossover_prob=0.0,
    )

    assert island_id == 0
    assert operator == "improve"
    assert len(selected_nodes) == 1
    assert trace is database.last_parent_selection
    assert trace["enabled"] is True
    assert trace["selected_node_ids"] == [selected_nodes[0].id]


def test_sandbox_task_async_eval_uses_router_url_override(monkeypatch):
    task = SandboxMLEBenchTask(
        {
            "data_dir": "/tmp/data/valid",
            "submit_dir": "submit",
            "higher_is_better": True,
            "task_description": "Predict the target.",
            "data_description": "Small data.",
            "sandbox": {
                "resource": "gpu",
                "base_url": "http://old-sandbox:6580",
                "job_timeout": 60,
                "wait_timeout": 60,
                "poll_interval": 1,
                "use_score2reward": False,
                "use_clear_run_log_score": True,
                "trust_model_validation_score": True,
            },
        }
    )

    calls = []

    async def fake_run_eval_async(
        code,
        *,
        data_dir,
        sandbox_base_url=None,
        eval_split=None,
        job_timeout=None,
        wait_timeout=None,
    ):
        calls.append(
            {
                "sandbox_base_url": sandbox_base_url,
                "eval_split": eval_split,
                "job_timeout": job_timeout,
                "wait_timeout": wait_timeout,
            }
        )
        return {
            "status_code": 200,
            "status": "success",
            "score": 0.5,
            "raw_score": 0.5,
            "reward": 0.5,
            "job_id": "job-1",
            "queue_time": 0.0,
            "run_time": 1.25,
            "raw_run_log": "Final Validation Score: 0.75",
            "clear_run_log": "Final Validation Score: 0.75",
            "feedback": "ok",
            "data_dir": data_dir,
            "sandbox_base_url": sandbox_base_url,
        }

    monkeypatch.setattr(task, "_run_eval_async", fake_run_eval_async)

    async def run_step():
        return await task.step_task_async(
            {},
            "print('ok')",
            sandbox_base_url="http://router.test",
        )

    _, eval_result = asyncio.run(run_step())

    assert eval_result[VALIDATION_FITNESS] == pytest.approx(0.75)
    assert (
        eval_result[AUX_EVAL_INFO]["sandbox_base_url"] == "http://router.test"
    )
    assert calls == [
        {
            "sandbox_base_url": "http://router.test",
            "eval_split": None,
            "job_timeout": 60,
            "wait_timeout": 60,
        }
    ]
    assert task.validation_time_used == pytest.approx(1.25)
