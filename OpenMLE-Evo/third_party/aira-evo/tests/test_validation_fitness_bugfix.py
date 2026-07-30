import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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

from dojo.core.interpreters.base import ExecutionResult
from dojo.core.solvers.utils.journal import Journal
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.metric import MetricValue
from dojo.core.solvers.utils.metric import WorstMetricValue
from dojo.core.tasks.constants import AUX_EVAL_INFO
from dojo.core.tasks.constants import EXECUTION_OUTPUT
from dojo.core.tasks.constants import TEST_FITNESS
from dojo.core.tasks.constants import VALIDATION_FITNESS
from dojo.solvers.evo.evo import Evolutionary
from dojo.solvers.evo.evo import SolutionsDatabase
from dojo.solvers.greedy.greedy import Greedy
from dojo.solvers.mcts.mcts import MCTS

EXAMPLES_MLE_BENCH_DIR = Path(__file__).resolve().parents[1] / "examples" / "mle_bench"
if str(EXAMPLES_MLE_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MLE_BENCH_DIR))

from base_task import SandboxMLEBenchTask


def build_solver(solver_cls):
    solver = solver_cls.__new__(solver_cls)
    solver.logger = logging.getLogger(solver_cls.__name__)
    solver.cfg = SimpleNamespace(use_test_score=False)
    solver.lower_is_better = False
    solver.analyze_calls = 0

    def analyze(node):
        solver.analyze_calls += 1
        return {}

    solver._analyze = analyze
    return solver


@pytest.mark.parametrize("solver_cls", [Evolutionary, Greedy, MCTS])
def test_validation_fitness_marks_node_as_non_buggy(solver_cls):
    solver = build_solver(solver_cls)
    node = Node(code="print('hello')")
    validation_score = 0.9123
    eval_result = {
        EXECUTION_OUTPUT: ExecutionResult(term_out=["ok"], exec_time=1.0, exit_code=0),
        VALIDATION_FITNESS: validation_score,
        AUX_EVAL_INFO: {"score": validation_score, "status": "success"},
    }

    solver.parse_eval_result(node, eval_result)

    assert node.is_buggy is False
    assert node.metric.value == pytest.approx(validation_score)
    assert node.metric.info["score"] == pytest.approx(validation_score)
    if solver_cls is Evolutionary:
        assert solver.analyze_calls == 1


def test_evolutionary_experience_mode_skips_analyze_for_validation_fitness():
    solver = build_solver(Evolutionary)
    solver.cfg.experience = {"enabled": True}
    node = Node(code="print('hello')")
    validation_score = 0.9123
    eval_result = {
        EXECUTION_OUTPUT: ExecutionResult(term_out=["ok"], exec_time=1.0, exit_code=0),
        VALIDATION_FITNESS: validation_score,
        AUX_EVAL_INFO: {"score": validation_score, "status": "success"},
    }

    solver.parse_eval_result(node, eval_result)

    assert node.is_buggy is False
    assert node.metric.value == pytest.approx(validation_score)
    assert solver.analyze_calls == 0


def test_evolutionary_validation_fitness_sanitizes_analysis_summary():
    solver = build_solver(Evolutionary)
    solver.cfg.experience = {
        "enabled": True,
        "prompt_score_sanitization": {"enabled": True},
    }
    node = Node(code="print('hello')")
    validation_score = 0.9123
    eval_result = {
        EXECUTION_OUTPUT: ExecutionResult(term_out=["ok"], exec_time=1.0, exit_code=0),
        VALIDATION_FITNESS: validation_score,
        AUX_EVAL_INFO: {
            "score": 0.4567,
            "status": "success",
            "feedback": (
                "Final Validation Score: 0.9123\n"
                "Final Score: 0.4567\n"
                "prefix ##SCORE##0.4567 suffix\n"
                "submission.csv Grader Feedback: ## Execution Result\n"
                "**Score**: 0.4567\n"
            ),
        },
    }

    solver.parse_eval_result(node, eval_result)

    assert node.is_buggy is False
    assert node.metric.value == pytest.approx(validation_score)
    assert "Final Validation Score: 0.9123" in node.analysis
    assert "Official sandbox score redacted" in node.analysis
    assert "Final Score: 0.4567" not in node.analysis
    assert "##SCORE##0.4567" not in node.analysis
    assert "**Score**: 0.4567" not in node.analysis


def build_sandbox_task(
    *,
    use_clear_run_log_score=True,
    trust_model_validation_score=False,
):
    return SandboxMLEBenchTask(
        {
            "data_dir": "/tmp/airaevo-valid/task",
            "submit_dir": "submit",
            "higher_is_better": True,
            "task_description": "",
            "data_description": "",
            "sandbox": {
                "resource": "gpu",
                "base_url": "http://sandbox.invalid",
                "job_timeout": 7200,
                "wait_timeout": 7200,
                "poll_interval": 5,
                "use_score2reward": False,
                "use_clear_run_log_score": use_clear_run_log_score,
                "trust_model_validation_score": trust_model_validation_score,
            },
        }
    )


def test_sandbox_task_does_not_trust_model_reported_validation_score_by_default():
    task = build_sandbox_task(use_clear_run_log_score=True)
    payload = {
        "status_code": 200,
        "score": 0.5,
        "raw_score": 0.5,
        "clear_run_log": "Final Validation Score: 0.12345\n##SCORE##0.5",
    }

    task._annotate_eval_scores(payload, phase="validation")

    assert payload["score"] == pytest.approx(0.5)
    assert payload["raw_score"] == pytest.approx(0.5)
    assert payload["raw_scores"]["model_final_validation_score"] == pytest.approx(
        0.12345
    )
    assert payload["validation_score"] == pytest.approx(0.5)
    assert payload["selection_score"] == pytest.approx(0.5)
    assert payload["selection_score_source"] == "sandbox_score"


def test_sandbox_task_can_explicitly_enable_model_reported_validation_score():
    task = build_sandbox_task(
        use_clear_run_log_score=True,
        trust_model_validation_score=True,
    )
    payload = {
        "status_code": 200,
        "score": 0.5,
        "raw_score": 0.5,
        "clear_run_log": "Final Validation Score: 0.12345\n##SCORE##0.5",
    }

    task._annotate_eval_scores(payload, phase="validation")

    assert payload["validation_score"] == pytest.approx(0.12345)
    assert payload["selection_score"] == pytest.approx(0.12345)
    assert payload["selection_score_source"] == "model_final_validation_score"


def test_sandbox_http_client_verifies_tls_by_default(monkeypatch):
    task = build_sandbox_task()
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_get_sandbox_result(**kwargs):
        return 500, {"error": "unit-test"}

    monkeypatch.setenv("SANDBOX_GPU_API_KEY", "unit-key")
    monkeypatch.setattr("base_task.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "base_task.get_sandbox_result",
        fake_get_sandbox_result,
    )

    import asyncio

    asyncio.run(
        task._run_eval_async(
            "print('ok')",
            data_dir="/tmp/airaevo-valid/task",
        )
    )

    assert captured["verify"] is True


def test_sandbox_step_task_uses_self_validation_for_validation_fitness():
    task = build_sandbox_task(use_clear_run_log_score=True)
    task.evaluate_code = lambda code, *, phase: {
        "status_code": 200,
        "status": "success",
        "score": 0.5,
        "raw_score": 0.5,
        "reward": 0.5,
        "validation_score": 0.12345,
        "validation_reward": 0.12345,
        "selection_score_source": "clear_run_log",
        "feedback": "ok",
    }

    _, eval_result = task.step_task({}, "print('ok')")

    assert eval_result[VALIDATION_FITNESS] == pytest.approx(0.12345)
    assert eval_result[AUX_EVAL_INFO]["score"] == pytest.approx(0.5)
    assert eval_result[AUX_EVAL_INFO]["raw_score"] == pytest.approx(0.5)


def test_sandbox_evaluate_fitness_reports_test_fitness():
    task = build_sandbox_task(use_clear_run_log_score=True)
    task.evaluate_code = lambda code, *, phase: {
        "status_code": 200,
        "status": "success",
        "score": 0.5,
        "raw_score": 0.5,
        "reward": 0.5,
        "feedback": "ok",
    }

    eval_result = task.evaluate_fitness("print('ok')")

    assert eval_result[TEST_FITNESS] == pytest.approx(0.5)
    assert VALIDATION_FITNESS not in eval_result


def test_evo_debug_cycle_logs_each_node_before_next_debug_attempt():
    solver = Evolutionary.__new__(Evolutionary)
    solver.cfg = SimpleNamespace(
        max_debug_depth=2,
        max_debug_time=999,
        experience={"enabled": True},
    )
    solver.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    solver.journal = Journal()
    solver.state = SimpleNamespace(current_step=0)
    logged_node_ids = []

    def log_journal():
        node = solver.journal.nodes[-1]
        node.experience_card = {"node_id": node.id}
        logged_node_ids.append(node.id)

    solver.log_journal = log_journal

    buggy_node = Node(code="raise RuntimeError('bug')", operators_used=["draft"])
    buggy_node.absorb_exec_result(
        ExecutionResult(term_out=["bug"], exec_time=1.0, exit_code=1)
    )
    buggy_node.is_buggy = True
    buggy_node.metric = WorstMetricValue(info={"status": "failed"})

    debug_attempts = []

    def debug(parent):
        assert parent in solver.journal.nodes
        assert getattr(parent, "experience_card", None) == {"node_id": parent.id}
        node = Node(
            code=f"print('debug {len(debug_attempts)}')",
            parents=[parent],
            operators_used=["debug"],
        )
        debug_attempts.append(node)
        return node

    def parse_eval_result(node, eval_result):
        node.absorb_exec_result(
            ExecutionResult(term_out=["still buggy"], exec_time=1.0, exit_code=1)
        )
        node.is_buggy = True
        node.metric = WorstMetricValue(info={"status": "failed"})
        node.analysis = "still buggy"

    class Task:
        def step_task(self, state, code):
            return state, {"dummy": True}

    solver._debug = debug
    solver.parse_eval_result = parse_eval_result

    _, debug_path, fixed_metric = solver.debug_cycle({}, Task(), buggy_node)

    assert fixed_metric is None
    assert debug_path == [buggy_node, *debug_attempts]
    assert [node.id for node in solver.journal.nodes] == [
        node.id for node in debug_path
    ]
    assert logged_node_ids == [node.id for node in debug_path]
    assert solver.state.current_step == len(debug_path)


def test_evo_debug_cycle_defers_journal_logging_without_experience_mode():
    solver = Evolutionary.__new__(Evolutionary)
    solver.cfg = SimpleNamespace(
        max_debug_depth=2,
        max_debug_time=999,
        experience={"enabled": False},
    )
    solver.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    solver.journal = Journal()
    solver.state = SimpleNamespace(current_step=0)
    logged_node_ids = []

    def log_journal():
        logged_node_ids.append(solver.journal.nodes[-1].id)

    solver.log_journal = log_journal

    buggy_node = Node(code="raise RuntimeError('bug')", operators_used=["draft"])
    buggy_node.absorb_exec_result(
        ExecutionResult(term_out=["bug"], exec_time=1.0, exit_code=1)
    )
    buggy_node.is_buggy = True
    buggy_node.metric = WorstMetricValue(info={"status": "failed"})

    debug_attempts = []

    def debug(parent):
        assert parent not in solver.journal.nodes
        node = Node(
            code=f"print('debug {len(debug_attempts)}')",
            parents=[parent],
            operators_used=["debug"],
        )
        debug_attempts.append(node)
        return node

    def parse_eval_result(node, eval_result):
        node.absorb_exec_result(
            ExecutionResult(term_out=["still buggy"], exec_time=1.0, exit_code=1)
        )
        node.is_buggy = True
        node.metric = WorstMetricValue(info={"status": "failed"})
        node.analysis = "still buggy"

    class Task:
        def step_task(self, state, code):
            return state, {"dummy": True}

    solver._debug = debug
    solver.parse_eval_result = parse_eval_result

    _, debug_path, fixed_metric = solver.debug_cycle({}, Task(), buggy_node)

    assert fixed_metric is None
    assert debug_path == [buggy_node, *debug_attempts]
    assert solver.journal.nodes == []
    assert logged_node_ids == []
    assert solver.state.current_step == 0


def test_evo_parent_selection_switches_between_original_and_experience_modes():
    class Logger:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    def make_search_node(score: float, official_score: float) -> Node:
        node = Node(code="print('ok')")
        node.metric = MetricValue(
            score,
            maximize=True,
            info={
                "score": official_score,
                "raw_score": official_score,
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

    nodes = [make_search_node(0.9, 0.1), make_search_node(0.7, 0.99)]

    original_db = SolutionsDatabase(
        num_islands=1,
        max_size=10,
        lower_is_better=False,
        logger=Logger(),
        experience_config={"enabled": False},
    )
    original_db.seed_islands_with_nodes(nodes, [0, 0])
    original_db.sample_in_context({"improve": 1}, temperature=1.0, crossover_prob=0.0)
    assert original_db.last_parent_selection["enabled"] is False
    assert original_db.last_parent_selection["candidates"] == []

    experience_db = SolutionsDatabase(
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
    experience_db.seed_islands_with_nodes(nodes, [0, 0])
    experience_db.sample_in_context(
        {"improve": 1},
        temperature=1.0,
        crossover_prob=0.0,
    )
    trace = experience_db.last_parent_selection

    assert trace["enabled"] is True
    assert len(trace["candidates"]) == 2
    assert all(
        candidate["score_source"] == "self_validation"
        for candidate in trace["candidates"]
    )
    assert all(
        candidate["official_score_used"] is False
        for candidate in trace["candidates"]
    )
