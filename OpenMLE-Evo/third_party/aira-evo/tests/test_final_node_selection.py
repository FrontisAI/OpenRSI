import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules["dotenv"] = dotenv_stub

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mle_bench"
    / "single_task_runner.py"
)
spec = importlib.util.spec_from_file_location("single_task_runner_under_test", RUNNER_PATH)
single_task_runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(single_task_runner)


class ComparableMetric:
    def __init__(self, value, info):
        self.value = value
        self.info = info

    def __lt__(self, other):
        self_value = self.value if self.value is not None else float("-inf")
        other_value = other.value if other.value is not None else float("-inf")
        return self_value < other_value


def make_node(
    *,
    node_id: str,
    metric: float | None,
    official_score: float | None,
    status: str,
    is_buggy: bool = False,
    validation_score: float | None = None,
):
    validation_score = metric if validation_score is None else validation_score
    return SimpleNamespace(
        id=node_id,
        metric=ComparableMetric(
            metric,
            {
                "score": official_score,
                "status": status,
                "validation_score": validation_score,
            },
        ),
        is_buggy=is_buggy,
        code=f"# {node_id}",
    )


def make_solver(nodes, *, lower_is_better: bool = False):
    class Journal:
        def __init__(self, nodes):
            self.nodes = nodes

        def get_best_node(self):
            return max(
                [node for node in self.nodes if not node.is_buggy],
                key=lambda node: node.metric.value if node.metric.value is not None else float("-inf"),
                default=None,
            )

        def is_root_node(self, _node):
            return False

    return SimpleNamespace(journal=Journal(nodes), lower_is_better=lower_is_better)


def test_final_selection_uses_journal_best_metric_not_official_score():
    higher_official_lower_metric_node = make_node(
        node_id="higher_official_lower_metric",
        metric=0.70,
        official_score=0.95,
        status="success",
    )
    lower_official_higher_metric_node = make_node(
        node_id="lower_official_higher_metric",
        metric=0.90,
        official_score=0.70,
        status="success",
    )
    solver = make_solver(
        [higher_official_lower_metric_node, lower_official_higher_metric_node],
        lower_is_better=False,
    )

    selected = single_task_runner._select_best_available_node(
        solver,
        seed=0,
        task_name="unit-test",
        sample_index=0,
    )

    assert selected is lower_official_higher_metric_node


def test_final_selection_falls_back_to_latest_non_root_when_journal_has_no_best_node():
    node = make_node(
        node_id="latest",
        metric=None,
        official_score=None,
        status="timeout",
        is_buggy=True,
        validation_score=None,
    )
    solver = make_solver([node], lower_is_better=False)

    selected = single_task_runner._select_best_available_node(
        solver,
        seed=0,
        task_name="unit-test",
        sample_index=0,
    )

    assert selected is node
