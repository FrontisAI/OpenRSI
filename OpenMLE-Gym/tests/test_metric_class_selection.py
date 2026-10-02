from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openmle_gym.metric_validation import evaluate_sample_submission, load_metric_class

BASE_AND_CONCRETE = (
    "class AbstractMetrics:\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        raise NotImplementedError\n"
    "\n"
    "class TitanicMetrics(AbstractMetrics):\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        return 0.5\n"
)

CONCRETE_FIRST_ALPHABETICALLY = (
    "class AAMetrics:\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        return 0.25\n"
    "\n"
    "class ZMetrics(AAMetrics):\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        return 0.75\n"
)

TWO_INDEPENDENT = (
    "class AlphaMetrics:\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        return 0.1\n"
    "\n"
    "class BetaMetrics:\n"
    "    def evaluate(self, y_true=None, y_pred=None):\n"
    "        return 0.2\n"
)


def _task_with_metric(directory: str, source: str) -> Path:
    task = Path(directory) / "task"
    (task / "utils").mkdir(parents=True)
    (task / "utils" / "metric.py").write_text(source, encoding="utf-8")
    return task


def _task_with_data(directory: str, source: str) -> Path:
    task = Path(directory) / "task"
    for sub in ("data/public", "data/private", "utils"):
        (task / sub).mkdir(parents=True)
    (task / "data/public/sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
    (task / "data/private/test_answer.csv").write_text("target\n1\n", encoding="utf-8")
    (task / "utils/metric.py").write_text(source, encoding="utf-8")
    return task


class MetricClassSelectionTests(unittest.TestCase):
    def test_concrete_class_wins_over_abstract_base(self) -> None:
        """A task with an abstract base must not score through the base."""
        with tempfile.TemporaryDirectory() as directory:
            task = _task_with_metric(directory, BASE_AND_CONCRETE)
            self.assertEqual(load_metric_class(task).__name__, "TitanicMetrics")

    def test_most_derived_class_wins_even_when_sorts_first(self) -> None:
        """Name order must not decide: the subclass overrides evaluate()."""
        with tempfile.TemporaryDirectory() as directory:
            task = _task_with_metric(directory, CONCRETE_FIRST_ALPHABETICALLY)
            self.assertEqual(load_metric_class(task).__name__, "ZMetrics")

    def test_single_class_is_still_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _task_with_metric(directory, TWO_INDEPENDENT)
            self.assertIn(load_metric_class(task).__name__, {"AlphaMetrics", "BetaMetrics"})

    def test_imported_class_is_not_selected(self) -> None:
        """Existing guarantee: only classes defined in metric.py are eligible."""
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            utils = task / "utils"
            utils.mkdir(parents=True)
            (utils / "helper.py").write_text(
                "class AImportedMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return 1.0\n",
                encoding="utf-8",
            )
            (utils / "metric.py").write_text(
                "from helper import AImportedMetrics\n"
                "class ZLocalMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return 0.5\n",
                encoding="utf-8",
            )
            import sys

            sys.path.insert(0, str(utils))
            try:
                metric_class = load_metric_class(task)
            finally:
                sys.path.remove(str(utils))
                sys.modules.pop("helper", None)
            self.assertEqual(metric_class.__name__, "ZLocalMetrics")

    def test_evaluate_sample_submission_uses_concrete_class(self) -> None:
        """End to end: the abstract base raises, so this must not raise."""
        with tempfile.TemporaryDirectory() as directory:
            task = _task_with_data(directory, BASE_AND_CONCRETE)
            result = evaluate_sample_submission(task)
            self.assertEqual(result["score"], 0.5)

    def test_no_metrics_class_still_raises_lookup_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _task_with_metric(directory, "class NotAMetric:\n    pass\n")
            with self.assertRaises(LookupError):
                load_metric_class(task)


if __name__ == "__main__":
    unittest.main()