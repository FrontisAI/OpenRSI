from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from openmle_gym.metric_validation import evaluate_sample_submission, load_metric_class


class MetricValidationTests(unittest.TestCase):
    def test_imported_metrics_class_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            utils = task / "utils"
            utils.mkdir(parents=True)
            (utils / "helper.py").write_text(
                "class AImportedMetrics:\n    pass\n",
                encoding="utf-8",
            )
            (utils / "metric.py").write_text(
                "from helper import AImportedMetrics\n"
                "class ZLocalMetrics:\n"
                "    pass\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(utils))
            try:
                metric_class = load_metric_class(task)
            finally:
                sys.path.remove(str(utils))
                sys.modules.pop("helper", None)
            self.assertEqual(metric_class.__name__, "ZLocalMetrics")

    def test_numpy_scalar_score_is_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            public = task / "data" / "public"
            private = task / "data" / "private"
            utils = task / "utils"
            public.mkdir(parents=True)
            private.mkdir(parents=True)
            utils.mkdir(parents=True)
            (public / "sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
            (private / "test_answer.csv").write_text("target\n1\n", encoding="utf-8")
            (utils / "metric.py").write_text(
                "import numpy as np\n"
                "class ScalarMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return np.float32(0.5)\n",
                encoding="utf-8",
            )
            result = evaluate_sample_submission(task)
            self.assertIsInstance(result["score"], float)
            self.assertEqual(result["score"], 0.5)

    def test_non_finite_metric_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task"
            public = task / "data" / "public"
            private = task / "data" / "private"
            utils = task / "utils"
            public.mkdir(parents=True)
            private.mkdir(parents=True)
            utils.mkdir(parents=True)
            (public / "sample_submission.csv").write_text(
                "target\n1\n",
                encoding="utf-8",
            )
            (private / "test_answer.csv").write_text(
                "target\n1\n",
                encoding="utf-8",
            )
            (utils / "metric.py").write_text(
                "class InvalidMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return float('nan')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite numeric score"):
                evaluate_sample_submission(task)


if __name__ == "__main__":
    unittest.main()
