from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openmle_gym.build import _run_all
from openmle_gym.common import atomic_write_json as real_atomic_write_json
from openmle_gym.process_runner import TaskProcessOutcome


HAS_BATCH_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("pandas", "tqdm")
)


def _make_task(root: Path, name: str, metric_code: str) -> Path:
    task = root / name
    public = task / "data" / "public"
    private = task / "data" / "private"
    utils = task / "utils"
    raw = task / "raw"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    raw.mkdir()
    (public / "description.txt").write_text("description", encoding="utf-8")
    (public / "train.csv").write_text("feature,target\n1,1\n", encoding="utf-8")
    (public / "test.csv").write_text("feature\n1\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
    (private / "test_answer.csv").write_text("target\n1\n", encoding="utf-8")
    (utils / "prepare.py").write_text("def prepare(*args):\n    pass\n", encoding="utf-8")
    (utils / "metric.py").write_text(metric_code, encoding="utf-8")
    return task


GOOD_METRIC = """
class GoodMetrics:
    def evaluate(self, y_true=None, y_pred=None):
        return 1.0
"""


@unittest.skipUnless(HAS_BATCH_DEPS, "batch dependencies are required")
class BatchIndependenceTests(unittest.TestCase):
    def test_build_collects_failure_without_cancelling_siblings(self) -> None:
        def fake_runner(operation, payload, **kwargs):
            slug = payload["slug"]
            if slug == "bad":
                return TaskProcessOutcome(ok=False, error="crashed", returncode=9)
            return TaskProcessOutcome(
                ok=True,
                result={"name": slug, "success": True, "errors": []},
                returncode=0,
            )

        with patch("openmle_gym.build.run_task_process", side_effect=fake_runner):
            results = asyncio.run(
                _run_all(
                    builder_root=Path("builder_core"),
                    batch_name="batch",
                    slugs=["good-a", "bad", "good-b"],
                    retry=0,
                    output_root=Path("output"),
                    info_csv=Path("info.csv"),
                    max_concurrency=3,
                    delete_raw=False,
                    execution_mode="process",
                    task_timeout=10,
                )
            )
        self.assertEqual([result["name"] for result in results], ["good-a", "bad", "good-b"])
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertTrue(results[2]["success"])

    def test_build_isolated_mode_keeps_outer_task_process(self) -> None:
        calls = []

        def fake_runner(operation, payload, **kwargs):
            calls.append((operation, payload, kwargs))
            return TaskProcessOutcome(
                ok=True,
                result={"name": payload["slug"], "success": True, "errors": []},
            )

        with patch("openmle_gym.build.run_task_process", side_effect=fake_runner):
            asyncio.run(
                _run_all(
                    builder_root=Path("builder_core"),
                    batch_name="batch",
                    slugs=["task"],
                    retry=0,
                    output_root=Path("output"),
                    info_csv=Path("info.csv"),
                    max_concurrency=1,
                    delete_raw=False,
                    execution_mode="isolated",
                    task_timeout=10,
                )
            )
        self.assertEqual(calls[0][2]["execution_mode"], "process")
        self.assertEqual(calls[0][1]["code_execution_mode"], "isolated")

    def test_overview_crash_keeps_sibling_rows(self) -> None:
        from openmle_gym.overview import generate_overview

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            _make_task(tasks, "good", GOOD_METRIC)
            _make_task(tasks, "crashed", "import os\nos._exit(7)\n")
            output = root / "overview.csv"
            generate_overview(tasks, output, workers=2, skip_llm=True, task_timeout=10)
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["Name"] for row in rows], ["crashed", "good"])
            self.assertTrue(rows[0]["NOTE"].startswith("#TASK_RUNTIME_ERROR:"))
            self.assertEqual(rows[1]["NOTE"], "")

    def test_overview_rejects_task_symlink_outside_batch_root(self) -> None:
        from openmle_gym.overview import generate_overview

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            outside = root / "outside"
            _make_task(root, "outside", GOOD_METRIC)
            (tasks / "escaped").symlink_to(outside, target_is_directory=True)
            output = root / "overview.csv"
            generate_overview(tasks, output, workers=1, skip_llm=True)
            with output.open(encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["Name"], "escaped")
            self.assertTrue(row["NOTE"].startswith("#TASK_RUNTIME_ERROR:"))

    def test_evaluate_crash_keeps_sibling_results(self) -> None:
        from openmle_gym.local_evaluator import evaluate_tasks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            _make_task(tasks, "good", GOOD_METRIC)
            _make_task(tasks, "crashed", "import os\nos._exit(7)\n")
            task_list = root / "tasks.txt"
            task_list.write_text("crashed\ngood\n", encoding="utf-8")
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["Name"])
                writer.writeheader()
                writer.writerow({"Name": "crashed"})
                writer.writerow({"Name": "good"})
            paths = evaluate_tasks(
                tasks,
                task_list,
                root / "evaluation",
                local_only=True,
                overview_csv=overview,
                workers=2,
                task_timeout=10,
            )
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            self.assertEqual([result["task_name"] for result in results], ["crashed", "good"])
            self.assertTrue(results[0]["task_process_failed"])
            self.assertNotIn("task_process_failed", results[1])
            self.assertEqual(results[1]["quality_scores"], {})
            self.assertEqual(
                results[1]["quality_evaluation"]["status"],
                "skipped",
            )
            with paths["summary"].open(encoding="utf-8", newline="") as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(summary_rows[1]["Overall Score"], "")
            self.assertEqual(summary_rows[1]["Recommendation"], "")
            self.assertEqual(summary_rows[1]["Validation Status"], "passed")
            self.assertEqual(summary_rows[1]["Evaluation Status"], "skipped")

    def test_evaluate_item_write_failure_keeps_sibling_and_aggregates(self) -> None:
        from openmle_gym.local_evaluator import evaluate_tasks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            _make_task(tasks, "unwritable", GOOD_METRIC)
            _make_task(tasks, "good", GOOD_METRIC)
            task_list = root / "tasks.txt"
            task_list.write_text("unwritable\ngood\n", encoding="utf-8")
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["Name"])
                writer.writeheader()
                writer.writerow({"Name": "unwritable"})
                writer.writerow({"Name": "good"})

            def flaky_write(path, value):
                if Path(path).name == "unwritable.json":
                    raise OSError("simulated item write failure")
                return real_atomic_write_json(path, value)

            with patch(
                "openmle_gym.local_evaluator.atomic_write_json",
                side_effect=flaky_write,
            ):
                paths = evaluate_tasks(
                    tasks,
                    task_list,
                    root / "evaluation",
                    local_only=True,
                    overview_csv=overview,
                    workers=2,
                    task_timeout=10,
                )
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            self.assertTrue(results[0]["task_process_failed"])
            self.assertIn("Failed to write task result", results[0]["errors"][0])
            self.assertNotIn("task_process_failed", results[1])
            self.assertTrue((root / "evaluation" / "good.json").is_file())

    def test_evaluate_duplicate_task_is_an_item_failure(self) -> None:
        from openmle_gym.local_evaluator import evaluate_tasks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            _make_task(tasks, "same", GOOD_METRIC)
            task_list = root / "tasks.txt"
            task_list.write_text("same\nsame\n", encoding="utf-8")
            overview = root / "overview.csv"
            with overview.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["Name"])
                writer.writeheader()
                writer.writerow({"Name": "same"})
            paths = evaluate_tasks(
                tasks,
                task_list,
                root / "evaluation",
                local_only=True,
                overview_csv=overview,
                workers=2,
            )
            results = json.loads(paths["combined"].read_text(encoding="utf-8"))
            self.assertNotIn("task_process_failed", results[0])
            self.assertTrue(results[1]["task_process_failed"])
            self.assertIn("Duplicate task name", results[1]["errors"][0])


if __name__ == "__main__":
    unittest.main()
