from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from openmle_gym.process_runner import (
    _container_command,
    _load_result_envelope,
    run_task_process,
)


HAS_PANDAS = importlib.util.find_spec("pandas") is not None


def _make_task(root: Path, name: str, metric_code: str) -> Path:
    task = root / name
    public = task / "data" / "public"
    private = task / "data" / "private"
    utils = task / "utils"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    (public / "sample_submission.csv").write_text("target\n1\n", encoding="utf-8")
    (private / "test_answer.csv").write_text("target\n1\n", encoding="utf-8")
    (utils / "metric.py").write_text(metric_code, encoding="utf-8")
    return task


GOOD_METRIC = """
class GoodMetrics:
    def validate_submission(self, submission, ground_truth):
        return None

    def evaluate(self, y_true=None, y_pred=None):
        return 1.0
"""


@unittest.skipUnless(HAS_PANDAS, "pandas is required for metric worker tests")
class ProcessIsolationTests(unittest.TestCase):
    def test_invalid_result_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text("{invalid", encoding="utf-8")
            with self.assertRaises(Exception):
                _load_result_envelope(result)

    def test_crashed_task_does_not_break_sibling_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good_a = _make_task(root, "good-a", GOOD_METRIC)
            crashed = _make_task(
                root,
                "crashed",
                "import os\nos._exit(9)\n",
            )
            good_b = _make_task(root, "good-b", GOOD_METRIC)

            def run(task: Path):
                return run_task_process(
                    "metric",
                    {"task_dir": str(task)},
                    timeout=10,
                    readonly_paths=(task,),
                )

            with ThreadPoolExecutor(max_workers=3) as executor:
                outcomes = list(executor.map(run, (good_a, crashed, good_b)))

            self.assertTrue(outcomes[0].ok)
            self.assertFalse(outcomes[1].ok)
            self.assertEqual(outcomes[1].returncode, 9)
            self.assertTrue(outcomes[2].ok)
            self.assertEqual(outcomes[0].result["score"], 1.0)
            self.assertEqual(outcomes[2].result["score"], 1.0)

    def test_same_dynamic_module_name_is_isolated_between_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_task(root / "first", "same-name", GOOD_METRIC)
            second = _make_task(
                root / "second",
                "same-name",
                "class OtherMetrics:\n"
                "    def evaluate(self, y_true=None, y_pred=None):\n"
                "        return 2.0\n",
            )

            def run(task: Path):
                return run_task_process(
                    "metric",
                    {"task_dir": str(task)},
                    timeout=10,
                    readonly_paths=(task,),
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(run, (first, second)))
            self.assertEqual(outcomes[0].result["score"], 1.0)
            self.assertEqual(outcomes[1].result["score"], 2.0)

    def test_timeout_is_local_to_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hanging = _make_task(root, "hanging", "while True:\n    pass\n")
            good = _make_task(root, "good", GOOD_METRIC)
            timed_out = run_task_process(
                "metric",
                {"task_dir": str(hanging)},
                timeout=0.2,
                readonly_paths=(hanging,),
            )
            succeeded = run_task_process(
                "metric",
                {"task_dir": str(good)},
                timeout=10,
                readonly_paths=(good,),
            )
            self.assertFalse(timed_out.ok)
            self.assertIn("timed out", timed_out.error or "")
            self.assertTrue(succeeded.ok)

    def test_isolated_mode_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory), "task", GOOD_METRIC)
            outcome = run_task_process(
                "metric",
                {"task_dir": str(task)},
                timeout=10,
                execution_mode="isolated",
                readonly_paths=(task,),
            )
            if outcome.ok:
                self.assertEqual(outcome.result["score"], 1.0)
            else:
                self.assertRegex(
                    outcome.error or "",
                    r"(docker|podman|OPENMLE_GYM_ISOLATED_IMAGE|exited)",
                )

    def test_isolated_command_has_no_network_or_source_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            result = root / "result.json"
            task_input = root / "input"
            task_output = root / "output"
            task_input.mkdir()
            task_output.mkdir()
            request.write_text("{}", encoding="utf-8")
            with (
                patch("openmle_gym.process_runner.shutil.which", return_value="/usr/bin/docker"),
                patch.dict(
                    os.environ,
                    {
                        "OPENMLE_GYM_ISOLATED_IMAGE": "openmle:test",
                        "OPENMLE_BUILD_LLM_API_KEY": "build-secret",
                        "OPENMLE_EVAL_LLM_API_KEY": "eval-secret",
                        "KAGGLE_KEY": "kaggle-secret",
                    },
                    clear=False,
                ),
            ):
                command = _container_command(
                    request,
                    result,
                    "metric",
                    (task_input,),
                    (task_output,),
                )
            rendered = " ".join(command)
            self.assertIn("--network none", rendered)
            self.assertIn("--read-only", command)
            self.assertIn("--cap-drop ALL", rendered)
            self.assertIn(f"{task_input.resolve()}:{task_input.resolve()}:ro", command)
            self.assertIn(f"{task_output.resolve()}:{task_output.resolve()}:rw", command)
            source_root = Path(__file__).resolve().parents[1]
            self.assertNotIn(f"{source_root}:{source_root}:ro", command)
            self.assertNotIn("OPENMLE_BUILD_LLM_API_KEY", command)
            self.assertNotIn("OPENMLE_EVAL_LLM_API_KEY", command)
            self.assertNotIn("KAGGLE_KEY", command)

    def test_prepare_operation_uses_its_own_result_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            public = root / "public"
            private = root / "private"
            raw.mkdir()
            public.mkdir()
            private.mkdir()
            prepare = root / "prepare.py"
            prepare.write_text(
                "def prepare(raw_dir, public_dir, private_dir):\n"
                "    (public_dir / 'done.txt').write_text('ok')\n",
                encoding="utf-8",
            )
            outcome = run_task_process(
                "prepare",
                {
                    "prepare_path": str(prepare),
                    "raw_dir": str(raw),
                    "public_dir": str(public),
                    "private_dir": str(private),
                },
                timeout=10,
            )
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.result, {"executed": True})
            self.assertEqual((public / "done.txt").read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
