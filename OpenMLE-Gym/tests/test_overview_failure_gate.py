from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from openmle_gym.overview import FIELDNAMES, analyze_task, overview_has_task_failures

BROKEN_METRIC = 'raise RuntimeError("metric is broken")\n'


def _write_rows(directory: str, rows: list[dict[str, str]]) -> Path:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
    path = Path(directory) / "overview.csv"
    path.write_text(output.getvalue(), encoding="utf-8")
    return path


def _task_with_broken_metric(directory: str) -> Path:
    task = Path(directory) / "crasher"
    public = task / "data" / "public"
    private = task / "data" / "private"
    utils = task / "utils"
    for path in (public, private, utils):
        path.mkdir(parents=True)
    (public / "description.txt").write_text("desc", encoding="utf-8")
    (public / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (public / "test.csv").write_text("id\n2\n", encoding="utf-8")
    (public / "sample_submission.csv").write_text("id,target\n2,0\n", encoding="utf-8")
    (private / "test_answer.csv").write_text("target\n0\n", encoding="utf-8")
    (utils / "prepare.py").write_text("pass", encoding="utf-8")
    (utils / "metric.py").write_text(BROKEN_METRIC, encoding="utf-8")
    return task


class OverviewFailureGateTests(unittest.TestCase):
    def test_metric_runtime_error_marks_the_row_as_failed(self) -> None:
        """analyze_task records the error in Metric; NOTE must match so the gate sees it."""
        with tempfile.TemporaryDirectory() as directory:
            row = analyze_task(_task_with_broken_metric(directory))
            self.assertTrue(row["Metric"].startswith("#MEVA_RUNTIME_ERROR:"))
            self.assertTrue(row["NOTE"].startswith("#TASK_RUNTIME_ERROR:"))

    def test_metric_runtime_error_is_detected_from_a_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rows(
                directory,
                [{"Name": "crasher", "Metric": "#MEVA_RUNTIME_ERROR: RuntimeError: boom", "NOTE": ""}],
            )
            self.assertTrue(overview_has_task_failures(path))

    def test_task_runtime_error_in_note_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rows(
                directory,
                [{"Name": "crasher", "Metric": "", "NOTE": "#TASK_RUNTIME_ERROR: boom"}],
            )
            self.assertTrue(overview_has_task_failures(path))

    def test_successful_batch_reports_no_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rows(
                directory,
                [{"Name": "good", "Metric": "@Evaluation result: {'score': 0.5}", "NOTE": ""}],
            )
            self.assertFalse(overview_has_task_failures(path))

    def test_other_note_markers_are_not_treated_as_failures(self) -> None:
        """#DESCRIPTION_NOT_FOUND and #PENDING_LLM are informational."""
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rows(
                directory,
                [
                    {"Name": "a", "Metric": "", "NOTE": "#DESCRIPTION_NOT_FOUND"},
                    {"Name": "b", "Metric": "", "NOTE": "#PENDING_LLM"},
                    {"Name": "c", "Metric": "", "NOTE": "#SKIPPED_LLM"},
                ],
            )
            self.assertFalse(overview_has_task_failures(path))

    def test_a_single_failure_among_successes_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rows(
                directory,
                [
                    {"Name": "good", "Metric": "@Evaluation result: {'score': 0.5}", "NOTE": ""},
                    {"Name": "bad", "Metric": "#MEVA_RUNTIME_ERROR: RuntimeError: boom", "NOTE": ""},
                ],
            )
            self.assertTrue(overview_has_task_failures(path))


if __name__ == "__main__":
    unittest.main()