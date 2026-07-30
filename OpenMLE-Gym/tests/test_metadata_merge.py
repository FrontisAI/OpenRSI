from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metadata_pipeline.common import merge_task_values


class MetadataMergeTests(unittest.TestCase):
    def test_values_are_joined_by_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_a = root / "a"
            task_b = root / "b"
            frame = pd.DataFrame({"Name": ["b", "a"]})
            merged = merge_task_values(
                frame,
                [task_a, task_b],
                {"Metric": ["metric-a", "metric-b"]},
            )
            self.assertEqual(merged["Metric"].tolist(), ["metric-b", "metric-a"])

    def test_duplicate_or_missing_names_are_rejected(self) -> None:
        frame = pd.DataFrame({"Name": ["a", "a"]})
        with self.assertRaises(ValueError):
            merge_task_values(frame, [Path("a"), Path("b")], {"Metric": [1, 2]})


if __name__ == "__main__":
    unittest.main()
