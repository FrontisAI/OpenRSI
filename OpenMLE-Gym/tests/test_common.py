from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openmle_gym.common import (
    atomic_write_json,
    json_safe,
    read_slug_entries,
    resolve_task_path,
    validate_task_name,
)


class _Scalar:
    def item(self):
        return 1.25


class CommonTests(unittest.TestCase):
    def test_task_name_validation_blocks_path_escape(self) -> None:
        self.assertEqual(validate_task_name("spaceship-titanic"), "spaceship-titanic")
        for value in ("", ".", "..", "../other", "a/b", "/tmp/task"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_task_name(value)

    def test_resolve_task_path_stays_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_task_path(root, "task-a"), root.resolve() / "task-a")
            with self.assertRaises(ValueError):
                resolve_task_path(root, "../task-b")

    def test_slug_reader_does_not_sanitize_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slugs.txt"
            path.write_text("../other\n/tmp/task\ncompetitions/good-task\n", encoding="utf-8")
            self.assertEqual(
                read_slug_entries(path),
                ["../other", "/tmp/task", "good-task"],
            )

    def test_json_safe_converts_scalar_and_exception(self) -> None:
        converted = json_safe({"score": _Scalar(), "error": ValueError("bad")})
        self.assertEqual(converted["score"], 1.25)
        self.assertFalse(converted["error"]["success"])

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            atomic_write_json(target, {"value": _Scalar()})
            self.assertEqual(json.loads(target.read_text())["value"], 1.25)
            self.assertEqual(list(target.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
