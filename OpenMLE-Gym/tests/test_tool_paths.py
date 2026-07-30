from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_TOOL_DEPS = importlib.util.find_spec("langchain_core") is not None


@unittest.skipUnless(HAS_TOOL_DEPS, "langchain-core is required")
class ToolPathTests(unittest.TestCase):
    def test_task_tools_cannot_cross_task_boundaries(self) -> None:
        from builder_core.tools.tools import (
            configure_task_paths,
            list_directory_contents,
            save,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_a = root / "task-a" / "raw"
            raw_b = root / "task-b" / "raw"
            raw_a.mkdir(parents=True)
            raw_b.mkdir(parents=True)
            (raw_b / "secret.txt").write_text("secret", encoding="utf-8")
            save_a = root / "task-a" / "forge" / "fileinfo.txt"
            save_b = root / "task-b" / "forge" / "fileinfo.txt"
            configure_task_paths(raw_a, save_a)

            listing = list_directory_contents.invoke({"path": str(raw_b)})
            self.assertIn("outside the current task", listing)
            with self.assertRaises(Exception):
                save.invoke({"info_path": str(save_b), "content": "wrong task"})
            result = save.invoke({"info_path": str(save_a), "content": "task a"})
            self.assertEqual(result, "Successfully saved to fileinfo.txt")
            self.assertFalse(save_b.exists())


if __name__ == "__main__":
    unittest.main()
