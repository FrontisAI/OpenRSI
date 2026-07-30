from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace


REQUIRED_MODULES = (
    "langchain_core",
    "langgraph",
    "py7zr",
    "requests",
    "tqdm",
)
HAS_BUILDER_DEPS = all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


@unittest.skipUnless(HAS_BUILDER_DEPS, "builder dependencies are required")
class PerceiveTests(unittest.TestCase):
    def _node(self, root: Path):
        from builder_core.utils.nodes import NodeExecutor

        forge = root / "forge" / "task"
        forge.mkdir(parents=True)
        (forge / "rawtree.txt").write_text("raw/\n", encoding="utf-8")
        (forge / "fileinfo.txt").write_text("", encoding="utf-8")
        node = NodeExecutor.__new__(NodeExecutor)
        node.todo = {"name": "task", "perceive": False, "errors": []}
        node.max_tool_call = 40
        node.tools = ["list_directory_contents", "get_csv_summary", "read_txt_md", "save"]
        node.structure = SimpleNamespace(comp_id_dir=lambda _: forge)
        return node

    def test_only_successful_save_tool_message_completes_normally(self) -> None:
        from langchain_core.messages import ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            state = {
                "messages": [
                    ToolMessage(
                        content="Successfully saved to fileinfo.txt",
                        tool_call_id="save-1",
                        name="save",
                    )
                ]
            }
            node.Perceive(state)
            self.assertTrue(node.todo["perceive"])
            self.assertEqual(node.todo["errors"], [])

    def test_saved_substring_from_other_tool_degrades(self) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            node.llm_provider_tools = SimpleNamespace(
                query=lambda _: AIMessage(content="done")
            )
            state = {
                "messages": [
                    ToolMessage(
                        content="The word saved appeared in a text file",
                        tool_call_id="read-1",
                        name="read_txt_md",
                    )
                ]
            }
            node.Perceive(state)
            self.assertTrue(node.todo["perceive"])
            self.assertTrue(node.todo["errors"])

    def test_no_tool_call_uses_fail_open_exit(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            node.llm_provider_tools = SimpleNamespace(
                query=lambda _: AIMessage(content="no tool")
            )
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("no tool calls", node.todo["errors"][0])

    def test_llm_exception_uses_fail_open_exit(self) -> None:
        from langchain_core.messages import HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))

            def fail(_):
                raise RuntimeError("simulated LLM failure")

            node.llm_provider_tools = SimpleNamespace(query=fail)
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("simulated LLM failure", node.todo["errors"][0])

    def test_filtered_tool_calls_use_fail_open_exit(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "outside_tool",
                        "args": {},
                        "id": "outside-1",
                        "type": "tool_call",
                    }
                ],
            )
            node.llm_provider_tools = SimpleNamespace(query=lambda _: response)
            node.Perceive({"messages": [HumanMessage(content="start")]})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("no allowed tool calls", node.todo["errors"][0])

    def test_tool_call_limit_uses_fail_open_exit(self) -> None:
        from langchain_core.messages import ToolMessage

        with tempfile.TemporaryDirectory() as directory:
            node = self._node(Path(directory))
            messages = [
                ToolMessage(
                    content="read",
                    tool_call_id=f"read-{index}",
                    name="read_txt_md",
                )
                for index in range(node.max_tool_call)
            ]
            node.Perceive({"messages": messages})
            self.assertTrue(node.todo["perceive"])
            self.assertIn("exceeded maximum", node.todo["errors"][0])

    def test_truncated_text_preview_is_larger_and_explicitly_one_shot(self) -> None:
        from builder_core.tools.tools import (
            configure_task_paths,
            read_txt_md,
        )
        from builder_core.utils.prompts import gen_perceiver

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            text = raw / "long.txt"
            text.write_text("x" * 7000, encoding="utf-8")
            configure_task_paths(raw, root / "fileinfo.txt")
            result = read_txt_md.invoke({"file_path": "long.txt"})
        self.assertIn("truncated to 6000 chars", result)
        self.assertIn("do not read the same path again", result)
        self.assertIn("A truncated preview is sufficient", gen_perceiver)

    def test_empty_fileinfo_preserves_raw_on_delete_request(self) -> None:
        from builder_core.utils.nodes import NodeExecutor
        from builder_core.utils.struct import Structure

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structure = Structure(root, "batch")
            raw = structure.raw_dir("task")
            raw.mkdir(parents=True)
            (raw / "data.csv").write_text("x\n1\n", encoding="utf-8")
            forge = structure.comp_id_dir("task")
            forge.mkdir(parents=True, exist_ok=True)
            (forge / "fileinfo.txt").write_text("", encoding="utf-8")
            node = NodeExecutor.__new__(NodeExecutor)
            node.structure = structure
            node.delete_raw = True
            node.todo = {
                "name": "task",
                "download": True,
                "copy": True,
                "web_info": True,
                "describe": True,
                "prepare": True,
                "metric": True,
                "success": False,
                "errors": [],
                "prepares": [],
            }
            node.Next({"messages": []})
            self.assertTrue(raw.is_dir())
            self.assertTrue(node.todo["success"])
            self.assertIn("Raw data retained", node.todo["errors"][-1])

    def test_archive_path_traversal_is_rejected(self) -> None:
        from builder_core.utils.nodes import NodeExecutor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            destination = root / "raw"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../outside.txt", "not allowed")
            node = NodeExecutor.__new__(NodeExecutor)
            with self.assertRaises(RuntimeError):
                node.extract(archive, destination, recursive=False)
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
