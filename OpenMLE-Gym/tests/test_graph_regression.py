from __future__ import annotations

import asyncio
import importlib.util
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


REQUIRED_MODULES = (
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "pandas",
    "py7zr",
)
HAS_GRAPH_DEPS = all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


PREPARE_CODE = """\
from pathlib import Path
import pandas as pd

def prepare(raw: Path, public: Path, private: Path):
    source = pd.read_csv(raw / "source.csv")
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    source.to_csv(public / "train.csv", index=False)
    source[["feature"]].to_csv(public / "test.csv", index=False)
    source[["target"]].to_csv(public / "sample_submission.csv", index=False)
    source[["target"]].to_csv(private / "test_answer.csv", index=False)
    (public / "description.txt").write_text("fixture task", encoding="utf-8")
"""


METRIC_CODE = """\
class FixtureMetrics:
    def evaluate(self, y_true=None, y_pred=None):
        return 1.0
"""


@unittest.skipUnless(HAS_GRAPH_DEPS, "full builder graph dependencies are required")
class GraphRegressionTests(unittest.TestCase):
    def test_full_graph_with_fixed_kaggle_and_llm_fixtures(self) -> None:
        from builder_core.design import Graph
        from builder_core.utils.nodes import NodeExecutor
        from builder_core.utils.state import AgentState
        from builder_core.utils.task import Task
        from langchain_core.messages import AIMessage

        def fake_download(self, comp_id, target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{comp_id}.zip"
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("source.csv", "feature,target\n1,0\n2,1\n")
            return {"success": True, "path": target}

        def fake_init(self, *args, **kwargs):
            self.tools = kwargs.get("tools") or []

        def fake_query(self, prompts):
            last_content = prompts[-1].content
            if "fileinfo.txt" in last_content:
                match = re.search(r"'fileinfo\.txt' PATH:'([^']+)'", last_content)
                if match is None:
                    raise AssertionError("Perceive fixture could not find fileinfo path")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save",
                            "args": {
                                "info_path": match.group(1),
                                "content": "source.csv: fixed two-row fixture",
                            },
                            "id": "save-fixture",
                            "type": "tool_call",
                        }
                    ],
                )
            if "based on the DESCRIPTION" in last_content:
                return AIMessage(content=f"<code>\n{PREPARE_CODE}\n</code>")
            if "based on the INFORMATION" in last_content:
                return AIMessage(content=f"<code>\n{METRIC_CODE}\n</code>")
            return AIMessage(content="Fixed fixture task description")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info_csv = root / "info.csv"
            info_csv.write_text(
                "Slug,Overview,DatasetDescription\n"
                "fixture-task,Fixed overview,Fixed dataset description\n",
                encoding="utf-8",
            )
            task = Task(
                name="fixture-task",
                retry=0,
                success=False,
                download=False,
                copy=False,
                web_info=False,
                perceive=False,
                describe=False,
                gen_prep=False,
                metric=False,
                prepare=False,
                prepares=[],
                errors=[],
            )
            with (
                patch.object(NodeExecutor, "download", fake_download),
                patch(
                    "builder_core.utils.nodes.OpenAILLMProvider.__init__",
                    fake_init,
                ),
                patch(
                    "builder_core.utils.nodes.OpenAILLMProvider.query",
                    fake_query,
                ),
            ):
                app = Graph(
                    batch_name="fixture-batch",
                    competition=task,
                    base_dir=root,
                    info_csv=info_csv,
                    sample_dir=Path(__file__).resolve().parents[1]
                    / "builder_core"
                    / "samples",
                ).compile()
                result = asyncio.run(app.ainvoke(AgentState({"messages": []})))

            task_result = result["task_result"]
            self.assertTrue(task_result["success"])
            self.assertTrue(task_result["perceive"])
            self.assertEqual(task_result["errors"], [])
            task_dir = root / "fixture-batch" / "data" / "fixture-task"
            self.assertTrue((task_dir / "data" / "public" / "train.csv").is_file())
            self.assertTrue((task_dir / "utils" / "prepare.py").is_file())
            self.assertTrue((task_dir / "utils" / "metric.py").is_file())


if __name__ == "__main__":
    unittest.main()
