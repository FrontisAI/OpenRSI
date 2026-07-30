from __future__ import annotations

import asyncio
import ast
from pathlib import Path

import pandas as pd
import pytest

from tts_search import eval_utils
from tts_search import reward_func_utils


def test_load_leaderboard_supports_release_layouts_and_missing_data(tmp_path):
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    pd.DataFrame({"Score": [0.9, 0.5]}).to_csv(
        flat_dir / "task-a.csv",
        index=False,
    )

    flat = eval_utils.load_leaderboard(
        {
            "task_name": "task-a",
            "data_dir": str(tmp_path / "unused"),
            "leaderboard_dir": str(flat_dir),
        }
    )
    missing = eval_utils.load_leaderboard(
        {
            "task_name": "missing-task",
            "data_dir": str(tmp_path / "missing-data"),
            "leaderboard_dir": str(tmp_path / "missing-leaderboards"),
        }
    )

    assert list(flat.columns) == ["score"]
    assert flat["score"].tolist() == [0.9, 0.5]
    assert missing is None


def test_summary_skips_corrupt_task_stat_and_continues(tmp_path, caplog):
    task_dir = tmp_path / "program_ep_0" / "broken-task"
    task_dir.mkdir(parents=True)
    (task_dir / "stat.json").write_text("{not valid json", encoding="utf-8")

    summary = eval_utils.write_summary_csv(
        tmp_path,
        {
            "broken-task": {
                "benchmark": "naturebench",
                "task_name": "broken-task",
                "data_dir": str(tmp_path),
            }
        },
        expected_task_names=["broken-task"],
    )

    assert summary.loc[0, "Task"] == "broken-task"
    assert "Skipping unreadable JSON file" in caplog.text


def test_linear_sign_preserves_valid_zero_score():
    metadata = {"higher_is_better": True}

    assert reward_func_utils.score2reward(
        0.0,
        metadata,
        mode="linear_sign",
    ) == pytest.approx(0.0)


def test_sandbox_poll_returns_persistent_client_error_immediately(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload
            self.headers = {}

        def json(self):
            return self._payload

    class FakeClient:
        base_url = "https://sandbox.example"

        async def post(self, *args, **kwargs):
            return FakeResponse(200, {"job_id": "job-1"})

        async def get(self, *args, **kwargs):
            return FakeResponse(401, {"detail": "invalid credentials"})

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(reward_func_utils.asyncio, "sleep", no_sleep)

    status_code, payload = asyncio.run(
        reward_func_utils.get_sandbox_result(
            client=FakeClient(),
            code_str="print('ok')",
            data_dir="/tmp/data",
            wait_timeout=10,
            poll_interval=0,
        )
    )

    assert status_code == 401
    assert payload["error"] == "poll_failed"
    assert payload["data"] == {"detail": "invalid credentials"}


def test_corrupt_completed_task_stat_is_not_treated_as_resumable(tmp_path):
    import importlib.util

    runner_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "runner.py"
    )
    spec = importlib.util.spec_from_file_location("review_runner", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    for filename in ("valid_code_final.py", "submit_code.py"):
        (tmp_path / filename).write_text("print('ok')", encoding="utf-8")
    (tmp_path / "stat.json").write_text("{broken", encoding="utf-8")

    assert runner._has_completed_task_outputs(tmp_path) is False


def test_final_submit_is_not_executed_from_a_finally_block():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py"
    )
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    calls_from_finally = []
    for try_node in (node for node in ast.walk(tree) if isinstance(node, ast.Try)):
        for node in try_node.finalbody:
            for descendant in ast.walk(node):
                if (
                    isinstance(descendant, ast.Call)
                    and isinstance(descendant.func, ast.Attribute)
                    and descendant.func.attr == "evaluate_code"
                ):
                    calls_from_finally.append(descendant)

    assert calls_from_finally == []
