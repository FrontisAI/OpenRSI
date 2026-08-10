from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "aira-evo" / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_fake_naturebench_task(
    root: Path,
    task_name: str = "fake-nature-task",
) -> Path:
    task_dir = root / "tasks" / task_name
    problem_dir = task_dir / "problem"
    data_dir = problem_dir / "data"
    data_dir.mkdir(parents=True)
    (problem_dir / "README.md").write_text(
        "# Fake Nature task\n\nPredict the target and write outputs under OUTPUT_DIR.",
        encoding="utf-8",
    )
    (problem_dir / "data_description.md").write_text(
        "The visible data lives under DATA_DIR and contains one small CSV file.",
        encoding="utf-8",
    )
    (data_dir / "input.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "domain": "unit-test",
                "resource": "cpu",
                "performance_entries": [
                    {
                        "dataset_name": "fake-instance",
                        "metrics": [
                            {
                                "name": "accuracy",
                                "is_primary": True,
                                "metric_direction": "higher_is_better",
                                "sota_score": 0.8,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return task_dir


def test_naturebench_build_tasks_writes_airaevo_config(tmp_path):
    build_tasks = _load_module(
        "naturebench_build_tasks",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_list": ["fake-nature-task"],
                    "eval_service_url": "http://127.0.0.1:8321",
                    "batch_name": "unit-batch",
                    "execution_mode": "local",
                    "execution_timeout": 123,
                    "local_python": "/opt/naturebench/bin/python",
                    "local_conda_env": None,
                    "local_terminate_grace_seconds": 2,
                    "candidate_env_allowlist": ["XDG_CACHE_HOME"],
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    assert task_cfg["benchmark"] == "naturebench"
    assert task_cfg["task_name"] == "fake-nature-task"
    assert task_cfg["higher_is_better"] is True
    assert task_cfg["operator_family"] == "naturebench"
    assert task_cfg["solver_defaults"] == "naturebench/evo.yaml"
    assert task_cfg["execution_mode"] == "local"
    assert task_cfg["eval_service_url"] == "http://127.0.0.1:8321"
    assert task_cfg["batch_name"] == "unit-batch"
    assert task_cfg["execution_timeout"] == 123
    assert task_cfg["local_python"] == "/opt/naturebench/bin/python"
    assert task_cfg["local_conda_env"] is None
    assert task_cfg["local_terminate_grace_seconds"] == 2
    assert task_cfg["candidate_env_allowlist"] == ["XDG_CACHE_HOME"]
    assert task_cfg["data_dir"].endswith("problem/data")
    assert "Predict the target" in task_cfg["task_description"]
    assert "DATA_DIR" in task_cfg["data_description"]
    assert "aggregate_improvement" in task_cfg["public_user_prompt"]
    assert task_cfg["visible_data_analysis"] == ""
    assert "Do not install packages" in task_cfg["task_family_guidance"]
    assert task_cfg["candidate_preflight"] is True
    assert task_cfg["candidate_preflight_imports"] is True


def test_naturebench_build_tasks_extracts_empirical_eda_addendum(tmp_path):
    build_tasks = _load_module(
        "naturebench_build_tasks_eda",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    eda_root = tmp_path / "eda"
    eda_root.mkdir()
    (eda_root / "fake-nature-task.md").write_text(
        "# Full document\n\n"
        "This section must not be injected.\n\n"
        "## 5. Empirical EDA Addendum\n\n"
        "Verified shape: (10, 2).\n"
        "Use the visible train.csv path.\n",
        encoding="utf-8",
    )
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_list": ["fake-nature-task"],
                    "execution_mode": "local",
                    "visible_data_analysis_root": str(eda_root),
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    assert task_cfg["visible_data_analysis"] == (
        "Verified shape: (10, 2).\nUse the visible train.csv path."
    )


def test_naturebench_build_tasks_uses_task_set_resource_labels(tmp_path):
    build_tasks = _load_module(
        "naturebench_build_tasks_resource",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    task_set_root = naturebench_root / "task-set"
    task_set_root.mkdir()
    (task_set_root / "gpu_high.txt").write_text("fake-nature-task\n", encoding="utf-8")
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_list": ["fake-nature-task"],
                    "execution_mode": "local",
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    assert task_cfg["resource"] == "gpu_high"


def test_naturebench_build_tasks_uses_official_scm_resource_lines(tmp_path):
    build_tasks = _load_module(
        "naturebench_build_tasks_resource_lines",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    for task_name in [
        "cpu-task",
        "thu-task",
        "thu-shared-task",
        "amd-task",
        "amd-shared-task",
    ]:
        _write_fake_naturebench_task(naturebench_root, task_name=task_name)
    task_set_root = naturebench_root / "task-set"
    task_set_root.mkdir()
    (task_set_root / "all.txt").write_text(
        "\n".join(
            [
                "cpu-task",
                "thu-task",
                "thu-shared-task",
                "amd-task",
                "amd-shared-task",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_set_root / "gemini_3.5_flash_cpu_thu.txt").write_text(
        "cpu-task\n", encoding="utf-8"
    )
    (task_set_root / "gemini_3.5_flash_gpu_thu_normal.txt").write_text(
        "thu-task\n", encoding="utf-8"
    )
    (task_set_root / "gemini_3.5_flash_gpu_thu_shared.txt").write_text(
        "thu-shared-task\n", encoding="utf-8"
    )
    (task_set_root / "gemini_3.5_flash_gpu_amd_normal.txt").write_text(
        "amd-task\n", encoding="utf-8"
    )
    (task_set_root / "gemini_3.5_flash_gpu_amd_shared.txt").write_text(
        "amd-shared-task\n", encoding="utf-8"
    )
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_set_path": "task-set/all.txt",
                    "execution_mode": "scm_docker",
                    "scm_host": "scm-primary",
                    "gpu_devices": "all",
                    "scm_resource_lines": {
                        "gpu_thu_normal": {
                            "gpu_devices": [0, 1],
                            "gpu_pool_file": "/tmp/thu-exclusive.json",
                        },
                        "gpu_thu_shared": {
                            "shared_gpu_device": 2,
                            "shared_gpu_slots": 4,
                            "shared_gpu_pool_file": "/tmp/thu-shared.json",
                        },
                        "gpu_amd_normal": {
                            "scm_host": "scm-secondary",
                            "gpu_devices": [0],
                            "gpu_pool_file": "/tmp/amd-exclusive.json",
                        },
                        "gpu_amd_shared": {
                            "scm_host": "scm-secondary",
                            "shared_gpu_device": 1,
                            "shared_gpu_slots": 3,
                            "shared_gpu_pool_file": "/tmp/amd-shared.json",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    cpu_cfg = yaml.safe_load((output_root / "cpu-task" / "config.yaml").read_text())
    thu_cfg = yaml.safe_load((output_root / "thu-task" / "config.yaml").read_text())
    thu_shared_cfg = yaml.safe_load(
        (output_root / "thu-shared-task" / "config.yaml").read_text()
    )
    amd_cfg = yaml.safe_load((output_root / "amd-task" / "config.yaml").read_text())
    amd_shared_cfg = yaml.safe_load(
        (output_root / "amd-shared-task" / "config.yaml").read_text()
    )

    assert cpu_cfg["resource_line"] == "cpu_thu"
    assert cpu_cfg["resource"] == "cpu"
    assert cpu_cfg["gpu_mode"] == "none"
    assert "gpu_devices" not in cpu_cfg

    assert thu_cfg["resource_line"] == "gpu_thu_normal"
    assert thu_cfg["resource"] == "gpu_thu_normal"
    assert thu_cfg["scm_host"] == "scm-primary"
    assert thu_cfg["gpu_mode"] == "exclusive"
    assert thu_cfg["gpu_devices"] == [0, 1]
    assert thu_cfg["gpu_pool_file"] == "/tmp/thu-exclusive.json"

    assert thu_shared_cfg["resource_line"] == "gpu_thu_shared"
    assert thu_shared_cfg["gpu_mode"] == "shared"
    assert thu_shared_cfg["shared_gpu_device"] == 2
    assert thu_shared_cfg["shared_gpu_slots"] == 4
    assert thu_shared_cfg["shared_gpu_pool_file"] == "/tmp/thu-shared.json"

    assert amd_cfg["resource_line"] == "gpu_amd_normal"
    assert amd_cfg["scm_host"] == "scm-secondary"
    assert amd_cfg["gpu_mode"] == "exclusive"
    assert amd_cfg["gpu_devices"] == [0]

    assert amd_shared_cfg["resource_line"] == "gpu_amd_shared"
    assert amd_shared_cfg["scm_host"] == "scm-secondary"
    assert amd_shared_cfg["gpu_mode"] == "shared"
    assert amd_shared_cfg["shared_gpu_device"] == 1
    assert amd_shared_cfg["shared_gpu_slots"] == 3


def test_naturebench_build_tasks_fails_when_scm_resource_lines_are_unreadable(
    tmp_path,
    monkeypatch,
):
    build_tasks = _load_module(
        "naturebench_build_tasks_resource_lines_failure",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_list": ["fake-nature-task"],
                    "execution_mode": "scm_docker",
                    "scm_task_set_host": "scm-primary",
                    "scm_task_set_root": "/missing/task-set",
                }
            }
        ),
        encoding="utf-8",
    )

    class FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "cat: /missing/task-set/gemini_3.5_flash_cpu_thu.txt: No such file"

    monkeypatch.setattr(
        build_tasks.subprocess, "run", lambda *args, **kwargs: FailedProcess()
    )

    with pytest.raises(
        RuntimeError, match="Failed to read NatureBench resource task-set"
    ):
        build_tasks.build_tasks(build_cfg_path, output_root=tmp_path / "airaevo_tasks")


def test_naturebench_build_tasks_allows_zero_submit_repeats(tmp_path):
    build_tasks = _load_module(
        "naturebench_build_tasks_zero_submit",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "build_tasks.py",
    )
    naturebench_root = tmp_path / "NatureBench"
    _write_fake_naturebench_task(naturebench_root)
    build_cfg_path = tmp_path / "build.yaml"
    build_cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "naturebench_root": str(naturebench_root),
                    "task_list": ["fake-nature-task"],
                    "execution_mode": "local",
                    "submit_repeats": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "airaevo_tasks"
    build_tasks.build_tasks(build_cfg_path, output_root=output_root)

    task_cfg = yaml.safe_load(
        (output_root / "fake-nature-task" / "config.yaml").read_text(encoding="utf-8")
    )
    assert task_cfg["submit_repeats"] == 0


def test_naturebench_task_uses_current_aggregate_improvement_as_fitness(tmp_path):
    base_task = _load_module(
        "naturebench_base_task",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    from dojo.core.tasks.constants import (
        AUX_EVAL_INFO,
        TASK_DESCRIPTION,
        VALIDATION_FITNESS,
    )

    class FakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            assert "hello naturebench" in code
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            return {
                "aggregate_improvement": 0.37,
                "best_aggregate_improvement": 0.42,
                "raw_scores": {"fake-instance": {"accuracy": 0.9}},
                "per_instance_improvement": {"fake-instance": 0.125},
                "attempt": 3,
            }

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = FakeNatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "higher_is_better": True,
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
            "task_description": "Do the fake task.",
            "data_description": "Use DATA_DIR.",
            "visible_data_analysis": "Verified shape: (10, 2).",
        }
    )

    state, task_info = task.prepare()
    state, result = task.step_task(state, "print('hello naturebench')")

    assert task_info["lower_is_better"] is False
    assert task_info["visible_data_analysis"] == "Verified shape: (10, 2)."
    assert "Verified Visible-Data Analysis" in task_info[TASK_DESCRIPTION]
    assert result[VALIDATION_FITNESS] == pytest.approx(0.37)
    assert result[AUX_EVAL_INFO]["aggregate_improvement"] == pytest.approx(0.37)
    assert result[AUX_EVAL_INFO]["best_aggregate_improvement"] == pytest.approx(0.42)
    assert result[AUX_EVAL_INFO]["score"] == pytest.approx(0.37)
    assert state["naturebench_attempt_index"] == 1


def test_naturebench_task_clips_selection_fitness_but_preserves_raw_aggregate(
    tmp_path,
):
    base_task = _load_module(
        "naturebench_base_task_clipped_fitness",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    from dojo.core.tasks.constants import AUX_EVAL_INFO, VALIDATION_FITNESS

    class FakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            return {
                "aggregate_improvement": 64.05,
                "best_aggregate_improvement": 64.05,
                "raw_scores": {"fake-instance": {"accuracy": 999.0}},
                "per_instance_improvement": {"fake-instance": 64.05},
                "attempt": 1,
            }

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = FakeNatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "higher_is_better": True,
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
            "task_description": "Do the fake task.",
            "data_description": "Use DATA_DIR.",
            "selection_score_clip": 1.0,
        }
    )

    state, _ = task.prepare()
    _, result = task.step_task(state, "print('hello naturebench')")

    assert result[VALIDATION_FITNESS] == pytest.approx(1.0)
    assert result[AUX_EVAL_INFO]["selection_score"] == pytest.approx(1.0)
    assert result[AUX_EVAL_INFO]["aggregate_improvement"] == pytest.approx(64.05)
    assert result[AUX_EVAL_INFO]["score"] == pytest.approx(64.05)


def test_naturebench_preflight_blocks_package_install_without_evaluation(tmp_path):
    base_task = _load_module(
        "naturebench_base_task_preflight",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    from dojo.core.tasks.constants import AUX_EVAL_INFO

    calls = []

    class CountingTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            calls.append(("run", code, phase))
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "ran candidate",
                "run_time": 0.25,
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            calls.append(("evaluate", str(output_dir)))
            return {"aggregate_improvement": 0.2}

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = CountingTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
            "candidate_preflight": True,
        }
    )

    state, _ = task.prepare()
    _, result = task.step_task(
        state,
        "import subprocess\nsubprocess.run(['pip', 'install', 'andi'])",
    )

    info = result[AUX_EVAL_INFO]
    assert calls == []
    assert info["status_code"] == 422
    assert info["preflight_category"] == "prohibited_package_install"
    assert "package installation is not allowed" in info["feedback"]


def test_naturebench_scm_gpu_wait_is_reported_separately(monkeypatch, tmp_path):
    base_task = _load_module(
        "naturebench_base_task_gpu_wait",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    class FakeCompletedProcess:
        returncode = 0
        stdout = "__AIREVO_GPU_WAIT_SECONDS__=7\ncandidate stdout\n"
        stderr = ""

    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            return None

        def _resolve_scm_task_root(self) -> str:
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            return FakeCompletedProcess()

    ticks = iter([100.0, 110.0])
    monkeypatch.setattr(base_task.time, "monotonic", lambda: next(ticks))
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = CapturingTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "resource": "gpu_thu_normal",
            "gpu_mode": "exclusive",
            "gpu_devices": [0],
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "execution_mode": "scm_docker",
            "scm_host": "scm-primary",
            "scm_workspace_root": "/remote/workspaces/unit",
            "scm_task_roots": ["/remote/tasks"],
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            "batch_name": "unit-batch",
            "execution_timeout": 60,
        }
    )

    attempt = task._start_attempt(phase="validation")
    result = task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    assert result["run_time"] == pytest.approx(10.0)
    assert result["gpu_wait_seconds"] == pytest.approx(7.0)
    assert result["active_run_time"] == pytest.approx(3.0)
    assert "__AIREVO_GPU_WAIT_SECONDS__" not in result["feedback"]


def test_naturebench_task_registers_eval_service_once_before_evaluation(tmp_path):
    base_task = _load_module(
        "naturebench_base_task_register",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    calls = []

    class RegisteringFakeNatureBenchTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": "ran candidate",
                "clear_run_log": "ran candidate",
                "feedback": "execution succeeded",
                "run_time": 0.25,
            }

        def _post_json(
            self,
            endpoint: str,
            payload: dict,
            *,
            timeout: int | float,
        ) -> dict:
            calls.append((endpoint, dict(payload), timeout))
            if endpoint == "register":
                return {"status": "ok"}
            if endpoint == "start_timer":
                return {"status": "ok"}
            return {
                "aggregate_improvement": 0.2,
                "best_aggregate_improvement": 0.2,
                "raw_scores": {},
                "per_instance_improvement": {},
                "attempt": len([call for call in calls if call[0] == "evaluate"]),
            }

    task_package_dir = tmp_path / "task-package"
    data_dir = task_package_dir / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = RegisteringFakeNatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "higher_is_better": True,
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(task_package_dir),
            "workspace_root": str(tmp_path / "workspace"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
            "task_description": "Do the fake task.",
            "data_description": "Use DATA_DIR.",
            "execution_timeout": 600,
        },
        time_budget=14400,
    )

    state, _ = task.prepare()
    task.step_task(state, "print('first')")
    task.step_task(state, "print('second')")

    assert [endpoint for endpoint, _, _ in calls] == [
        "register",
        "start_timer",
        "evaluate",
        "evaluate",
    ]
    register_payload = calls[0][1]
    assert register_payload["task_name"] == "fake-nature-task"
    assert register_payload["data_dir"] == str(task_package_dir.resolve())
    assert register_payload["timeout"] == 14400
    assert register_payload["batch_name"] == "unit-batch"
    assert register_payload["out_dir"].endswith("unit-batch/fake-nature-task")
    assert calls[1][1] == {
        "task_name": "fake-nature-task",
        "batch_name": "unit-batch",
    }


def test_naturebench_scm_docker_uses_exclusive_gpu_pool_not_all_flag(tmp_path):
    base_task = _load_module(
        "naturebench_base_task_gpu",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            captured["remote_workspace"] = remote_workspace

        def _resolve_scm_task_root(self) -> str:
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            captured["remote_command"] = remote_command
            captured["ssh_timeout"] = kwargs.get("timeout")
            return FakeCompletedProcess()

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = CapturingTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "resource": "gpu_thu_normal",
            "gpu_mode": "exclusive",
            "gpu_devices": [0, 1],
            "gpu_pool_file": "/tmp/airaevo-unit-exclusive",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "execution_mode": "scm_docker",
            "scm_host": "scm-primary",
            "scm_workspace_root": "/remote/workspaces/unit",
            "scm_task_roots": ["/remote/tasks"],
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            "batch_name": "unit-batch",
            "execution_timeout": 60,
        }
    )

    attempt = task._start_attempt(phase="validation")
    task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    assert "--gpus all" not in captured["remote_command"]
    assert "device=all" not in captured["remote_command"]
    assert "AIREVO_GPU_ID" in captured["remote_command"]
    assert "0 1" in captured["remote_command"]
    assert "device=$AIREVO_GPU_ID" in captured["remote_command"]
    assert "timeout -k 30s 60s docker" in captured["remote_command"]
    assert captured["ssh_timeout"] >= 14400
    assert "nvidia-smi" in captured["remote_command"]
    assert "AIREVO_GPU_SKIP_BUSY_MB=2000" in captured["remote_command"]
    assert "AIREVO_GPU_SKIP_BUSY_UTIL=80" in captured["remote_command"]


def test_naturebench_scm_docker_uses_shared_gpu_slot_pool(tmp_path):
    base_task = _load_module(
        "naturebench_base_task_shared_gpu",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = "ok"
        stderr = ""

    class CapturingTask(base_task.NatureBenchTask):
        def _sync_workspace_to_scm(
            self,
            workspace: Path,
            remote_workspace: str,
        ) -> None:
            captured["remote_workspace"] = remote_workspace

        def _resolve_scm_task_root(self) -> str:
            return "/remote/task"

        def _ssh(self, remote_command: str, **kwargs):
            captured["remote_command"] = remote_command
            return FakeCompletedProcess()

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = CapturingTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-shared-task",
            "resource": "gpu_thu_shared",
            "gpu_mode": "shared",
            "shared_gpu_device": 9,
            "shared_gpu_slots": 4,
            "shared_gpu_pool_file": "/tmp/airaevo-unit-shared",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "execution_mode": "scm_docker",
            "scm_host": "scm-primary",
            "scm_workspace_root": "/remote/workspaces/unit",
            "scm_task_roots": ["/remote/tasks"],
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            "batch_name": "unit-batch",
            "execution_timeout": 60,
        }
    )

    attempt = task._start_attempt(phase="validation")
    task._run_scm_docker_solution(
        "print('hello naturebench')",
        attempt=attempt,
    )

    assert "--gpus all" not in captured["remote_command"]
    assert "AIREVO_GPU_ID=9" in captured["remote_command"]
    assert "AIREVO_SHARED_SLOT" in captured["remote_command"]
    assert "AIREVO_SHARED_GPU_SLOTS=4" in captured["remote_command"]
    assert "slot_${AIREVO_SLOT}.lock" in captured["remote_command"]
    assert "device=9" in captured["remote_command"]
    assert "nvidia-smi" in captured["remote_command"]
    assert "airaevo_gpu_has_live_shared_holder" in captured["remote_command"]
    assert "AIREVO_GPU_SKIP_BUSY_MB=2000" in captured["remote_command"]


def test_naturebench_scm_ssh_retries_transient_connection_close(
    monkeypatch,
    tmp_path,
):
    base_task = _load_module(
        "naturebench_base_task_ssh_retry",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )

    calls = []

    class FakeCompletedProcess:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return FakeCompletedProcess(
                255,
                stderr="Connection closed by UNKNOWN port 65535",
            )
        return FakeCompletedProcess(0, stdout="ok")

    monkeypatch.setattr(base_task.subprocess, "run", fake_run)
    monkeypatch.setattr(base_task.time, "sleep", lambda delay: None)

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = base_task.NatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "fake-nature-task",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "execution_mode": "scm_docker",
            "scm_host": "scm-primary",
            "scm_workspace_root": "/remote/workspaces/unit",
            "scm_task_roots": ["/remote/tasks"],
            "eval_service_url": "http://localhost:8321",
            "scm_eval_service_url": "http://localhost:8321",
            "scm_container_eval_service_url": "http://host.docker.internal:8321",
            "batch_name": "unit-batch",
            "scm_ssh_retries": 2,
        }
    )

    process = task._ssh("echo ok")

    assert process.returncode == 0
    assert process.stdout == "ok"
    assert len(calls) == 2


def test_naturebench_runner_uses_naturebench_task_and_prompt_paths():
    single_task_runner = _load_module(
        "airaevo_single_task_runner_for_naturebench",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    assert (
        single_task_runner._benchmark_name({"benchmark": "naturebench"})
        == "naturebench"
    )
    assert single_task_runner._solver_defaults_relative_path(
        {"benchmark": "naturebench"}
    ) == Path("naturebench/evo.yaml")

    ordinary_paths = single_task_runner._operator_paths_for_benchmark(
        "naturebench",
        experience_enabled=False,
    )
    assert ordinary_paths["draft"] == "naturebench/aira_operators/draft.yaml"
    assert ordinary_paths["improve"] == "naturebench/aira_operators/improve.yaml"
    assert ordinary_paths["debug"] == "naturebench/aira_operators/debug.yaml"
    assert ordinary_paths["crossover"] == "naturebench/aira_operators/crossover.yaml"
    assert ordinary_paths["analyze"] == "naturebench/aira_operators/analyze.yaml"

    experience_paths = single_task_runner._operator_paths_for_benchmark(
        "naturebench",
        experience_enabled=True,
    )
    assert (
        experience_paths["improve"]
        == "naturebench/aira_operators/improve_experience.yaml"
    )
    assert (
        experience_paths["debug"] == "naturebench/aira_operators/debug_experience.yaml"
    )
    assert (
        experience_paths["crossover"]
        == "naturebench/aira_operators/crossover_experience.yaml"
    )
    assert experience_paths["rich_memory_summary"] == (
        "naturebench/aira_operators/rich_memory_summary.yaml"
    )


def test_naturebench_operator_templates_accept_verified_visible_data_analysis():
    operators_root = (
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "src"
        / "dojo"
        / "configs"
        / "solver"
        / "operators"
        / "naturebench"
        / "aira_operators"
    )
    for filename, operator_name in [
        ("draft.yaml", "draft"),
        ("improve.yaml", "improve"),
        ("improve_experience.yaml", "improve"),
        ("debug.yaml", "debug"),
        ("debug_experience.yaml", "debug"),
        ("crossover.yaml", "crossover"),
        ("crossover_experience.yaml", "crossover"),
    ]:
        payload = yaml.safe_load((operators_root / filename).read_text())
        prompt = payload[operator_name]["init_user_message_prompt_template"]
        assert "Verified Visible-Data Analysis" in prompt["template"]
        assert "{{visible_data_analysis}}" in prompt["template"]
        assert "visible_data_analysis" in prompt["input_variables"]

    analyze_payload = yaml.safe_load((operators_root / "analyze.yaml").read_text())
    analyze_prompt = analyze_payload["analyze"]["system_message_prompt_template"]
    assert "{{task_desc}}" in analyze_prompt["template"]


def test_naturebench_final_submit_is_disabled_by_default_and_uses_search_score():
    single_task_runner = _load_module(
        "airaevo_single_task_runner_final_submit",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    assert (
        single_task_runner._should_run_final_submit(
            benchmark="naturebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={},
        )
        is False
    )
    assert (
        single_task_runner._should_run_final_submit(
            benchmark="naturebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={"solver": {"final_submit": True}},
        )
        is True
    )
    assert (
        single_task_runner._should_run_final_submit(
            benchmark="mlebench",
            submit_repeats=1,
            task_cfg={},
            runner_cfg={},
        )
        is True
    )
    assert single_task_runner._effective_submit_score(
        benchmark="naturebench",
        submit_score=None,
        final_selection_score=0.23,
        final_submit_enabled=False,
    ) == pytest.approx(0.23)


def test_naturebench_final_selection_ignores_failed_high_score_nodes():
    single_task_runner = _load_module(
        "airaevo_single_task_runner_selection",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    class Metric:
        def __init__(self, value: float, info: dict):
            self.value = value
            self.info = info

    class Node:
        def __init__(self, node_id: str, value: float, info: dict):
            self.id = node_id
            self.step = len(node_id)
            self.code = f"print({node_id!r})"
            self.metric = Metric(value, info)

    class Journal:
        def __init__(self, nodes: list[Node]):
            self.nodes = nodes

        def is_root_node(self, node: Node) -> bool:
            return False

    class Solver:
        def __init__(self, nodes: list[Node]):
            self.journal = Journal(nodes)

    failed_high = Node(
        "failed-high",
        10.0,
        {"selection_score": 10.0, "status": "failed", "status_code": 500},
    )
    success_low = Node(
        "success-low",
        0.2,
        {"selection_score": 0.2, "status": "success", "status_code": 200},
    )

    selected = single_task_runner._select_best_available_node(
        Solver([failed_high, success_low]),
        seed=7,
        task_name="fake-task",
        sample_index=0,
        benchmark="naturebench",
    )

    assert selected is success_low
    assert selected._final_selection_source == "validation"


def test_naturebench_leaderboard_flags_use_official_thresholds():
    single_task_runner = _load_module(
        "airaevo_single_task_runner_leaderboard_flags",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    assert single_task_runner._naturebench_leaderboard_flags(-0.01) == {
        "match_sota": False,
        "surpass_sota": False,
    }
    assert single_task_runner._naturebench_leaderboard_flags(0.0) == {
        "match_sota": True,
        "surpass_sota": False,
    }
    assert single_task_runner._naturebench_leaderboard_flags(0.1) == {
        "match_sota": True,
        "surpass_sota": False,
    }
    assert single_task_runner._naturebench_leaderboard_flags(0.1001) == {
        "match_sota": True,
        "surpass_sota": True,
    }


def test_solutions_database_can_force_a_fresh_draft_after_repeated_error():
    from dojo.solvers.evo.evo import SolutionsDatabase

    class Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    class Node:
        def __init__(self, node_id: str, value: float):
            self.id = node_id
            self.metric = type("Metric", (), {"value": value})()

    database = SolutionsDatabase(
        num_islands=1,
        max_size=4,
        lower_is_better=False,
        logger=Logger(),
    )
    database._islands[0].nodes.append(Node("first", 0.1))
    database.request_fresh_draft("repeated_timeout")

    nodes, island_id, operator = database.sample_in_context(
        {"improve": 1, "crossover": 2},
        temperature=1.0,
        crossover_prob=0.5,
        fresh_draft_prob=0.0,
    )

    assert nodes == []
    assert island_id == 0
    assert operator == "draft"
    assert database.last_parent_selection["reason"] == "repeated_timeout"

    nodes, island_id, operator = database.sample_in_context(
        {"improve": 1, "crossover": 2},
        temperature=1.0,
        crossover_prob=1.0,
        fresh_draft_prob=0.0,
    )

    assert island_id == 0
    assert operator == "improve"
    assert [node.id for node in nodes] == ["first"]
    assert database.last_parent_selection["operator_fallback_reason"] == (
        "crossover_parent_shortage"
    )


def test_naturebench_valid_final_score_requires_successful_node():
    single_task_runner = _load_module(
        "airaevo_single_task_runner_valid_score",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py",
    )

    class Metric:
        def __init__(self, info: dict):
            self.info = info

    class Node:
        def __init__(self, info: dict):
            self.metric = Metric(info)

    failed_node = Node(
        {"aggregate_improvement": 0.9, "status": "failed", "status_code": 500}
    )
    successful_node = Node(
        {"aggregate_improvement": 0.2, "status": "success", "status_code": 200}
    )

    assert single_task_runner._naturebench_valid_score_from_node(failed_node) is None
    assert single_task_runner._naturebench_valid_score_from_node(
        successful_node
    ) == pytest.approx(0.2)


def test_naturebench_skipped_submit_outputs_count_as_completed(tmp_path):
    runner = _load_module(
        "airaevo_runner_resume_naturebench",
        REPO_ROOT / "third_party" / "aira-evo" / "examples" / "mle_bench" / "runner.py",
    )

    (tmp_path / "valid_code_final.py").write_text("print('best')", encoding="utf-8")
    (tmp_path / "submit_code.py").write_text("print('best')", encoding="utf-8")
    (tmp_path / "stat.json").write_text(
        json.dumps(
            {
                "benchmark": "naturebench",
                "final_submit_skipped": True,
                "final_score": 0.23,
                "submit_score": 0.23,
                "best_aggregate_improvement": 0.23,
            }
        ),
        encoding="utf-8",
    )

    assert runner._has_completed_task_outputs(tmp_path) is True


def test_naturebench_operator_prompts_use_output_dir_not_submission_csv():
    prompts_root = (
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "src"
        / "dojo"
        / "configs"
        / "solver"
        / "operators"
        / "naturebench"
        / "aira_operators"
    )
    for prompt_name in [
        "draft.yaml",
        "improve.yaml",
        "debug.yaml",
        "crossover.yaml",
        "improve_experience.yaml",
        "debug_experience.yaml",
        "crossover_experience.yaml",
    ]:
        prompt_text = (prompts_root / prompt_name).read_text(encoding="utf-8")
        assert "DATA_DIR" in prompt_text
        assert "OUTPUT_DIR" in prompt_text
        assert "aggregate_improvement" in prompt_text
        assert "{{packages}}" in prompt_text
        assert "h5py" in prompt_text
        assert "/layers/spliced" in prompt_text
        assert "submission.csv" not in prompt_text

    analyze_prompt = (prompts_root / "analyze.yaml").read_text(encoding="utf-8")
    assert "NatureBench" in analyze_prompt
    assert "DATA_DIR" in analyze_prompt
    assert "OUTPUT_DIR" in analyze_prompt
    assert "aggregate_improvement" in analyze_prompt
    assert "submission.csv" not in analyze_prompt

    solver_cfg = yaml.safe_load(
        (
            REPO_ROOT
            / "third_party"
            / "aira-evo"
            / "src"
            / "dojo"
            / "configs"
            / "solver"
            / "naturebench"
            / "evo.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "h5py" in solver_cfg["available_packages"]


def test_evaluate_naturebench_payloads_point_to_naturebench_builder_and_runner(
    tmp_path,
):
    evaluate_naturebench = _load_module(
        "evaluate_naturebench",
        REPO_ROOT / "scripts" / "evaluate_naturebench.py",
    )
    cfg = OmegaConf.create(
        {
            "experiment_name": "naturebench_smoke",
            "output_dir": str(tmp_path / "outputs"),
            "seed": 7,
            "time_budget": 14400,
            "model_plus_sandbox_time_budget": 14400,
            "n_samples_per_task": 1,
            "candidates_per_step": 1,
            "max_steps": 2,
            "llm_concurrency": 1,
            "sandbox": {"concurrency": 1},
            "litellm": {
                "model_list": [
                    {
                        "model_name": "unit-model",
                        "litellm_params": {
                            "api_key": "unit-key",
                            "base_url": "http://127.0.0.1:30000/v1",
                            "temperature": 0.2,
                        },
                    }
                ]
            },
            "data": {
                "naturebench_root": "/tmp/NatureBench",
                "task_list": ["fake-nature-task"],
                "eval_service_url": "http://127.0.0.1:8321",
                "execution_mode": "scm_docker",
                "batch_name": "unit-batch",
                "submit_repeats": 0,
                "scm_host": "scm-primary",
                "scm_workspace_root": "/remote/workspaces/unit",
                "scm_task_roots": ["/remote/naturebench/tasks/part1"],
                "scm_eval_service_url": "http://localhost:8321",
                "scm_container_eval_service_url": "http://host.docker.internal:8321",
            },
            "search": {
                "runner": {
                    "package_root": "third_party/aira-evo",
                    "task_concurrency": 1,
                    "strict_resume": False,
                    "llm": {"api": "litellm", "provider": "selfhosted"},
                    "solver": {"execution_timeout": 60},
                    "interpreter": {"use_symlinks": True},
                    "logger": {"use_console": True},
                }
            },
        }
    )

    prepare_payload, runner_payload = evaluate_naturebench._build_payloads(
        cfg,
        output_dir=tmp_path / "outputs",
        task_root=tmp_path / "tasks",
    )

    assert prepare_payload["data"]["naturebench_root"] == "/tmp/NatureBench"
    assert prepare_payload["data"]["task_list"] == ["fake-nature-task"]
    assert prepare_payload["data"]["execution_mode"] == "scm_docker"
    assert prepare_payload["data"]["submit_repeats"] == 0
    assert prepare_payload["data"]["scm_host"] == "scm-primary"
    assert prepare_payload["data"]["scm_workspace_root"] == "/remote/workspaces/unit"
    assert prepare_payload["data"]["scm_task_roots"] == [
        "/remote/naturebench/tasks/part1"
    ]
    assert prepare_payload["data"]["scm_eval_service_url"] == "http://localhost:8321"
    assert (
        prepare_payload["data"]["scm_container_eval_service_url"]
        == "http://host.docker.internal:8321"
    )
    assert runner_payload["task_root"] == str(tmp_path / "tasks")
    assert runner_payload["llm"]["model_id"] == "unit-model"
    assert "api_key" not in runner_payload["llm"]
    assert runner_payload["solver"]["step_limit"] == 2
    assert runner_payload["solver"]["execution_timeout"] == 60


def test_naturebench_default_configs_skip_submit_and_reduce_search_artifacts():
    data_cfg = OmegaConf.load(
        REPO_ROOT / "tts_search" / "configs" / "data" / "naturebench_scm_all.yaml"
    )
    search_cfg = yaml.safe_load(
        (
            REPO_ROOT / "tts_search" / "configs" / "search" / "airaevo_naturebench.yaml"
        ).read_text(encoding="utf-8")
    )

    assert data_cfg["submit_repeats"] == 0
    assert data_cfg["candidate_preflight"] is True
    assert data_cfg["candidate_preflight_imports"] is True
    assert data_cfg["candidate_preflight_timeout"] == 45
    assert data_cfg["scm_gpu_wait_timeout"] == 3600
    assert search_cfg["runner"]["solver"]["final_submit"] is False
    assert search_cfg["runner"]["solver"]["export_search_results"] is False
    assert search_cfg["runner"]["logger"]["use_console"] is False
    assert search_cfg["runner"]["solver"]["max_debug_depth"] == 2
    assert search_cfg["runner"]["solver"]["max_debug_time"] == 1200
    assert search_cfg["runner"]["solver"]["fresh_draft_prob"] == pytest.approx(0.2)
    assert search_cfg["runner"]["solver"]["max_wall_time_secs"] == 21600


def test_naturebench_summary_does_not_require_mle_leaderboard(tmp_path):
    from tts_search import eval_utils

    task_dir = tmp_path / "program_ep_0" / "fake-task"
    task_dir.mkdir(parents=True)
    (task_dir / "stat.json").write_text(
        json.dumps(
            {
                "benchmark": "naturebench",
                "task_name": "fake-task",
                "submit_score": 0.123,
                "submit_grade": 1.0,
                "status_count": {"success": 1},
                "total_cost": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary = eval_utils.write_summary_csv(
        tmp_path,
        {
            "fake-task": {
                "benchmark": "naturebench",
                "task_name": "fake-task",
                "data_dir": str(tmp_path / "no-mle-leaderboard-here"),
            }
        },
        expected_task_names=["fake-task"],
    )

    assert (tmp_path / "summary.csv").exists()
    assert summary.loc[0, "Task"] == "fake-task"
    assert summary.loc[0, "score_best@k"] == pytest.approx(0.123)
    assert summary.loc[0, "score_avg@k_medal"] == "N/A"


def test_generic_llm_converts_generation_kwargs_to_plain_containers(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("LOGGING_DIR", str(tmp_path / "logs"))

    from dojo.core.solvers.llm_helpers import generic_llm

    captured_kwargs = {}

    class FakeClient:
        client_content_key = "content"

        def query(self, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return "```python\nprint(1)\n```", {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
            }

    monkeypatch.setattr(generic_llm, "get_client", lambda client_cfg: FakeClient())

    cfg = OmegaConf.create(
        {
            "llm": {
                "client": {
                    "api": "litellm",
                    "model_id": "unit-model",
                    "base_url": "http://127.0.0.1:30002/v1",
                    "use_azure_client": False,
                    "provider": "selfhosted",
                },
                "generation_kwargs": {
                    "temperature": 0.2,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                },
            },
            "system_message_prompt_template": {
                "template": "{{public_system_prompt}}",
                "input_variables": ["public_system_prompt"],
                "partial_variables": {},
            },
            "init_user_message_prompt_template": {
                "template": "{{task_description}}",
                "input_variables": ["task_description"],
                "partial_variables": {},
            },
            "user_message_prompt_template": {
                "template": "",
                "input_variables": [],
                "partial_variables": {},
            },
        }
    )

    llm = generic_llm.GenericLLM(cfg)
    output, _ = llm(
        query_data={
            "public_system_prompt": "",
            "task_description": "write code",
        }
    )

    assert output.startswith("```python")
    assert isinstance(llm.generation_kwargs["extra_body"], dict)
    assert not OmegaConf.is_config(llm.generation_kwargs["extra_body"])
    assert isinstance(captured_kwargs["extra_body"], dict)
    assert captured_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_naturebench_concurrent_attempts_keep_distinct_workspaces(tmp_path):
    base_task = _load_module(
        "naturebench_base_task_concurrent_attempts",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    from dojo.core.tasks.constants import AUX_EVAL_INFO

    barrier = threading.Barrier(2)

    class ConcurrentTask(base_task.NatureBenchTask):
        def _run_solution(self, code: str, *, phase: str, attempt=None) -> dict:
            barrier.wait(timeout=5)
            output_dir = (
                attempt.output_dir if attempt is not None else self._active_output_dir
            )
            return {
                "status_code": 0,
                "status": "success",
                "raw_run_log": code,
                "clear_run_log": code,
                "feedback": code,
                "run_time": 0.01,
                "output_dir": str(output_dir),
            }

        def _post_evaluate(self, output_dir: Path) -> dict:
            return {
                "aggregate_improvement": 0.1,
                "best_aggregate_improvement": 0.1,
                "raw_scores": {},
                "per_instance_improvement": {},
            }

    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = ConcurrentTask(
        {
            "benchmark": "naturebench",
            "task_name": "race-task",
            "higher_is_better": True,
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspace"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
            "candidate_preflight": False,
        }
    )

    def run_candidate(index: int):
        state, result = task.step_task({}, f"print({index})")
        return state, result[AUX_EVAL_INFO]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_candidate, (1, 2)))

    output_dirs = {payload["output_dir"] for _, payload in results}
    attempt_indices = {state["naturebench_attempt_index"] for state, _ in results}
    assert len(output_dirs) == 2
    assert attempt_indices == {1, 2}


def test_naturebench_docker_timeout_forcibly_removes_named_container(
    monkeypatch,
    tmp_path,
):
    base_task = _load_module(
        "naturebench_base_task_docker_timeout",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs["timeout"],
                output="partial stdout",
                stderr="partial stderr",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(base_task.subprocess, "run", fake_run)
    data_dir = tmp_path / "problem" / "data"
    workspace = tmp_path / "workspace"
    output_dir = workspace / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    task = base_task.NatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "docker-timeout-task",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspaces"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "docker",
            "execution_timeout": 10,
        }
    )

    result = task._run_docker_solution(
        "print('hello')",
        workspace=workspace,
        output_dir=output_dir,
        attempt_index=7,
    )

    run_command = calls[0]
    container_name = run_command[run_command.index("--name") + 1]
    assert "--network" not in run_command
    assert "host.docker.internal:host-gateway" in run_command
    assert "EVAL_SERVICE_URL=http://host.docker.internal:8321" in run_command
    assert result["status_code"] == 504
    assert result["status"] == "timeout"
    assert calls[1] == ["docker", "rm", "-f", container_name]


def test_naturebench_remote_tar_command_quotes_workspace_component():
    base_task = _load_module(
        "naturebench_base_task_remote_quote",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    remote_workspace = "/remote/work dir;touch /tmp/should-not-run"

    remote_command = base_task.NatureBenchTask._remote_tar_extract_command(
        remote_workspace
    )

    assert remote_command == ("tar -C " + shlex.quote(remote_workspace) + " -xf -")


def test_naturebench_local_candidate_env_does_not_inherit_secrets(
    monkeypatch,
    tmp_path,
):
    base_task = _load_module(
        "naturebench_base_task_candidate_env",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-copy")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-copy")
    data_dir = tmp_path / "problem" / "data"
    data_dir.mkdir(parents=True)
    task = base_task.NatureBenchTask(
        {
            "benchmark": "naturebench",
            "task_name": "candidate-env-task",
            "data_dir": str(data_dir),
            "problem_dir": str(data_dir.parent),
            "task_dir": str(tmp_path),
            "workspace_root": str(tmp_path / "workspaces"),
            "eval_service_url": "http://127.0.0.1:8321",
            "batch_name": "unit-batch",
            "execution_mode": "local",
        }
    )

    env = task._solution_env(tmp_path / "output")

    assert env["DATA_DIR"] == str(data_dir)
    assert "PATH" in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "OPENAI_API_KEY" not in env


def test_runner_config_redaction_removes_nested_api_keys():
    runner = _load_module(
        "airaevo_runner_redaction",
        REPO_ROOT / "third_party" / "aira-evo" / "examples" / "mle_bench" / "runner.py",
    )
    payload = {
        "llm": {"api_key": "top-secret", "model_id": "unit-model"},
        "nested": [{"token": "also-secret", "safe": 3}],
        "overrides": ["litellm.model_list.0.litellm_params.api_key=cli-secret"],
    }

    redacted = runner._redact_sensitive_config(payload)

    assert redacted["llm"]["api_key"] is None
    assert redacted["nested"][0]["token"] is None
    assert "top-secret" not in yaml.safe_dump(redacted)
    assert "also-secret" not in yaml.safe_dump(redacted)
    assert "cli-secret" not in yaml.safe_dump(redacted)
