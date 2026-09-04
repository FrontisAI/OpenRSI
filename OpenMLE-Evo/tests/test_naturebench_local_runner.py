from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _base_task_module():
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "aira-evo" / "src"))
    return _load_module(
        "naturebench_base_task_local",
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "nature_bench"
        / "base_task.py",
    )


def _task_config(tmp_path: Path, **overrides):
    task_dir = tmp_path / "task"
    data_dir = task_dir / "problem" / "data"
    data_dir.mkdir(parents=True)
    config = {
        "task_name": "local-test",
        "data_dir": str(data_dir),
        "problem_dir": str(data_dir.parent),
        "task_dir": str(task_dir),
        "execution_mode": "local",
        "eval_service_url": "http://127.0.0.1:8321",
        "workspace_root": str(tmp_path / "workspaces"),
        "candidate_preflight": True,
        "candidate_preflight_imports": True,
        "execution_timeout": 5,
    }
    config.update(overrides)
    return config


def test_local_python_command_prefers_explicit_interpreter(tmp_path):
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_python="/opt/naturebench/bin/python",
            local_conda_env="ignored-env",
        )
    )

    assert task._local_python_command("-c", "print(1)") == [
        "/opt/naturebench/bin/python",
        "-c",
        "print(1)",
    ]


def test_local_python_command_supports_conda_environment(tmp_path):
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_conda_env="naturebench-local",
            local_conda_executable="/opt/conda/bin/conda",
        )
    )

    assert task._local_python_command("run.py") == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "naturebench-local",
        "python",
        "run.py",
    ]


def test_local_import_preflight_uses_candidate_runtime(tmp_path):
    task = _base_task_module().NatureBenchTask(
        _task_config(tmp_path, local_python=sys.executable)
    )

    passed = task._candidate_preflight("import json\nprint(json.dumps({}))")
    blocked = task._candidate_preflight("import definitely_missing_naturebench_module")

    assert passed["ok"] is True
    assert blocked["ok"] is False
    assert blocked["category"] == "unavailable_import"
    assert "definitely_missing_naturebench_module" in blocked["feedback"]


def test_local_solution_uses_configured_python(tmp_path):
    task = _base_task_module().NatureBenchTask(
        _task_config(tmp_path, local_python=sys.executable)
    )
    attempt = task._start_attempt(phase="validation")

    result = task._run_local_solution(
        "import os, sys\nprint(sys.executable)\nprint(os.environ['DATA_DIR'])",
        workspace=attempt.workspace,
        output_dir=attempt.output_dir,
    )

    assert result["status"] == "success"
    assert sys.executable in result["raw_run_log"]
    assert task.validation_data_dir in result["raw_run_log"]


def test_current_eval_contract_adds_control_header_only_to_control_endpoints(
    monkeypatch,
    tmp_path,
):
    base_task = _base_task_module()
    control_token_file = tmp_path / "eval-control-token"
    control_token_file.write_text("private-control-token\n", encoding="utf-8")
    task = base_task.NatureBenchTask(
        _task_config(
            tmp_path,
            eval_control_token_file=str(control_token_file),
        )
    )
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status":"ok"}'

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(base_task.urllib.request, "urlopen", fake_urlopen)

    task._post_json("register", {"task_name": "local-test"}, timeout=10)
    task._post_json("evaluate", {"eval_token": "opaque"}, timeout=10)

    register_headers = {
        key.lower(): value for key, value in requests[0][0].header_items()
    }
    evaluate_headers = {
        key.lower(): value for key, value in requests[1][0].header_items()
    }
    assert register_headers["x-naturebench-control-token"] == "private-control-token"
    assert "x-naturebench-control-token" not in evaluate_headers


def test_local_timeout_terminates_process_group(tmp_path):
    if os.name != "posix":
        return
    task = _base_task_module().NatureBenchTask(
        _task_config(
            tmp_path,
            local_python=sys.executable,
            execution_timeout=1,
            local_terminate_grace_seconds=0.1,
        )
    )
    attempt = task._start_attempt(phase="validation")
    child_pid_path = attempt.workspace / "child.pid"
    child_term_path = attempt.workspace / "child.terminated"
    child_code = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(pathlib.Path(sys.argv[1]).write_text('terminated'), sys.exit(0))); "
        "time.sleep(60)"
    )

    result = task._run_local_solution(
        "import pathlib, subprocess, sys, time\n"
        f"pid_path = pathlib.Path({str(child_pid_path)!r})\n"
        f"term_path = {str(child_term_path)!r}\n"
        f"child_code = {child_code!r}\n"
        "child = subprocess.Popen([sys.executable, '-c', child_code, term_path])\n"
        "pid_path.write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        workspace=attempt.workspace,
        output_dir=attempt.output_dir,
    )

    assert result["status"] == "timeout"
    assert child_pid_path.is_file()
    assert child_term_path.read_text(encoding="utf-8") == "terminated"


def test_local_launcher_helpers_validate_task_packages(tmp_path):
    launcher = _load_module(
        "naturebench_local_launcher",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    task_id = "s42256-023-00611-x"
    task_root = tmp_path / "tasks" / task_id
    (task_root / "problem" / "data").mkdir(parents=True)
    (task_root / "evaluation").mkdir()
    (task_root / "metadata.json").write_text("{}", encoding="utf-8")
    (task_root / "evaluation" / "evaluator.py").write_text("", encoding="utf-8")
    task_set = tmp_path / "tasks.txt"
    task_set.write_text(f"# local\n{task_id}\n", encoding="utf-8")

    assert launcher._read_task_set(task_set) == [task_id]
    assert (
        launcher._resolve_runtime_python(
            local_python=sys.executable,
            conda_env=None,
            conda_executable="conda",
        )
        == Path(sys.executable).absolute()
    )
    launcher._verify_task_packages(tmp_path, [task_id])
    assert json.loads(launcher._hydra_list([task_id])) == [task_id]


def test_local_launcher_selects_control_token_file_for_current_service(tmp_path):
    launcher = _load_module(
        "naturebench_local_launcher_control_token",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    naturebench_repo = tmp_path / "NatureBench"
    naturebench_repo.mkdir()
    (naturebench_repo / "eval_service.py").write_text(
        'parser.add_argument("--control-token-file")\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    new_service_token = launcher._resolve_eval_control_token_file(
        naturebench_repo=naturebench_repo,
        output_dir=output_dir,
        configured=None,
        reusing_service=False,
    )
    assert new_service_token == output_dir / ".eval_service" / "control_token"

    default_token = naturebench_repo / "eval_logs" / "eval_control_token"
    default_token.parent.mkdir()
    default_token.write_text("existing-token\n", encoding="utf-8")
    reused_token = launcher._resolve_eval_control_token_file(
        naturebench_repo=naturebench_repo,
        output_dir=output_dir,
        configured=None,
        reusing_service=True,
    )
    assert reused_token == default_token


def test_local_launcher_rejects_current_service_without_readable_control_token(
    tmp_path,
):
    launcher = _load_module(
        "naturebench_local_launcher_missing_control_token",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    naturebench_repo = tmp_path / "NatureBench"
    naturebench_repo.mkdir()
    (naturebench_repo / "eval_service.py").write_text(
        'parser.add_argument("--control-token-file")\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="control token"):
        launcher._resolve_eval_control_token_file(
            naturebench_repo=naturebench_repo,
            output_dir=tmp_path / "output",
            configured=None,
            reusing_service=True,
        )


def test_local_quick_example_targets_only_counterfactual_task():
    launcher = _load_module(
        "naturebench_local_launcher_quick_task",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    assert launcher.DEFAULT_QUICK_TASK == "s42256-023-00611-x"
    assert launcher._read_task_set(launcher.DEFAULT_TASK_SET) == [
        launcher.DEFAULT_QUICK_TASK
    ]
    assert launcher.DEFAULT_TASK_SET.parent.name == "naturebench_local_quick"


def test_runtime_python_resolution_ignores_outer_virtualenv(monkeypatch, tmp_path):
    launcher = _load_module(
        "naturebench_local_launcher_conda",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )
    python_path = tmp_path / "conda" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    captured_environment = None

    def fake_run(*args, **kwargs):
        nonlocal captured_environment
        captured_environment = kwargs["env"]
        return subprocess.CompletedProcess(args[0], 0, f"{python_path}\n", "")

    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/project-venv")
    monkeypatch.setenv("PATH", "/tmp/project-venv/bin:/usr/bin:/bin")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    resolved = launcher._resolve_runtime_python(
        local_python=None,
        conda_env="naturebench-local",
        conda_executable="conda",
    )

    assert resolved == python_path
    assert captured_environment is not None
    assert "VIRTUAL_ENV" not in captured_environment
    assert captured_environment["PATH"] == "/usr/bin:/bin"


def test_local_launcher_inspects_real_imports(tmp_path):
    launcher = _load_module(
        "naturebench_local_launcher_imports",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    available, unavailable = launcher._inspect_runtime_packages(
        Path(sys.executable),
        environment=os.environ.copy(),
        package_imports={
            "json": "json",
            "missing": "definitely_missing_naturebench_module",
        },
    )

    assert available == ["json"]
    assert "ModuleNotFoundError" in unavailable["missing"]


def test_local_launcher_builds_ssh_model_tunnel_and_hydra_overrides():
    launcher = _load_module(
        "naturebench_local_launcher_ssh",
        REPO_ROOT / "scripts" / "run_naturebench_local.py",
    )

    command = launcher._build_ssh_model_tunnel_command(
        ssh_host="model-host",
        local_host="127.0.0.1",
        local_port=31010,
        remote_host="127.0.0.1",
        remote_port=30010,
    )

    assert command[-4:] == [
        "-N",
        "-L",
        "127.0.0.1:31010:127.0.0.1:30010",
        "model-host",
    ]
    assert "ExitOnForwardFailure=yes" in command
    assert launcher._model_hydra_overrides(
        "served-model", "http://127.0.0.1:31010/v1/"
    ) == [
        "litellm.model_list.0.model_name=served-model",
        "litellm.model_list.0.litellm_params.model=openai/served-model",
        "litellm.model_list.0.litellm_params.base_url=http://127.0.0.1:31010/v1",
    ]


def test_local_adapter_against_current_naturebench_service(tmp_path):
    naturebench_repo_value = os.environ.get("NATUREBENCH_REPO")
    if not naturebench_repo_value:
        pytest.skip("set NATUREBENCH_REPO to run the external contract test")
    naturebench_repo = Path(naturebench_repo_value).expanduser().resolve()
    eval_service = naturebench_repo / "eval_service.py"
    if not eval_service.is_file():
        pytest.skip(f"NatureBench eval service is unavailable: {eval_service}")

    task_dir = tmp_path / "task-package"
    data_dir = task_dir / "problem" / "data"
    evaluation_dir = task_dir / "evaluation"
    data_dir.mkdir(parents=True)
    evaluation_dir.mkdir()
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "performance_entries": [
                    {
                        "dataset_name": "fake-instance",
                        "metrics": [
                            {
                                "name": "accuracy",
                                "is_primary": True,
                                "metric_direction": "higher_is_better",
                                "sota_score": 0.5,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "evaluator.py").write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "def run_evaluation():\n"
        "    score_path = Path(os.environ['OUTPUT_DIR']) / 'score.json'\n"
        "    score = json.loads(score_path.read_text())['score']\n"
        "    return {'fake-instance': {'accuracy': score}}\n",
        encoding="utf-8",
    )
    control_token_file = tmp_path / "control-token"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    service = subprocess.Popen(
        [
            sys.executable,
            str(eval_service),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--control-token-file",
            str(control_token_file),
        ],
        cwd=naturebench_repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=0.5
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if service.poll() is not None:
                break
            time.sleep(0.05)
        else:
            pytest.fail("current NatureBench eval service did not become healthy")
        assert service.poll() is None

        task = _base_task_module().NatureBenchTask(
            _task_config(
                tmp_path,
                task_name="current-contract-test",
                task_dir=str(task_dir),
                data_dir=str(data_dir),
                problem_dir=str(data_dir.parent),
                eval_service_url=f"http://127.0.0.1:{port}",
                eval_control_token_file=str(control_token_file),
                local_python=sys.executable,
                batch_name="current-contract-batch",
            ),
            time_budget=60,
        )
        candidate = (
            "import json, os\n"
            "from pathlib import Path\n"
            "output = Path(os.environ['OUTPUT_DIR'])\n"
            "output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'score.json').write_text(json.dumps({'score': SCORE}))\n"
        )
        first = task.evaluate_code(
            candidate.replace("SCORE", "0.8"), phase="validation"
        )
        second = task.evaluate_code(
            candidate.replace("SCORE", "0.6"), phase="validation"
        )

        assert first["aggregate_improvement"] == pytest.approx(0.6)
        assert second["aggregate_improvement"] == pytest.approx(0.2)
        assert second["best_aggregate_improvement"] == pytest.approx(0.6)
        assert first["attempt"] == 1
        assert second["attempt"] == 2
        assert Path(first["output_dir"]).parent.name == "workspace"
        assert Path(second["output_dir"]).parent.name == "workspace"
        assert first["output_dir"] != second["output_dir"]
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait(timeout=5)
