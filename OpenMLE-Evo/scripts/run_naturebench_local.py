#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_SET = REPO_ROOT / "benchmarks" / "naturebench_local_quick" / "tasks.txt"
DEFAULT_QUICK_TASK = "s42256-023-00611-x"
LOCAL_PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "h5py": "h5py",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "anndata": "anndata",
    "scanpy": "scanpy",
    "umap-learn": "umap",
    "loompy": "loompy",
    "networkx": "networkx",
    "igraph": "igraph",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}
REQUIRED_LOCAL_PACKAGES = {"numpy", "scipy", "pandas", "scikit-learn"}
PACKAGE_INSPECTION_MARKER = "__NATUREBENCH_LOCAL_PACKAGES__="


def _read_task_set(path: Path) -> list[str]:
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tasks:
        raise ValueError(f"NatureBench task set is empty: {path}")
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"NatureBench task set contains duplicate ids: {path}")
    return tasks


def _resolve_naturebench_repo(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    env_value = os.environ.get("NATUREBENCH_REPO")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend([REPO_ROOT.parent / "NatureBench", Path.cwd() / "NatureBench"])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "run_naturebench.py").is_file() and (
            resolved / "eval_service.py"
        ).is_file():
            return resolved
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the NatureBench repository. Pass --naturebench-repo "
        f"or set NATUREBENCH_REPO. Checked: {rendered}"
    )


def _resolve_runtime_python(
    *,
    local_python: str | None,
    conda_env: str | None,
    conda_executable: str,
) -> Path:
    if local_python:
        path = Path(os.path.abspath(Path(local_python).expanduser()))
        if not path.is_file():
            raise FileNotFoundError(f"Local Python does not exist: {path}")
        return path
    if not conda_env:
        return Path(sys.executable).absolute()

    conda_environment = os.environ.copy()
    outer_virtualenv = conda_environment.pop("VIRTUAL_ENV", None)
    if outer_virtualenv:
        virtualenv_bin = os.path.normpath(str(Path(outer_virtualenv) / "bin"))
        path_entries = conda_environment.get("PATH", "").split(os.pathsep)
        conda_environment["PATH"] = os.pathsep.join(
            entry for entry in path_entries if os.path.normpath(entry) != virtualenv_bin
        )
    process = subprocess.run(
        [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-c",
            "import sys; print(sys.executable)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=conda_environment,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(
            f"Conda environment {conda_env!r} is unavailable: {detail}. "
            "Create it with: conda env create -f environments/naturebench-local.yml"
        )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"Could not resolve Python for Conda environment {conda_env!r}"
        )
    path = Path(os.path.abspath(Path(lines[-1]).expanduser()))
    if not path.is_file():
        raise FileNotFoundError(f"Resolved Conda Python does not exist: {path}")
    return path


def _health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def _service_is_healthy(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(host, port), timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return response.status == 200 and payload.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _eval_service_supports_control_token(naturebench_repo: Path) -> bool:
    source = (naturebench_repo / "eval_service.py").read_text(
        encoding="utf-8", errors="replace"
    )
    return "--control-token-file" in source


def _resolve_eval_control_token_file(
    *,
    naturebench_repo: Path,
    output_dir: Path,
    configured: str | None,
    reusing_service: bool,
) -> Path | None:
    if not _eval_service_supports_control_token(naturebench_repo):
        if configured:
            raise RuntimeError(
                "--eval-control-token-file was provided, but this NatureBench "
                "eval service does not support control tokens"
            )
        return None

    if configured:
        token_path = Path(configured).expanduser().resolve()
    elif reusing_service:
        token_path = naturebench_repo / "eval_logs" / "eval_control_token"
    else:
        token_path = output_dir / ".eval_service" / "control_token"

    if token_path.exists() and not token_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(
            f"NatureBench eval control token file is empty: {token_path}"
        )
    if reusing_service and not token_path.is_file():
        raise RuntimeError(
            "A current NatureBench eval service is already running, but its "
            f"control token file is unavailable: {token_path}. Pass the file "
            "used to start that service with --eval-control-token-file, or use "
            "a free --eval-port."
        )
    if not reusing_service:
        token_path.parent.mkdir(parents=True, exist_ok=True)
    return token_path


def _find_free_loopback_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _build_ssh_model_tunnel_command(
    *,
    ssh_host: str,
    local_host: str,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> list[str]:
    forward = f"{local_host}:{local_port}:{remote_host}:{remote_port}"
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-N",
        "-L",
        forward,
        ssh_host,
    ]


def _fetch_model_ids(
    base_url: str,
    *,
    api_key: str,
    timeout: float = 2,
) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(model["id"]) for model in payload.get("data", []) if model.get("id")]


def _select_model_id(requested: str | None, available: list[str]) -> str:
    if requested:
        if requested not in available:
            raise RuntimeError(
                f"Requested model {requested!r} is not served. Available: {available}"
            )
        return requested
    if len(available) == 1:
        return available[0]
    raise RuntimeError(
        "The model service must expose exactly one model when --model-id is "
        f"omitted. Available: {available}"
    )


def _wait_for_model_api(
    process: subprocess.Popen[str],
    *,
    base_url: str,
    api_key: str,
    requested_model_id: str | None,
    timeout: float,
    log_path: Path,
) -> str:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            available = _fetch_model_ids(base_url, api_key=api_key)
            if available:
                return _select_model_id(requested_model_id, available)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if process.poll() is not None:
            break
        time.sleep(0.25)
    detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    raise RuntimeError(
        f"Model API did not become healthy at {base_url}. "
        f"Last error: {last_error}. SSH log tail:\n{detail}"
    )


def _model_hydra_overrides(model_id: str, base_url: str) -> list[str]:
    return [
        f"litellm.model_list.0.model_name={model_id}",
        f"litellm.model_list.0.litellm_params.model=openai/{model_id}",
        f"litellm.model_list.0.litellm_params.base_url={base_url.rstrip('/')}",
    ]


def _wait_for_service(
    process: subprocess.Popen[str],
    *,
    host: str,
    port: int,
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _service_is_healthy(host, port):
            return
        if process.poll() is not None:
            break
        time.sleep(0.25)
    detail = ""
    if log_path.is_file():
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    raise RuntimeError(
        f"NatureBench eval service did not become healthy at "
        f"{_health_url(host, port)}. Log tail:\n{detail}"
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()


def _verify_task_packages(data_root: Path, tasks: list[str]) -> None:
    tasks_root = data_root / "tasks" if (data_root / "tasks").is_dir() else data_root
    required = (
        Path("metadata.json"),
        Path("problem") / "data",
        Path("evaluation") / "evaluator.py",
    )
    missing = [
        str(tasks_root / task / relative)
        for task in tasks
        for relative in required
        if not (tasks_root / task / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "NatureBench task packages are incomplete:\n- " + "\n- ".join(missing)
        )


def _inspect_runtime_packages(
    runtime_python: Path,
    *,
    environment: dict[str, str],
    package_imports: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    imports = package_imports if package_imports is not None else LOCAL_PACKAGE_IMPORTS
    script = (
        "import importlib, json\n"
        f"imports = {json.dumps(imports, sort_keys=True)}\n"
        "errors = {}\n"
        "for package, module in imports.items():\n"
        "    try:\n"
        "        importlib.import_module(module)\n"
        "    except Exception as exc:\n"
        "        errors[package] = f'{type(exc).__name__}: {exc}'\n"
        f"print({PACKAGE_INSPECTION_MARKER!r} + json.dumps(errors, sort_keys=True))\n"
    )
    process = subprocess.run(
        [str(runtime_python), "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if process.returncode != 0:
        detail = ((process.stderr or "") + (process.stdout or ""))[-2000:]
        raise RuntimeError(f"Failed to inspect local runtime packages: {detail}")
    payload = next(
        (
            line[len(PACKAGE_INSPECTION_MARKER) :]
            for line in reversed(process.stdout.splitlines())
            if line.startswith(PACKAGE_INSPECTION_MARKER)
        ),
        None,
    )
    try:
        unavailable = dict(json.loads(payload))
    except (IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Local runtime package inspection returned invalid output: "
            f"{process.stdout[-1000:]}"
        ) from exc
    available = [package for package in imports if package not in unavailable]
    return available, unavailable


def _hydra_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AIRA-Evo NatureBench tasks entirely on the local machine."
    )
    parser.add_argument("--naturebench-repo", default=None)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / ".naturebench" / "data"))
    parser.add_argument("--task-set", default=str(DEFAULT_TASK_SET))
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dataset-revision", default=None)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--local-python", default=None)
    runtime.add_argument("--conda-env", default="naturebench-local")
    parser.add_argument("--conda-executable", default="conda")
    parser.add_argument("--eval-host", default="127.0.0.1")
    parser.add_argument("--eval-port", type=int, default=8321)
    parser.add_argument("--eval-start-timeout", type=float, default=30)
    parser.add_argument(
        "--eval-control-token-file",
        default=None,
        help=(
            "Control-token file used by a current NatureBench eval service. "
            "For a service started by this launcher, a private file under the "
            "experiment output is used automatically."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model ID exposed by the OpenAI-compatible endpoint (the SGLang --served-model-name value).",
    )
    model_endpoint = parser.add_mutually_exclusive_group()
    model_endpoint.add_argument(
        "--model-base-url",
        default=None,
        help="Direct OpenAI-compatible /v1 URL for a hosted API or an already-running local/remote service.",
    )
    model_endpoint.add_argument(
        "--model-ssh-host",
        default=None,
        help="SSH host for tunneling to an already-running remote SGLang service; this launcher does not deploy it.",
    )
    parser.add_argument(
        "--model-ssh-port",
        type=int,
        default=None,
        help="API port of the already-running SGLang service on --model-ssh-host.",
    )
    parser.add_argument(
        "--model-ssh-remote-host",
        default="127.0.0.1",
        help="Host visible from the SSH server where SGLang is listening.",
    )
    parser.add_argument("--model-ssh-local-host", default="127.0.0.1")
    parser.add_argument("--model-ssh-local-port", type=int, default=None)
    parser.add_argument("--model-ssh-start-timeout", type=float, default=30)
    parser.add_argument("--config-name", default="experiment/naturebench_local_quick")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one task and one search node for an end-to-end check.",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides after '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.model_ssh_host and args.model_ssh_port is None:
        raise ValueError("--model-ssh-port is required with --model-ssh-host")
    naturebench_repo = _resolve_naturebench_repo(args.naturebench_repo)
    runtime_python = _resolve_runtime_python(
        local_python=args.local_python,
        conda_env=args.conda_env,
        conda_executable=args.conda_executable,
    )

    if args.task:
        tasks = list(dict.fromkeys(args.task))
    elif args.smoke:
        tasks = [DEFAULT_QUICK_TASK]
    else:
        tasks = _read_task_set(Path(args.task_set).expanduser().resolve())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"naturebench_local_quick_{stamp}"
    output_dir = (
        Path(args.output_dir or REPO_ROOT / "output" / experiment_name)
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "task_set.txt"
    selection_path.write_text("\n".join(tasks) + "\n", encoding="utf-8")

    data_root = Path(args.data_dir).expanduser().resolve()
    if not args.skip_download:
        download_command = [
            str(runtime_python),
            str(naturebench_repo / "run_naturebench.py"),
            "--data-dir",
            str(data_root),
            "--tasks",
            str(selection_path),
            "--download-only",
        ]
        if args.dataset_revision:
            download_command.extend(["--dataset-revision", args.dataset_revision])
        print(
            f"Downloading or refreshing {len(tasks)} NatureBench task(s)...", flush=True
        )
        subprocess.run(download_command, cwd=naturebench_repo, check=True)
    _verify_task_packages(data_root, tasks)

    runtime_env = os.environ.copy()
    cache_root = data_root.parent / "cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    runtime_env.setdefault("XDG_CACHE_HOME", str(cache_root))
    runtime_env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    available_packages, unavailable_packages = _inspect_runtime_packages(
        runtime_python,
        environment=runtime_env,
    )
    missing_required = sorted(REQUIRED_LOCAL_PACKAGES & unavailable_packages.keys())
    if missing_required:
        details = "; ".join(
            f"{package}: {unavailable_packages[package]}"
            for package in missing_required
        )
        raise RuntimeError(
            "Local NatureBench runtime is missing required packages: "
            f"{details}. Recreate it with environments/naturebench-local.yml."
        )
    print(
        "Available local candidate packages: " + ", ".join(available_packages),
        flush=True,
    )
    if unavailable_packages:
        print(
            "Omitting unavailable optional packages from the agent prompt: "
            + ", ".join(sorted(unavailable_packages)),
            flush=True,
        )

    child_env = runtime_env.copy()
    if not child_env.get("PRIMARY_KEY") and args.model_ssh_host:
        child_env["PRIMARY_KEY"] = "EMPTY"
    if not child_env.get("PRIMARY_KEY"):
        raise RuntimeError(
            "Missing model API key. Set PRIMARY_KEY in the "
            "environment. SSH-tunneled models without auth use EMPTY automatically."
        )

    model_tunnel_process: subprocess.Popen[str] | None = None
    model_tunnel_log_handle = None
    model_base_url = args.model_base_url
    model_id = args.model_id
    eval_process: subprocess.Popen[str] | None = None
    eval_log_handle = None
    try:
        if args.model_ssh_host:
            local_port = args.model_ssh_local_port or _find_free_loopback_port(
                args.model_ssh_local_host
            )
            model_base_url = f"http://{args.model_ssh_local_host}:{local_port}/v1"
            tunnel_log_path = output_dir / "model_ssh_tunnel.log"
            model_tunnel_log_handle = tunnel_log_path.open("a", encoding="utf-8")
            tunnel_command = _build_ssh_model_tunnel_command(
                ssh_host=args.model_ssh_host,
                local_host=args.model_ssh_local_host,
                local_port=local_port,
                remote_host=args.model_ssh_remote_host,
                remote_port=args.model_ssh_port,
            )
            model_tunnel_process = subprocess.Popen(
                tunnel_command,
                stdout=model_tunnel_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name == "posix",
            )
            model_id = _wait_for_model_api(
                model_tunnel_process,
                base_url=model_base_url,
                api_key=child_env["PRIMARY_KEY"],
                requested_model_id=model_id,
                timeout=args.model_ssh_start_timeout,
                log_path=tunnel_log_path,
            )
            print(
                f"Connected model {model_id} through SSH tunnel "
                f"{args.model_ssh_host}:{args.model_ssh_port} -> {model_base_url}",
                flush=True,
            )
        elif model_base_url:
            available = _fetch_model_ids(
                model_base_url,
                api_key=child_env["PRIMARY_KEY"],
                timeout=args.model_ssh_start_timeout,
            )
            model_id = _select_model_id(model_id, available)
            print(f"Using model {model_id} at {model_base_url}", flush=True)

        reusing_eval_service = _service_is_healthy(args.eval_host, args.eval_port)
        eval_control_token_file = _resolve_eval_control_token_file(
            naturebench_repo=naturebench_repo,
            output_dir=output_dir,
            configured=args.eval_control_token_file,
            reusing_service=reusing_eval_service,
        )
        if reusing_eval_service:
            print(
                f"Reusing healthy NatureBench eval service at "
                f"http://{args.eval_host}:{args.eval_port}",
                flush=True,
            )
        else:
            eval_log_path = output_dir / "eval_service.log"
            eval_log_handle = eval_log_path.open("a", encoding="utf-8")
            eval_command = [
                str(runtime_python),
                str(naturebench_repo / "eval_service.py"),
                "--host",
                args.eval_host,
                "--port",
                str(args.eval_port),
            ]
            if eval_control_token_file is not None:
                eval_command.extend(
                    ["--control-token-file", str(eval_control_token_file)]
                )
            eval_process = subprocess.Popen(
                eval_command,
                cwd=naturebench_repo,
                stdout=eval_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=runtime_env,
                start_new_session=os.name == "posix",
            )
            _wait_for_service(
                eval_process,
                host=args.eval_host,
                port=args.eval_port,
                timeout=args.eval_start_timeout,
                log_path=eval_log_path,
            )
            print(
                f"Started local NatureBench eval service at "
                f"http://{args.eval_host}:{args.eval_port}",
                flush=True,
            )

        if eval_control_token_file is not None:
            try:
                control_token = eval_control_token_file.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as exc:
                raise RuntimeError(
                    "NatureBench eval service became healthy without a readable "
                    f"control token file: {eval_control_token_file}"
                ) from exc
            if not control_token:
                raise RuntimeError(
                    "NatureBench eval service control token file is empty: "
                    f"{eval_control_token_file}"
                )
            child_env["NATUREBENCH_EVAL_CONTROL_TOKEN_FILE"] = str(
                eval_control_token_file
            )

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_naturebench.py"),
            "--config-name",
            args.config_name,
            f"experiment_name={experiment_name}",
            f"output_dir={output_dir}",
            f"data.naturebench_root={data_root}",
            "data.tasks_root=tasks",
            f"data.task_list={_hydra_list(tasks)}",
            f"data.eval_service_url=http://{args.eval_host}:{args.eval_port}",
            f"data.batch_name={experiment_name}",
            "data.execution_mode=local",
            f"data.workspace_root={output_dir / 'workspaces'}",
            f"data.local_python={runtime_python}",
            "data.local_conda_env=null",
            "search.runner.solver.available_packages="
            f"{_hydra_list(available_packages)}",
        ]
        if model_base_url:
            if model_id is None:
                raise RuntimeError("Could not resolve model id")
            command.extend(_model_hydra_overrides(model_id, model_base_url))
        if args.smoke:
            command.extend(
                [
                    "max_steps=2",
                    "time_budget=1800",
                    "model_plus_sandbox_time_budget=1800",
                    "llm_concurrency=1",
                    "search.runner.task_concurrency=1",
                    "search.runner.solver.step_limit=2",
                    "search.runner.solver.num_generations=1",
                    "search.runner.solver.individuals_per_generation=1",
                    "search.runner.solver.execution_timeout=600",
                    "data.execution_timeout=600",
                ]
            )
        command.extend(value for value in args.hydra_overrides if value != "--")
        print(
            f"Running {len(tasks)} local NatureBench task(s); output: {output_dir}",
            flush=True,
        )
        subprocess.run(command, cwd=REPO_ROOT, env=child_env, check=True)
    finally:
        if eval_process is not None:
            _terminate_process_group(eval_process)
        if eval_log_handle is not None:
            eval_log_handle.close()
        if model_tunnel_process is not None:
            _terminate_process_group(model_tunnel_process)
        if model_tunnel_log_handle is not None:
            model_tunnel_log_handle.close()


if __name__ == "__main__":
    main()
