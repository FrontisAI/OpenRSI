from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_write_json


FORWARDED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_COMPAT_API_KEY",
    "ANTHROPIC_MODEL",
    "KAGGLE_CONFIG_DIR",
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "MODEL",
    "OPENMLE_BUILD_LLM_API_KEY",
    "OPENMLE_BUILD_LLM_BASE_URL",
    "OPENMLE_BUILD_LLM_MODEL",
    "OPENMLE_EVAL_LLM_API_KEY",
    "OPENMLE_EVAL_LLM_BASE_URL",
    "OPENMLE_EVAL_LLM_MODEL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
)


@dataclass(frozen=True)
class TaskProcessOutcome:
    ok: bool
    result: Any = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


def _load_result_envelope(path: Path) -> dict[str, Any]:
    import json

    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Task result must be a JSON object")
    return envelope


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()


def _container_command(
    request_path: Path,
    result_path: Path,
    operation: str,
    readonly_paths: Iterable[Path],
    writable_paths: Iterable[Path],
) -> list[str]:
    runtime = shutil.which("docker") or shutil.which("podman")
    image = os.environ.get("OPENMLE_GYM_ISOLATED_IMAGE")
    if runtime is None:
        raise RuntimeError("isolated execution requires docker or podman")
    if not image:
        raise RuntimeError("isolated execution requires OPENMLE_GYM_ISOLATED_IMAGE")

    uid = os.getuid() or 65534
    gid = os.getgid() or 65534
    command = [
        runtime,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--pids-limit",
        "128",
        "--memory",
        os.environ.get("OPENMLE_GYM_MEMORY_LIMIT", "8g"),
        "--cpus",
        os.environ.get("OPENMLE_GYM_CPU_LIMIT", "2"),
        "--user",
        f"{uid}:{gid}",
        "--workdir",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
    ]
    mounted: set[tuple[str, str]] = set()

    def mount(path: Path, mode: str) -> None:
        resolved = path.resolve()
        key = (str(resolved), mode)
        if key in mounted:
            return
        mounted.add(key)
        command.extend(["--volume", f"{resolved}:{resolved}:{mode}"])

    mount(request_path.parent, "rw")
    for path in readonly_paths:
        mount(path, "ro")
    for path in writable_paths:
        mount(path, "rw")
    forwarded_env = () if operation in {"prepare", "metric"} else FORWARDED_ENV
    for name in forwarded_env:
        if os.environ.get(name):
            command.extend(["--env", name])
    command.extend(
        [
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            image,
            "python",
            "-m",
            "openmle_gym.task_worker",
            operation,
            str(request_path),
            str(result_path),
        ]
    )
    return command


def run_task_process(
    operation: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    execution_mode: str = "process",
    readonly_paths: Iterable[str | Path] = (),
    writable_paths: Iterable[str | Path] = (),
) -> TaskProcessOutcome:
    """Run one task in a disposable process and return a non-throwing outcome."""
    if execution_mode not in {"process", "isolated"}:
        return TaskProcessOutcome(ok=False, error=f"Unknown execution mode: {execution_mode}")

    source_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="openmle-task-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        request_path = temporary_root / "request.json"
        result_path = temporary_root / "result.json"
        atomic_write_json(request_path, payload)

        try:
            if execution_mode == "isolated":
                command = _container_command(
                    request_path,
                    result_path,
                    operation,
                    [Path(path) for path in readonly_paths],
                    [Path(path) for path in writable_paths],
                )
            else:
                command = [
                    sys.executable,
                    "-m",
                    "openmle_gym.task_worker",
                    operation,
                    str(request_path),
                    str(result_path),
                ]
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                stdout, stderr = process.communicate()
                return TaskProcessOutcome(
                    ok=False,
                    error=f"Task timed out after {timeout:g}s",
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"Failed to start task process: {type(exc).__name__}: {exc}",
            )

        if process.returncode != 0:
            return TaskProcessOutcome(
                ok=False,
                error=f"Task process exited with code {process.returncode}",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        if not result_path.is_file():
            return TaskProcessOutcome(
                ok=False,
                error="Task process did not produce a result file",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        try:
            envelope = _load_result_envelope(result_path)
        except Exception as exc:
            return TaskProcessOutcome(
                ok=False,
                error=f"Invalid task result JSON: {type(exc).__name__}: {exc}",
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            error = envelope.get("error") if isinstance(envelope, dict) else None
            return TaskProcessOutcome(
                ok=False,
                error=str(error or "Task worker reported failure"),
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )
        return TaskProcessOutcome(
            ok=True,
            result=envelope.get("result"),
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
        )
