from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("LOGGING_DIR", str(Path.cwd() / ".airaevo_logs"))

from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
)


class _RawShell(str):
    pass


def _shell_join(parts: list[Any]) -> str:
    return " ".join(
        str(part) if isinstance(part, _RawShell) else shlex.quote(str(part))
        for part in parts
    )


_GPU_WAIT_MARKER_RE = re.compile(
    r"(?m)^__AIREVO_GPU_WAIT_SECONDS__=(?P<seconds>\d+(?:\.\d+)?)\s*$"
)
_GPU_WAIT_TIMEOUT_MARKER = "__AIREVO_GPU_WAIT_TIMED_OUT__=1"
_IMPORT_PREFLIGHT_MARKER = "__AIREVO_IMPORT_PREFLIGHT__="
_EVAL_CONTROL_TOKEN_HEADER = "X-NatureBench-Control-Token"
_EVAL_CONTROL_ENDPOINTS = frozenset(
    {"register", "start_timer", "resume_timer", "pause_timer"}
)
_CANDIDATE_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "TMPDIR",
    }
)


class _AttemptContext(NamedTuple):
    index: int
    phase: str
    workspace: Path
    output_dir: Path
    scm_workspace: str | None


class NatureBenchTask(Task):
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        time_budget: float | None = None,
    ) -> None:
        super().__init__(cfg)
        self.cfg = dict(cfg)
        self.task_name = str(self.cfg["task_name"])
        self.validation_data_dir = str(self.cfg["data_dir"])
        self.submit_data_dir = self.validation_data_dir
        self.problem_dir = str(self.cfg["problem_dir"])
        self.task_dir = str(self.cfg["task_dir"])
        self.evaluation_protocol = "naturebench"
        self.execution_mode = str(self.cfg.get("execution_mode", "docker")).lower()
        self.eval_service_url = str(self.cfg["eval_service_url"]).rstrip("/")
        self.eval_control_token_file = str(
            self.cfg.get("eval_control_token_file")
            or os.environ.get("NATUREBENCH_EVAL_CONTROL_TOKEN_FILE")
            or ""
        ).strip()
        self.scm_host = str(self.cfg.get("scm_host") or "")
        self.scm_workspace_root = str(self.cfg.get("scm_workspace_root") or "")
        self.scm_eval_service_url = str(
            self.cfg.get("scm_eval_service_url") or "http://localhost:8321"
        ).rstrip("/")
        self.scm_container_eval_service_url = str(
            self.cfg.get("scm_container_eval_service_url")
            or "http://host.docker.internal:8321"
        ).rstrip("/")
        self.scm_task_roots = self._normalize_scm_task_roots(
            self.cfg.get("scm_task_roots")
        )
        if self._uses_scm():
            missing = [
                name
                for name, value in (
                    ("scm_host", self.scm_host),
                    ("scm_workspace_root", self.scm_workspace_root),
                    ("scm_task_roots", self.scm_task_roots),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "SCM execution requires explicit deployment configuration: "
                    + ", ".join(missing)
                )
        self.batch_name = str(self.cfg.get("batch_name") or "airaevo-naturebench")
        self.public_system_prompt = str(self.cfg.get("public_system_prompt") or "")
        self.public_user_prompt = str(self.cfg.get("public_user_prompt") or "")
        self.raw_task_description = str(self.cfg.get("task_description") or "")
        self.data_description = str(self.cfg.get("data_description") or "")
        self.visible_data_analysis = str(self.cfg.get("visible_data_analysis") or "")
        self.task_family_guidance = str(self.cfg.get("task_family_guidance") or "")
        self.task_description = self._build_task_description()
        self.lower_is_better = False
        submit_repeats = self.cfg.get("submit_repeats", 1)
        self.submit_repeats = int(1 if submit_repeats is None else submit_repeats)
        self.eval_timeout = int(self.cfg.get("eval_timeout", 3600) or 3600)
        self.time_budget = float(time_budget) if time_budget is not None else None
        self._state_lock = threading.RLock()
        self._eval_service_lock = threading.RLock()
        self.validation_time_used = 0.0
        self.validation_gpu_wait_time = 0.0
        self.validation_preflight_time_used = 0.0
        self.stop_requested = False
        self.attempt_index = 0
        self._eval_service_registered = False
        self._eval_service_timer_started = False
        self._eval_control_token: str | None = None
        self._eval_token: str | None = None
        self._scm_task_root_cache: str | None = None
        self._import_preflight_cache: dict[str, str | None] = {}

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric

    def _selection_score_from_aggregate(self, aggregate: float | None) -> float | None:
        if aggregate is None:
            return None
        clip = self._coerce_score(self.cfg.get("selection_score_clip", 1.0))
        if clip is None or clip <= 0:
            return aggregate
        return max(-clip, min(clip, aggregate))

    def get_budget_exempt_wait_seconds(self) -> float:
        """Return scheduler delay that should not consume the NatureBench search budget."""
        with self._state_lock:
            return max(0.0, float(self.validation_gpu_wait_time))

    def _candidate_preflight_enabled(self) -> bool:
        return bool(self.cfg.get("candidate_preflight", True))

    def _candidate_preflight_imports_enabled(self) -> bool:
        return bool(
            self.cfg.get(
                "candidate_preflight_imports",
                self._uses_scm() or self.execution_mode == "local",
            )
        )

    def _candidate_preflight_timeout_seconds(self) -> int:
        configured = self._coerce_score(self.cfg.get("candidate_preflight_timeout"))
        return max(1, int(45 if configured is None or configured <= 0 else configured))

    @staticmethod
    def _candidate_import_roots(tree: ast.AST) -> list[str]:
        roots: set[str] = set()
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = str(module).split(".", 1)[0].strip()
                if root and root not in stdlib:
                    roots.add(root)
        return sorted(roots)

    @staticmethod
    def _contains_package_install(tree: ast.AST, code: str) -> bool:
        command_functions = {
            "system",
            "run",
            "call",
            "check_call",
            "check_output",
            "popen",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                function_name = func.attr.lower()
            elif isinstance(func, ast.Name):
                function_name = func.id.lower()
            else:
                continue
            if function_name not in command_functions:
                continue
            call_source = (ast.get_source_segment(code, node) or "").lower()
            if "pip" in call_source and "install" in call_source:
                return True
        return False

    @staticmethod
    def _preflight_failure(
        *,
        category: str,
        feedback: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "category": category,
            "feedback": feedback,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        }

    def _run_scm_import_preflight(self, modules: list[str]) -> dict[str, Any]:
        unknown_modules = [
            module for module in modules if module not in self._import_preflight_cache
        ]
        if unknown_modules:
            script = (
                "import importlib, json\n"
                f"modules = {json.dumps(unknown_modules)}\n"
                "missing = {}\n"
                "for module in modules:\n"
                "    try:\n"
                "        importlib.import_module(module)\n"
                "    except Exception as exc:\n"
                "        missing[module] = f'{type(exc).__name__}: {exc}'\n"
                f"print({_IMPORT_PREFLIGHT_MARKER!r} + json.dumps({{'missing': missing}}, sort_keys=True))\n"
            )
            image = str(self.cfg.get("docker_image") or "cnsbench-base:v3")
            timeout_seconds = self._candidate_preflight_timeout_seconds()
            command = _shell_join(
                [
                    "timeout",
                    "-k",
                    "5s",
                    f"{timeout_seconds}s",
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    image,
                    "python",
                    "-c",
                    script,
                ]
            )
            started = time.monotonic()
            try:
                process = self._ssh(command, timeout=timeout_seconds + 60)
            except subprocess.TimeoutExpired as exc:
                return {
                    "ok": True,
                    "status": "skipped",
                    "reason": "container_import_preflight_timed_out",
                    "elapsed_seconds": time.monotonic() - started,
                    "detail": str(exc),
                }
            elapsed = time.monotonic() - started
            raw_log = (process.stdout or "") + (process.stderr or "")
            marker_payload = None
            for line in (process.stdout or "").splitlines():
                if line.startswith(_IMPORT_PREFLIGHT_MARKER):
                    marker_payload = line[len(_IMPORT_PREFLIGHT_MARKER) :]
            if process.returncode != 0 or marker_payload is None:
                return {
                    "ok": True,
                    "status": "skipped",
                    "reason": "container_import_preflight_unavailable",
                    "elapsed_seconds": elapsed,
                    "detail": raw_log[-1000:],
                }
            try:
                missing = dict(json.loads(marker_payload).get("missing") or {})
            except (TypeError, ValueError, json.JSONDecodeError):
                return {
                    "ok": True,
                    "status": "skipped",
                    "reason": "container_import_preflight_unparseable",
                    "elapsed_seconds": elapsed,
                    "detail": raw_log[-1000:],
                }
            for module in unknown_modules:
                detail = missing.get(module)
                self._import_preflight_cache[module] = str(detail) if detail else None
        unavailable = {
            module: self._import_preflight_cache[module]
            for module in modules
            if self._import_preflight_cache.get(module)
        }
        return {
            "ok": not unavailable,
            "missing": unavailable,
            "elapsed_seconds": 0.0,
        }

    def _local_python_command(self, *arguments: str) -> list[str]:
        local_python = str(self.cfg.get("local_python") or "").strip()
        if local_python:
            prefix = [str(Path(os.path.abspath(Path(local_python).expanduser())))]
        else:
            conda_env = str(self.cfg.get("local_conda_env") or "").strip()
            if conda_env:
                prefix = [
                    str(self.cfg.get("local_conda_executable") or "conda"),
                    "run",
                    "--no-capture-output",
                    "-n",
                    conda_env,
                    "python",
                ]
            else:
                prefix = [sys.executable]
        return [*prefix, *arguments]

    def _run_local_import_preflight(self, modules: list[str]) -> dict[str, Any]:
        unknown_modules = [
            module for module in modules if module not in self._import_preflight_cache
        ]
        if unknown_modules:
            script = (
                "import importlib, json\n"
                f"modules = {json.dumps(unknown_modules)}\n"
                "missing = {}\n"
                "for module in modules:\n"
                "    try:\n"
                "        importlib.import_module(module)\n"
                "    except Exception as exc:\n"
                "        missing[module] = f'{type(exc).__name__}: {exc}'\n"
                f"print({_IMPORT_PREFLIGHT_MARKER!r} + json.dumps({{'missing': missing}}, sort_keys=True))\n"
            )
            timeout_seconds = self._candidate_preflight_timeout_seconds()
            started = time.monotonic()
            workspace_root = self._workspace_root()
            workspace_root.mkdir(parents=True, exist_ok=True)
            try:
                process = subprocess.run(
                    self._local_python_command("-c", script),
                    cwd=workspace_root,
                    env=self._solution_env(workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {
                    "ok": False,
                    "missing": {
                        module: f"local runtime unavailable: {exc}"
                        for module in unknown_modules
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }

            marker_payload = None
            for line in (process.stdout or "").splitlines():
                if line.startswith(_IMPORT_PREFLIGHT_MARKER):
                    marker_payload = line[len(_IMPORT_PREFLIGHT_MARKER) :]
            if process.returncode != 0 or marker_payload is None:
                detail = ((process.stderr or "") + (process.stdout or ""))[-1000:]
                return {
                    "ok": False,
                    "missing": {
                        module: f"local import preflight failed: {detail}"
                        for module in unknown_modules
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }
            try:
                missing = dict(json.loads(marker_payload).get("missing") or {})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return {
                    "ok": False,
                    "missing": {
                        module: f"local import preflight returned invalid JSON: {exc}"
                        for module in unknown_modules
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }
            for module in unknown_modules:
                detail = missing.get(module)
                self._import_preflight_cache[module] = str(detail) if detail else None

        unavailable = {
            module: self._import_preflight_cache[module]
            for module in modules
            if self._import_preflight_cache.get(module)
        }
        return {
            "ok": not unavailable,
            "missing": unavailable,
            "elapsed_seconds": 0.0,
        }

    def _candidate_preflight(self, code: str) -> dict[str, Any]:
        started = time.monotonic()
        if not self._candidate_preflight_enabled():
            return {"ok": True, "status": "disabled", "elapsed_seconds": 0.0}
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno else "an unknown line"
            return self._preflight_failure(
                category="syntax_error",
                feedback=(
                    "Candidate preflight blocked execution: Python syntax error at "
                    f"{location}: {exc.msg}."
                ),
                elapsed_seconds=time.monotonic() - started,
            )
        if self._contains_package_install(tree, code):
            return self._preflight_failure(
                category="prohibited_package_install",
                feedback=(
                    "Candidate preflight blocked execution: package installation is "
                    "not allowed. Use an already available dependency or a standard "
                    "library fallback instead of invoking pip install."
                ),
                elapsed_seconds=time.monotonic() - started,
            )
        modules = self._candidate_import_roots(tree)
        if not self._candidate_preflight_imports_enabled() or not modules:
            return {
                "ok": True,
                "status": "passed",
                "elapsed_seconds": time.monotonic() - started,
            }
        if self._uses_scm():
            import_result = self._run_scm_import_preflight(modules)
            runtime_name = "container"
        elif self.execution_mode == "local":
            import_result = self._run_local_import_preflight(modules)
            runtime_name = "local runtime"
        else:
            return {
                "ok": True,
                "status": "passed",
                "elapsed_seconds": time.monotonic() - started,
            }
        elapsed = time.monotonic() - started
        if not import_result.get("ok", True):
            missing = dict(import_result.get("missing") or {})
            details = "; ".join(
                f"{name}: {detail}" for name, detail in sorted(missing.items())
            )
            return self._preflight_failure(
                category="unavailable_import",
                feedback=(
                    f"Candidate preflight blocked execution because this {runtime_name} "
                    f"cannot import: {details}. Choose an available fallback."
                ),
                elapsed_seconds=elapsed,
            )
        return {
            "ok": True,
            "status": str(import_result.get("status") or "passed"),
            "elapsed_seconds": elapsed,
            "imports": modules,
            "note": import_result.get("reason"),
        }

    def _build_task_description(self) -> str:
        sections = [
            ("Task Description", self.raw_task_description),
            ("Data Description", self.data_description),
            ("Verified Visible-Data Analysis", self.visible_data_analysis),
            ("Task-Family Execution Guidance", self.task_family_guidance),
            (
                "NatureBench Runtime Contract",
                "Read visible inputs from DATA_DIR, write required outputs under "
                "OUTPUT_DIR, and optimize aggregate_improvement. The hidden "
                "evaluation directory is only available to the host eval service.",
            ),
        ]
        return "\n\n".join(
            f"{title}:\n{content.strip()}"
            for title, content in sections
            if content.strip()
        )

    def prepare(self, **task_args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        state = dict(task_args)
        state["init_obs"] = {}
        state["validation_time_used"] = self.validation_time_used
        state["validation_gpu_wait_time"] = self.validation_gpu_wait_time
        state["naturebench_attempt_index"] = self.attempt_index
        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": self.lower_is_better,
            "naturebench_raw_task_description": self.raw_task_description,
            "data_description": self.data_description,
            "visible_data_analysis": self.visible_data_analysis,
            "task_family_guidance": self.task_family_guidance,
            "public_system_prompt": self.public_system_prompt,
            "public_user_prompt": self.public_user_prompt,
        }
        return state, task_info

    def build_submit_code(self, code: str) -> str:
        return code

    def _workspace_root(self) -> Path:
        root = self.cfg.get("workspace_root")
        if root:
            return Path(str(root)).expanduser().resolve()
        return Path(self.task_dir).resolve() / ".airaevo_workspaces"

    @staticmethod
    def _safe_path_component(value: str) -> str:
        raw_value = str(value)
        safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value).strip("._-")
        safe_value = safe_value or "item"
        if safe_value == raw_value:
            return safe_value
        digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:8]
        return f"{safe_value[:80]}-{digest}"

    @staticmethod
    def _normalize_scm_task_roots(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part for part in value.split(":") if part]
        if isinstance(value, list):
            return [str(part) for part in value if str(part)]
        return [str(value)]

    def _uses_scm(self) -> bool:
        return self.execution_mode in {"scm", "scm_docker", "remote_scm"}

    def _eval_service_out_dir(self) -> Path:
        configured = self.cfg.get("eval_service_out_dir")
        if configured:
            return Path(str(configured)).expanduser().resolve()
        return (
            self._workspace_root()
            / "_eval_service"
            / self._safe_path_component(self.batch_name)
            / self._safe_path_component(self.task_name)
        ).resolve()

    def _eval_service_out_dir_value(self) -> str:
        if self._uses_scm():
            configured = self.cfg.get("eval_service_out_dir")
            if configured:
                return str(configured)
            return "/".join(
                [
                    self.scm_workspace_root.rstrip("/"),
                    "_eval_service",
                    self._safe_path_component(self.batch_name),
                    self._safe_path_component(self.task_name),
                ]
            )
        return str(self._eval_service_out_dir())

    def _uses_token_eval_contract(self) -> bool:
        return bool(self.eval_control_token_file)

    def _load_eval_control_token(self) -> str:
        if self._eval_control_token is not None:
            return self._eval_control_token
        if not self.eval_control_token_file:
            raise RuntimeError("NatureBench eval control token file is not configured")
        token_path = Path(self.eval_control_token_file).expanduser()
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Could not read NatureBench eval control token file: {token_path}"
            ) from exc
        if not token:
            raise RuntimeError(
                f"NatureBench eval control token file is empty: {token_path}"
            )
        self._eval_control_token = token
        return token

    def _load_or_create_eval_token(self) -> str:
        if self._eval_token is not None:
            return self._eval_token
        control_token = self._load_eval_control_token()
        token_path = self._eval_service_out_dir() / "eval_token.txt"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            if token and not secrets.compare_digest(token, control_token):
                self._eval_token = token
                return token

        token = secrets.token_urlsafe(32)
        while secrets.compare_digest(token, control_token):
            token = secrets.token_urlsafe(32)
        temporary_path = token_path.with_name(
            f".{token_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary_path.write_text(token + "\n", encoding="utf-8")
        temporary_path.chmod(0o600)
        temporary_path.replace(token_path)
        self._eval_token = token
        return token

    def _solve_timeout_seconds(self) -> int:
        candidates = [
            self.time_budget,
            self.cfg.get("time_budget"),
            self.cfg.get("model_plus_sandbox_time_budget"),
            self.cfg.get("execution_timeout"),
        ]
        for value in candidates:
            numeric = self._coerce_score(value)
            if numeric is not None and numeric > 0:
                return int(numeric)
        return 14400

    def _execution_timeout_seconds(self) -> int:
        for value in (
            self.cfg.get("execution_timeout"),
            self.cfg.get("job_timeout"),
        ):
            numeric = self._coerce_score(value)
            if numeric is not None and numeric > 0:
                return int(numeric)
        return 600

    def _scm_docker_command_timeout_seconds(self, execution_timeout: int) -> int:
        configured = self._coerce_score(self.cfg.get("scm_docker_command_timeout"))
        if configured is not None and configured > 0:
            return int(configured)
        wait_timeout = self._coerce_score(self.cfg.get("scm_gpu_wait_timeout"))
        if wait_timeout is None or wait_timeout <= 0:
            wait_timeout = 14400.0
        return int(wait_timeout) + int(execution_timeout) + 90

    def _scm_gpu_wait_timeout_seconds(self) -> int:
        configured = self._coerce_score(self.cfg.get("scm_gpu_wait_timeout"))
        return max(0, int(14400 if configured is None else configured))

    def _start_attempt(self, *, phase: str) -> _AttemptContext:
        with self._state_lock:
            self.attempt_index += 1
            attempt_index = self.attempt_index
        attempt_root = self._workspace_root() / (
            f"{self._safe_path_component(self.task_name)}_"
            f"{self._safe_path_component(phase)}_{attempt_index}"
        )
        workspace = (
            attempt_root / "workspace"
            if self._uses_token_eval_contract()
            else attempt_root
        )
        output_dir = workspace / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        scm_workspace = None
        if self._uses_scm():
            scm_attempt_root = "/".join(
                [
                    self.scm_workspace_root.rstrip("/"),
                    self._safe_path_component(self.batch_name),
                    self._safe_path_component(self.task_name),
                    f"{self._safe_path_component(phase)}_{attempt_index}",
                ]
            )
            scm_workspace = (
                f"{scm_attempt_root}/workspace"
                if self._uses_token_eval_contract()
                else scm_attempt_root
            )
        return _AttemptContext(
            index=attempt_index,
            phase=phase,
            workspace=workspace,
            output_dir=output_dir,
            scm_workspace=scm_workspace,
        )

    def _solution_env(self, output_dir: Path) -> dict[str, str]:
        configured_allowlist = self.cfg.get("candidate_env_allowlist") or []
        if isinstance(configured_allowlist, str):
            configured_allowlist = [
                item.strip() for item in configured_allowlist.split(",") if item.strip()
            ]
        allowed_names = _CANDIDATE_ENV_ALLOWLIST.union(
            str(item) for item in configured_allowlist
        )
        env = {name: os.environ[name] for name in allowed_names if name in os.environ}
        env.setdefault("PATH", os.defpath)
        env.update(
            {
                "DATA_DIR": self.validation_data_dir,
                "OUTPUT_DIR": str(output_dir),
                "EVAL_SERVICE_URL": self.eval_service_url,
                "NONINTERACTIVE": "1",
                "CI": "1",
            }
        )
        return env

    def _container_eval_service_url(self) -> str:
        configured = self.cfg.get("container_eval_service_url")
        if configured:
            return str(configured).rstrip("/")
        parsed = urllib.parse.urlsplit(self.eval_service_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return self.eval_service_url
        host = "host.docker.internal"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit(
            (
                parsed.scheme or "http",
                host,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        ).rstrip("/")

    def _terminate_local_process(self, process: subprocess.Popen[str]) -> None:
        grace_seconds = max(
            0.0,
            float(self.cfg.get("local_terminate_grace_seconds", 5) or 0),
        )
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=grace_seconds)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()

    def _run_local_solution(
        self, code: str, *, workspace: Path, output_dir: Path
    ) -> dict[str, Any]:
        run_path = workspace / "run.py"
        run_path.write_text(code, encoding="utf-8")
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                self._local_python_command(str(run_path)),
                cwd=workspace,
                env=self._solution_env(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
            )
            stdout, stderr = process.communicate(
                timeout=self._execution_timeout_seconds()
            )
            run_time = time.monotonic() - started
            raw_log = (stdout or "") + (stderr or "")
            return {
                "status_code": 200 if process.returncode == 0 else 500,
                "status": "success" if process.returncode == 0 else "failed",
                "raw_run_log": raw_log,
                "clear_run_log": raw_log,
                "feedback": raw_log,
                "run_time": run_time,
                "output_dir": str(output_dir),
            }
        except subprocess.TimeoutExpired:
            assert process is not None
            self._terminate_local_process(process)
            stdout, stderr = process.communicate()
            raw_log = (stdout or "") + (stderr or "")
            return {
                "status_code": 504,
                "status": "timeout",
                "raw_run_log": raw_log,
                "clear_run_log": raw_log,
                "feedback": raw_log or "candidate timed out",
                "run_time": time.monotonic() - started,
                "output_dir": str(output_dir),
            }
        except OSError as exc:
            raw_log = f"Failed to start local NatureBench runtime: {exc}"
            return {
                "status_code": 500,
                "status": "failed",
                "raw_run_log": raw_log,
                "clear_run_log": raw_log,
                "feedback": raw_log,
                "run_time": time.monotonic() - started,
                "output_dir": str(output_dir),
            }

    @staticmethod
    def _timeout_stream_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _run_docker_solution(
        self,
        code: str,
        *,
        workspace: Path,
        output_dir: Path,
        attempt_index: int,
    ) -> dict[str, Any]:
        run_path = workspace / "run.py"
        run_path.write_text(code, encoding="utf-8")
        image = str(self.cfg.get("docker_image") or "naturebench-base:v3")
        container_name = "-".join(
            [
                "airaevo",
                self._safe_path_component(self.batch_name)[:32],
                self._safe_path_component(self.task_name)[:48],
                str(attempt_index),
            ]
        )
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            "DATA_DIR=/task/problem/data",
            "-e",
            "OUTPUT_DIR=/workspace/output",
            "-e",
            f"EVAL_SERVICE_URL={self._container_eval_service_url()}",
            "-e",
            "NONINTERACTIVE=1",
            "-e",
            "CI=1",
            "-v",
            f"{Path(self.problem_dir).resolve()}:/task/problem:ro",
            "-v",
            f"{workspace.resolve()}:/workspace",
        ]
        gpu_devices = self.cfg.get("gpu_devices")
        resource = str(self.cfg.get("resource") or "cpu").lower()
        if resource != "cpu" and gpu_devices:
            devices = (
                ",".join(str(item) for item in gpu_devices)
                if isinstance(gpu_devices, list)
                else str(gpu_devices)
            )
            cmd.extend(
                ["--gpus", "all" if devices.lower() == "all" else f"device={devices}"]
            )
        cmd.extend([image, "bash", "-lc", "python /workspace/run.py"])

        started = time.monotonic()
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._execution_timeout_seconds(),
            )
        except subprocess.TimeoutExpired as exc:
            raw_log = self._timeout_stream_text(exc.stdout) + self._timeout_stream_text(
                exc.stderr
            )
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                cleanup_log = (cleanup.stdout or "") + (cleanup.stderr or "")
                cleanup_failed = cleanup.returncode != 0
            except (OSError, subprocess.TimeoutExpired) as cleanup_exc:
                cleanup_log = str(cleanup_exc)
                cleanup_failed = True
            if cleanup_failed:
                raw_log = f"{raw_log}\nContainer cleanup failed: {cleanup_log}".strip()
            return {
                "status_code": 504,
                "status": "timeout",
                "raw_run_log": raw_log,
                "clear_run_log": raw_log,
                "feedback": raw_log or "candidate timed out",
                "run_time": time.monotonic() - started,
                "output_dir": str(output_dir),
                "container_name": container_name,
            }
        raw_log = (process.stdout or "") + (process.stderr or "")
        return {
            "status_code": 200 if process.returncode == 0 else 500,
            "status": "success" if process.returncode == 0 else "failed",
            "raw_run_log": raw_log,
            "clear_run_log": raw_log,
            "feedback": raw_log,
            "run_time": time.monotonic() - started,
            "output_dir": str(output_dir),
        }

    def _ssh(
        self,
        remote_command: str,
        *,
        input_text: str | None = None,
        timeout: int | float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self._run_ssh_with_retries(
            ["ssh", *self._ssh_options(), self.scm_host, remote_command],
            input_text=input_text,
            timeout=timeout,
        )

    @staticmethod
    def _ssh_options() -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]

    def _ssh_retry_attempts(self) -> int:
        return max(1, int(self.cfg.get("scm_ssh_retries", 5) or 5))

    def _ssh_retry_delay(self, attempt: int) -> float:
        jitter = (sum(ord(char) for char in self.task_name) % 100) / 200.0
        return min(8.0, 0.5 * (2**attempt)) + jitter

    @staticmethod
    def _is_transient_ssh_failure(output: str, returncode: int) -> bool:
        if returncode != 255:
            return False
        transient_markers = (
            "Connection closed",
            "UNKNOWN port 65535",
            "kex_exchange_identification",
            "Connection reset",
            "Connection refused",
            "stdio forwarding failed",
            "Timeout",
            "timed out",
        )
        return any(marker in output for marker in transient_markers)

    def _run_ssh_with_retries(
        self,
        command: list[str] | str,
        *,
        input_text: str | None = None,
        timeout: int | float | None = None,
        shell: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        attempts = self._ssh_retry_attempts()
        last_process: subprocess.CompletedProcess[str] | None = None
        for attempt in range(attempts):
            process = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=shell,
            )
            last_process = process
            output = (process.stdout or "") + (process.stderr or "")
            if not self._is_transient_ssh_failure(output, process.returncode):
                return process
            if attempt + 1 < attempts:
                time.sleep(self._ssh_retry_delay(attempt))
        assert last_process is not None
        return last_process

    def _remote_json_request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None,
        *,
        timeout: int | float,
    ) -> dict[str, Any]:
        script = r"""
import json
import sys
import urllib.error
import urllib.request

req = json.loads(sys.stdin.read())
data = None if req["payload"] is None else json.dumps(req["payload"]).encode("utf-8")
headers = dict(req.get("headers") or {})
if data is not None:
    headers["Content-Type"] = "application/json"
request = urllib.request.Request(
    req["url"],
    data=data,
    headers=headers,
    method=req["method"],
)
try:
    with urllib.request.urlopen(request, timeout=req["timeout"]) as response:
        body = response.read().decode("utf-8")
        print(json.dumps({"status": response.status, "body": body}))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(json.dumps({"status": exc.code, "body": body}))
except Exception as exc:
    print(json.dumps({"status": 0, "body": repr(exc)}))
"""
        envelope = {
            "method": "POST" if payload is not None else "GET",
            "url": f"{self.scm_eval_service_url}/{endpoint.lstrip('/')}",
            "payload": payload,
            "timeout": timeout,
            "headers": self._eval_request_headers(endpoint),
        }
        process = self._ssh(
            "python3 -c " + shlex.quote(script),
            input_text=json.dumps(envelope),
            timeout=float(timeout) + 15,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"SCM eval request failed on {self.scm_host}: "
                f"{process.stderr[-1000:] or process.stdout[-1000:]}"
            )
        result = json.loads(process.stdout)
        status = int(result.get("status") or 0)
        body = str(result.get("body") or "")
        if status < 200 or status >= 300:
            raise RuntimeError(
                f"NatureBench SCM eval service /{endpoint} failed with "
                f"HTTP {status}: {body}"
            )
        return json.loads(body) if body.strip() else {}

    def _resolve_scm_task_root(self) -> str:
        if self._scm_task_root_cache is not None:
            return self._scm_task_root_cache
        configured = self.cfg.get("scm_task_root")
        if configured:
            self._scm_task_root_cache = str(configured).rstrip("/")
            return self._scm_task_root_cache
        script = r"""
import json
from pathlib import Path
import sys

payload = json.loads(sys.stdin.read())
task_name = payload["task_name"]
for root in payload["roots"]:
    candidate = Path(root) / task_name
    if candidate.is_dir():
        print(str(candidate))
        raise SystemExit(0)
print("", end="")
raise SystemExit(2)
"""
        process = self._ssh(
            "python3 -c " + shlex.quote(script),
            input_text=json.dumps(
                {"task_name": self.task_name, "roots": self.scm_task_roots}
            ),
            timeout=20,
        )
        if process.returncode != 0 or not process.stdout.strip():
            raise RuntimeError(
                f"Could not locate {self.task_name} under SCM task roots "
                f"{self.scm_task_roots}: {process.stderr[-1000:]}"
            )
        self._scm_task_root_cache = process.stdout.strip()
        return self._scm_task_root_cache

    @staticmethod
    def _remote_tar_extract_command(remote_workspace: str) -> str:
        return _shell_join(["tar", "-C", remote_workspace, "-xf", "-"])

    def _sync_workspace_to_scm(self, workspace: Path, remote_workspace: str) -> None:
        remote_q = shlex.quote(remote_workspace)
        prep = f"rm -rf {remote_q} && mkdir -p {remote_q}"
        prep_proc = self._ssh(prep, timeout=30)
        if prep_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to prepare SCM workspace: {prep_proc.stderr[-1000:]}"
            )
        remote_tar_command = self._remote_tar_extract_command(remote_workspace)
        tar_cmd = (
            f"tar -C {shlex.quote(str(workspace.resolve()))} -cf - . "
            f"| ssh {' '.join(shlex.quote(option) for option in self._ssh_options())} "
            f"{shlex.quote(self.scm_host)} "
            f"{shlex.quote(remote_tar_command)}"
        )
        sync_proc = self._run_ssh_with_retries(
            tar_cmd,
            shell=True,
            timeout=120,
        )
        if sync_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to sync workspace to SCM: "
                f"{sync_proc.stderr[-1000:] or sync_proc.stdout[-1000:]}"
            )

    @staticmethod
    def _normalize_gpu_devices(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            if value.strip().lower() == "all":
                return ["all"]
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part) for part in value if str(part)]
        return [str(value)]

    def _gpu_lock_root(self, key: str, configured: Any) -> str:
        if configured:
            return str(configured)
        host = self._safe_path_component(self.scm_host)
        batch = self._safe_path_component(self.batch_name)
        return f"/tmp/airaevo_naturebench_{host}_{batch}_{key}"

    def _gpu_skip_busy_limits(self) -> tuple[int, int]:
        skip_mb = self._coerce_score(self.cfg.get("gpu_skip_busy_mb"))
        skip_util = self._coerce_score(self.cfg.get("gpu_skip_busy_util"))
        return (
            int(2000 if skip_mb is None else skip_mb),
            int(80 if skip_util is None else skip_util),
        )

    def _gpu_external_busy_probe_script(self) -> str:
        skip_mb, skip_util = self._gpu_skip_busy_limits()
        return f"""
AIREVO_GPU_SKIP_BUSY_MB={skip_mb}
AIREVO_GPU_SKIP_BUSY_UTIL={skip_util}
airaevo_gpu_is_available() {{
  if [ "$AIREVO_GPU_SKIP_BUSY_MB" -le 0 ]; then
    return 0
  fi
  AIREVO_GPU_STATS="$(nvidia-smi -i "$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -z "$AIREVO_GPU_STATS" ]; then
    return 0
  fi
  AIREVO_GPU_MEM="$(printf '%s\\n' "$AIREVO_GPU_STATS" | awk -F, 'NR==1 {{gsub(/ /,"",$1); print $1}}')"
  AIREVO_GPU_UTIL="$(printf '%s\\n' "$AIREVO_GPU_STATS" | awk -F, 'NR==1 {{gsub(/ /,"",$2); print $2}}')"
  AIREVO_GPU_MEM="${{AIREVO_GPU_MEM:-0}}"
  AIREVO_GPU_UTIL="${{AIREVO_GPU_UTIL:-0}}"
  if [ "$AIREVO_GPU_MEM" -gt "$AIREVO_GPU_SKIP_BUSY_MB" ]; then
    return 1
  fi
  if [ "$AIREVO_GPU_UTIL" -gt "$AIREVO_GPU_SKIP_BUSY_UTIL" ]; then
    return 1
  fi
  return 0
}}
""".strip()

    def _exclusive_gpu_acquire_script(self, devices: list[str]) -> str:
        lock_root = self._gpu_lock_root(
            "exclusive_gpu_pool", self.cfg.get("gpu_pool_file")
        )
        gpu_ids = " ".join(shlex.quote(device) for device in devices)
        wait_timeout = self._scm_gpu_wait_timeout_seconds()
        return f"""
AIREVO_GPU_LOCK_ROOT={shlex.quote(lock_root)}
AIREVO_GPU_IDS={shlex.quote(gpu_ids)}
AIREVO_GPU_WAIT_TIMEOUT={wait_timeout}
{self._gpu_external_busy_probe_script()}
mkdir -p "$AIREVO_GPU_LOCK_ROOT"
while :; do
  if [ "$AIREVO_GPU_WAIT_TIMEOUT" -gt 0 ] && [ $(( ${{SECONDS:-0}} - ${{AIREVO_GPU_WAIT_STARTED:-0}} )) -ge "$AIREVO_GPU_WAIT_TIMEOUT" ]; then
    AIREVO_GPU_WAIT_SECONDS=$(( ${{SECONDS:-0}} - ${{AIREVO_GPU_WAIT_STARTED:-0}} ))
    printf '__AIREVO_GPU_WAIT_SECONDS__=%s\\n' "$AIREVO_GPU_WAIT_SECONDS"
    printf '__AIREVO_GPU_WAIT_TIMED_OUT__=1\\n' >&2
    exit 75
  fi
  for AIREVO_GPU_ID in $AIREVO_GPU_IDS; do
    AIREVO_GPU_LOCK_DIR="$AIREVO_GPU_LOCK_ROOT/gpu_${{AIREVO_GPU_ID}}.lock"
    if mkdir "$AIREVO_GPU_LOCK_DIR" 2>/dev/null; then
      if ! airaevo_gpu_is_available "$AIREVO_GPU_ID"; then
        rm -rf "$AIREVO_GPU_LOCK_DIR"
        continue
      fi
      printf '%s %s\\n' "$$" "$AIREVO_CONTAINER" > "$AIREVO_GPU_LOCK_DIR/holder"
      export AIREVO_GPU_ID AIREVO_GPU_LOCK_DIR
      break 2
    fi
    if [ -f "$AIREVO_GPU_LOCK_DIR/holder" ]; then
      AIREVO_HOLDER_PID="$(awk '{{print $1}}' "$AIREVO_GPU_LOCK_DIR/holder" 2>/dev/null || true)"
      if [ -n "$AIREVO_HOLDER_PID" ] && ! kill -0 "$AIREVO_HOLDER_PID" 2>/dev/null; then
        rm -rf "$AIREVO_GPU_LOCK_DIR"
      fi
    else
      rm -rf "$AIREVO_GPU_LOCK_DIR"
    fi
  done
  sleep 2
done
""".strip()

    def _shared_gpu_acquire_script(self) -> str:
        gpu_id = str(self.cfg.get("shared_gpu_device"))
        slots = max(1, int(self.cfg.get("shared_gpu_slots", 1) or 1))
        lock_root = self._gpu_lock_root(
            "shared_gpu_pool", self.cfg.get("shared_gpu_pool_file")
        )
        wait_timeout = self._scm_gpu_wait_timeout_seconds()
        return f"""
AIREVO_GPU_ID={shlex.quote(gpu_id)}
AIREVO_GPU_LOCK_ROOT={shlex.quote(lock_root)}
AIREVO_SHARED_GPU_SLOTS={slots}
AIREVO_GPU_WAIT_TIMEOUT={wait_timeout}
{self._gpu_external_busy_probe_script()}
airaevo_gpu_has_live_shared_holder() {{
  for AIREVO_HOLDER_FILE in "$AIREVO_GPU_LOCK_ROOT"/gpu_${{AIREVO_GPU_ID}}_slot_*.lock/holder; do
    if [ ! -e "$AIREVO_HOLDER_FILE" ]; then
      continue
    fi
    AIREVO_HOLDER_PID="$(awk '{{print $1}}' "$AIREVO_HOLDER_FILE" 2>/dev/null || true)"
    AIREVO_HOLDER_DIR="$(dirname "$AIREVO_HOLDER_FILE")"
    if [ -n "$AIREVO_HOLDER_PID" ] && kill -0 "$AIREVO_HOLDER_PID" 2>/dev/null; then
      return 0
    fi
    rm -rf "$AIREVO_HOLDER_DIR"
  done
  return 1
}}
mkdir -p "$AIREVO_GPU_LOCK_ROOT"
while :; do
  if [ "$AIREVO_GPU_WAIT_TIMEOUT" -gt 0 ] && [ $(( ${{SECONDS:-0}} - ${{AIREVO_GPU_WAIT_STARTED:-0}} )) -ge "$AIREVO_GPU_WAIT_TIMEOUT" ]; then
    AIREVO_GPU_WAIT_SECONDS=$(( ${{SECONDS:-0}} - ${{AIREVO_GPU_WAIT_STARTED:-0}} ))
    printf '__AIREVO_GPU_WAIT_SECONDS__=%s\\n' "$AIREVO_GPU_WAIT_SECONDS"
    printf '__AIREVO_GPU_WAIT_TIMED_OUT__=1\\n' >&2
    exit 75
  fi
  if ! airaevo_gpu_has_live_shared_holder && ! airaevo_gpu_is_available "$AIREVO_GPU_ID"; then
    sleep 2
    continue
  fi
  AIREVO_SLOT=0
  while [ "$AIREVO_SLOT" -lt "$AIREVO_SHARED_GPU_SLOTS" ]; do
    AIREVO_GPU_LOCK_DIR="$AIREVO_GPU_LOCK_ROOT/gpu_${{AIREVO_GPU_ID}}_slot_${{AIREVO_SLOT}}.lock"
    if mkdir "$AIREVO_GPU_LOCK_DIR" 2>/dev/null; then
      printf '%s %s\\n' "$$" "$AIREVO_CONTAINER" > "$AIREVO_GPU_LOCK_DIR/holder"
      AIREVO_SHARED_SLOT="$AIREVO_SLOT"
      export AIREVO_GPU_ID AIREVO_SHARED_SLOT AIREVO_GPU_LOCK_DIR
      break 2
    fi
    if [ -f "$AIREVO_GPU_LOCK_DIR/holder" ]; then
      AIREVO_HOLDER_PID="$(awk '{{print $1}}' "$AIREVO_GPU_LOCK_DIR/holder" 2>/dev/null || true)"
      if [ -n "$AIREVO_HOLDER_PID" ] && ! kill -0 "$AIREVO_HOLDER_PID" 2>/dev/null; then
        rm -rf "$AIREVO_GPU_LOCK_DIR"
      fi
    else
      rm -rf "$AIREVO_GPU_LOCK_DIR"
    fi
    AIREVO_SLOT=$((AIREVO_SLOT + 1))
  done
  sleep 2
done
""".strip()

    def _scm_gpu_setup_and_docker_args(self) -> tuple[str, list[Any]]:
        resource = str(self.cfg.get("resource") or "cpu").lower()
        gpu_mode = str(self.cfg.get("gpu_mode") or "").lower()
        if resource == "cpu" or gpu_mode == "none":
            return "", []

        if gpu_mode == "exclusive":
            devices = self._normalize_gpu_devices(self.cfg.get("gpu_devices"))
            if not devices:
                return "", []
            if devices == ["all"]:
                return "", ["--gpus", "all"]
            return self._exclusive_gpu_acquire_script(devices), [
                "--gpus",
                _RawShell("device=$AIREVO_GPU_ID"),
            ]

        if gpu_mode == "shared":
            if self.cfg.get("shared_gpu_device") is None:
                raise ValueError(
                    "NatureBench shared GPU tasks require shared_gpu_device"
                )
            return self._shared_gpu_acquire_script(), [
                "--gpus",
                f"device={self.cfg.get('shared_gpu_device')}",
            ]

        gpu_devices = self.cfg.get("gpu_devices")
        if gpu_devices:
            devices = (
                ",".join(str(item) for item in gpu_devices)
                if isinstance(gpu_devices, list)
                else str(gpu_devices)
            )
            return "", [
                "--gpus",
                "all" if devices.lower() == "all" else f"device={devices}",
            ]
        return "", []

    def _wrap_scm_docker_command(
        self,
        *,
        docker_command: str,
        container_name: str,
        gpu_setup_script: str,
    ) -> str:
        gpu_wait_measurement = (
            [
                "AIREVO_GPU_WAIT_STARTED=${SECONDS:-0}",
                gpu_setup_script,
                "AIREVO_GPU_WAIT_SECONDS=$((${SECONDS:-0} - AIREVO_GPU_WAIT_STARTED))",
                'printf "__AIREVO_GPU_WAIT_SECONDS__=%s\\n" "$AIREVO_GPU_WAIT_SECONDS"',
            ]
            if gpu_setup_script
            else []
        )
        script = "\n".join(
            part
            for part in [
                "set -u",
                f"AIREVO_CONTAINER={shlex.quote(container_name)}",
                (
                    "cleanup() { "
                    'docker rm -f "$AIREVO_CONTAINER" >/dev/null 2>&1 || true; '
                    'if [ -n "${AIREVO_GPU_LOCK_DIR:-}" ]; then '
                    'rm -rf "$AIREVO_GPU_LOCK_DIR"; '
                    "fi; }"
                ),
                "trap cleanup EXIT INT TERM",
                *gpu_wait_measurement,
                docker_command,
                "AIREVO_RC=$?",
                "exit $AIREVO_RC",
            ]
            if part
        )
        return "bash -lc " + shlex.quote(script)

    @staticmethod
    def _extract_gpu_wait_seconds(raw_log: str) -> tuple[float, bool, str]:
        wait_seconds = 0.0
        for match in _GPU_WAIT_MARKER_RE.finditer(raw_log):
            try:
                wait_seconds = max(wait_seconds, float(match.group("seconds")))
            except (TypeError, ValueError):
                continue
        timed_out = _GPU_WAIT_TIMEOUT_MARKER in raw_log
        clear_log = _GPU_WAIT_MARKER_RE.sub("", raw_log).replace(
            _GPU_WAIT_TIMEOUT_MARKER,
            "",
        )
        return wait_seconds, timed_out, clear_log

    def _run_scm_docker_solution(
        self,
        code: str,
        *,
        attempt: _AttemptContext,
    ) -> dict[str, Any]:
        workspace = attempt.workspace
        output_dir = attempt.output_dir
        run_path = workspace / "run.py"
        run_path.write_text(code, encoding="utf-8")
        remote_workspace = attempt.scm_workspace
        if remote_workspace is None:
            raise RuntimeError("SCM attempt is missing its remote workspace")
        self._sync_workspace_to_scm(workspace, remote_workspace)
        remote_task_root = self._resolve_scm_task_root()
        remote_problem = f"{remote_task_root}/problem"
        image = str(self.cfg.get("docker_image") or "cnsbench-base:v3")
        timeout_seconds = self._execution_timeout_seconds()
        container_name = "-".join(
            [
                "airaevo",
                self._safe_path_component(self.batch_name)[:40],
                self._safe_path_component(self.task_name)[:80],
                str(attempt.index),
            ]
        )
        docker_parts = [
            "timeout",
            "-k",
            "30s",
            f"{timeout_seconds}s",
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--add-host=host.docker.internal:host-gateway",
            "-e",
            "DATA_DIR=/task/problem/data",
            "-e",
            "OUTPUT_DIR=/workspace/output",
            "-e",
            f"EVAL_SERVICE_URL={self.scm_container_eval_service_url}",
            "-e",
            "NONINTERACTIVE=1",
            "-e",
            "CI=1",
            "-e",
            "NVIDIA_DISABLE_REQUIRE=1",
        ]
        gpu_setup_script, gpu_docker_args = self._scm_gpu_setup_and_docker_args()
        docker_parts.extend(gpu_docker_args)
        docker_parts.extend(
            [
                "-v",
                f"{remote_problem}:/task/problem:ro",
                "-v",
                f"{remote_workspace}:/workspace",
                image,
                "bash",
                "-lc",
                "python /workspace/run.py",
            ]
        )
        docker_command = _shell_join(docker_parts)
        remote_command = self._wrap_scm_docker_command(
            docker_command=docker_command,
            container_name=container_name,
            gpu_setup_script=gpu_setup_script,
        )
        started = time.monotonic()
        process = self._ssh(
            remote_command,
            timeout=self._scm_docker_command_timeout_seconds(timeout_seconds),
        )
        raw_log = (process.stdout or "") + (process.stderr or "")
        run_time = time.monotonic() - started
        gpu_wait_seconds, gpu_wait_timed_out, clear_log = (
            self._extract_gpu_wait_seconds(raw_log)
        )
        gpu_wait_seconds = min(max(0.0, gpu_wait_seconds), run_time)
        status_code = 200 if process.returncode == 0 else 500
        if process.returncode == 124 or gpu_wait_timed_out:
            status_code = 504
        return {
            "status_code": status_code,
            "status": (
                "success"
                if process.returncode == 0
                else ("timeout" if status_code == 504 else "failed")
            ),
            "raw_run_log": raw_log,
            "clear_run_log": clear_log,
            "feedback": clear_log,
            "run_time": run_time,
            "active_run_time": max(0.0, run_time - gpu_wait_seconds),
            "gpu_wait_seconds": gpu_wait_seconds,
            "output_dir": str(output_dir),
            "remote_workspace": remote_workspace,
            "eval_output_dir": f"{remote_workspace}/output",
        }

    def _run_solution(
        self,
        code: str,
        *,
        phase: str,
        attempt: _AttemptContext,
    ) -> dict[str, Any]:
        workspace = attempt.workspace
        output_dir = attempt.output_dir
        if self.execution_mode == "local":
            return self._run_local_solution(
                code, workspace=workspace, output_dir=output_dir
            )
        if self._uses_scm():
            return self._run_scm_docker_solution(code, attempt=attempt)
        return self._run_docker_solution(
            code,
            workspace=workspace,
            output_dir=output_dir,
            attempt_index=attempt.index,
        )

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: int | float,
    ) -> dict[str, Any]:
        if self._uses_scm():
            return self._remote_json_request(endpoint, payload, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        headers.update(self._eval_request_headers(endpoint))
        request = urllib.request.Request(
            f"{self.eval_service_url}/{endpoint.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"NatureBench eval service /{endpoint} failed with "
                f"HTTP {exc.code}: {body}"
            ) from exc

    def _eval_request_headers(self, endpoint: str) -> dict[str, str]:
        normalized_endpoint = endpoint.strip("/")
        if (
            self._uses_token_eval_contract()
            and normalized_endpoint in _EVAL_CONTROL_ENDPOINTS
        ):
            return {_EVAL_CONTROL_TOKEN_HEADER: self._load_eval_control_token()}
        return {}

    def _registration_out_dir_for_output(self, output_dir: str | Path) -> str:
        if not self._uses_token_eval_contract():
            return self._eval_service_out_dir_value()
        output_path = Path(str(output_dir))
        workspace = output_path.parent
        if output_path.name != "output" or workspace.name != "workspace":
            raise RuntimeError(
                "Token-based NatureBench evaluation requires an "
                "<attempt>/workspace/output directory"
            )
        return str(workspace.parent)

    def _ensure_eval_service_registered(
        self,
        *,
        out_dir: str | None = None,
        refresh: bool = False,
    ) -> None:
        with self._eval_service_lock:
            if self._eval_service_registered and not refresh:
                return
            payload: dict[str, Any] = {
                "task_name": self.task_name,
                "data_dir": (
                    self._resolve_scm_task_root()
                    if self._uses_scm()
                    else str(Path(self.task_dir).resolve())
                ),
                "timeout": self._solve_timeout_seconds(),
                "out_dir": out_dir or self._eval_service_out_dir_value(),
                "batch_name": self.batch_name,
            }
            if self._uses_token_eval_contract():
                payload["eval_token"] = self._load_or_create_eval_token()
            if (
                bool(self.cfg.get("force_eval_register", False))
                and not self._eval_service_registered
            ):
                payload["force"] = True
            response = self._post_json("register", payload, timeout=10)
            status = response.get("status")
            if status not in (None, "ok"):
                raise RuntimeError(
                    f"NatureBench eval service register failed: {response}"
                )
            self._eval_service_registered = True
            self._start_eval_service_timer()

    def _start_eval_service_timer(self) -> None:
        with self._eval_service_lock:
            if self._eval_service_timer_started:
                return
            self._post_json(
                "start_timer",
                {"task_name": self.task_name, "batch_name": self.batch_name},
                timeout=10,
            )
            self._eval_service_timer_started = True

    def _post_evaluate(self, output_dir: str | Path) -> dict[str, Any]:
        with self._eval_service_lock:
            if self._uses_token_eval_contract():
                self._ensure_eval_service_registered(
                    out_dir=self._registration_out_dir_for_output(output_dir),
                    refresh=True,
                )
            else:
                self._ensure_eval_service_registered()
            payload = {
                "task_name": self.task_name,
                "batch_name": self.batch_name,
                "output_dir": str(output_dir),
            }
            if self._uses_token_eval_contract():
                payload["eval_token"] = self._load_or_create_eval_token()
            return self._post_json("evaluate", payload, timeout=self.eval_timeout)

    def _annotate_eval_payload(
        self,
        run_payload: dict[str, Any],
        eval_payload: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        aggregate = self._coerce_score(eval_payload.get("aggregate_improvement"))
        best = self._coerce_score(eval_payload.get("best_aggregate_improvement"))
        run_status_code = int(run_payload.get("status_code", 500))
        run_succeeded = run_status_code in {0, 200}
        status_code = (
            200 if run_succeeded and aggregate is not None else run_status_code
        )
        selection_score = (
            self._selection_score_from_aggregate(aggregate)
            if phase == "validation"
            else None
        )
        selection_reward = (
            float(selection_score) if selection_score is not None else 0.0
        )
        reward = (
            selection_reward
            if phase == "validation"
            else (float(aggregate) if aggregate is not None else 0.0)
        )
        selection_score_source = None
        if phase == "validation":
            selection_score_source = (
                "aggregate_improvement_clipped"
                if selection_score is not None and selection_score != aggregate
                else "aggregate_improvement"
            )
        feedback_parts = [
            str(run_payload.get("feedback") or ""),
            f"aggregate_improvement={aggregate}",
            f"best_aggregate_improvement={best}",
        ]
        payload = {
            **run_payload,
            "status_code": status_code,
            "score_protocol": "naturebench",
            "score": aggregate,
            "raw_score": aggregate,
            "aggregate_improvement": aggregate,
            "raw_aggregate_improvement": aggregate,
            "best_aggregate_improvement": best,
            "raw_scores": eval_payload.get("raw_scores") or {},
            "per_instance_improvement": eval_payload.get("per_instance_improvement")
            or {},
            "selection_score": selection_score,
            "selection_reward": selection_reward,
            "selection_score_source": selection_score_source,
            "selection_score_clip": self._coerce_score(
                self.cfg.get("selection_score_clip", 1.0)
            ),
            "validation_score": selection_score,
            "validation_reward": selection_reward,
            "submit_score": aggregate if phase == "test" else None,
            "submit_reward": reward if phase == "test" else 0.0,
            "submit_score_source": "aggregate_improvement" if phase == "test" else None,
            "reward": reward,
            "attempt": eval_payload.get("attempt"),
            "phase": phase,
            "feedback": "\n".join(part for part in feedback_parts if part),
        }
        return payload

    def _preflight_failure_payload(
        self,
        preflight: dict[str, Any],
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        elapsed = max(0.0, float(preflight.get("elapsed_seconds") or 0.0))
        feedback = str(preflight.get("feedback") or "Candidate preflight failed.")
        return {
            "status_code": 422,
            "status": "preflight_failed",
            "raw_run_log": feedback,
            "clear_run_log": feedback,
            "feedback": feedback,
            "run_time": elapsed,
            "active_run_time": elapsed,
            "gpu_wait_seconds": 0.0,
            "preflight": dict(preflight),
            "preflight_category": str(preflight.get("category") or "unknown"),
            "output_dir": str(output_dir),
        }

    def _record_validation_timing(
        self,
        payload: dict[str, Any],
    ) -> tuple[float, float, float]:
        run_time = max(0.0, float(payload.get("run_time") or 0.0))
        active_value = payload.get("active_run_time")
        active_run_time = max(
            0.0,
            float(run_time if active_value is None else active_value),
        )
        gpu_wait_seconds = min(
            run_time,
            max(0.0, float(payload.get("gpu_wait_seconds") or 0.0)),
        )
        preflight = payload.get("preflight")
        preflight_time = 0.0
        if isinstance(preflight, dict):
            preflight_time = max(0.0, float(preflight.get("elapsed_seconds") or 0.0))
        with self._state_lock:
            self.validation_time_used += active_run_time
            self.validation_gpu_wait_time += gpu_wait_seconds
            self.validation_preflight_time_used += preflight_time
            if (
                self.time_budget is not None
                and self.validation_time_used >= self.time_budget
            ):
                self.stop_requested = True
            return (
                self.validation_time_used,
                self.validation_gpu_wait_time,
                self.validation_preflight_time_used,
            )

    def evaluate_code(self, code: str, *, phase: str) -> dict[str, Any]:
        attempt = self._start_attempt(phase=phase)
        output_dir = attempt.output_dir
        preflight = self._candidate_preflight(code)
        if preflight.get("ok", False):
            run_payload = self._run_solution(
                code,
                phase=phase,
                attempt=attempt,
            )
            run_payload["preflight"] = dict(preflight)
        else:
            run_payload = self._preflight_failure_payload(
                preflight, output_dir=output_dir
            )
        run_payload.setdefault("output_dir", str(output_dir))
        run_payload.setdefault("gpu_wait_seconds", 0.0)
        run_payload.setdefault("active_run_time", run_payload.get("run_time") or 0.0)
        if preflight.get("ok", False):
            eval_output_dir = str(run_payload.get("eval_output_dir") or output_dir)
            eval_payload = self._post_evaluate(eval_output_dir)
        else:
            eval_payload = {}
        payload = self._annotate_eval_payload(run_payload, eval_payload, phase=phase)
        payload["attempt_index"] = attempt.index
        payload["workspace"] = str(attempt.workspace)
        if phase == "validation":
            (
                validation_time_used,
                validation_gpu_wait_time,
                validation_preflight_time_used,
            ) = self._record_validation_timing(payload)
            payload["validation_time_used"] = validation_time_used
            payload["validation_gpu_wait_time"] = validation_gpu_wait_time
            payload["validation_preflight_time_used"] = validation_preflight_time_used
        return payload

    def _build_execution_result(self, eval_payload: dict[str, Any]) -> ExecutionResult:
        output = str(
            eval_payload.get("clear_run_log")
            or eval_payload.get("feedback")
            or json.dumps(eval_payload, ensure_ascii=False)
        )
        active_value = eval_payload.get("active_run_time")
        return ExecutionResult(
            term_out=[output],
            exec_time=float(
                active_value
                if active_value is not None
                else (eval_payload.get("run_time") or 0.0)
            ),
            exit_code=0 if eval_payload.get("status_code") == 200 else 1,
            timed_out=bool(eval_payload.get("status_code") == 504),
        )

    def step_task(
        self,
        state: dict[str, Any],
        action: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        eval_payload = self.evaluate_code(str(action), phase="validation")
        state["validation_time_used"] = float(
            eval_payload.get("validation_time_used") or 0.0
        )
        state["validation_gpu_wait_time"] = float(
            eval_payload.get("validation_gpu_wait_time") or 0.0
        )
        state["naturebench_attempt_index"] = int(eval_payload["attempt_index"])
        valid_solution = eval_payload["status_code"] == 200
        validation_fitness = (
            eval_payload.get("selection_score") if valid_solution else None
        )
        return state, {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            VALIDATION_FITNESS: validation_fitness,
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }

    def evaluate_fitness(
        self,
        solution: Any | None = None,
        state: dict[str, Any] | None = None,
        interpreter: Interpreter | None = None,
        aux_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if solution is None:
            return {}
        eval_payload = self.evaluate_code(str(solution), phase="test")
        valid_solution = eval_payload["status_code"] == 200
        return {
            EXECUTION_OUTPUT: self._build_execution_result(eval_payload),
            TEST_FITNESS: (
                eval_payload.get("aggregate_improvement") if valid_solution else None
            ),
            AUX_EVAL_INFO: dict(eval_payload),
            VALID_SOLUTION: valid_solution,
            VALID_SOLUTION_FEEDBACK: str(eval_payload.get("feedback") or ""),
        }

    def close(self, state: dict[str, Any]) -> None:
        for interp_key in ("solver_interpreter", "eval_interpreter"):
            interpreter = state.get(interp_key)
            if interpreter is None:
                continue
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()
            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
