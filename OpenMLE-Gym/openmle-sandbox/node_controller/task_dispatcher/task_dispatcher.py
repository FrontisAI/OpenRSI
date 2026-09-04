import contextlib
import json
import os
import re
import shlex
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import psycopg2
import redis
from psycopg2.extras import RealDictCursor

try:
    from agent_sandbox import Sandbox as LegacySandbox
except ImportError:
    LegacySandbox = None


# --- 配置 ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "sandbox")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_SANDBOX_CONFIG_FILE = CONFIG_DIR / "sandbox_config.json"
SANDBOX_CONFIG_FILE = Path(os.getenv("SANDBOX_CONFIG_FILE", str(DEFAULT_SANDBOX_CONFIG_FILE)))


def _sanitize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def _deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _normalize_host(value: Optional[str]) -> str:
    if not value:
        raise RuntimeError("host is required in sandbox config entries.")
    host = value.strip()
    if not host:
        raise RuntimeError("host is required in sandbox config entries.")
    return host


def _load_sandbox_config(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise RuntimeError(f"Sandbox config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse sandbox config file {path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read sandbox config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Sandbox config file {path} must contain a JSON object.")
    return data


def _expand_range_entry(entry: dict) -> List[str]:
    host = _normalize_host(entry.get("host"))
    try:
        start = int(entry["start"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("range entries require integer 'start'.") from exc

    end_value = entry.get("end")
    count_value = entry.get("count")
    if end_value is not None and count_value is not None:
        raise RuntimeError("range entry cannot define both 'end' and 'count'.")
    if end_value is None and count_value is None:
        raise RuntimeError("range entry must define either 'end' or 'count'.")

    if end_value is not None:
        try:
            end = int(end_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("'end' must be an integer.") from exc
    else:
        try:
            count = int(count_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("'count' must be an integer.") from exc
        if count <= 0:
            raise RuntimeError("'count' must be positive.")
        end = start + count - 1

    if end < start:
        raise RuntimeError("'end' must be greater than or equal to 'start'.")

    return [f"http://{host}:{port}" for port in range(start, end + 1)]


def _expand_ports_entry(entry: dict) -> List[str]:
    host = _normalize_host(entry.get("host"))
    ports = entry.get("ports")
    if not isinstance(ports, list) or not ports:
        raise RuntimeError("'ports' entry must include a non-empty list of ports.")

    endpoints: List[str] = []
    for port in ports:
        try:
            endpoints.append(f"http://{host}:{int(port)}")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid port value '{port}' for host {host}.") from exc
    return endpoints


def _collect_endpoints(config: Dict[str, dict], resource_type: str, *, allow_empty: bool = False) -> List[str]:
    section = config.get(resource_type)
    if not isinstance(section, dict):
        raise RuntimeError(f"Sandbox config missing '{resource_type}' section.")

    endpoints: List[str] = []
    for raw in section.get("endpoints") or []:
        if not isinstance(raw, str):
            raise RuntimeError(f"'endpoints' entries must be strings ({resource_type}).")
        endpoints.append(_sanitize_endpoint(raw.strip()))
    for entry in section.get("ranges") or []:
        if not isinstance(entry, dict):
            raise RuntimeError(f"'ranges' entries must be objects ({resource_type}).")
        endpoints.extend(_expand_range_entry(entry))
    for entry in section.get("hosts") or []:
        if not isinstance(entry, dict):
            raise RuntimeError(f"'hosts' entries must be objects ({resource_type}).")
        endpoints.extend(_expand_ports_entry(entry))

    endpoints = [_sanitize_endpoint(ep) for ep in endpoints if ep]
    endpoints = _deduplicate_preserve_order(endpoints)
    if not endpoints:
        if allow_empty:
            return []
        raise RuntimeError(f"No endpoints configured for resource type '{resource_type}'.")
    return endpoints


_SANDBOX_CONFIG = _load_sandbox_config(SANDBOX_CONFIG_FILE)
GPU_SANDBOX_ENDPOINTS = _collect_endpoints(_SANDBOX_CONFIG, "gpu", allow_empty=True)
CPU_SANDBOX_ENDPOINTS = _collect_endpoints(_SANDBOX_CONFIG, "cpu", allow_empty=True)
if not GPU_SANDBOX_ENDPOINTS and not CPU_SANDBOX_ENDPOINTS:
    raise RuntimeError("No endpoints configured for any resource type (gpu/cpu).")

SANDBOX_ENDPOINTS_BY_TYPE: Dict[str, List[str]] = {
    "gpu": GPU_SANDBOX_ENDPOINTS,
    "cpu": CPU_SANDBOX_ENDPOINTS,
}

JOB_QUEUE_CPU_P1 = "job_queue_cpu_p1"
JOB_QUEUE_CPU_P2 = "job_queue_cpu_p2"
JOB_QUEUE_GPU_P1 = "job_queue_gpu_p1"
JOB_QUEUE_GPU_P2 = "job_queue_gpu_p2"
JOB_QUEUE_BY_RESOURCE_PRIORITY = {
    "cpu": {1: JOB_QUEUE_CPU_P1, 2: JOB_QUEUE_CPU_P2},
    "gpu": {1: JOB_QUEUE_GPU_P1, 2: JOB_QUEUE_GPU_P2},
}
GPU_WORKER_QUEUE_KEY = "available_workers_gpu"
CPU_WORKER_QUEUE_KEY = "available_workers_cpu"
WORKER_QUEUE_MAP = {"gpu": GPU_WORKER_QUEUE_KEY, "cpu": CPU_WORKER_QUEUE_KEY}
WORKER_TYPE_REGISTRY: Dict[str, str] = {}
RESOURCE_TYPES = {"gpu", "cpu"}
PRIORITY_LEVELS = {1, 2}
CONTROL_ENV_KEYS = {"EXECUTION_MODE", "EXECUTION_COMMAND", "EXECUTION_WORKDIR"}
DEFAULT_STORAGE_ROOT = Path("/mnt/pubdatasets2/mlsandbox")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", str(DEFAULT_STORAGE_ROOT))).resolve()
JOB_BASE_PATH = STORAGE_PATH / "jobs"
ENABLE_LOCAL_JOB_SCRATCH = os.getenv("ENABLE_LOCAL_JOB_SCRATCH", "0").lower() in {"1", "true", "yes", "on"}
LOCAL_SCRATCH_ROOT = Path(os.getenv("LOCAL_SCRATCH_ROOT", "/mnt/local_sandbox_workdir")).resolve()
LOCAL_SCRATCH_JOB_SUBDIR = os.getenv("LOCAL_SCRATCH_JOB_SUBDIR", "jobs").strip("/") or "jobs"
LOCAL_SCRATCH_MIN_FREE_BYTES = int(os.getenv("LOCAL_SCRATCH_MIN_FREE_BYTES", str(100 * 1024 * 1024 * 1024)))
LOCAL_SCRATCH_STAGE_TIMEOUT_SECONDS = int(os.getenv("LOCAL_SCRATCH_STAGE_TIMEOUT_SECONDS", "120"))
LOCAL_DISPATCHER_LOG_ROOT = Path(os.getenv("LOCAL_DISPATCHER_LOG_ROOT", "/tmp/mlsandbox_dispatcher_job_logs"))
EVALUATION_SCRIPT = Path(
    os.getenv(
        "EVALUATION_SCRIPT_PATH",
        "/mnt/pubdatasets2/mlsandbox/workdir/evaluation/read_and_metric.py",
    )
).resolve()

RESULT_SUCCESS = "success"
RESULT_NO_CODE = "code_missing"
RESULT_CODE_ERROR = "code_execution_error"
RESULT_NO_SUBMISSION = "submission_missing"
RESULT_MISSING_EVAL_RESOURCE = "metric_or_answer_missing"
RESULT_SCORING_ERROR = "scoring_failed"
RESULT_TIMEOUT = "timeout"
RESULT_SANDBOX_UNCONFIRMED = "sandbox_unconfirmed"

DEFAULT_SHELL_POLL_SECONDS = max(1, int(os.getenv("SANDBOX_POLL_INTERVAL", "5")))
COMMAND_DONE_STATUSES = {"completed", "terminated", "hard_timeout"}
COMMAND_PENDING_STATUSES = {"running", "no_change_timeout"}
HTTP_REQUEST_TIMEOUT = float(os.getenv("SANDBOX_HTTP_TIMEOUT", "120"))
MAX_CONSECUTIVE_HTTP_ERRORS = int(os.getenv("SANDBOX_MAX_POLL_ERRORS", "5"))
SANDBOX_STDOUT_FILENAME = os.getenv("SANDBOX_STDOUT_FILENAME", "sandbox_stdout.log")
SANDBOX_STDOUT_TAIL_BYTES = 1024 * 1024
GEM_USER_ENV_CLEAN_PATHS = (
    "/home/gem/.local/lib",
    "/home/gem/.local/bin",
    "/home/gem/.cache/pip",
    "/home/gem/.pip",
    "/home/gem/.config/pip",
    "/home/gem/.ipython",
    "/home/gem/.keras",
    "/home/gem/.triton",
    "/home/gem/.nv",
    "/home/gem/nltk_data",
    "/home/gem/.bash_history",
    "/home/gem/.cache/huggingface",
    "/home/gem/.cache/torch",
    "/home/gem/.cache/matplotlib",
    "/home/gem/.cache/joblib",
)
FAKE_SUCCESS_MAX_ATTEMPTS = max(1, int(os.getenv("FAKE_SUCCESS_MAX_ATTEMPTS", "10")))
EXEC_MARKER_VISIBILITY_WAIT_SECONDS = max(0.0, float(os.getenv("EXEC_MARKER_VISIBILITY_WAIT_SECONDS", "5")))
SUBMISSION_VISIBILITY_WAIT_SECONDS = max(0.0, float(os.getenv("SUBMISSION_VISIBILITY_WAIT_SECONDS", "5")))
FILE_VISIBILITY_POLL_SECONDS = max(0.1, float(os.getenv("FILE_VISIBILITY_POLL_SECONDS", "0.2")))
NO_WORKER_SLEEP_SECONDS = 0.2
WORKER_QUARANTINE_SECONDS = max(1, int(os.getenv("WORKER_QUARANTINE_SECONDS", "60")))
WORKER_RECOVERY_CHECK_SECONDS = max(1.0, float(os.getenv("WORKER_RECOVERY_CHECK_SECONDS", "15")))
WORKER_RECOVERY_HTTP_TIMEOUT = float(os.getenv("WORKER_RECOVERY_HTTP_TIMEOUT", "10"))
WORKER_RECOVERY_COMMAND = os.getenv("WORKER_RECOVERY_COMMAND", "bash -lc 'true'")
WORKER_RECOVERY_GPU_COMMAND = os.getenv(
    "WORKER_RECOVERY_GPU_COMMAND",
    """python - <<'PY'
import torch
assert torch.cuda.is_available(), "cuda not available"
x = torch.tensor([1.0], device="cuda")
torch.cuda.synchronize()
print(x.item())
PY""",
)
QUARANTINED_WORKERS: Dict[str, Dict[str, object]] = {}
QUARANTINED_WORKERS_LOCK = threading.Lock()


def local_job_root(job_id: str, job_storage_dir: Optional[Path | str] = None) -> Path:
    if job_storage_dir is not None:
        try:
            relative = Path(job_storage_dir).resolve().relative_to(JOB_BASE_PATH.resolve())
            if relative.parts and relative.parts[-1] == job_id:
                return LOCAL_SCRATCH_ROOT / LOCAL_SCRATCH_JOB_SUBDIR / relative
        except ValueError:
            pass
    return LOCAL_SCRATCH_ROOT / LOCAL_SCRATCH_JOB_SUBDIR / job_id


def local_code_dir(job_id: str, job_storage_dir: Optional[Path | str] = None) -> Path:
    return local_job_root(job_id, job_storage_dir) / "code"


def local_outputs_dir(job_id: str, job_storage_dir: Optional[Path | str] = None) -> Path:
    return local_job_root(job_id, job_storage_dir) / "outputs"


def local_stdout_path(job_id: str, job_storage_dir: Optional[Path | str] = None) -> Path:
    return local_job_root(job_id, job_storage_dir) / SANDBOX_STDOUT_FILENAME


def local_submission_path(job_id: str, job_storage_dir: Optional[Path | str] = None) -> Path:
    return local_code_dir(job_id, job_storage_dir) / "submission.csv"


def local_dispatcher_log_path(job_id: str) -> Path:
    return LOCAL_DISPATCHER_LOG_ROOT / f"{job_id}.logs.txt"


def build_gem_user_env_cleanup_command() -> str:
    paths = " ".join(shlex.quote(path) for path in GEM_USER_ENV_CLEAN_PATHS)
    return (
        f"if ! rm -rf -- {paths}; then "
        'echo "[WARN] failed to clean gem user environment before job execution" >&2; '
        "fi; "
    )


def build_stage_to_local_command(
    *,
    job_id: str,
    source_code_path: Path,
    job_storage_dir: Path,
    dataset_label: Optional[str],
    trace_id: Optional[str],
    done_marker_path: Path,
) -> str:
    root = local_job_root(job_id, job_storage_dir)
    code_dir = local_code_dir(job_id, job_storage_dir)
    outputs_dir = local_outputs_dir(job_id, job_storage_dir)
    source = Path(source_code_path)
    dest = code_dir / source.name
    metadata = {
        "job_id": job_id,
        "source_code_path": str(source),
        "local_job_root": str(root),
        "dataset_label": dataset_label,
        "trace_id": trace_id,
        "created_at": current_timestamp(),
    }
    script = (
        "set -euo pipefail; "
        f"test -d {shlex.quote(str(LOCAL_SCRATCH_ROOT))}; "
        f"test -w {shlex.quote(str(LOCAL_SCRATCH_ROOT))}; "
        f"avail=$(df -PB1 {shlex.quote(str(LOCAL_SCRATCH_ROOT))} | awk 'NR==2 {{print $4}}'); "
        f"if [ \"${{avail:-0}}\" -lt {LOCAL_SCRATCH_MIN_FREE_BYTES} ]; then "
        f"echo 'local scratch free bytes below threshold: '${{avail:-0}} >&2; exit 75; fi; "
        f"rm -rf {shlex.quote(str(root))}; "
        f"mkdir -p {shlex.quote(str(code_dir))} {shlex.quote(str(outputs_dir))}; "
        f"cp {shlex.quote(str(source))} {shlex.quote(str(dest))}; "
        f"cat > {shlex.quote(str(root / 'metadata.json'))} <<'JSON'\n"
        f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n"
        "JSON\n"
        f"chmod -R 777 {shlex.quote(str(root))}; "
        f"test -f {shlex.quote(str(dest))}; "
        f"touch {shlex.quote(str(done_marker_path))}; "
    )
    return "bash -lc " + shlex.quote(script)


class NoWorkerConfiguredError(RuntimeError):
    def __init__(self, resource_type: str):
        message = f"No {resource_type.upper()} sandbox workers are configured."
        super().__init__(message)
        self.resource_type = resource_type


class SandboxStartupError(RuntimeError):
    def __init__(self, worker_endpoint: str, message: str):
        super().__init__(message)
        self.worker_endpoint = _sanitize_endpoint(worker_endpoint)


class RequeueJobOnWorkerStartupError(RuntimeError):
    def __init__(self, job_data: dict, worker_endpoint: str, message: str):
        super().__init__(message)
        self.job_data = job_data
        self.worker_endpoint = _sanitize_endpoint(worker_endpoint)
        self.job_id = job_data.get("job_id")



def normalize_resource_type(value: Optional[str]) -> str:
    if not value:
        return "cpu"
    value = value.strip().lower()
    return value if value in RESOURCE_TYPES else "cpu"


def normalize_priority(value) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError):
        return 1
    return priority if priority in PRIORITY_LEVELS else 1


def queue_for_resource(resource_type: str, priority: int) -> str:
    normalized = normalize_resource_type(resource_type)
    normalized_priority = normalize_priority(priority)
    return JOB_QUEUE_BY_RESOURCE_PRIORITY[normalized][normalized_priority]


def has_available_worker(redis_client: redis.Redis, resource_type: str) -> bool:
    queue_name = WORKER_QUEUE_MAP.get(resource_type)
    if not queue_name:
        return False
    try:
        return redis_client.llen(queue_name) > 0
    except Exception:
        return False


def current_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def worker_id_from_endpoint(endpoint: str) -> str:
    return endpoint.replace("://", "_").replace(":", "_")


def record_worker_state(
    redis_client: redis.Redis, worker_endpoint: str, status: str, resource_type: Optional[str] = None
) -> None:
    worker_endpoint = _sanitize_endpoint(worker_endpoint)
    worker_id = worker_id_from_endpoint(worker_endpoint)
    resource_type = resource_type or WORKER_TYPE_REGISTRY.get(worker_endpoint)
    timestamp = current_timestamp()
    redis_client.set(f"worker:{worker_id}:status", status)
    redis_client.set(f"worker:{worker_id}:heartbeat", timestamp)
    redis_client.set(f"worker:{worker_id}:endpoint", worker_endpoint)
    if resource_type:
        redis_client.set(f"worker:{worker_id}:resource_type", resource_type)


def prepare_job_storage(job_data: dict, job_id: str) -> Tuple[Path, Path]:
    job_storage_dir = Path(job_data.get("job_storage_dir") or (JOB_BASE_PATH / job_id))
    job_storage_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_storage_dir / "logs.txt"
    return job_storage_dir, log_path


def fail_job_for_unavailable_worker(job_data: dict, message: str) -> None:
    job_id = job_data.get("job_id") or "unknown"
    _, log_path = prepare_job_storage(job_data, job_id)
    append_job_log(log_path, "FAILED", message)

    conn = get_db_connection()
    finish_time = current_timestamp()
    result_payload = {
        "score": None,
        "run_log": message,
        "result": RESULT_CODE_ERROR,
    }
    try:
        update_job_status(
            conn,
            job_id,
            "failed",
            error_message=message,
            updated_at=finish_time,
            completed_at=finish_time,
            result=json.dumps(result_payload, ensure_ascii=False),
        )
    finally:
        with contextlib.suppress(Exception):
            conn.close()
    print(f"[dispatcher] job {job_id} failed to start: {message}")


def acquire_worker_for_job(redis_client: redis.Redis, resource_type: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = normalize_resource_type(resource_type)
    endpoints = SANDBOX_ENDPOINTS_BY_TYPE.get(normalized) or []
    if not endpoints:
        raise NoWorkerConfiguredError(normalized)
    queue_name = WORKER_QUEUE_MAP[normalized]
    worker_endpoint = redis_client.lpop(queue_name)
    if not worker_endpoint:
        return None, None
    return _sanitize_endpoint(worker_endpoint), normalized


def requeue_job_to_original_queue(redis_client: redis.Redis, db_conn, job_data: dict, reason: str) -> str:
    job_id = job_data.get("job_id") or "unknown"
    resource_type = normalize_resource_type(job_data.get("resource_type"))
    priority = normalize_priority(job_data.get("priority"))
    queue_name = queue_for_resource(resource_type, priority)
    job_payload = json.dumps(job_data, ensure_ascii=False)
    redis_client.rpush(queue_name, job_payload)
    update_job_status(
        db_conn,
        job_id,
        "queued",
        worker_id=None,
        started_at=None,
        completed_at=None,
        updated_at=current_timestamp(),
        error_message=None,
        result=None,
    )
    print(
        f"[dispatcher-requeue] action=requeue job={job_id} queue={queue_name} "
        f"resource={resource_type} priority={priority} reason={reason}"
    )
    return queue_name


def quarantine_worker(
    redis_client: redis.Redis,
    worker_endpoint: str,
    worker_type: str,
    reason: str,
    *,
    job_id: Optional[str] = None,
) -> None:
    worker_endpoint = _sanitize_endpoint(worker_endpoint)
    queue_name = WORKER_QUEUE_MAP.get(worker_type)
    retry_after = time.monotonic() + WORKER_QUARANTINE_SECONDS
    metadata = {
        "resource_type": worker_type,
        "retry_after": retry_after,
        "reason": reason,
        "job_id": job_id,
        "quarantined_at": current_timestamp(),
    }
    with QUARANTINED_WORKERS_LOCK:
        QUARANTINED_WORKERS[worker_endpoint] = metadata

    if queue_name:
        redis_client.lrem(queue_name, 0, worker_endpoint)
    record_worker_state(redis_client, worker_endpoint, "quarantined", worker_type)
    print(
        f"[worker-quarantine] action=isolate worker={worker_endpoint} type={worker_type} "
        f"seconds={WORKER_QUARANTINE_SECONDS} job={job_id or '-'} reason={reason}"
    )


def get_due_quarantined_workers() -> List[Tuple[str, dict]]:
    now = time.monotonic()
    with QUARANTINED_WORKERS_LOCK:
        return [
            (endpoint, dict(metadata))
            for endpoint, metadata in QUARANTINED_WORKERS.items()
            if float(metadata.get("retry_after", 0.0)) <= now
        ]


def probe_worker_ready(worker_endpoint: str, worker_type: str) -> None:
    controller = SandboxHTTPController(worker_endpoint, timeout_seconds=WORKER_RECOVERY_HTTP_TIMEOUT)
    session_id = None
    command = WORKER_RECOVERY_GPU_COMMAND if worker_type == "gpu" else WORKER_RECOVERY_COMMAND
    try:
        payload = controller.exec_command(command=command, exec_dir=None)
        session_id = payload.get("session_id")
        if session_id:
            response = controller.wait(session_id, max(1, int(WORKER_RECOVERY_HTTP_TIMEOUT)))
            if response:
                status = response.get("status")
                if status not in COMMAND_DONE_STATUSES:
                    raise SandboxStartupError(
                        worker_endpoint,
                        f"recovery probe did not finish cleanly (status={status})",
                    )
            snapshot = controller.view(session_id)
            if snapshot:
                status = snapshot.get("status")
                exit_code = snapshot.get("exit_code")
                stderr_text = snapshot.get("stderr") or snapshot.get("error")
                if status not in COMMAND_DONE_STATUSES or exit_code not in (None, 0):
                    detail = f"recovery probe failed (status={status}, exit_code={exit_code})"
                    if isinstance(stderr_text, str) and stderr_text.strip():
                        detail += f": {stderr_text.strip()}"
                    raise SandboxStartupError(worker_endpoint, detail)
    except SandboxHTTPError as exc:
        raise SandboxStartupError(worker_endpoint, str(exc)) from exc
    finally:
        if session_id:
            with contextlib.suppress(Exception):
                controller.cleanup(session_id)


def _worker_recovery_loop():
    redis_client = get_redis_connection()
    print("Worker recovery loop started.")

    while True:
        try:
            due_workers = get_due_quarantined_workers()
            if not due_workers:
                time.sleep(WORKER_RECOVERY_CHECK_SECONDS)
                continue

            for worker_endpoint, metadata in due_workers:
                worker_type = normalize_resource_type(str(metadata.get("resource_type")))
                print(
                    f"[worker-recovery] action=probe worker={worker_endpoint} type={worker_type} "
                    f"last_reason={metadata.get('reason', '')}"
                )
                try:
                    probe_worker_ready(worker_endpoint, worker_type)
                except Exception as exc:
                    with QUARANTINED_WORKERS_LOCK:
                        if worker_endpoint in QUARANTINED_WORKERS:
                            QUARANTINED_WORKERS[worker_endpoint]["retry_after"] = (
                                time.monotonic() + WORKER_QUARANTINE_SECONDS
                            )
                            QUARANTINED_WORKERS[worker_endpoint]["reason"] = str(exc)
                    record_worker_state(redis_client, worker_endpoint, "quarantined", worker_type)
                    print(
                        f"[worker-recovery] action=retry_later worker={worker_endpoint} type={worker_type} "
                        f"seconds={WORKER_QUARANTINE_SECONDS} error={exc}"
                    )
                    continue

                queue_name = WORKER_QUEUE_MAP.get(worker_type)
                with QUARANTINED_WORKERS_LOCK:
                    QUARANTINED_WORKERS.pop(worker_endpoint, None)
                if queue_name:
                    redis_client.lrem(queue_name, 0, worker_endpoint)
                    redis_client.rpush(queue_name, worker_endpoint)
                record_worker_state(redis_client, worker_endpoint, "idle", worker_type)
                print(
                    f"[worker-recovery] action=restore worker={worker_endpoint} type={worker_type} "
                    f"queue={queue_name or '-'}"
                )

            time.sleep(WORKER_RECOVERY_CHECK_SECONDS)
        except Exception as exc:
            print(f"[worker-recovery] loop error: {exc}")
            print(traceback.format_exc())
            time.sleep(5)
            redis_client = get_redis_connection()


def append_job_log(log_path: Path, header: str, body: str | None = None) -> None:
    log_path = Path(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(f"[{current_timestamp()}] {header}\n")
            if body:
                fp.write(body.rstrip() + "\n")
            fp.write("\n")
    except OSError as exc:
        print(f"写入日志失败: {log_path} ({exc})")


def to_builtin(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return to_builtin(value.model_dump())
        except Exception as e:
            print(f"Warning: Failed to convert model_dump for {type(value)}: {e}")
    if hasattr(value, "dict"):
        try:
            return to_builtin(value.dict())
        except Exception as e:
            print(f"Warning: Failed to convert dict for {type(value)}: {e}")
    if hasattr(value, "__dict__"):
        return {
            k: to_builtin(v)
            for k, v in value.__dict__.items()
            if not callable(v) and not k.startswith("_")
        }
    return str(value)


def serialize_result(result, execution_mode: str):
    payload = None
    log_sections: list[str] = []

    if hasattr(result, "data"):
        payload = to_builtin(result.data)
    else:
        payload = to_builtin(result)

    if payload is not None:
        try:
            log_sections.append(json.dumps(payload, ensure_ascii=False, indent=2))
        except TypeError:
            log_sections.append(str(payload))

    if isinstance(payload, dict):
        output = payload.get("output")
        if output:
            log_sections.append(
                "\n--- OUTPUT START ---\n"
                + (output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)) + "\n--- OUTPUT END ---\n"
            )
        stderr = payload.get("stderr") or payload.get("error")
        if stderr:
            log_sections.append(
                "--- STDERR ---\n"
                + (stderr if isinstance(stderr, str) else json.dumps(stderr, ensure_ascii=False))
            )

    log_text = "\n\n".join(log_sections) if log_sections else ""
    return payload, log_text


class SandboxHTTPError(RuntimeError):
    pass


class SandboxSessionLostError(RuntimeError):
    def __init__(self, session_id: str, message: str, view=None):
        super().__init__(message)
        self.session_id = session_id
        self.view = view


class SandboxCommandTimeoutError(RuntimeError):
    def __init__(self, session_id: str, message: str, view=None):
        super().__init__(message)
        self.session_id = session_id
        self.view = view


class SandboxCommandCancelledError(RuntimeError):
    def __init__(self, session_id: str, message: str, view=None):
        super().__init__(message)
        self.session_id = session_id
        self.view = view


class SandboxResultWrapper:
    def __init__(self, data: dict | None):
        self.data = data or {}


def read_sandbox_stdout(path: Path, max_bytes: int = SANDBOX_STDOUT_TAIL_BYTES) -> Optional[str]:
    try:
        with path.open("rb") as fp:
            if max_bytes is not None and max_bytes > 0:
                fp.seek(0, os.SEEK_END)
                size = fp.tell()
                if size > max_bytes:
                    fp.seek(-max_bytes, os.SEEK_END)
                else:
                    fp.seek(0)
            content = fp.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    content = content.strip()
    return content or None


def wait_for_file_exists(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = FILE_VISIBILITY_POLL_SECONDS,
    deadline: Optional[float] = None,
) -> bool:
    wait_until = time.monotonic() + max(0.0, timeout_seconds)
    if deadline is not None:
        wait_until = min(wait_until, deadline)

    while True:
        if path.exists():
            return True
        remaining = wait_until - time.monotonic()
        if remaining <= 0:
            return path.exists()
        time.sleep(min(poll_seconds, remaining))


def _remaining_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _extract_payload(response_json):
    if isinstance(response_json, dict):
        data = response_json.get("data")
        if isinstance(data, dict):
            return data
    return response_json


class SandboxHTTPController:
    def __init__(self, base_url: str, timeout_seconds: Optional[float] = None):
        self.base_url = base_url.rstrip("/")
        timeout_seconds = timeout_seconds if timeout_seconds is not None else HTTP_REQUEST_TIMEOUT
        self.timeout = httpx.Timeout(timeout_seconds, connect=10, read=timeout_seconds, write=timeout_seconds)

    def _request(self, path: str, payload: dict, *, allow_status: Optional[set[int]] = None):
        url = f"{self.base_url}{path}"
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if allow_status and exc.response is not None and exc.response.status_code in allow_status:
                return {"__http_status__": exc.response.status_code}
            raise SandboxHTTPError(f"请求 {url} 失败: {exc}") from exc
        except httpx.RequestError as exc:
            raise SandboxHTTPError(f"请求 {url} 失败: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SandboxHTTPError(f"沙盒返回非 JSON 数据: {url}") from exc

    def exec_command(self, *, command: str, exec_dir: Optional[str]) -> dict:
        payload = {
            "id": "",
            "command": command,
            "exec_dir": exec_dir,
            "async_mode": True,
            "timeout": HTTP_REQUEST_TIMEOUT,
        }
        data = _extract_payload(self._request("/v1/shell/exec", payload))
        if not isinstance(data, dict) or not data.get("session_id"):
            raise SandboxHTTPError("沙盒未返回有效的 session_id。")
        print(f"[sandbox-http] exec {self.base_url} -> session {data['session_id']}")
        return data

    def wait(self, session_id: str, seconds: int) -> Optional[dict]:
        payload = {"id": session_id, "seconds": seconds}
        data = self._request("/v1/shell/wait", payload, allow_status={404})
        if data and "__http_status__" in data:
            return None
        payload = _extract_payload(data)
        if payload:
            status = payload.get("status")
            print(f"[sandbox-http] wait {session_id} -> {status}")
        return payload

    def view(self, session_id: str) -> Optional[dict]:
        payload = {"id": session_id}
        data = self._request("/v1/shell/view", payload, allow_status={404})
        if data and "__http_status__" in data:
            return None
        payload = _extract_payload(data)
        if payload:
            status = payload.get("status")
            print(f"[sandbox-http] view {session_id} -> {status}")
        return payload

    def kill(self, session_id: str) -> None:
        payload = {"id": session_id}
        with contextlib.suppress(Exception):
            self._request("/v1/shell/kill", payload, allow_status={404})
            print(f"[sandbox-http] kill {session_id}")

    def cleanup(self, session_id: str) -> None:
        with contextlib.suppress(Exception):
            url = f"{self.base_url}/v1/shell/sessions/{session_id}"
            httpx.delete(url, timeout=self.timeout)


def execute_shell_command_with_deadline(
    *,
    worker_endpoint: str,
    command: str,
    exec_dir: Optional[str],
    job_id: str,
    redis_client: redis.Redis,
    db_conn,
    job_deadline: Optional[float],
    on_started: Optional[Callable[[str], None]] = None,
) -> SandboxResultWrapper:
    controller = SandboxHTTPController(worker_endpoint)
    try:
        exec_payload = controller.exec_command(command=command, exec_dir=exec_dir)
    except SandboxHTTPError as exc:
        raise SandboxStartupError(worker_endpoint, str(exc)) from exc
    session_id = exec_payload["session_id"]
    if on_started is not None:
        on_started(session_id)
    print(f"[dispatcher] job {job_id} session_id={session_id}")
    last_snapshot = exec_payload
    consecutive_errors = 0

    try:
        status = exec_payload.get("status")
        if status in COMMAND_DONE_STATUSES:
            final_view = controller.view(session_id)
            if final_view:
                last_snapshot = final_view
            print(f"[sandbox-http] job {job_id} session {session_id} completed immediately")
            return SandboxResultWrapper(last_snapshot)

        while True:
            remaining = _remaining_seconds(job_deadline)
            if remaining is not None and remaining <= 0:
                controller.kill(session_id)
                print(f"[sandbox-http] job {job_id} session {session_id} timeout")
                raise SandboxCommandTimeoutError(
                    session_id,
                    f"Job {job_id} exceeded timeout while executing command.",
                    view=SandboxResultWrapper(last_snapshot),
                )

            if job_cancelled(redis_client, db_conn, job_id):
                controller.kill(session_id)
                print(f"[sandbox-http] job {job_id} session {session_id} cancelled")
                raise SandboxCommandCancelledError(
                    session_id,
                    f"Job {job_id} was cancelled.",
                    view=SandboxResultWrapper(last_snapshot),
                )

            wait_seconds = DEFAULT_SHELL_POLL_SECONDS
            if remaining is not None:
                wait_seconds = max(1, min(DEFAULT_SHELL_POLL_SECONDS, int(remaining)))

            wait_response = controller.wait(session_id, wait_seconds)
            if wait_response:
                status = wait_response.get("status", status)

            view_response = controller.view(session_id)
            if view_response:
                last_snapshot = view_response
                status = view_response.get("status", status)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_HTTP_ERRORS:
                    print(f"[sandbox-http] job {job_id} session {session_id} lost after {consecutive_errors} errors")
                    raise SandboxSessionLostError(
                        session_id,
                        "Lost connection to sandbox session.",
                        view=SandboxResultWrapper(last_snapshot),
                    )

            if status in COMMAND_DONE_STATUSES:
                print(f"[sandbox-http] job {job_id} session {session_id} finished with status {status}")
                return SandboxResultWrapper(last_snapshot)

            if status not in COMMAND_PENDING_STATUSES:
                print(f"[sandbox-http] job {job_id} session {session_id} unexpected status {status}")
                return SandboxResultWrapper(last_snapshot)
    finally:
        controller.cleanup(session_id)


def extract_score_from_output(output: str) -> float | None:
    if not output:
        return None
    # 匹配形如 ##SCORE##0.8156，允许前后有其他字符
    pattern = r"##SCORE##\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, output)
    if not matches:
        return None
    try:
        return float(matches[-1])  # 如果有多次打印，取最后一次
    except ValueError:
        return None


def job_cancelled(redis_client: redis.Redis, db_conn, job_id: str) -> bool:
    if redis_client.get(f"job:{job_id}:cancelled") == "1":
        return True
    try:
        with db_conn.cursor() as cursor:
            cursor.execute("SELECT status FROM jobs WHERE job_id = %s", (job_id,))
            row = cursor.fetchone()
            if row and row.get("status") == "cancelled":
                return True
    except psycopg2.Error as exc:
        print(f"查询任务状态失败 (Job: {job_id}): {exc}")
        db_conn.rollback()
    return False


def load_code_content(path: str) -> str:
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read()


def update_job_status(conn, job_id: str, status: str, **kwargs) -> None:
    fields = [f"{key} = %s" for key in kwargs.keys()]
    set_clause_parts = ["status = %s"] + fields
    query = f"UPDATE jobs SET {', '.join(set_clause_parts)} WHERE job_id = %s"
    params = [status] + list(kwargs.values()) + [job_id]

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()
    except Exception as exc:
        print(f"数据库更新失败 (Job: {job_id}): {exc}")
        conn.rollback()


def get_db_connection():
    for _ in range(5):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                cursor_factory=RealDictCursor,
            )
            print("数据库连接成功")
            return conn
        except psycopg2.OperationalError as exc:
            print(f"数据库连接失败，正在重试... ({exc})")
            time.sleep(3)
    print("无法连接到数据库，退出。")
    sys.exit(1)


def get_redis_connection():
    for _ in range(5):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            print("Redis 连接成功")
            return r
        except redis.exceptions.ConnectionError as exc:
            print(f"Redis 连接失败，正在重试... ({exc})")
            time.sleep(3)
    print("无法连接到 Redis，退出。")
    sys.exit(1)


def initialize_worker_queue(redis_client: redis.Redis):
    for queue_name in WORKER_QUEUE_MAP.values():
        redis_client.delete(queue_name)
    WORKER_TYPE_REGISTRY.clear()

    total_workers = 0
    for resource_type, endpoints in SANDBOX_ENDPOINTS_BY_TYPE.items():
        queue_name = WORKER_QUEUE_MAP[resource_type]
        sanitized_endpoints = [_sanitize_endpoint(endpoint) for endpoint in endpoints]
        for endpoint in sanitized_endpoints:
            WORKER_TYPE_REGISTRY[endpoint] = resource_type
            redis_client.rpush(queue_name, endpoint)
            record_worker_state(redis_client, endpoint, "idle", resource_type)
        total_workers += len(sanitized_endpoints)
        if sanitized_endpoints:
            print(f"已初始化 {len(sanitized_endpoints)} 个 {resource_type.upper()} worker 到队列中。")
    if total_workers == 0:
        print("警告: 未配置任何 sandbox worker endpoint。")
    else:
        print(f"初始化成功，共有 worker {total_workers} 个。")



def execute_job(redis_client: redis.Redis, db_conn, worker_endpoint: str, job_data: dict):
    job_id = job_data["job_id"]
    code_file_path = job_data.get("code_file_path")
    environment: Dict[str, str] = job_data.get("environment") or {}
    execution_mode = environment.get("EXECUTION_MODE", "jupyter")
    custom_command = environment.get("EXECUTION_COMMAND")
    user_working_dir = job_data.get("working_dir")
    trace_id = job_data.get("trace_id") or environment.get("TRACE_ID")
    use_local_scratch = ENABLE_LOCAL_JOB_SCRATCH and execution_mode == "shell"
    job_storage_dir = Path(job_data.get("job_storage_dir") or (JOB_BASE_PATH / job_id))
    job_storage_dir.mkdir(parents=True, exist_ok=True)
    log_path = local_dispatcher_log_path(job_id) if use_local_scratch else job_storage_dir / "logs.txt"
    print(f"[dispatcher] job {job_id} storage_dir={job_storage_dir} logs={log_path}")
    sandbox_stdout_path = job_storage_dir / SANDBOX_STDOUT_FILENAME
    if use_local_scratch:
        sandbox_stdout_path = local_stdout_path(job_id, job_storage_dir)
    else:
        with contextlib.suppress(OSError):
            sandbox_stdout_path.write_text("", encoding="utf-8")
            sandbox_stdout_path.chmod(0o666)

    if job_cancelled(redis_client, db_conn, job_id):
        append_job_log(log_path, "CANCELLED", "任务在调度前已取消")
        update_job_status(db_conn, job_id, "cancelled", worker_id=None, updated_at=current_timestamp())
        redis_client.delete(f"job:{job_id}:cancelled")
        print(f"Job {job_id} 被标记为已取消，未执行。")
        return

    working_dir = environment.get("EXECUTION_WORKDIR")
    runtime_env = {
        key: value
        for key, value in environment.items()
        if key not in CONTROL_ENV_KEYS and value is not None
    }
    data_dir = job_data.get("data_dir") or environment.get("DATA_DIR") or environment.get("SANDBOX_DATA_DIR")
    code_file_path = job_data["code_file_path"]
    timeout_raw = job_data.get("timeout")
    job_timeout: Optional[int]
    try:
        job_timeout = int(timeout_raw) if timeout_raw is not None else None
    except (TypeError, ValueError):
        job_timeout = None
    if job_timeout is not None and job_timeout <= 0:
        job_timeout = None
    job_deadline = time.monotonic() + job_timeout if job_timeout else None
    job_marked_running = False

    def mark_job_running_once(session_id: Optional[str] = None) -> None:
        nonlocal job_marked_running
        if job_marked_running:
            return
        start_time = current_timestamp()
        update_job_status(
            db_conn,
            job_id,
            "running",
            worker_id=worker_endpoint,
            started_at=start_time,
            updated_at=start_time,
        )
        message = f"Worker {worker_endpoint}, mode={execution_mode}"
        if session_id:
            message += f", session={session_id}"
        append_job_log(log_path, "RUNNING", message)
        print(
            f"[dispatcher] job {job_id} started on {worker_endpoint}, "
            f"mode={execution_mode}, timeout={job_timeout}, session={session_id or '-'}"
        )
        job_marked_running = True

    run_log_sections: list[str] = []
    score: float | None = None
    result_status = RESULT_SUCCESS
    error_message: Optional[str] = None
    submission_marker_path: Optional[Path] = None

    nfs_code_path = Path(code_file_path)
    code_path = nfs_code_path
    if code_path.is_dir():
        submission_path = code_path / "submission.csv"
    else:
        submission_path = code_path.parent / "submission.csv"

    code_empty = False
    if code_path.is_file():
        try:
            code_text = code_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            code_text = code_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            code_text = None
        if code_text is not None and not code_text.strip():
            code_empty = True

    try:
        evaluation_notes: list[str] = []
        legacy_client = None
        if execution_mode != "shell":
            if LegacySandbox is None:
                raise RuntimeError("当前模式需要 agent_sandbox 支持，但未安装。")
            legacy_client = LegacySandbox(base_url=worker_endpoint)

        if use_local_scratch:
            if custom_command:
                raise RuntimeError("local scratch canary does not support custom EXECUTION_COMMAND.")
            if user_working_dir:
                raise RuntimeError("local scratch canary does not support custom working_dir.")
            if not nfs_code_path.is_file():
                raise RuntimeError(f"local scratch canary only supports single code file: {nfs_code_path}")

            max_stage_attempts = FAKE_SUCCESS_MAX_ATTEMPTS
            stage_attempt = 1
            while stage_attempt <= max_stage_attempts:
                stage_done_marker_path = job_storage_dir / f".sandbox_stage_done_attempt_{stage_attempt}"
                stage_start = time.monotonic()
                stage_command = build_stage_to_local_command(
                    job_id=job_id,
                    source_code_path=nfs_code_path,
                    job_storage_dir=job_storage_dir,
                    dataset_label=job_data.get("dataset_label"),
                    trace_id=trace_id,
                    done_marker_path=stage_done_marker_path,
                )
                append_job_log(log_path, "LOCAL_STAGE", f"attempt={stage_attempt}: {stage_command}")
                stage_deadline = time.monotonic() + max(1, LOCAL_SCRATCH_STAGE_TIMEOUT_SECONDS)
                if job_deadline is not None:
                    stage_deadline = min(stage_deadline, job_deadline)
                try:
                    stage_result = execute_shell_command_with_deadline(
                        worker_endpoint=worker_endpoint,
                        command=stage_command,
                        exec_dir=None,
                        job_id=job_id,
                        redis_client=redis_client,
                        db_conn=db_conn,
                        job_deadline=stage_deadline,
                        on_started=mark_job_running_once,
                    )
                except SandboxStartupError as exc:
                    if not job_marked_running:
                        requeue_message = (
                            f"worker={worker_endpoint} startup_failed_before_local_stage: {exc}"
                        )
                        append_job_log(log_path, "REQUEUE", requeue_message)
                        print(f"[dispatcher] job {job_id} will be requeued: {requeue_message}")
                        raise RequeueJobOnWorkerStartupError(job_data, worker_endpoint, str(exc)) from exc
                    raise

                stage_payload, stage_log = serialize_result(stage_result, "shell")
                stage_exit_code = stage_payload.get("exit_code") if isinstance(stage_payload, dict) else None
                stage_duration = time.monotonic() - stage_start
                append_job_log(log_path, "DURATION", f"stage_to_local attempt={stage_attempt} took {stage_duration:.2f}s")
                print(
                    f"[dispatcher] [DURATION]job {job_id} stage_to_local attempt={stage_attempt} "
                    f"took {stage_duration:.2f}s on {worker_endpoint}"
                )
                if stage_exit_code not in (None, 0):
                    raise RuntimeError(
                        "stage code to local scratch failed"
                        + (f": {stage_log}" if stage_log else "")
                    )

                stage_attempt_completed = wait_for_file_exists(
                    stage_done_marker_path,
                    timeout_seconds=EXEC_MARKER_VISIBILITY_WAIT_SECONDS,
                    deadline=job_deadline,
                )
                if not stage_attempt_completed and stage_attempt < max_stage_attempts:
                    retry_msg = (
                        "疑似 local scratch stage 假完成："
                        f"exit_code={stage_exit_code}，"
                        f"stage marker 缺失，stage 用时 {stage_duration:.2f}s，"
                        f"自动重试（attempt {stage_attempt+1}/{max_stage_attempts}）。"
                    )
                    append_job_log(log_path, "RETRY", retry_msg)
                    run_log_sections.append("--- RETRY ---\n" + retry_msg)
                    print(f"[dispatcher] [LOCAL-STAGE-RETRY] job {job_id}: {retry_msg}")
                    stage_attempt += 1
                    time.sleep(3.0)
                    continue
                if not stage_attempt_completed:
                    msg = (
                        "sandbox_unconfirmed: stage main.py 到 local scratch 阶段无法确认完成；"
                        f"已尝试执行 {stage_attempt}/{max_stage_attempts} 次，"
                        f"最后一次 exit_code={stage_exit_code}，"
                        f"缺少 stage marker: {stage_done_marker_path}"
                    )
                    append_job_log(log_path, "FAILED", msg)
                    run_log_sections.append("--- SANDBOX UNCONFIRMED ---\n" + msg)
                    result_status = RESULT_SANDBOX_UNCONFIRMED
                    error_message = msg
                break

            if result_status == RESULT_SUCCESS:
                local_code_path = local_code_dir(job_id, job_storage_dir) / nfs_code_path.name
                code_path = local_code_path
                code_file_path = str(local_code_path)
                submission_path = local_submission_path(job_id, job_storage_dir)
                sandbox_stdout_path = local_stdout_path(job_id, job_storage_dir)
                working_dir = str(local_code_dir(job_id, job_storage_dir))

        if result_status != RESULT_SUCCESS:
            pass
        elif code_empty:
            mark_job_running_once()
            msg = "提交的代码内容为空，未执行。"
            run_log_sections.append("--- EXECUTION ---\n" + msg)
            error_message = msg
            result_status = RESULT_NO_CODE
        else:
            if execution_mode == "shell":
                if custom_command is None and not use_local_scratch and not code_path.is_file():
                    raise FileNotFoundError(f"code file not found: {code_file_path}")
                base_command = custom_command or f"python {shlex.quote(str(code_file_path))}"
                command_parts = []
                for key, value in runtime_env.items():
                    command_parts.append(f"export {key}={shlex.quote(str(value))}")
                if command_parts:
                    command_parts.append(base_command)
                    command = " && ".join(command_parts)
                else:
                    command = base_command

                # 只添加DATA_DIR这个环境变量. client上传的代码里可以使用这个变量来访问数据目录
                # base_command_with_datadir = f"export DATA_DIR={shlex.quote(str(data_dir))}" + " && " + base_command
                base_command_with_datadir = f"export DATA_DIR={shlex.quote(os.path.join(str(data_dir), 'data', 'public'))}" + " && " + base_command

                def build_shell_command(done_marker_path: Path, submission_marker_path: Optional[Path] = None) -> str:
                    submission_marker_clause = ""
                    if submission_marker_path is not None:
                        submission_marker_clause = (
                            f"if [ -f {shlex.quote(str(submission_path))} ]; then "
                            f"touch {shlex.quote(str(submission_marker_path))}; fi; "
                        )
                    clean_gem_env_command = build_gem_user_env_cleanup_command()
                    stdout_capture = (
                        "set +e; "
                        f"{{ {clean_gem_env_command}{base_command_with_datadir}; }} 2>&1 | tee -a {shlex.quote(str(sandbox_stdout_path))}; "
                        "command_status=${PIPESTATUS[0]}; "
                        "set -e; "
                        f"if [ \"$command_status\" -eq 0 ]; then "
                        f"touch {shlex.quote(str(done_marker_path))}; "
                        f"{submission_marker_clause}"
                        "fi; "
                    )
                    return (
                        "bash -lc '"
                        "set -euo pipefail; "
                        "export PYTHONUNBUFFERED=1; "
                        # 心跳放后台（写到 stderr；间隔可通过 HEARTBEAT_INTERVAL 调整，默认 60s）
                        "( while :; do echo \"[HB] $(date +%s) This is Heart Beat to avoid no_change_timeout bug of AIO Sandbox.[HB]\" >&2; "
                        "  sleep ${HEARTBEAT_INTERVAL:-20}; "
                        "done ) & hb=$!; "
                        # 收到停止信号时先关心跳
                        "trap \"kill -TERM $hb 2>/dev/null\" TERM INT; "
                        # 让用户代码作为前台主进程运行；正常返回后写入本次 attempt 完成标记
                        f"{stdout_capture}"
                        # 收尾：停掉心跳，归还真实退出码
                        "kill $hb 2>/dev/null || true; "
                        "wait $hb 2>/dev/null || true; "
                        "exit $command_status'"
                    )

                result = None

                max_attempts = FAKE_SUCCESS_MAX_ATTEMPTS
                unnormal_time_threshold = 3.0
                last_duration = None
                execution_confirmed = False
                attempt = 1
                while attempt <= max_attempts:
                    result = None
                    done_marker_path = job_storage_dir / f".sandbox_exec_done_attempt_{attempt}"
                    submission_marker_path = (
                        job_storage_dir / f".sandbox_submission_ready_attempt_{attempt}"
                        if use_local_scratch
                        else None
                    )
                    command = build_shell_command(done_marker_path, submission_marker_path)
                    append_job_log(log_path, "COMMAND", f"attempt={attempt}: {command}")
                    start_run_code = time.monotonic()
                    try:
                        result = execute_shell_command_with_deadline(
                            worker_endpoint=worker_endpoint,
                            command=command,
                            exec_dir=working_dir,
                            job_id=job_id,
                            redis_client=redis_client,
                            db_conn=db_conn,
                            job_deadline=job_deadline,
                            on_started=mark_job_running_once,
                        )
                    except SandboxStartupError as exc:
                        if not job_marked_running:
                            requeue_message = (
                                f"worker={worker_endpoint} startup_failed_before_running: {exc}"
                            )
                            append_job_log(log_path, "REQUEUE", requeue_message)
                            print(f"[dispatcher] job {job_id} will be requeued: {requeue_message}")
                            raise RequeueJobOnWorkerStartupError(job_data, worker_endpoint, str(exc)) from exc
                        raise
                    except SandboxCommandCancelledError as exc:
                        snapshot = exc.view
                        cancel_message = "任务执行过程中被取消，沙盒进程已终止。"
                        append_job_log(log_path, "CANCELLED", cancel_message)
                        additional_log = None
                        if snapshot:
                            _, additional_log = serialize_result(snapshot, "shell")
                        combined_log = cancel_message
                        if additional_log:
                            combined_log = cancel_message + "\n" + additional_log
                            run_log_sections.append("--- EXECUTION ---\n" + additional_log)
                        stdout_excerpt = read_sandbox_stdout(sandbox_stdout_path)
                        if stdout_excerpt:
                            combined_log = combined_log + "\n--- SANDBOX STDOUT START ---\n" + stdout_excerpt + "\n--- SANDBOX STDOUT END ---\n"
                        finish_time = current_timestamp()
                        update_job_status(
                            db_conn,
                            job_id,
                            "cancelled",
                            worker_id=None,
                            updated_at=finish_time,
                            completed_at=finish_time,
                            result=json.dumps(
                                {
                                    "score": None,
                                    "run_log": combined_log,
                                    "result": RESULT_CODE_ERROR,
                                },
                                ensure_ascii=False,
                            ),
                        )
                        redis_client.delete(f"job:{job_id}:cancelled")
                        print(f"Job {job_id} 在执行过程中被取消，session={exc.session_id}.")
                        return
                    except SandboxCommandTimeoutError as exc:
                        timeout_message = f"任务执行超时，沙盒进程 {exc.session_id} 已终止。"
                        append_job_log(log_path, "TIMEOUT", timeout_message)
                        snapshot = exc.view
                        extra_log = None
                        if snapshot:
                            _, extra_log = serialize_result(snapshot, "shell")
                        combined_lines = [timeout_message]
                        if extra_log:
                            combined_lines.append(extra_log)
                        run_log_sections.append("--- EXECUTION ---\n" + "\n".join(combined_lines))
                        stdout_excerpt = read_sandbox_stdout(sandbox_stdout_path)
                        if stdout_excerpt:
                            run_log_sections.append("\n--- SANDBOX STDOUT START ---\n" + stdout_excerpt + "\n--- SANDBOX STDOUT END ---\n")
                        result_status = RESULT_TIMEOUT
                        error_message = timeout_message
                        result = snapshot
                    except SandboxSessionLostError as exc:
                        snapshot = exc.view
                        lost_message = f"与沙盒会话 {exc.session_id} 的连接中断，任务状态未知。"
                        append_job_log(log_path, "FAILED", lost_message)
                        if snapshot:
                            _, additional_log = serialize_result(snapshot, "shell")
                            if additional_log:
                                run_log_sections.append("--- EXECUTION ---\n" + additional_log)
                        result_status = RESULT_CODE_ERROR
                        error_message = lost_message
                        result = snapshot
                    finally:
                        duration_run_code = time.monotonic() - start_run_code
                        last_duration = duration_run_code
                        append_job_log(log_path, "DURATION", f"shell execution took {duration_run_code:.2f}s")
                        print(f"[dispatcher] [DURATION]job {job_id} shell execution took {duration_run_code:.2f}s on {worker_endpoint}")

                    exec_payload = None
                    exit_code = None
                    if result_status == RESULT_SUCCESS and result is not None:
                        exec_payload, _ = serialize_result(result, "shell")
                        if isinstance(exec_payload, dict):
                            exit_code = exec_payload.get("exit_code")

                    attempt_completed = False
                    if result_status == RESULT_SUCCESS and result is not None and exit_code in (None, 0):
                        attempt_completed = wait_for_file_exists(
                            done_marker_path,
                            timeout_seconds=EXEC_MARKER_VISIBILITY_WAIT_SECONDS,
                            deadline=job_deadline,
                        )
                        if attempt_completed:
                            execution_confirmed = True

                    if result_status != RESULT_SUCCESS or result is None:
                        break

                    if exit_code not in (None, 0):
                        break

                    if not attempt_completed:
                        if attempt < max_attempts:
                            retry_msg = (
                                "检测到异常：疑似沙盒假完成："
                                f"exit_code={exit_code}，"
                                f"本次 attempt 完成标记缺失，执行用时 {last_duration:.2f}s，"
                                f"自动重试（attempt {attempt+1}/{max_attempts}）。"
                            )
                            append_job_log(log_path, "RETRY", retry_msg)
                            run_log_sections.append("--- RETRY ---\n" + retry_msg)
                            print(f"[dispatcher] [RUNCODE-RETRY] job {job_id}: {retry_msg}")
                            attempt += 1
                            time.sleep(unnormal_time_threshold)
                            continue
                        if not execution_confirmed:
                            msg = (
                                "sandbox_unconfirmed: 执行用户代码阶段无法确认完成；"
                                f"exit_code={exit_code}，"
                                f"已尝试执行 {max_attempts}/{max_attempts} 次，均缺少执行完成标记，"
                                f"缺少 exec marker: {done_marker_path}"
                            )
                            append_job_log(log_path, "FAILED", msg)
                            run_log_sections.append("--- SANDBOX UNCONFIRMED ---\n" + msg)
                            result_status = RESULT_SANDBOX_UNCONFIRMED
                            error_message = msg
                        break

                    submission_attempt_completed = False
                    submission_ready_candidate = (
                        submission_marker_path
                        if use_local_scratch and submission_marker_path is not None
                        else submission_path
                    )
                    submission_attempt_completed = wait_for_file_exists(
                        submission_ready_candidate,
                        timeout_seconds=SUBMISSION_VISIBILITY_WAIT_SECONDS,
                        deadline=job_deadline,
                    )
                    if submission_attempt_completed:
                        break

                    short_runtime_retry = last_duration is not None and last_duration < unnormal_time_threshold
                    if short_runtime_retry and (attempt < max_attempts):
                        retry_msg = (
                            "检测到异常："
                            f"本次执行已确认完成，但用时 {last_duration:.2f}s < {unnormal_time_threshold}s，"
                            "且未确认产生 submission，"
                            f"自动重试（attempt {attempt+1}/{max_attempts}）。"
                        )
                        append_job_log(log_path, "RETRY", retry_msg)
                        run_log_sections.append("--- RETRY ---\n" + retry_msg)
                        print(f"[dispatcher] [RUNCODE-RETRY] job {job_id}: {retry_msg}")
                        attempt += 1
                        time.sleep(unnormal_time_threshold)
                        continue
                    # 已确认代码执行完成；如果仍未产出 submission，后续统一归类为 submission_missing。
                    break
                        
            else:
                source_path = code_path
                if not source_path.is_file():
                    raise FileNotFoundError(f"code file not found: {code_file_path}")
                code_content = load_code_content(code_file_path)
                if runtime_env:
                    assignments = "\n".join(
                        f"os.environ[{repr(key)}] = {repr(str(value))}"
                        for key, value in runtime_env.items()
                    )
                    code_content = f"import os\n{assignments}\n{code_content}"
                if legacy_client is None:
                    raise RuntimeError("Jupyter 模式不可用：缺少 Legacy sandbox client。")
                mark_job_running_once()
                result = legacy_client.jupyter.execute_code(code=code_content)

            if result_status == RESULT_SUCCESS:
                exec_payload, log_text = serialize_result(result, execution_mode)
                if log_text:
                    run_log_sections.append("--- EXECUTION ---\n" + log_text)

                execution_failed = False
                failure_details = None

                if execution_mode == "shell":
                    exit_code = None
                    stderr_text = None
                    if isinstance(exec_payload, dict):
                        exit_code = exec_payload.get("exit_code")
                        stderr_text = exec_payload.get("stderr") or exec_payload.get("error")
                    if exit_code not in (None, 0):
                        execution_failed = True
                        failure_details = f"代码执行失败 (exit_code={exit_code})."
                        if isinstance(stderr_text, str) and stderr_text.strip():
                            failure_details += "\n" + stderr_text.strip()
                else:
                    detail_text = None
                    if isinstance(exec_payload, dict):
                        status_value = str(exec_payload.get("status") or "").lower()
                        detail_text = exec_payload.get("error") or exec_payload.get("stderr") or exec_payload.get("traceback")
                        if status_value in {"error", "failed"} or (isinstance(detail_text, str) and detail_text.strip()):
                            execution_failed = True
                            failure_details = "代码执行失败。"
                            if isinstance(detail_text, str) and detail_text.strip():
                                failure_details += "\n" + detail_text.strip()

                if execution_failed:
                    result_status = RESULT_CODE_ERROR
                    error_message = failure_details or "代码执行失败。"
                    if failure_details:
                        evaluation_notes.append(failure_details)
                    evaluation_notes.append("代码执行失败，未进行评测。")

        if result_status == RESULT_SUCCESS:
            submission_ready_path = submission_marker_path if use_local_scratch and submission_marker_path else submission_path
            if not wait_for_file_exists(
                submission_ready_path,
                timeout_seconds=SUBMISSION_VISIBILITY_WAIT_SECONDS,
                deadline=job_deadline,
            ):
                result_status = RESULT_NO_SUBMISSION
                msg = f"未找到提交文件: {submission_path}"
                if use_local_scratch:
                    msg += f" (marker missing: {submission_ready_path})"
                evaluation_notes.append(msg)
                error_message = msg
            elif not data_dir:
                result_status = RESULT_MISSING_EVAL_RESOURCE
                msg = "未提供 DATA_DIR，无法执行评测。"
                evaluation_notes.append(msg)
                error_message = msg
            else:
                answer_path = Path(data_dir) / "data/private/test_answer.csv"
                metric_path = Path(data_dir) / "utils/metric.py"
                missing_resources = []
                if not answer_path.exists():
                    missing_resources.append(str(answer_path))
                if not metric_path.exists():
                    missing_resources.append(str(metric_path))

                if missing_resources:
                    result_status = RESULT_MISSING_EVAL_RESOURCE
                    msg = f"评测资源缺失: {', '.join(missing_resources)}"
                    evaluation_notes.append(msg)
                    error_message = msg
                else:
                    base_evaluation_cmd = (
                        f"python3.11 {shlex.quote(str(EVALUATION_SCRIPT))}"
                        f" -d {shlex.quote(data_dir)}"
                        f" -a {shlex.quote(str(answer_path))}"
                        f" -p {shlex.quote(str(submission_path))}"
                    )

                    def build_evaluation_shell_command(done_marker_path: Path) -> str:
                        return (
                            "bash -lc '"
                            "set -euo pipefail; "
                            "export PYTHONUNBUFFERED=1; "
                            "export PYTHONNOUSERSITE=1; "
                            "unset PYTHONPATH; "
                            "unset PYTHONHOME; "
                            "( while :; do echo \"[HB] $(date +%s) This is Heart Beat to avoid no_change_timeout bug of AIO Sandbox.[HB]\" >&2; "
                            "  sleep ${HEARTBEAT_INTERVAL:-20}; "
                            "done ) & hb=$!; "
                            "trap \"kill -TERM $hb 2>/dev/null\" TERM INT; "
                            "set +e; "
                            f"{base_evaluation_cmd}; "
                            "code=$?; "
                            "set -e; "
                            f"if [ \"$code\" -eq 0 ]; then touch {shlex.quote(str(done_marker_path))}; fi; "
                            "kill $hb 2>/dev/null || true; "
                            "wait $hb 2>/dev/null || true; "
                            "exit $code'"
                        )

                    evaluation_result = None
                    max_eval_attempts = FAKE_SUCCESS_MAX_ATTEMPTS
                    unnormal_eval_time_threshold = 3.0
                    eval_attempt = 1
                    last_eval_duration = None
                    while eval_attempt <= max_eval_attempts:
                        evaluation_result = None
                        eval_done_marker_path = job_storage_dir / f".sandbox_eval_done_attempt_{eval_attempt}"
                        evaluation_cmd = build_evaluation_shell_command(eval_done_marker_path)
                        append_job_log(log_path, "COMMAND", f"evaluation attempt={eval_attempt}: {evaluation_cmd}")
                        start_eval = time.monotonic()
                        try:
                            evaluation_result = execute_shell_command_with_deadline(
                                worker_endpoint=worker_endpoint,
                                command=evaluation_cmd,
                                exec_dir=working_dir,
                                job_id=job_id,
                                redis_client=redis_client,
                                db_conn=db_conn,
                                job_deadline=job_deadline,
                            )
                        except SandboxCommandCancelledError as exc:
                            cancel_message = "任务在评测阶段被取消，沙盒进程已终止。"
                            append_job_log(log_path, "CANCELLED", cancel_message)
                            snapshot = exc.view
                            if snapshot:
                                _, eval_log_text = serialize_result(snapshot, "shell")
                                if eval_log_text:
                                    evaluation_notes.append(eval_log_text)
                            evaluation_notes.append(cancel_message)
                            combined_sections = list(run_log_sections)
                            if evaluation_notes:
                                combined_sections.append(
                                    "\n--- EVALUATION START ---\n" + "\n".join(evaluation_notes) + "\n--- EVALUATION END ---\n"
                                )
                            stdout_excerpt = read_sandbox_stdout(sandbox_stdout_path)
                            if stdout_excerpt:
                                combined_sections.append(
                                    "\n--- SANDBOX STDOUT START ---\n"
                                    + stdout_excerpt
                                    + "\n--- SANDBOX STDOUT END ---\n"
                                )
                            combined_log = "\n\n".join(section for section in combined_sections if section and section.strip())
                            finish_time = current_timestamp()
                            update_job_status(
                                db_conn,
                                job_id,
                                "cancelled",
                                worker_id=None,
                                updated_at=finish_time,
                                completed_at=finish_time,
                                result=json.dumps(
                                    {
                                        "score": None,
                                        "run_log": combined_log or cancel_message,
                                        "result": RESULT_CODE_ERROR,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            redis_client.delete(f"job:{job_id}:cancelled")
                            print(f"Job {job_id} 在评测阶段被取消，session={exc.session_id}.")
                            return
                        except SandboxCommandTimeoutError as exc:
                            timeout_message = f"评测阶段超时，沙盒进程 {exc.session_id} 已终止。"
                            append_job_log(log_path, "TIMEOUT", timeout_message)
                            snapshot = exc.view
                            if snapshot:
                                _, eval_log_text = serialize_result(snapshot, "shell")
                                if eval_log_text:
                                    evaluation_notes.append(eval_log_text)
                            evaluation_notes.append(timeout_message)
                            stdout_excerpt = read_sandbox_stdout(sandbox_stdout_path)
                            if stdout_excerpt:
                                evaluation_notes.append(
                                    "\n--- SANDBOX STDOUT START ---\n" + stdout_excerpt + "\n--- SANDBOX STDOUT END ---\n"
                                )
                            result_status = RESULT_TIMEOUT
                            error_message = timeout_message
                            evaluation_result = None
                        except SandboxSessionLostError as exc:
                            lost_message = f"评测阶段与沙盒会话 {exc.session_id} 的连接中断。"
                            append_job_log(log_path, "FAILED", lost_message)
                            snapshot = exc.view
                            if snapshot:
                                _, eval_log_text = serialize_result(snapshot, "shell")
                                if eval_log_text:
                                    evaluation_notes.append(eval_log_text)
                            evaluation_notes.append(lost_message)
                            result_status = RESULT_CODE_ERROR
                            error_message = lost_message
                            evaluation_result = None
                        except Exception as exc:
                            result_status = RESULT_SCORING_ERROR
                            msg = f"评测过程中出现异常: {exc}"
                            evaluation_notes.append(msg)
                            error_message = msg
                        finally:
                            duration_eval = time.monotonic() - start_eval
                            last_eval_duration = duration_eval
                            append_job_log(
                                log_path,
                                "DURATION",
                                f"evaluation execution took {duration_eval:.2f}s",
                            )
                            print(
                                f"[dispatcher] [DURATION]job {job_id} evaluation execution took {duration_eval:.2f}s on {worker_endpoint}"
                            )

                        eval_payload_for_retry = None
                        eval_exit_code = None
                        if result_status == RESULT_SUCCESS and evaluation_result is not None:
                            eval_payload_for_retry, _ = serialize_result(evaluation_result, "shell")
                            if isinstance(eval_payload_for_retry, dict):
                                eval_exit_code = eval_payload_for_retry.get("exit_code")

                        eval_attempt_completed = False
                        if result_status == RESULT_SUCCESS and evaluation_result is not None and eval_exit_code in (None, 0):
                            eval_attempt_completed = wait_for_file_exists(
                                eval_done_marker_path,
                                timeout_seconds=EXEC_MARKER_VISIBILITY_WAIT_SECONDS,
                                deadline=job_deadline,
                            )
                        if eval_attempt_completed:
                            break

                        fake_eval_success_retry = (
                            result_status == RESULT_SUCCESS
                            and evaluation_result is not None
                            and eval_exit_code in (None, 0)
                            and not eval_attempt_completed
                        )

                        if fake_eval_success_retry and eval_attempt < max_eval_attempts:
                            retry_msg = (
                                "检测到异常：疑似评测假完成："
                                f"exit_code={eval_exit_code}，"
                                f"本次 attempt 完成标记缺失，评测用时 {last_eval_duration:.2f}s，"
                                f"自动重试（attempt {eval_attempt+1}/{max_eval_attempts}）。"
                            )
                            append_job_log(log_path, "RETRY", retry_msg)
                            evaluation_notes.append(retry_msg)
                            print(f"[dispatcher] [EVAL-RETRY] job {job_id}: {retry_msg}")
                            eval_attempt += 1
                            time.sleep(unnormal_eval_time_threshold)
                            result_status = RESULT_SUCCESS
                            error_message = None
                            continue
                        if fake_eval_success_retry:
                            msg = (
                                "sandbox_unconfirmed: 评测阶段无法确认完成；"
                                f"exit_code={eval_exit_code}，"
                                f"已尝试执行 {max_eval_attempts}/{max_eval_attempts} 次，均缺少评测完成标记，"
                                f"缺少 eval marker: {eval_done_marker_path}"
                            )
                            append_job_log(log_path, "FAILED", msg)
                            evaluation_notes.append(msg)
                            result_status = RESULT_SANDBOX_UNCONFIRMED
                            error_message = msg
                        break
                    if result_status == RESULT_SUCCESS and evaluation_result is not None:
                        eval_payload, eval_log = serialize_result(evaluation_result, "shell")

                        collected_outputs: list[str] = []
                        seen_outputs: set[str] = set()

                        def _collect(value):
                            if not value:
                                return
                            if isinstance(value, str):
                                text_value = value.strip()
                            else:
                                try:
                                    text_value = json.dumps(value, ensure_ascii=False)
                                except Exception:
                                    text_value = str(value)
                            if not text_value or text_value in seen_outputs:
                                return
                            seen_outputs.add(text_value)
                            collected_outputs.append(text_value)

                        if isinstance(eval_payload, dict):
                            for key in ("output", "stdout", "stderr", "error", "text"):
                                _collect(eval_payload.get(key))
                            for entry in eval_payload.get("console") or []:
                                command_text = entry.get("command")
                                output_text = entry.get("output")
                                if command_text:
                                    _collect(f"$ {command_text}")
                                _collect(output_text)
                        elif isinstance(eval_payload, list):
                            for item in eval_payload:
                                _collect(item)
                        else:
                            _collect(eval_payload)

                        _collect(eval_log)

                        evaluation_output = "\n".join(collected_outputs).strip()
                        if evaluation_output:
                            evaluation_notes.append(evaluation_output)

                        score = extract_score_from_output(evaluation_output)
                        if score is None:
                            result_status = RESULT_SCORING_ERROR
                            msg = "评测完成但未能解析得分。"
                            evaluation_notes.append(msg)
                            error_message = msg
                        else:
                            result_status = RESULT_SUCCESS
                            evaluation_notes.append(f"解析得分: {score}")

        if evaluation_notes:
            run_log_sections.append("\n--- EVALUATION START ---\n" + "\n".join(evaluation_notes) + "\n--- EVALUATION END ---\n")

        combined_log = "\n\n".join(section for section in run_log_sections if section.strip())
        log_header = (
            "SUCCESS" if result_status == RESULT_SUCCESS else "TIMEOUT" if result_status == RESULT_TIMEOUT else "COMPLETED"
        )
        append_job_log(log_path, log_header, combined_log or "任务执行完成。")

        result_record = {
            "score": score,
            "run_log": combined_log or "任务执行完成。",
            "result": result_status,
        }
        if use_local_scratch:
            result_record.update(
                {
                    "artifact_policy": "local_scratch",
                    "worker_endpoint": worker_endpoint,
                    "local_scratch_path": str(local_job_root(job_id, job_storage_dir)),
                    "local_submission_path": str(local_submission_path(job_id, job_storage_dir)),
                    "nfs_job_storage_dir": str(job_storage_dir),
                    "dispatcher_log_path": str(log_path),
                }
            )
        finish_time = current_timestamp()
        status_kwargs = {
            "result": json.dumps(result_record, ensure_ascii=False),
            "completed_at": finish_time,
            "updated_at": finish_time,
        }
        final_status = "completed" if result_status == RESULT_SUCCESS else "failed"
        if error_message and final_status != "completed":
            status_kwargs["error_message"] = error_message

        update_job_status(
            db_conn,
            job_id,
            final_status,
            **status_kwargs,
        )
        redis_client.delete(f"job:{job_id}:cancelled")
        if final_status == "completed":
            print(f"Job {job_id} 执行并评测完成，Score={score}.")
        else:
            print(f"Job {job_id} 完成但评测失败: {error_message or '未知原因'}")
        print(f"[dispatcher] job {job_id} finished with status {final_status}")
    except RequeueJobOnWorkerStartupError:
        raise
    except Exception as exc:
        error_trace = ''.join(traceback.format_exception(exc))
        append_job_log(log_path, "FAILED", error_trace)
        finish_time = current_timestamp()
        result_record = {
            "score": None,
            "run_log": error_trace,
            "result": RESULT_CODE_ERROR,
        }
        if use_local_scratch:
            result_record.update(
                {
                    "artifact_policy": "local_scratch",
                    "worker_endpoint": worker_endpoint,
                    "local_scratch_path": str(local_job_root(job_id, job_storage_dir)),
                    "local_submission_path": str(local_submission_path(job_id, job_storage_dir)),
                    "nfs_job_storage_dir": str(job_storage_dir),
                    "dispatcher_log_path": str(log_path),
                }
            )
        update_job_status(
            db_conn,
            job_id,
            "failed",
            error_message=str(exc),
            result=json.dumps(result_record, ensure_ascii=False),
            completed_at=finish_time,
            updated_at=finish_time,
        )
        print(f"Job {job_id} 执行失败: {exc}")
        redis_client.delete(f"job:{job_id}:cancelled")


def _run_worker_job(worker_endpoint: str, worker_type: str, job_data: dict):
    redis_conn = get_redis_connection()
    db_conn = get_db_connection()
    should_return_worker = True
    try:
        execute_job(redis_conn, db_conn, worker_endpoint, job_data)
    except RequeueJobOnWorkerStartupError as exc:
        should_return_worker = False
        queue_name = requeue_job_to_original_queue(redis_conn, db_conn, exc.job_data, str(exc))
        quarantine_worker(
            redis_conn,
            worker_endpoint,
            worker_type,
            str(exc),
            job_id=exc.job_id,
        )
        print(
            f"[dispatcher] startup failure handled for job {exc.job_id or '-'} on {worker_endpoint}; "
            f"requeued to {queue_name}"
        )
    except Exception as exc:
        print(f"[dispatcher] worker thread error on {worker_endpoint}: {exc}")
        print(traceback.format_exc())
    finally:
        try:
            if should_return_worker:
                record_worker_state(redis_conn, worker_endpoint, "idle", worker_type)
                queue_name = WORKER_QUEUE_MAP.get(worker_type)
                if queue_name:
                    redis_conn.rpush(queue_name, worker_endpoint)
        except Exception as inner_exc:
            print(f"[dispatcher] failed to return worker {worker_endpoint}: {inner_exc}")
        finally:
            with contextlib.suppress(Exception):
                redis_conn.close()
            with contextlib.suppress(Exception):
                db_conn.close()


def _dispatch_job_async(worker_endpoint: str, worker_type: str, job_data: dict):
    thread_name = f"worker-{worker_type}-{worker_endpoint.split(':')[-1]}"
    thread = threading.Thread(
        target=_run_worker_job,
        args=(worker_endpoint, worker_type, job_data),
        daemon=True,
        name=thread_name,
    )
    thread.start()
    print(f"[dispatcher] dispatched job {job_data.get('job_id')} to {worker_endpoint} (thread {thread.name})")


def _resource_dispatch_loop(resource_type: str):
    redis_client = get_redis_connection()
    queue_names = [
        queue_for_resource(resource_type, 1),
        queue_for_resource(resource_type, 2),
    ]
    queue_names = [name for name in queue_names if name]

    print(f"Task dispatcher loop started for {resource_type.upper()} workers.")

    while True:
        try:
            if not has_available_worker(redis_client, resource_type):
                time.sleep(NO_WORKER_SLEEP_SECONDS)
                continue

            job_item = redis_client.blpop(queue_names, timeout=1)
            if not job_item:
                continue

            job_data_json = job_item[1]
            try:
                job_data = json.loads(job_data_json)
            except json.JSONDecodeError:
                print(f"Unable to decode job JSON: {job_data_json}")
                continue

            job_resource = normalize_resource_type(job_data.get("resource_type"))
            job_priority = normalize_priority(job_data.get("priority"))
            target_queue = queue_for_resource(job_resource, job_priority)
            if job_resource != resource_type:
                redis_client.rpush(target_queue, job_data_json)
                continue

            try:
                worker_endpoint, worker_type = acquire_worker_for_job(redis_client, resource_type)
            except NoWorkerConfiguredError as exc:
                fail_job_for_unavailable_worker(job_data, str(exc))
                continue

            if not worker_endpoint or not worker_type:
                redis_client.rpush(target_queue, job_data_json)
                time.sleep(NO_WORKER_SLEEP_SECONDS)
                continue

            record_worker_state(redis_client, worker_endpoint, "busy", worker_type)
            _dispatch_job_async(worker_endpoint, worker_type, job_data)
        except Exception as exc:
            print(f"Task dispatcher loop error ({resource_type}): {exc}")
            print(traceback.format_exc())
            time.sleep(5)
            redis_client = get_redis_connection()


def main_loop():
    redis_client = get_redis_connection()
    initialize_worker_queue(redis_client)

    print("Task dispatcher started (Sync, v3-Queues)... waiting for jobs...")

    recovery_thread = threading.Thread(
        target=_worker_recovery_loop,
        daemon=True,
        name="worker-recovery",
    )
    recovery_thread.start()

    threads = []
    for resource_type in ("cpu", "gpu"):
        endpoints = SANDBOX_ENDPOINTS_BY_TYPE.get(resource_type) or []
        if not endpoints:
            continue
        thread = threading.Thread(
            target=_resource_dispatch_loop,
            args=(resource_type,),
            daemon=True,
            name=f"dispatcher-{resource_type}",
        )
        thread.start()
        threads.append(thread)

    if not threads:
        print("No sandbox workers configured. Exiting dispatcher.")
        return

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main_loop()
