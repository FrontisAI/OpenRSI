from __future__ import annotations

import asyncio
import contextlib
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import errors as pg_errors
from psycopg2 import pool as pg_pool
import redis
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from psycopg2.extras import RealDictCursor

app = FastAPI(title="Sandbox ML Training API", version="1.0.0")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "sandbox")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
DEFAULT_STORAGE_ROOT = Path("/mnt/pubdatasets2/mlsandbox")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", str(DEFAULT_STORAGE_ROOT))).resolve()
UPLOAD_ROOT = Path(os.getenv("UPLOAD_ROOT", str(STORAGE_PATH / "uploads"))).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_POLL_INTERVAL = int(os.getenv("JOB_WAIT_POLL_INTERVAL", "3"))
DEFAULT_WAIT_TIMEOUT = int(os.getenv("JOB_WAIT_TIMEOUT", "1800"))
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
CLIENT_TASK_KEY = "CLIENT_TASK_ID"
ENABLE_OBS = os.getenv("ENABLE_OBS", "1").lower() not in {"0", "false", "no", "off"}
OBS_LOG_DIR_RAW = os.getenv("OBS_LOG_DIR", "").strip()
OBS_LOG_DIR = Path(OBS_LOG_DIR_RAW).resolve() if OBS_LOG_DIR_RAW else None
SUBMIT_MAX_INFLIGHT_PER_PROCESS = int(os.getenv("SUBMIT_MAX_INFLIGHT_PER_PROCESS", "0"))
SUBMIT_SLOT_WAIT_SECONDS = float(os.getenv("SUBMIT_SLOT_WAIT_SECONDS", "2"))
SUBMIT_RETRY_AFTER_SECONDS = max(1, int(os.getenv("SUBMIT_RETRY_AFTER_SECONDS", "2")))
ENABLE_IDEMPOTENCY = os.getenv("ENABLE_IDEMPOTENCY", "1").lower() not in {"0", "false", "no", "off"}
JOB_QUEUE_CPU_P1 = "job_queue_cpu_p1"
JOB_QUEUE_CPU_P2 = "job_queue_cpu_p2"
JOB_QUEUE_GPU_P1 = "job_queue_gpu_p1"
JOB_QUEUE_GPU_P2 = "job_queue_gpu_p2"
JOB_QUEUE_BY_RESOURCE_PRIORITY = {
    "cpu": {1: JOB_QUEUE_CPU_P1, 2: JOB_QUEUE_CPU_P2},
    "gpu": {1: JOB_QUEUE_GPU_P1, 2: JOB_QUEUE_GPU_P2},
}

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
_DB_POOL: Optional[pg_pool.ThreadedConnectionPool] = None
_DB_POOL_LOCK = threading.Lock()
_SUBMIT_SEMAPHORE = (
    threading.BoundedSemaphore(SUBMIT_MAX_INFLIGHT_PER_PROCESS)
    if SUBMIT_MAX_INFLIGHT_PER_PROCESS > 0
    else None
)


def _obs_log(event: str, **fields) -> None:
    if not ENABLE_OBS:
        return
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if OBS_LOG_DIR is not None:
        try:
            OBS_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (OBS_LOG_DIR / "api_server.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except OSError as exc:
            print(f"[obs-log] failed to write {OBS_LOG_DIR}: {exc}", flush=True)


def _stage_log(
    stage: str,
    start_time: float,
    *,
    trace_id: Optional[str] = None,
    job_id: Optional[str] = None,
    **fields,
) -> None:
    _obs_log(
        "api.stage",
        stage=stage,
        elapsed_ms=round((time.monotonic() - start_time) * 1000, 3),
        trace_id=trace_id,
        job_id=job_id,
        **fields,
    )


@contextmanager
def submit_admission_slot():
    if _SUBMIT_SEMAPHORE is None:
        yield
        return
    acquired = _SUBMIT_SEMAPHORE.acquire(timeout=max(0.0, SUBMIT_SLOT_WAIT_SECONDS))
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail={"error": "server_busy", "retry_after": SUBMIT_RETRY_AFTER_SECONDS},
            headers={"Retry-After": str(SUBMIT_RETRY_AFTER_SECONDS)},
        )
    try:
        yield
    finally:
        _SUBMIT_SEMAPHORE.release()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _normalize_dataset_label(source: Optional[str]) -> str:
    if not source:
        return "default"
    try:
        candidate = Path(source.rstrip("/")).name
    except Exception:
        candidate = ""
    if not candidate:
        stripped = source.strip("/ ")
        candidate = stripped.split("/")[-1] if stripped else ""
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    return label or "default"


def _relocate_source_to_job(src: Path, code_root: Path) -> Path:
    if _is_relative_to(src, UPLOAD_ROOT):
        relative = src.relative_to(UPLOAD_ROOT)
        upload_root = UPLOAD_ROOT / relative.parts[0]
        code_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(upload_root), str(code_root))
        moved_root = code_root / relative.parts[0]
        if len(relative.parts) > 1:
            return moved_root.joinpath(*relative.parts[1:])
        return moved_root

    if src.is_dir():
        dest = code_root / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest

    code_root.mkdir(parents=True, exist_ok=True)
    dest_file = code_root / src.name
    shutil.copy2(src, dest_file)
    return dest_file


def _make_world_writable(path: Path) -> None:
    if not path.exists():
        return

    def _chmod(target: Path) -> None:
        try:
            target.chmod(0o777)
        except (PermissionError, OSError):
            pass

    _chmod(path)
    if path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            _chmod(root_path)
            for name in dirs + files:
                _chmod(root_path / name)


def _load_json(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


class JobSubmission(BaseModel):
    task_id: Optional[str] = Field(None, description="客户端自定义任务标识")
    idempotency_key: Optional[str] = Field(
        None,
        max_length=128,
        description="客户端幂等键；同一逻辑任务重试时复用",
    )
    name: str = Field(..., description="任务名称")
    code: Optional[str] = Field(None, description="训练代码内容")
    code_file_path: Optional[str] = Field(None, description="共享存储上的代码路径")
    data_dir: Optional[str] = Field(None, description="沙盒内可访问的数据目录绝对路径")
    requirements: List[str] = Field(default_factory=list, description="依赖包列表")
    data_paths: List[str] = Field(default_factory=list, description="数据路径")
    gpu_count: int = Field(1, ge=1, le=8, description="所需 GPU 数量")
    timeout: int = Field(3600, description="超时时间(秒)")
    environment: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    execution_command: Optional[str] = Field(
        default=None, description="自定义 shell 命令 (仅 shell 模式使用)"
    )
    working_dir: Optional[str] = Field(
        default=None, description="自定义工作目录 (Sandbox 内部路径)"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=2,
        description="任务优先级: 1(高)/2(低)，默认 1"
    )
    resource_type: str = Field(
        default="cpu",
        description="资源类型: gpu/cpu (默认 cpu)"
    )


def _canonical_submit_payload(job: JobSubmission) -> Dict:
    return {
        "name": job.name,
        "task_id": job.task_id,
        "code": job.code,
        "code_file_path": job.code_file_path,
        "data_dir": job.data_dir,
        "requirements": list(job.requirements or []),
        "data_paths": list(job.data_paths or []),
        "gpu_count": job.gpu_count,
        "timeout": job.timeout,
        "environment": dict(job.environment or {}),
        "execution_command": job.execution_command,
        "working_dir": job.working_dir,
        "priority": int(job.priority) if job.priority in (1, 2) else 1,
        "resource_type": (job.resource_type or "cpu").strip().lower(),
    }


def _submit_payload_hash(job: JobSubmission) -> str:
    canonical = _canonical_submit_payload(job)
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_idempotency_key(value: Optional[str]) -> Optional[str]:
    if not ENABLE_IDEMPOTENCY or value is None:
        return None
    key = value.strip()
    return key or None


def fetch_idempotent_job(api_key: str, idempotency_key: str):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM jobs
                WHERE api_key = %s AND idempotency_key = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (api_key, idempotency_key),
            )
            return cursor.fetchone()


def _task_id_from_job_row(row) -> Optional[str]:
    environment_data = _load_json(row.get("environment")) if row else None
    if isinstance(environment_data, dict):
        return environment_data.get(CLIENT_TASK_KEY)
    return None


def _created_at_for_response(row) -> str:
    created_at = row.get("created_at")
    if hasattr(created_at, "isoformat"):
        return created_at.isoformat() + "Z"
    return str(created_at)



class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    estimated_start: Optional[str] = None
    task_id: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    name: str
    status: str
    progress: int
    worker_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    updated_at: str
    completed_at: Optional[str]
    logs_url: str
    result: Optional[Dict]
    metrics: Optional[Dict]
    task_id: Optional[str]


@app.on_event("startup")
def startup_event():
    global _DB_POOL
    if _DB_POOL is not None:
        return
    with _DB_POOL_LOCK:
        if _DB_POOL is not None:
            return
        try:
            minconn = max(1, DB_POOL_MIN)
            maxconn = max(minconn, DB_POOL_MAX)
            _DB_POOL = pg_pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                cursor_factory=RealDictCursor,
            )
            print(f"Database connection pool initialized ({minconn}-{maxconn}).")
        except Exception as exc:
            print(f"Error initializing connection pool: {exc}")


@contextmanager
def get_db_connection():
    if _DB_POOL is None:
        startup_event()
    if _DB_POOL is None:
        raise HTTPException(status_code=503, detail="database pool not initialized, please retry")
    try:
        conn = _DB_POOL.getconn()
    except pg_pool.PoolError as exc:
        raise HTTPException(
            status_code=503,
            detail="database connection pool exhausted, please retry",
        ) from exc
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        # Psycopg2 starts a transaction on first statement. Ensure pooled
        # connections are returned clean even for read-only requests.
        try:
            if not conn.closed:
                conn.rollback()
        except Exception:
            pass
        _DB_POOL.putconn(conn)


def verify_api_key(x_api_key: str = Header(...)):
    valid_keys = {"mlsandbox-oss-3fad186210e530e6b4cc53576cd2723e"}
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


def ensure_storage_path(path_str: str) -> Path:
    path = Path(path_str).resolve()
    try:
        path.relative_to(STORAGE_PATH)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"path must be under {STORAGE_PATH}",
        ) from exc
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"path does not exist: {path}")
    return path


def fetch_job_row(job_id: str, api_key: str):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM jobs WHERE job_id = %s AND api_key = %s",
                (job_id, api_key),
            )
            return cursor.fetchone()


def fetch_job_status_row(job_id: str, api_key: str):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    job_id, name, status, progress, worker_id, api_key,
                    created_at, started_at, updated_at, completed_at,
                    environment,
                    NULL::jsonb AS result,
                    NULL::jsonb AS metrics
                FROM jobs
                WHERE job_id = %s AND api_key = %s
                """,
                (job_id, api_key),
            )
            return cursor.fetchone()


async def wait_for_job_completion(
    job_id: str,
    api_key: str,
    timeout: int,
    poll_interval: int,
) -> "JobStatus":
    loop = asyncio.get_running_loop()
    deadline = time.monotonic() + timeout
    while True:
        job = await loop.run_in_executor(None, fetch_job_status_row, job_id, api_key)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        status = job["status"]
        if status in TERMINAL_STATUSES:
            job = await loop.run_in_executor(None, fetch_job_row, job_id, api_key)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            return serialize_job_row(job)
        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=202,
                detail={
                    "job_id": job_id,
                    "status": status,
                    "message": "job is still running",
                },
            )
        await asyncio.sleep(poll_interval)


def serialize_job_row(job: Dict) -> JobStatus:
    result_data = _load_json(job.get("result"))
    metrics_data = _load_json(job.get("metrics"))
    environment_data = _load_json(job.get("environment"))
    task_id = None
    if isinstance(environment_data, dict):
        task_id = environment_data.get(CLIENT_TASK_KEY)

    return JobStatus(
        job_id=job["job_id"],
        name=job["name"],
        status=job["status"],
        progress=job.get("progress", 0),
        worker_id=job.get("worker_id"),
        created_at=job["created_at"].isoformat() + "Z",
        started_at=job.get("started_at").isoformat() + "Z"
        if job.get("started_at")
        else None,
        updated_at=job["updated_at"].isoformat() + "Z",
        completed_at=job.get("completed_at").isoformat() + "Z"
        if job.get("completed_at")
        else None,
        logs_url=f"/api/v1/jobs/{job['job_id']}/logs",
        result=result_data,
        metrics=metrics_data,
        task_id=task_id,
    )


def create_job_entry(job: JobSubmission, api_key: str, trace_id: Optional[str] = None) -> Tuple[str, str, Optional[str], str]:
    if job.code is None and not job.code_file_path:
        raise HTTPException(
            status_code=400, detail="Must provide either 'code' or 'code_file_path'"
        )

    trace_id = trace_id or uuid.uuid4().hex
    total_start = time.monotonic()
    idempotency_key = _normalize_idempotency_key(job.idempotency_key)
    payload_hash = _submit_payload_hash(job) if idempotency_key else None

    if idempotency_key:
        existing = fetch_idempotent_job(api_key, idempotency_key)
        if existing:
            existing_hash = existing.get("idempotency_payload_hash")
            if existing_hash != payload_hash:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "idempotency_key_conflict",
                        "message": "idempotency_key already exists with different payload",
                    },
                )
            _obs_log(
                "api.submit.idempotency_hit",
                trace_id=trace_id,
                job_id=existing.get("job_id"),
                idempotency_key=idempotency_key,
                status=existing.get("status"),
            )
            return (
                existing["job_id"],
                _created_at_for_response(existing),
                _task_id_from_job_row(existing),
                existing.get("status") or "queued",
            )

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    created_at = datetime.utcnow().isoformat() + "Z"

    resource_type = (job.resource_type or "cpu").strip().lower()
    if resource_type not in {'cpu', 'gpu'}:
        raise HTTPException(status_code=400, detail="resource_type must be one of 'cpu','gpu'")
    priority = int(job.priority) if job.priority in (1, 2) else 1
    merged_environment = dict(job.environment or {})
    if job.data_dir:
        merged_environment["DATA_DIR"] = job.data_dir
    data_dir_value = merged_environment.get("DATA_DIR")
    if data_dir_value:
        merged_environment.setdefault("SANDBOX_DATA_DIR", data_dir_value)
    merged_environment['RESOURCE_TYPE'] = resource_type
    merged_environment['PRIORITY'] = str(priority)
    merged_environment["TRACE_ID"] = trace_id
    dataset_label = _normalize_dataset_label(data_dir_value)

    date_label = datetime.utcnow().strftime("%Y-%m-%d")
    job_dir = STORAGE_PATH / "jobs" / date_label / dataset_label / job_id
    # job_dir = STORAGE_PATH / "jobs" / dataset_label / job_id
    code_root = job_dir / "code"
    outputs_dir = job_dir / "outputs"
    stage_start = time.monotonic()
    code_root.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _stage_log("submit.mkdir", stage_start, trace_id=trace_id, job_id=job_id)

    working_src = None
    if job.working_dir:
        working_src = ensure_storage_path(job.working_dir)

    submitted_code_chars = None
    submitted_code_sha256 = None
    stage_start = time.monotonic()
    if job.code is not None:
        code_text = job.code or ""
        submitted_code_chars = len(code_text)
        submitted_code_sha256 = hashlib.sha256(code_text.encode("utf-8")).hexdigest()
        code_path = code_root / "main.py"
        code_path.write_text(code_text, encoding="utf-8")
    else:
        source_path = ensure_storage_path(job.code_file_path)
        code_path = _relocate_source_to_job(source_path, code_root)
    _stage_log("submit.write_code", stage_start, trace_id=trace_id, job_id=job_id)

    if working_src:
        if _is_relative_to(working_src, UPLOAD_ROOT):
            relative = working_src.relative_to(UPLOAD_ROOT)
            moved_root = code_root / relative.parts[0]
            if len(relative.parts) > 1:
                final_working_dir = moved_root.joinpath(*relative.parts[1:])
            else:
                final_working_dir = moved_root
        else:
            final_working_dir = working_src
    else:
        final_working_dir = code_path.parent

    merged_environment.setdefault("JOB_ID", job_id)
    merged_environment.setdefault("JOB_DIR", str(job_dir))
    merged_environment.setdefault("JOB_CODE_DIR", str(code_root))
    merged_environment.setdefault("JOB_OUTPUT_DIR", str(outputs_dir))
    merged_environment["JOB_DATASET_LABEL"] = dataset_label
    merged_environment["JOB_STORAGE_DIR"] = str(job_dir)
    merged_environment["EXECUTION_WORKDIR"] = str(final_working_dir)
    if job.execution_command:
        merged_environment["EXECUTION_COMMAND"] = job.execution_command
    if job.task_id:
        merged_environment[CLIENT_TASK_KEY] = job.task_id

    stage_start = time.monotonic()
    _make_world_writable(job_dir)
    _stage_log("submit.chmod", stage_start, trace_id=trace_id, job_id=job_id)

    job_data = {
        "job_id": job_id,
        "name": job.name,
        "code_file_path": str(code_path),
        "requirements": job.requirements,
        "data_paths": job.data_paths,
        "data_dir": data_dir_value,
        "gpu_count": job.gpu_count,
        "timeout": job.timeout,
        "environment": merged_environment,
        "status": "queued",
        "resource_type": resource_type,
        "priority": priority,
        "created_at": created_at,
        "api_key": api_key,
        "dataset_label": dataset_label,
        "job_storage_dir": str(job_dir),
        "trace_id": trace_id,
        "idempotency_key": idempotency_key,
        "idempotency_payload_hash": payload_hash,
        "working_dir": job.working_dir,
    }
    if job.data_dir:
        job_data["data_dir"] = job.data_dir
    if job.task_id:
        job_data["task_id"] = job.task_id

    stage_start = time.monotonic()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO jobs (
                        job_id, name, code_file_path, requirements, data_paths,
                        gpu_count, timeout, environment, status, created_at, api_key,
                        idempotency_key, idempotency_payload_hash,
                        submitted_code_chars, submitted_code_sha256
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        job.name,
                        str(code_path),
                        json.dumps(job.requirements),
                        json.dumps(job.data_paths),
                        job.gpu_count,
                        job.timeout,
                        json.dumps(merged_environment),
                        "queued",
                        created_at,
                        api_key,
                        idempotency_key,
                        payload_hash,
                        submitted_code_chars,
                        submitted_code_sha256,
                    ),
                )
            conn.commit()
    except pg_errors.UniqueViolation:
        if idempotency_key:
            existing = fetch_idempotent_job(api_key, idempotency_key)
            if existing and existing.get("idempotency_payload_hash") == payload_hash:
                with contextlib.suppress(Exception):
                    shutil.rmtree(job_dir)
                _obs_log(
                    "api.submit.idempotency_race_hit",
                    trace_id=trace_id,
                    job_id=existing.get("job_id"),
                    idempotency_key=idempotency_key,
                )
                return (
                    existing["job_id"],
                    _created_at_for_response(existing),
                    _task_id_from_job_row(existing),
                    existing.get("status") or "queued",
                )
        raise HTTPException(status_code=409, detail={"error": "idempotency_key_conflict"})
    _stage_log("submit.db_insert", stage_start, trace_id=trace_id, job_id=job_id)

    queue_name = JOB_QUEUE_BY_RESOURCE_PRIORITY[resource_type][priority]
    stage_start = time.monotonic()
    redis_client.rpush(queue_name, json.dumps(job_data))
    _stage_log("submit.redis_rpush", stage_start, trace_id=trace_id, job_id=job_id, queue=queue_name)
    _stage_log("submit.total", total_start, trace_id=trace_id, job_id=job_id)

    return job_id, created_at, job.task_id, "queued"


@app.get("/health")
def health_check():
    try:
        redis_client.ping()
        with get_db_connection():
            pass
        return {"status": "healthy", "redis": "connected", "database": "connected"}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


@app.post("/api/v1/jobs", response_model=JobResponse)
def submit_job(
    job: JobSubmission,
    api_key: str = Depends(verify_api_key),
    x_trace_id: Optional[str] = Header(None),
):
    trace_id = x_trace_id or uuid.uuid4().hex
    with submit_admission_slot():
        job_id, created_at, task_id, status = create_job_entry(job, api_key, trace_id=trace_id)
    return JobResponse(
        job_id=job_id,
        status=status,
        created_at=created_at,
        estimated_start=None,
        task_id=task_id,
    )


@app.post("/api/v1/jobs/submit_and_wait", response_model=JobStatus)
async def submit_and_wait(
    job: JobSubmission,
    api_key: str = Depends(verify_api_key),
    x_trace_id: Optional[str] = Header(None),
    timeout: Optional[int] = None,
    poll_interval: Optional[int] = None,
):
    trace_id = x_trace_id or uuid.uuid4().hex
    with submit_admission_slot():
        job_id, _, _, _ = create_job_entry(job, api_key, trace_id=trace_id)
    wait_timeout = timeout if timeout is not None else DEFAULT_WAIT_TIMEOUT
    poll = poll_interval if poll_interval is not None else DEFAULT_POLL_INTERVAL
    try:
        return await wait_for_job_completion(job_id, api_key, wait_timeout, poll)
    except HTTPException as exc:
        if exc.status_code == 202 and isinstance(exc.detail, dict):
            exc.detail.setdefault("job_id", job_id)
        raise


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    job = fetch_job_status_row(job_id, api_key)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in TERMINAL_STATUSES:
        job = fetch_job_row(job_id, api_key)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    return serialize_job_row(job)


@app.get("/api/v1/jobs/{job_id}/logs")
def get_job_logs(job_id: str, api_key: str = Depends(verify_api_key)):
    job = fetch_job_row(job_id, api_key)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_data = _load_json(job.get("result"))
    if isinstance(result_data, dict):
        run_log = result_data.get("run_log")
        if isinstance(run_log, str) and run_log.strip():
            return StreamingResponse(iter([run_log]), media_type="text/plain")

    environment_data = _load_json(job.get("environment"))
    log_dir: Optional[Path] = None
    if isinstance(environment_data, dict):
        storage_dir = environment_data.get("JOB_STORAGE_DIR")
        if storage_dir:
            log_dir = Path(storage_dir)
        else:
            dataset_label = environment_data.get("JOB_DATASET_LABEL")
            if dataset_label:
                log_dir = STORAGE_PATH / "jobs" / dataset_label / job_id
    if log_dir is None:
        log_dir = STORAGE_PATH / "jobs" / job_id

    log_file = log_dir / "logs.txt"
    if not log_file.exists():
        return {"logs": "No logs available yet"}

    def iter_file():
        with log_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                yield line

    return StreamingResponse(iter_file(), media_type="text/plain")


@app.get("/api/v1/jobs")
def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            query = "SELECT * FROM jobs WHERE api_key = %s"
            params = [api_key]
            if status:
                query += " AND status = %s"
                params.append(status)
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cursor.execute(query, params)
            jobs = cursor.fetchall()

    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/api/v1/jobs/{job_id}")
def cancel_job(job_id: str, api_key: str = Depends(verify_api_key)):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', updated_at = %s
                WHERE job_id = %s AND api_key = %s AND status IN ('queued', 'running')
                """,
                (datetime.utcnow().isoformat() + "Z", job_id, api_key),
            )
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, detail="Job not found or cannot be cancelled"
                )
        conn.commit()

    removed = False
    try:
        queue_keys = [JOB_QUEUE_CPU_P1, JOB_QUEUE_CPU_P2, JOB_QUEUE_GPU_P1, JOB_QUEUE_GPU_P2]
        for queue_key in queue_keys:
            queue_items = redis_client.lrange(queue_key, 0, -1)
            for raw in queue_items:
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if item.get("job_id") == job_id:
                    redis_client.lrem(queue_key, 1, raw)
                    removed = True
                    break
            if removed:
                break
        redis_client.setex(f"job:{job_id}:cancelled", 3600, "1")
    except Exception:
        pass

    return {"message": "Job cancelled successfully", "removed_from_queue": removed}


@app.get("/api/v1/workers/status")
def get_worker_status():
    workers = []
    for key in redis_client.scan_iter("worker:*:heartbeat"):
        worker_id = key.split(":")[1]
        heartbeat = redis_client.get(key)
        status_key = f"worker:{worker_id}:status"
        status = redis_client.get(status_key) or "unknown"
        endpoint = redis_client.get(f"worker:{worker_id}:endpoint")
        resource_type = (redis_client.get(f"worker:{worker_id}:resource_type") or "unknown").lower()
        workers.append(
            {
                "worker_id": worker_id,
                "status": status,
                "endpoint": endpoint,
                "resource_type": resource_type,
                "last_heartbeat": heartbeat,
            }
        )

    workers_show = sorted(
        (
            {
                "endpoint": w.get("endpoint"),
                "resource_type": w.get("resource_type"),
                "status": w.get("status"),
            }
            for w in workers
        ),
        key=lambda x: x.get("endpoint") or "",
    )

    total = len(workers_show)
    cpu_total = sum(1 for w in workers_show if w.get("resource_type") == "cpu")
    gpu_total = sum(1 for w in workers_show if w.get("resource_type") == "gpu")
    cpu_idle = sum(1 for w in workers_show if w.get("resource_type") == "cpu" and w.get("status") == "idle")
    gpu_idle = sum(1 for w in workers_show if w.get("resource_type") == "gpu" and w.get("status") == "idle")
    cpu_quarantined = sum(
        1 for w in workers_show if w.get("resource_type") == "cpu" and w.get("status") == "quarantined"
    )
    gpu_quarantined = sum(
        1 for w in workers_show if w.get("resource_type") == "gpu" and w.get("status") == "quarantined"
    )

    summary = {
        "total": total,
        "cpu_total": cpu_total,
        "gpu_total": gpu_total,
        "cpu_idle": cpu_idle,
        "gpu_idle": gpu_idle,
        "cpu_busy": max(0, cpu_total - cpu_idle - cpu_quarantined),
        "gpu_busy": max(0, gpu_total - gpu_idle - gpu_quarantined),
        "cpu_quarantined": cpu_quarantined,
        "gpu_quarantined": gpu_quarantined,
    }

    stats = {
        f"total:{total}, cpu_total:{cpu_total}, gpu_total:{gpu_total}; "
        f"cpu_idle:{cpu_idle}/{cpu_total}, gpu_idle:{gpu_idle}/{gpu_total}; "
        f"cpu_quarantined:{cpu_quarantined}/{cpu_total}, gpu_quarantined:{gpu_quarantined}/{gpu_total}"
    }

    return {
        "workers_show": workers_show,
        "stats": stats,
        "summary": summary,
    }



@app.post("/api/v1/uploads")
def upload_code_archive(
    file: UploadFile = File(...),
    extract: bool = True,
    api_key: str = Depends(verify_api_key),
):
    upload_id = f"upload_{uuid.uuid4().hex[:16]}"
    dest_dir = UPLOAD_ROOT / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dest_dir / file.filename
    with archive_path.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)

    extraction_root = dest_dir
    extracted = False
    if extract:
        try:
            shutil.unpack_archive(str(archive_path), str(dest_dir))
            extracted = True
            archive_path.unlink(missing_ok=True)
        except (shutil.ReadError, ValueError):
            pass

    if extracted:
        entries = [
            p for p in dest_dir.iterdir() if not p.name.startswith("__MACOSX")
        ]
        if len(entries) == 1 and entries[0].is_dir():
            extraction_root = entries[0]

    default_entry = None
    for candidate in ("main.py", "train.py", "run.py", "run.sh"):
        candidate_path = extraction_root / candidate
        if candidate_path.exists():
            default_entry = candidate_path
            break

    sample_files: List[str] = []
    try:
        for path in extraction_root.glob("**/*"):
            rel = path.relative_to(extraction_root)
            if rel.parts and len(sample_files) < 50:
                sample_files.append(str(rel))
            if len(sample_files) >= 50:
                break
    except OSError:
        pass

    return {
        "upload_id": upload_id,
        "base_path": str(extraction_root),
        "default_entry": str(default_entry) if default_entry else None,
        "sample_files": sample_files,
    }
