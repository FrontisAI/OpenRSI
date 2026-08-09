from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "backends.yaml"
DEFAULT_DB_PATH = SCRIPT_DIR / "gateway_routes.sqlite3"


@dataclass(frozen=True)
class Backend:
    id: str
    base_url: str
    enabled: bool = True
    weight: int = 1


@dataclass(frozen=True)
class GatewayConfig:
    backends: List[Backend]
    strategy: str = "idle_gpu_first"
    fallback: str = "round_robin"
    worker_status_ttl_seconds: float = 2.0
    request_timeout_seconds: float = 120.0
    health_timeout_seconds: float = 5.0
    route_store_path: Path = DEFAULT_DB_PATH


class RouteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_routes (
                    job_id TEXT PRIMARY KEY,
                    backend_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS upload_routes (
                    upload_id TEXT PRIMARY KEY,
                    backend_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def set_job_backend(self, job_id: str, backend_id: str) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO job_routes(job_id, backend_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        backend_id=excluded.backend_id,
                        created_at=excluded.created_at
                    """,
                    (job_id, backend_id, self._now()),
                )

    async def get_job_backend(self, job_id: str) -> Optional[str]:
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT backend_id FROM job_routes WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
        return str(row["backend_id"]) if row else None

    async def set_upload_backend(self, upload_id: str, backend_id: str) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO upload_routes(upload_id, backend_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(upload_id) DO UPDATE SET
                        backend_id=excluded.backend_id,
                        created_at=excluded.created_at
                    """,
                    (upload_id, backend_id, self._now()),
                )

    async def get_upload_backend(self, upload_id: str) -> Optional[str]:
        async with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT backend_id FROM upload_routes WHERE upload_id = ?",
                    (upload_id,),
                ).fetchone()
        return str(row["backend_id"]) if row else None


class GatewayState:
    def __init__(self, config_path: Path, config: GatewayConfig):
        self.config_path = config_path
        self.config_mtime_ns = self._config_mtime_ns()
        self.config = config
        self.all_backends: Dict[str, Backend] = {}
        self.backends: Dict[str, Backend] = {}
        self.backend_list: List[Backend] = []
        self.route_store = RouteStore(config.route_store_path)
        self._client: Optional[httpx.AsyncClient] = None
        self._rr_index = 0
        self._virtual_load: Dict[str, int] = {}
        self._worker_status_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
        self._select_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self.apply_config(config)

    def _config_mtime_ns(self) -> int:
        return self.config_path.stat().st_mtime_ns

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=min(10.0, self.config.request_timeout_seconds),
            )
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def apply_config(self, config: GatewayConfig) -> None:
        configured = {backend.id: backend for backend in config.backends}
        historical = {
            backend_id: backend
            for backend_id, backend in self.all_backends.items()
            if backend_id not in configured
        }
        self.config = config
        self.all_backends = {**historical, **configured}
        self.backends = {
            backend.id: backend
            for backend in config.backends
            if backend.enabled
        }
        self.backend_list = list(self.backends.values())
        self._virtual_load = {
            backend.id: self._virtual_load.get(backend.id, 0)
            for backend in self.backend_list
        }
        self._worker_status_cache = {
            backend_id: value
            for backend_id, value in self._worker_status_cache.items()
            if backend_id in self.all_backends
        }

    async def reload_config_if_needed(self, *, force: bool = False) -> bool:
        try:
            current_mtime_ns = self._config_mtime_ns()
        except FileNotFoundError as exc:
            if force:
                raise HTTPException(status_code=500, detail=f"gateway config missing: {self.config_path}") from exc
            return False

        if not force and current_mtime_ns == self.config_mtime_ns:
            return False

        async with self._reload_lock:
            current_mtime_ns = self._config_mtime_ns()
            if not force and current_mtime_ns == self.config_mtime_ns:
                return False

            new_config = load_config(self.config_path)
            timeout_changed = (
                new_config.request_timeout_seconds != self.config.request_timeout_seconds
                or new_config.health_timeout_seconds != self.config.health_timeout_seconds
            )
            if new_config.route_store_path != self.route_store.db_path:
                self.route_store = RouteStore(new_config.route_store_path)
            self.apply_config(new_config)
            self.config_mtime_ns = current_mtime_ns
            if timeout_changed:
                await self.close()
            return True

    def require_backends(self) -> None:
        if not self.backend_list:
            raise HTTPException(status_code=503, detail="no enabled backends configured")

    def get_backend(self, backend_id: str) -> Optional[Backend]:
        return self.all_backends.get(backend_id)

    async def choose_backend(self, api_key: str) -> Backend:
        await self.reload_config_if_needed()
        self.require_backends()
        statuses = None
        if self.config.strategy == "idle_gpu_first":
            statuses = await asyncio.gather(
                *(self.fetch_worker_status(backend, api_key) for backend in self.backend_list),
                return_exceptions=True,
            )
        async with self._select_lock:
            if self.config.strategy == "idle_gpu_first" and statuses is not None:
                selected = self._choose_by_idle_gpu(statuses)
                if selected is not None:
                    self._virtual_load[selected.id] = self._virtual_load.get(selected.id, 0) + 1
                    return selected
            selected = self._choose_round_robin()
            self._virtual_load[selected.id] = self._virtual_load.get(selected.id, 0) + 1
            return selected

    def _choose_round_robin(self) -> Backend:
        self.require_backends()
        weighted: List[Backend] = []
        for backend in self.backend_list:
            weighted.extend([backend] * max(1, int(backend.weight)))
        backend = weighted[self._rr_index % len(weighted)]
        self._rr_index += 1
        return backend

    def _choose_by_idle_gpu(self, statuses: List[Any]) -> Optional[Backend]:
        scored: List[Tuple[int, int, Backend]] = []
        for backend, status in zip(self.backend_list, statuses):
            if isinstance(status, Exception) or not isinstance(status, dict):
                continue
            summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
            idle = int(summary.get("gpu_idle") or summary.get("idle") or 0)
            total = int(summary.get("gpu_total") or summary.get("total") or 0)
            score = idle - self._virtual_load.get(backend.id, 0)
            # Tie-break with total capacity, then the stable configured order.
            scored.append((score, total, backend))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    async def fetch_worker_status(self, backend: Backend, api_key: str) -> Optional[Dict[str, Any]]:
        cached = self._worker_status_cache.get(backend.id)
        now = time.monotonic()
        if cached and now - cached[0] < self.config.worker_status_ttl_seconds:
            return cached[1]
        client = await self.client()
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/workers/status",
                headers={"X-API-Key": api_key},
            )
            if response.status_code >= 400:
                self._worker_status_cache[backend.id] = (now, None)
                return None
            data = response.json()
        except Exception:
            self._worker_status_cache[backend.id] = (now, None)
            return None
        self._worker_status_cache[backend.id] = (now, data)
        self._virtual_load[backend.id] = 0
        return data


def load_config(path: Path) -> GatewayConfig:
    if not path.exists():
        raise RuntimeError(f"gateway config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("gateway config must be a YAML mapping")

    backend_items = raw.get("backends") or []
    if not isinstance(backend_items, list):
        raise RuntimeError("config.backends must be a list")
    backends: List[Backend] = []
    for item in backend_items:
        if not isinstance(item, dict):
            raise RuntimeError("each backend must be a mapping")
        backend_id = str(item.get("id") or "").strip()
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        if not backend_id or not base_url:
            raise RuntimeError("each backend requires id and base_url")
        backends.append(
            Backend(
                id=backend_id,
                base_url=base_url,
                enabled=bool(item.get("enabled", True)),
                weight=int(item.get("weight", 1)),
            )
        )

    routing = raw.get("routing") or {}
    if not isinstance(routing, dict):
        routing = {}
    request = raw.get("request") or {}
    if not isinstance(request, dict):
        request = {}
    storage = raw.get("storage") or {}
    if not isinstance(storage, dict):
        storage = {}

    db_path_raw = os.getenv("SANDBOX_GATEWAY_DB") or storage.get("route_store_path") or str(DEFAULT_DB_PATH)
    db_path = Path(str(db_path_raw))
    if not db_path.is_absolute():
        db_path = (path.parent / db_path).resolve()

    return GatewayConfig(
        backends=backends,
        strategy=str(routing.get("strategy", "idle_gpu_first")),
        fallback=str(routing.get("fallback", "round_robin")),
        worker_status_ttl_seconds=float(routing.get("worker_status_ttl_seconds", 2.0)),
        request_timeout_seconds=float(request.get("timeout_seconds", 120.0)),
        health_timeout_seconds=float(request.get("health_timeout_seconds", 5.0)),
        route_store_path=db_path,
    )


def config_path_from_env() -> Path:
    return Path(os.getenv("SANDBOX_GATEWAY_CONFIG", str(DEFAULT_CONFIG_PATH))).resolve()


CONFIG_PATH = config_path_from_env()
config = load_config(CONFIG_PATH)
state = GatewayState(CONFIG_PATH, config)
app = FastAPI(title="Sandbox Gateway API", version="0.1.0")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await state.close()


def forward_headers(request: Request, api_key: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    accept = request.headers.get("accept")
    if accept:
        headers["Accept"] = accept
    return headers


def proxy_response(response: httpx.Response) -> Response:
    content_type = response.headers.get("content-type", "application/json")
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type.split(";")[0],
    )


async def request_backend(
    backend: Backend,
    method: str,
    path: str,
    request: Request,
    api_key: Optional[str],
    *,
    content: Optional[bytes] = None,
) -> httpx.Response:
    client = await state.client()
    if content is None and method.upper() in {"POST", "PUT", "PATCH"}:
        content = await request.body()
    try:
        return await client.request(
            method,
            f"{backend.base_url}{path}",
            params=request.query_params,
            headers=forward_headers(request, api_key),
            content=content,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"backend request failed: backend={backend.id}, error={exc}",
        ) from exc


async def resolve_backend_for_job(job_id: str, api_key: str) -> Backend:
    await state.reload_config_if_needed()
    backend_id = await state.route_store.get_job_backend(job_id)
    if backend_id:
        backend = state.get_backend(backend_id)
        if backend:
            return backend
        raise HTTPException(status_code=503, detail=f"mapped backend is not enabled: {backend_id}")

    client = await state.client()
    for backend in state.all_backends.values():
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/jobs/{job_id}",
                headers={"X-API-Key": api_key},
            )
        except Exception:
            continue
        if response.status_code == 200:
            await state.route_store.set_job_backend(job_id, backend.id)
            return backend
    raise HTTPException(status_code=404, detail="Job not found")


def summarize_worker_payloads(payloads: Iterable[Tuple[Backend, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    workers_show: List[Dict[str, Any]] = []
    backend_summaries: Dict[str, Any] = {}
    total = gpu_total = cpu_total = 0
    gpu_idle = cpu_idle = gpu_quarantined = cpu_quarantined = 0

    for backend, payload in payloads:
        if not isinstance(payload, dict):
            backend_summaries[backend.id] = {"ok": False}
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        backend_summaries[backend.id] = {"ok": True, "summary": summary}
        total += int(summary.get("total") or 0)
        gpu_total += int(summary.get("gpu_total") or 0)
        cpu_total += int(summary.get("cpu_total") or 0)
        gpu_idle += int(summary.get("gpu_idle") or 0)
        cpu_idle += int(summary.get("cpu_idle") or 0)
        gpu_quarantined += int(summary.get("gpu_quarantined") or 0)
        cpu_quarantined += int(summary.get("cpu_quarantined") or 0)
        for worker in payload.get("workers_show") or []:
            if isinstance(worker, dict):
                enriched = dict(worker)
                enriched["backend_id"] = backend.id
                workers_show.append(enriched)

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
    workers_show.sort(key=lambda item: (item.get("backend_id") or "", item.get("endpoint") or ""))
    return {
        "workers_show": workers_show,
        "stats": stats,
        "summary": summary,
        "backends": backend_summaries,
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    await state.reload_config_if_needed()
    client = await state.client()
    results: Dict[str, Any] = {}
    ok_count = 0
    for backend in state.backend_list:
        try:
            response = await client.get(
                f"{backend.base_url}/health",
                timeout=state.config.health_timeout_seconds,
            )
            ok = response.status_code == 200
            body: Any
            try:
                body = response.json()
            except Exception:
                body = response.text[:500]
            results[backend.id] = {"ok": ok, "status_code": response.status_code, "body": body}
            ok_count += 1 if ok else 0
        except Exception as exc:
            results[backend.id] = {"ok": False, "error": str(exc)}
    return {
        "status": "healthy" if ok_count > 0 else "unhealthy",
        "configured_backends": len(state.all_backends),
        "enabled_backends": len(state.backend_list),
        "healthy_backends": ok_count,
        "backends": results,
    }


@app.get("/api/v1/workers/status")
async def get_workers_status(x_api_key: str = Header(...)) -> Dict[str, Any]:
    await state.reload_config_if_needed()
    state.require_backends()
    payloads = await asyncio.gather(
        *(state.fetch_worker_status(backend, x_api_key) for backend in state.backend_list),
        return_exceptions=True,
    )
    normalized: List[Tuple[Backend, Optional[Dict[str, Any]]]] = []
    for backend, payload in zip(state.backend_list, payloads):
        normalized.append((backend, payload if isinstance(payload, dict) else None))
    return summarize_worker_payloads(normalized)


@app.post("/api/v1/jobs/submit_and_wait")
async def submit_and_wait_not_implemented(x_api_key: str = Header(...)) -> JSONResponse:
    _ = x_api_key
    return JSONResponse(
        status_code=501,
        content={
            "detail": "submit_and_wait is not implemented in the gateway; use POST /api/v1/jobs and poll /api/v1/jobs/{job_id}",
        },
    )


def require_admin(x_api_key: Optional[str], x_gateway_admin_token: Optional[str]) -> None:
    admin_token = os.getenv("SANDBOX_GATEWAY_ADMIN_TOKEN", "")
    if admin_token:
        if x_gateway_admin_token != admin_token:
            raise HTTPException(status_code=401, detail="invalid gateway admin token")
        return
    if x_api_key not in {"mlsandbox-router-oss-71c4bb62f1084d208e7d783589168cd4"}:
        raise HTTPException(status_code=401, detail="invalid gateway admin credential")


def backend_config_snapshot() -> Dict[str, Any]:
    return {
        "config_path": str(state.config_path),
        "config_mtime_ns": state.config_mtime_ns,
        "route_store_path": str(state.route_store.db_path),
        "strategy": state.config.strategy,
        "fallback": state.config.fallback,
        "configured_backends": [
            {
                "id": backend.id,
                "base_url": backend.base_url,
                "enabled": backend.enabled,
                "weight": backend.weight,
            }
            for backend in state.all_backends.values()
        ],
        "enabled_backends": [backend.id for backend in state.backend_list],
    }


@app.get("/admin/config")
async def admin_config(
    x_api_key: Optional[str] = Header(None),
    x_gateway_admin_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    require_admin(x_api_key, x_gateway_admin_token)
    await state.reload_config_if_needed()
    return backend_config_snapshot()


@app.post("/admin/reload")
async def admin_reload(
    x_api_key: Optional[str] = Header(None),
    x_gateway_admin_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    require_admin(x_api_key, x_gateway_admin_token)
    reloaded = await state.reload_config_if_needed(force=True)
    snapshot = backend_config_snapshot()
    snapshot["reloaded"] = reloaded
    return snapshot


@app.post("/api/v1/uploads")
async def uploads_not_implemented(x_api_key: str = Header(...)) -> JSONResponse:
    _ = x_api_key
    return JSONResponse(
        status_code=501,
        content={"detail": "uploads are not implemented in the gateway"},
    )


@app.post("/api/v1/jobs")
async def submit_job(request: Request, x_api_key: str = Header(...)) -> Response:
    backend = await state.choose_backend(x_api_key)
    response = await request_backend(backend, "POST", "/api/v1/jobs", request, x_api_key)
    if response.status_code < 400:
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("job_id"):
            await state.route_store.set_job_backend(str(data["job_id"]), backend.id)
    return proxy_response(response)


@app.get("/api/v1/jobs")
async def list_jobs(request: Request, x_api_key: str = Header(...)) -> Dict[str, Any]:
    await state.reload_config_if_needed()
    state.require_backends()
    requested_limit = int(request.query_params.get("limit", "50"))
    requested_offset = int(request.query_params.get("offset", "0"))
    query = dict(request.query_params)
    query["limit"] = str(max(requested_limit + requested_offset, requested_limit))
    query["offset"] = "0"
    client = await state.client()

    async def fetch(backend: Backend) -> Tuple[Backend, Optional[Dict[str, Any]]]:
        try:
            response = await client.get(
                f"{backend.base_url}/api/v1/jobs",
                params=query,
                headers={"X-API-Key": x_api_key},
            )
            if response.status_code >= 400:
                return backend, None
            return backend, response.json()
        except Exception:
            return backend, None

    results = await asyncio.gather(*(fetch(backend) for backend in state.backend_list))
    jobs: List[Dict[str, Any]] = []
    backend_counts: Dict[str, int] = {}
    for backend, payload in results:
        backend_jobs = payload.get("jobs") if isinstance(payload, dict) else []
        backend_counts[backend.id] = len(backend_jobs) if isinstance(backend_jobs, list) else 0
        if isinstance(backend_jobs, list):
            for job in backend_jobs:
                if isinstance(job, dict):
                    enriched = dict(job)
                    enriched["backend_id"] = backend.id
                    jobs.append(enriched)
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    sliced = jobs[requested_offset : requested_offset + requested_limit]
    return {"jobs": sliced, "count": len(sliced), "fetched_count": len(jobs), "backend_counts": backend_counts}


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "GET", f"/api/v1/jobs/{job_id}", request, x_api_key)
    return proxy_response(response)


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "GET", f"/api/v1/jobs/{job_id}/logs", request, x_api_key)
    return proxy_response(response)


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request, x_api_key: str = Header(...)) -> Response:
    backend = await resolve_backend_for_job(job_id, x_api_key)
    response = await request_backend(backend, "DELETE", f"/api/v1/jobs/{job_id}", request, x_api_key)
    return proxy_response(response)
