import argparse
import asyncio
import textwrap

import httpx


BASE_URL = "http://203.0.113.10:6580"  # 占位地址：替换为你的控制器地址
API_KEY = "mlsandbox-oss-3fad186210e530e6b4cc53576cd2723e"
DEFAULT_DATA_DIR = "/mnt/pubdatasets2/MLTasks/Selected_Dojo/titanic"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}


def build_gpu_hold_code(vram_gb: float, hold_seconds: int) -> str:
    target_bytes = max(1, int(vram_gb * 1024 * 1024 * 1024))
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import sys
        import time
        from pathlib import Path

        import pandas as pd
        import torch

        TARGET_BYTES = {target_bytes}
        HOLD_SECONDS = {hold_seconds}


        def main() -> int:
            if not torch.cuda.is_available():
                print("[job] CUDA is not available", file=sys.stderr, flush=True)
                return 1

            device = torch.device("cuda:0")
            print(
                f"[job] allocating about {{TARGET_BYTES / 1024**3:.2f}} GiB on {{device}}",
                flush=True,
            )

            tensor = torch.empty(int(TARGET_BYTES), dtype=torch.uint8, device=device)
            print(
                f"[job] allocation done, requested={{TARGET_BYTES / 1024**3:.2f}} GiB, "
                f"allocated={{torch.cuda.memory_allocated(device) / 1024**3:.2f}} GiB",
                flush=True,
            )

            for sec in range(HOLD_SECONDS):
                if sec % 5 == 0 or sec == HOLD_SECONDS - 1:
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(
                        f"[job] holding gpu memory: t={{sec + 1}}/{{HOLD_SECONDS}}s, "
                        f"allocated={{allocated:.2f}} GiB",
                        flush=True,
                    )
                time.sleep(1)

            del tensor
            torch.cuda.empty_cache()

            submission = pd.DataFrame({{"PassengerId": [1], "Survived": [0]}})
            submission_path = Path(__file__).resolve().parent / "submission.csv"
            submission.to_csv(submission_path, index=False)
            print(f"[job] wrote submission to {{submission_path}}", flush=True)
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip()


def safe_json(response: httpx.Response) -> dict:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_headers(api_key: str) -> dict:
    return {"X-API-Key": api_key}


async def get_job_status(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
) -> tuple[int, dict]:
    response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
    return response.status_code, safe_json(response)


async def submit_long_running_job(
    client: httpx.AsyncClient,
    headers: dict,
    data_dir: str,
    resource_type: str,
    priority: int,
    job_timeout: int,
    code_str: str,
) -> str:
    payload = {
        "name": "cancel-job-gpu-memory-test",
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": resource_type,
        "priority": priority,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
            "HF_ENDPOINT": "https://hf-mirror.com",
        },
    }
    response = await client.post("/api/v1/jobs", json=payload, headers=headers)
    data = safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"submit failed: status={response.status_code}, data={data}")
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"submit succeeded but job_id missing: data={data}")
    print(f"[submit] job_id={job_id}, data={data}")
    return job_id


async def cancel_job(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
) -> tuple[int, dict]:
    response = await client.delete(f"/api/v1/jobs/{job_id}", headers=headers)
    data = safe_json(response)
    print(f"[cancel] status_code={response.status_code}, data={data}")
    return response.status_code, data


async def wait_for_status(
    client: httpx.AsyncClient,
    headers: dict,
    job_id: str,
    *,
    target_statuses: set[str] | None,
    timeout_seconds: int,
    poll_seconds: float,
) -> tuple[int, dict]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_status = None
    while asyncio.get_running_loop().time() < deadline:
        status_code, data = await get_job_status(client, headers, job_id)
        status = str(data.get("status", "")).lower()
        if status != last_status:
            print(f"[poll] job_id={job_id}, http={status_code}, status={status or '<empty>'}")
            last_status = status

        if target_statuses and status in target_statuses:
            return status_code, data
        if status in TERMINAL_STATUSES:
            return status_code, data

        await asyncio.sleep(poll_seconds)

    raise TimeoutError(
        f"timed out waiting for job {job_id}, last_status={last_status or '<unknown>'}"
    )


async def run(args: argparse.Namespace) -> int:
    headers = build_headers(args.api_key)
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        job_id = args.job_id

        if not job_id:
            code_str = build_gpu_hold_code(
                vram_gb=args.vram_gb,
                hold_seconds=args.hold_seconds,
            )
            job_id = await submit_long_running_job(
                client=client,
                headers=headers,
                data_dir=args.data_dir,
                resource_type=args.resource_type,
                priority=args.priority,
                job_timeout=args.job_timeout,
                code_str=code_str,
            )

            try:
                await wait_for_status(
                    client,
                    headers,
                    job_id,
                    target_statuses={"running"},
                    timeout_seconds=args.start_timeout,
                    poll_seconds=args.poll_seconds,
                )
                print(f"[flow] job {job_id} entered running, sleep {args.cancel_after}s then cancel")
            except TimeoutError:
                print(
                    f"[flow] job {job_id} did not enter running within {args.start_timeout}s; "
                    f"will still try to cancel",
                )

            await asyncio.sleep(args.cancel_after)

        await cancel_job(client, headers, job_id)

        if args.watch_after_cancel > 0:
            try:
                status_code, data = await wait_for_status(
                    client,
                    headers,
                    job_id,
                    target_statuses={"cancelled", "canceled"},
                    timeout_seconds=args.watch_after_cancel,
                    poll_seconds=args.poll_seconds,
                )
                print(f"[done] job_id={job_id}, http={status_code}, final_status={data.get('status')}")
                result = data.get("result")
                if result:
                    print(f"[done] result={result}")
            except TimeoutError as exc:
                print(f"[warn] {exc}")
                return 2

        print(f"[done] cancel flow finished for job_id={job_id}")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit or cancel a sandbox job for cancellation testing.",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Sandbox API base URL")
    parser.add_argument("--api-key", default=API_KEY, help="API key")
    parser.add_argument(
        "--job-id",
        default="",
        help="Existing job_id to cancel. If omitted, the script submits a long-running job first.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset path used by the test job")
    parser.add_argument("--resource-type", default="gpu", choices=["cpu", "gpu"], help="Resource type")
    parser.add_argument("--priority", type=int, default=1, choices=[1, 2], help="Job priority")
    parser.add_argument("--job-timeout", type=int, default=300, help="Timeout passed to the job")
    parser.add_argument("--start-timeout", type=int, default=60, help="How long to wait for running status")
    parser.add_argument("--cancel-after", type=int, default=30, help="Seconds to wait before cancelling")
    parser.add_argument(
        "--watch-after-cancel",
        type=int,
        default=60,
        help="Seconds to keep polling after cancellation request",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--vram-gb", type=float, default=10.0, help="Approximate GPU memory to hold in GiB")
    parser.add_argument("--hold-seconds", type=int, default=120, help="How long the job keeps the GPU memory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("[exit] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
