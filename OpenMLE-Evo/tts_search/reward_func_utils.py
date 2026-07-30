from __future__ import annotations

import os
import time
import httpx
import asyncio
import random
import re
import logging
import black
import math
import json
import pandas as pd
from pathlib import Path
import uuid
import warnings

# Configure logging format with timestamp at module level
logger = logging.getLogger('mle_agent')
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

BASE_URL = os.environ.get("SANDBOX_URL", "")
API_KEY = os.environ.get("SANDBOX_GPU_API_KEY", "")

RETRYABLE_SANDBOX_STATUS_CODES = {429, 502, 503, 504}


def _build_retryable_sandbox_payload(
    *,
    error: str,
    detail: str,
    status_code: int | None = None,
    job_id: str | None = None,
    data: dict | None = None,
    error_type: str | None = None,
) -> dict:
    payload = {
        "error": error,
        "detail": detail,
        "retryable": True,
    }
    if status_code is not None:
        payload["status_code"] = int(status_code)
    if job_id is not None:
        payload["job_id"] = job_id
    if data is not None:
        payload["data"] = data
    if error_type is not None:
        payload["type"] = error_type
    return payload

async def get_sandbox_result_deprecated(
    client: httpx.AsyncClient,         # ✅ 复用的客户端从外部传入
    code_str: str,
    data_dir: str,
    *,
    resource_type: str,
    eval_split=None,
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    # wait_timeout: int = 100,
    poll_interval: int = 5,
):
    warnings.warn(
        "get_sandbox_result_deprecated() uses the pre-v0.9.32 sandbox client. "
        "Use get_sandbox_result() for idempotent submit retry and safer polling.",
        DeprecationWarning,
        stacklevel=2,
    )
    normalized_resource = str(resource_type).lower()
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": int(priority),  # 1 (high) | 2 (low)
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
        },
    }
    headers = {"X-API-Key": API_KEY}

    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    # 1) 只提交，不等待（带重试）
    submit_error_retries = 0
    while True:
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",              # ✅ 使用 base_url，path 写相对地址
                json=payload,
                headers=headers,
            )
            if submit_resp.status_code in RETRYABLE_SANDBOX_STATUS_CODES:
                if submit_error_retries < 5:
                    submit_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(f"WARNING: transient error {submit_resp.status_code} when submitting job, retrying...")
                    continue
                detail = (
                    f"Sandbox API returned retryable status {submit_resp.status_code} "
                    "when submitting job after retries"
                )
                logger.info(f"ERROR: {detail}")
                return submit_resp.status_code, _build_retryable_sandbox_payload(
                    error="submit_transient_failed",
                    detail=detail,
                    status_code=submit_resp.status_code,
                    data=safe_json(submit_resp),
                )
            break
        except Exception as e:
            if submit_error_retries < 5:
                submit_error_retries += 1
                await asyncio.sleep(float(1))
                logger.info(f"WARNING: exception {type(e).__name__} when submitting job, retrying...")
                continue
            error_msg = f"Failed to connect to sandbox API at {client.base_url}: {type(e).__name__}: {str(e)}"
            logger.info(f"ERROR: {error_msg}")
            return 503, _build_retryable_sandbox_payload(
                error="connection_failed",
                detail=error_msg,
                status_code=503,
                error_type=type(e).__name__,
            )
    submit_data = safe_json(submit_resp)
    logger.info("submit_data: %s", submit_data)
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {"error": "no job_id returned", "data": submit_data}

    # 2) 轮询直到结束或完全超时
    wait_status = ["running", "queued"]
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if r.status_code in RETRYABLE_SANDBOX_STATUS_CODES:
                if poll_error_retries < 5:
                    poll_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(f"WARNING: transient error {r.status_code} when polling job status, retrying...")
                    continue
                detail = (
                    f"Sandbox API returned retryable status {r.status_code} "
                    "when polling job status after retries"
                )
                logger.info(f"ERROR: {detail}")
                return r.status_code, _build_retryable_sandbox_payload(
                    error="poll_transient_failed",
                    detail=detail,
                    status_code=r.status_code,
                    job_id=job_id,
                    data=safe_json(r),
                )
        except Exception as e:
            error_msg = f"Failed to poll job status: {type(e).__name__}: {str(e)}"
            logger.info(f"ERROR: {error_msg}")
            return 503, _build_retryable_sandbox_payload(
                error="poll_failed",
                detail=error_msg,
                status_code=503,
                job_id=job_id,
                error_type=type(e).__name__,
            )
        if r.status_code >= 400:
            return r.status_code, {
                "error": "poll failed",
                "detail": safe_json(r),
                "job_id": job_id,
            }
        data = safe_json(r)
        status = data.get("status")
        logger.info(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        await asyncio.sleep(poll_interval)
        if status in wait_status:
            continue
        else:
            return 200, data

    return 504, {"error": "wait_timeout exceeded", "job_id": job_id}


async def get_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "gpu",
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    poll_interval: int = 10,
    eval_split: str | None = None,
):
    normalized_resource = (resource_type or "cpu").lower()
    normalized_priority = priority if priority in (1, 2) else 1
    trace_id = uuid.uuid4().hex
    base_url = str(client.base_url).rstrip("/")
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": normalized_priority,
        "idempotency_key": trace_id,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
        },
    }
    if eval_split:
        normalized_eval_split = str(eval_split).strip().lower()
        payload["eval_split"] = normalized_eval_split
        payload["environment"]["EVAL_SPLIT"] = normalized_eval_split
    headers = {"X-API-Key": API_KEY, "X-Trace-ID": trace_id}

    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    def format_httpx_error(exc: Exception, fallback_url: str) -> str:
        parts = [f"type={type(exc).__name__}"]
        req = getattr(exc, "request", None)
        if req is not None:
            method = getattr(req, "method", "UNKNOWN")
            url = getattr(req, "url", fallback_url)
            parts.append(f"method={method}")
            parts.append(f"url={url}")
        else:
            parts.append(f"url={fallback_url}")

        message = str(exc).strip()
        parts.append(f"message={message}" if message else f"repr={exc!r}")

        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        return "; ".join(str(part) for part in parts)

    def retry_after_seconds(resp: httpx.Response, fallback: float) -> float:
        value = resp.headers.get("Retry-After")
        if value:
            try:
                return max(0.0, min(float(value), 30.0))
            except ValueError:
                pass
        return fallback

    submit_connect_retries = 50
    transient_submit_statuses = {429, 500, 502, 503, 504}
    submit_resp = None
    for submit_attempt in range(1, submit_connect_retries + 1):
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",
                json=payload,
                headers=headers,
            )
            if (
                submit_resp.status_code in transient_submit_statuses
                and submit_attempt < submit_connect_retries
            ):
                retry_delay = retry_after_seconds(
                    submit_resp,
                    min(30.0, 1.5**submit_attempt) + random.uniform(0.0, 1.0),
                )
                logger.info(
                    "WARNING: transient submit status %s trace_id=%s; retry %d/%d in %.2fs",
                    submit_resp.status_code,
                    trace_id,
                    submit_attempt,
                    submit_connect_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            break
        except (
            httpx.ConnectTimeout,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.TransportError,
        ) as exc:
            error_detail = format_httpx_error(exc, base_url)
            if submit_attempt < submit_connect_retries:
                retry_delay = min(30.0, 1.5**submit_attempt) + random.uniform(0.0, 1.0)
                logger.info(
                    "WARNING: transient submit error to %s: %s; trace_id=%s; retry %d/%d in %.2fs",
                    base_url,
                    error_detail,
                    trace_id,
                    submit_attempt,
                    submit_connect_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to submit job to sandbox API at {base_url}: {error_detail}"
            logger.info("ERROR: %s", error_msg)
            return 503, {
                "error": "submit_failed_after_retries",
                "detail": error_msg,
                "type": type(exc).__name__,
                "trace_id": trace_id,
            }

    if submit_resp is None:
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at {base_url}",
            "type": "ConnectError",
            "trace_id": trace_id,
        }

    submit_data = safe_json(submit_resp)
    logger.info("submit_data: %s", submit_data)
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {
            "error": "submit failed",
            "data": submit_data,
            "trace_id": trace_id,
        }

    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {
            "error": "no job_id returned",
            "data": submit_data,
            "trace_id": trace_id,
        }

    wait_status = ["running", "queued"]
    finished_status = ["completed", "failed", "cancelled"]
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    max_poll_error_retries = 20
    malformed_poll_retries = 0
    max_malformed_poll_retries = 3
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if response.status_code in (502, 503, 504, 429):
                if poll_error_retries < max_poll_error_retries:
                    poll_error_retries += 1
                    retry_delay = (
                        min(5.0, 1.0 + 0.25 * poll_error_retries)
                        + random.uniform(0.0, 0.5)
                    )
                    logger.info(
                        "WARNING: transient poll status %s for %s, retry %d/%d in %.2fs",
                        response.status_code,
                        job_id,
                        poll_error_retries,
                        max_poll_error_retries,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
            if 400 <= response.status_code < 500:
                return response.status_code, {
                    "error": "poll_failed",
                    "job_id": job_id,
                    "trace_id": trace_id,
                    "data": safe_json(response),
                }
        except httpx.TransportError as exc:
            error_detail = format_httpx_error(exc, f"{base_url}/api/v1/jobs/{job_id}")
            if poll_error_retries < max_poll_error_retries:
                poll_error_retries += 1
                retry_delay = (
                    min(5.0, 1.0 + 0.25 * poll_error_retries)
                    + random.uniform(0.0, 0.5)
                )
                logger.info(
                    "WARNING: transient poll transport error for %s: %s; retry %d/%d in %.2fs",
                    job_id,
                    error_detail,
                    poll_error_retries,
                    max_poll_error_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to poll job status: {error_detail}"
            logger.info("ERROR: %s", error_msg)
            return 503, {
                "error": "poll_failed",
                "detail": error_msg,
                "job_id": job_id,
                "trace_id": trace_id,
                "type": type(exc).__name__,
            }

        poll_error_retries = 0
        data = safe_json(response)
        status = data.get("status")
        logger.info("job_id:%s, status(%s):%s", job_id, time.monotonic(), status)
        if status in wait_status:
            await asyncio.sleep(poll_interval + random.uniform(0.0, 1.0))
            continue
        if status in finished_status:
            data.setdefault("trace_id", trace_id)
            return 200, data

        malformed_poll_retries += 1
        if malformed_poll_retries <= max_malformed_poll_retries:
            retry_delay = (
                min(5.0, 1.0 + malformed_poll_retries)
                + random.uniform(0.0, 0.5)
            )
            logger.info(
                "WARNING: unexpected poll body for %s: status=%s; retry %d/%d in %.2fs",
                job_id,
                status,
                malformed_poll_retries,
                max_malformed_poll_retries,
                retry_delay,
            )
            await asyncio.sleep(retry_delay)
            continue
        return 502, {
            "error": "unexpected_poll_response",
            "job_id": job_id,
            "trace_id": trace_id,
            "data": data,
        }

    return 504, {
        "error": "wait_timeout exceeded",
        "job_id": job_id,
        "trace_id": trace_id,
    }

def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False

def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code

def extract_code(text):
    """Extract python code blocks from the text."""
    # logger.info(f"raw text: {text}")
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    return format_code("\n\n".join(valid_code_blocks))


_RECENT = {}
_RECENT_MAXLEN = 20

def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    higher = meta.get("higher_is_better", True)
    s = float(score)
    return s if higher else -s


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def _bounds_signed(meta: dict):
    higher = meta.get("higher_is_better", True)

    def to_signed(v):
        if not _finite(v):
            return None
        v = float(v)
        return v if higher else -v

    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    cands = [tmin, tmax, lmin, lmax]
    cands = [x for x in cands if x is not None]
    if len(cands) < 2:
        return None, None

    best = max(cands)
    worst = min(cands)

    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        if lmax is not None and lmin is not None and (lmax - lmin) <= MAX_RANGE:
            best, worst = max(lmax, lmin), min(lmax, lmin)
        else:
            worst = best - MAX_RANGE

    return best, worst

def score2reward(score, metadata, mode="logistic"):
    """
    将 metric 的 score 映射为 reward。
    """
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    def safe_max(*vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def safe_min(*vals):
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    # assert mode=="linear_sign" , f"Unsupported mode: {mode}"

    if mode == "linear_sign":
        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)


    s = _signed(score, metadata)


    # A：Power Sigmoid on MinMax
    if mode == "power_sigmoid":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return _stable_sigmoid(s)

        rng = max(best - worst, 1e-9)
        # p in [0,1]
        p = (s - worst) / rng
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)

        T = 0.50
        x = (p - 0.5) / max(T, 1e-9)
        r = _stable_sigmoid(x)  # (0,1)

        alpha = 2.0
        return r ** alpha

    # B：Tanh Margin Shaping
    if mode == "margin_tanh":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)  # 兜底

        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        m = (s - center) / scale

        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y               # (0,1)

        gamma = 2.0
        return r ** gamma

    # C：Online Percentile
    if mode == "online_percentile":
        key = str(metadata.get("data_dir", "unknown_task"))
        dq = _RECENT.get(key)
        if dq is None:
            from collections import deque
            dq = deque(maxlen=_RECENT_MAXLEN)
            _RECENT[key] = dq

        if len(dq) < 10:
            dq.append(s)
            return _stable_sigmoid(s)

        sorted_vals = sorted(dq)
        # rank = count(vals <= s)
        import bisect
        rank = bisect.bisect_right(sorted_vals, s)
        p = rank / len(sorted_vals)  # [0,1]

        dq.append(s)

        gamma = 2.0
        return p ** gamma

    # initial sigmoid reward
    if higher_is_better:
        # best score
        score_range_best_score = (
            theoretical_max
            if theoretical_max is not None
            else leaderboard_max
        )

        # worst score
        score_range_worst_score = safe_max(
            leaderboard_min,
            theoretical_min,
            -100
        )

    else:
        # best score
        score_range_best_score = (
            theoretical_min
            if theoretical_min is not None
            else leaderboard_min
        )

        # worst score
        score_range_worst_score = safe_min(
            leaderboard_max,
            theoretical_max,
            100
        )

    # 若仍然无法确定区间，直接返回中性奖励
    if score_range_best_score is None or score_range_worst_score is None:
        return 0.5

    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        return 0.5

    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    return reward



def test_score2reward(base_dir: str = "./Selected_Dojo"):
    """
    Test score2reward function for all tasks in Selected_Dojo.

    For each task, displays score and reward at:
    - Rank 1 (best)
    - 20th percentile
    - 40th percentile
    - 60th percentile
    - 80th percentile
    - Last rank (worst)

    Args:
        base_dir: Base directory containing task folders
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # Find all task directories
    task_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            metadata_path = item / "info" / "task_metadata.json"
            leaderboard_path = item / "info" / "public_leaderboard.csv"
            if metadata_path.exists() and leaderboard_path.exists():
                task_dirs.append(item)

    if not task_dirs:
        print(f"No valid tasks found in {base_dir}")
        return

    print(f"Found {len(task_dirs)} tasks to test\n")
    print("=" * 100)

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        metadata_path = task_dir / "info" / "task_metadata.json"
        leaderboard_path = task_dir / "info" / "public_leaderboard.csv"

        # Load metadata
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {task_name}: {e}")
            continue

        # Load leaderboard
        try:
            df = pd.read_csv(leaderboard_path)
            if "score" not in df.columns:
                print(f"Warning: {task_name} leaderboard does not have 'score' column")
                continue
            scores = df["score"].dropna().astype(float).tolist()
            if not scores:
                print(f"Warning: {task_name} leaderboard has no valid scores")
                continue
        except Exception as e:
            print(f"Error loading leaderboard for {task_name}: {e}")
            continue

        # Sort scores based on higher_is_better
        higher_is_better = metadata.get("higher_is_better", True)
        if higher_is_better:
            scores_sorted = sorted(scores, reverse=True)  # Best to worst
        else:
            scores_sorted = sorted(scores)  # Best (lowest) to worst (highest)

        total_entries = len(scores_sorted)
        if total_entries == 0:
            continue

        # Calculate indices for percentiles
        # Use floor to get the index, ensuring we get the entry at or below the percentile
        def percentile_index(pct: float) -> int:
            """Calculate index for a given percentile (0.0 to 1.0)."""
            idx = int((total_entries - 1) * pct)
            return max(0, min(idx, total_entries - 1))

        indices = [
            0,  # Rank 1 (best)
            percentile_index(0.2),  # 20%
            percentile_index(0.4),  # 40%
            percentile_index(0.6),  # 60%
            percentile_index(0.8),  # 80%
            total_entries - 1,  # Last (worst)
        ]

        # Get scores at these positions
        selected_scores = [scores_sorted[i] for i in indices]
        selected_ranks = [i + 1 for i in indices]  # Rank is 1-indexed

        # Calculate rewards
        rewards = []
        for score in selected_scores:
            try:
                reward = score2reward(score, metadata)
                rewards.append(reward)
            except Exception as e:
                print(f"  Warning: Failed to calculate reward for score {score:.6f}: {e}")
                rewards.append(float('nan'))

        # Display results
        print(f"\nTask: {task_name}")
        print(f"  UUID: {metadata.get('uuid', 'N/A')}")
        print(f"  Higher is better: {higher_is_better}")
        print(f"  Theoretical min: {metadata.get('theoretical_min', 'N/A')}")
        print(f"  Theoretical max: {metadata.get('theoretical_max', 'N/A')}")
        print(f"  Total entries: {total_entries}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"\n  {'Position':<12} {'Rank':<8} {'Score':<15} {'Reward':<10}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 15} {'-' * 10}")

        labels = ["1st (best)", "20%", "40%", "60%", "80%", "Last (worst)"]
        for label, rank, score, reward in zip(labels, selected_ranks, selected_scores, rewards):
            if math.isnan(reward):
                reward_str = "N/A"
            else:
                reward_str = f"{reward:.6f}"
            print(f"  {label:<12} {rank:<8} {score:<15.6f} {reward_str:<10}")

        print("=" * 100)

def get_clear_log(run_log: str | None) -> str:
    """
    从 run_log 中提取核心输出，基于新的包裹标记：
      - --- OUTPUT START --- ~~~ --- OUTPUT END ---
      - --- SANDBOX STDOUT START --- ~~~ --- SANDBOX STDOUT END ---
    并删除所有 [HB] ... [HB] 包裹的日志内容。
    """
    if not run_log:
        return ""

    markers = [
        ("--- OUTPUT START ---", "--- OUTPUT END ---"),
        ("--- SANDBOX STDOUT START ---", "--- SANDBOX STDOUT END ---"),
    ]

    hb_marker = "[HB]"
    hb_len = len(hb_marker)

    sections: list[str] = []

    for start_marker, end_marker in markers:
        search_pos = 0
        while True:
            start_idx = run_log.find(start_marker, search_pos)
            if start_idx == -1:
                break

            content_start = start_idx + len(start_marker)
            if content_start < len(run_log) and run_log[content_start] == "\n":
                content_start += 1

            end_idx = run_log.find(end_marker, content_start)
            if end_idx == -1:
                content = run_log[content_start:].strip()
            else:
                content = run_log[content_start:end_idx].strip()

            # === 删除所有 [HB] ... [HB] 块（用 find，不用 re） ===
            if content:
                parts: list[str] = []
                pos = 0
                while True:
                    s = content.find(hb_marker, pos)
                    if s == -1:
                        parts.append(content[pos:])
                        break
                    parts.append(content[pos:s])
                    e = content.find(hb_marker, s + hb_len)
                    if e == -1:
                        # 未闭合：从 s 到末尾丢弃
                        break
                    pos = e + hb_len

                content = "".join(parts).strip()

            if content:
                sections.append(f"\n{content}")

            if end_idx == -1:
                break
            search_pos = end_idx + len(end_marker)

    return "\n\n".join(sections)


def parse_final_validation_score(run_log: str | None) -> float | None:
    if not run_log:
        return None

    matches = re.findall(
        r"Final Validation Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        run_log,
    )
    if not matches:
        return None

    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None

def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    if status_code == 200:
        result = payload.get("result") or {}
        status = payload.get("status", "unknown")
        run_log = get_clear_log(result.get("run_log"))
        run_result = result.get("result", "")
        score = result.get("score")

        feedback = f"## Execution Result\n"
        feedback += f"**Status**: {status}\n\n"

        if score is not None:
            feedback += f"**Score**: {score}\n\n"

        if run_result:
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        feedback = f"## Connection Error\n"
        feedback += f"**Type**: {error_type}\n"
        feedback += f"**Detail**: {error_detail}\n\n"
        feedback += "Please check your code and try again."
    else:
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        feedback = f"## Execution Error\n"
        feedback += f"**Error**: {error_msg}\n"
        if isinstance(detail, dict):
            if detail.get("status"):
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                feedback += f"**Message**: {detail.get('message')}\n"
        feedback += "\nPlease fix the errors and try again."

    return feedback

if __name__ == "__main__":
    # Test score2reward for all tasks
    test_score2reward()
