import time
import os
import httpx
import asyncio
import re
import logging
import black
import math
import json
import random
import pandas as pd
from pathlib import Path

# Configure logging format with timestamp at module level
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

def normalize_sandbox_resource_type(
    resource_type: str | None, default: str = "gpu"
) -> str:
    """Normalize sandbox resource type to cpu/gpu with a safe default."""
    normalized_default = str(default or "gpu").strip().lower()
    if normalized_default not in {"cpu", "gpu"}:
        normalized_default = "gpu"

    if resource_type is None:
        return normalized_default

    normalized_resource = str(resource_type).strip().lower()
    if normalized_resource in {"cpu", "gpu"}:
        return normalized_resource
    return normalized_default


def resolve_sandbox_resource_type(
    metadata: dict | None,
    override: str | None = None,
    default: str = "gpu",
) -> str:
    """Resolve effective sandbox resource type from override or task metadata."""
    if override is not None:
        return normalize_sandbox_resource_type(override, default=default)
    task_resource = (metadata or {}).get("cpu_gpu")
    return normalize_sandbox_resource_type(task_resource, default=default)


async def get_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "gpu",
    priority: int = 1,
    job_timeout: int = 3600,
    wait_timeout: int = 7200,
    # wait_timeout: int = 100,
    poll_interval: int = 5,
):
    normalized_resource = normalize_sandbox_resource_type(resource_type, default="gpu")
    normalized_priority = priority if priority in (1, 2) else 1
    client_base_url = str(getattr(client, "base_url", "")).rstrip("/")
    if not client_base_url:
        raise ValueError("The sandbox HTTP client must define a base URL.")
    sandbox_api_key = (
        os.getenv(f"SANDBOX_{normalized_resource.upper()}_API_KEY")
        or os.getenv("SANDBOX_API_KEY")
    )
    if not sandbox_api_key:
        raise ValueError(
            f"SANDBOX_{normalized_resource.upper()}_API_KEY or SANDBOX_API_KEY "
            "is required."
        )
    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "priority": normalized_priority,
        "environment": {
            "EXECUTION_MODE": "shell",
            "DATA_DIR": data_dir,
            "SANDBOX_DATA_DIR": data_dir,
        },
    }
    headers = {"X-API-Key": sandbox_api_key}

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
        if message:
            parts.append(f"message={message}")
        else:
            parts.append(f"repr={exc!r}")

        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cause_message = str(cause).strip()
            if cause_message:
                parts.append(f"cause={type(cause).__name__}: {cause_message}")
            else:
                parts.append(f"cause={cause!r}")
        return "; ".join(str(p) for p in parts)

    # Submit the job without waiting for completion.
    submit_connect_retries = 0
    max_submit_connect_retries = 20
    submit_resp = None
    while True:
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",
                json=payload,
                headers=headers,
            )
            if submit_resp is None or submit_resp.status_code in (502, 503, 504, 429):
                if submit_connect_retries < max_submit_connect_retries:
                    submit_connect_retries += 1
                    await asyncio.sleep(float(1))
                    status_code = (
                        submit_resp.status_code if submit_resp is not None else "none"
                    )
                    print(
                        f"WARNING: transient error {status_code} when submitting job, retrying...",
                        flush=True,
                    )
                    continue
            break
        except Exception as e:
            error_detail = format_httpx_error(e, client_base_url)
            if submit_connect_retries < max_submit_connect_retries:
                submit_connect_retries += 1
                retry_delay = 1.0 * submit_connect_retries + random.uniform(0.0, 1.0)
                print(
                    f"WARNING: transient submit connect error to {client_base_url}: {error_detail}; "
                    f"retry {submit_connect_retries}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = (
                f"Failed to connect to sandbox API at {client_base_url}: {error_detail}"
            )
            print(f"ERROR: {error_msg}")
            return 503, {"error": "connection_failed", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TimeoutException as e:
        #     error_msg = f"Failed to submit job to sandbox API at {client_base_url}: {format_httpx_error(e, client_base_url)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_timeout", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TransportError as e:
        #     error_msg = f"Failed to submit job to sandbox API at {client_base_url}: {format_httpx_error(e, client_base_url)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_transport_failed", "detail": error_msg, "type": type(e).__name__}

    if submit_resp is None:
        return 503, {
            "error": "connection_failed",
            "detail": f"Failed to connect to sandbox API at xxx",
            "type": "ConnectError",
        }
    submit_data = safe_json(submit_resp)
    print("submit_data:", submit_data)
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {"error": "no job_id returned", "data": submit_data}

    # Poll until the job completes or the overall timeout expires.
    wait_status = ["running", "queued"]
    finished_status = ["success", "failed", "timeout", "canceled", "completed"]
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    max_poll_error_retries = 20
    total_running_time = 0.0
    last_status = None
    last_status_poll_at = None
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if r is None or r.status_code in (502, 503, 504, 429):
                if poll_error_retries < max_poll_error_retries:
                    poll_error_retries += 1
                    retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + random.uniform(0.0, 0.5)
                    print(
                        f"WARNING: transient poll status {r.status_code} for {job_id}, "
                        f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                        flush=True,
                    )
                    await asyncio.sleep(retry_delay)
                    continue
        except Exception as e:
            error_detail = format_httpx_error(
                e, f"{client_base_url}/api/v1/jobs/{job_id}"
            )
            if poll_error_retries < max_poll_error_retries:
                poll_error_retries += 1
                retry_delay = min(5.0, 1.0 + 0.25 * poll_error_retries) + random.uniform(0.0, 0.5)
                print(
                    f"WARNING: transient poll transport error for {job_id}: {error_detail}; "
                    f"retry {poll_error_retries}/{max_poll_error_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to poll job status: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "poll_failed", "detail": error_msg, "job_id": job_id, "type": type(e).__name__}
        poll_error_retries = 0
        data = safe_json(r)
        status = data.get("status")
        current_poll_at = time.monotonic()
        if last_status == "running" and last_status_poll_at is not None:
            total_running_time += max(0.0, current_poll_at - last_status_poll_at)
        last_status = status
        last_status_poll_at = current_poll_at
        print(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        if status in wait_status:
            await asyncio.sleep(poll_interval + random.uniform(0.0, 0.5))
            continue
        data["running_time"] = total_running_time
        if status in finished_status:
            return 200, data
        else:
            return 200, data

    return 504, {"error": "wait_timeout exceeded", "job_id": job_id, "running_time": total_running_time}



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
    """Map a metric score to a normalized reward."""
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
        if abs(float(score)) < 1e-7:
            return -100.0

        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)

    s = _signed(score, metadata)

    # A: Power sigmoid on min-max bounds.
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
        return r**alpha

    # B: Tanh margin shaping.
    if mode == "margin_tanh":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)

        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        m = (s - center) / scale

        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y  # (0,1)

        gamma = 2.0
        return r**gamma

    # C: Online percentile.
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
        return p**gamma

    # initial sigmoid reward
    if higher_is_better:
        # best score
        score_range_best_score = (
            theoretical_max if theoretical_max is not None else leaderboard_max
        )

        # worst score
        score_range_worst_score = safe_max(leaderboard_min, theoretical_min, -100)

    else:
        # best score
        score_range_best_score = (
            theoretical_min if theoretical_min is not None else leaderboard_min
        )

        # worst score
        score_range_worst_score = safe_min(leaderboard_max, theoretical_max, 100)

    # Return a neutral reward when the score range remains unknown.
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


def get_clear_log(run_log: str | None) -> str:
    """Extract marked output sections and remove heartbeat log blocks."""
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

            # Remove every heartbeat block with index searches.
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
                        # Drop an unterminated heartbeat block through end of text.
                        break
                    pos = e + hb_len

                content = "".join(parts).strip()

            if content:
                sections.append(f"\n{content}")

            if end_idx == -1:
                break
            search_pos = end_idx + len(end_marker)

    return "\n\n".join(sections)


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
