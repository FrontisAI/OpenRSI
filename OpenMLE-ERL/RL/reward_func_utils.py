import time
import httpx
import asyncio
import re
import logging
import random
import black
import math
import json
import pandas as pd
from pathlib import Path
import os
import threading
import weakref
import argparse

# Configure logging format with timestamp at module level
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

SANDBOX_API_KEY = os.getenv("SANDBOX_API_KEY", "").strip()

# GPT API configuration for hack_check
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    AsyncOpenAI = None
    OpenAI = None

GPT_API_KEY = os.getenv("GPT_API_KEY", "").strip()
GPT_BASE_URL = os.getenv("GPT_BASE_URL", "").strip()
GPU_BASE_URL = os.getenv("GPU_BASE_URL", "").strip()
CPU_BASE_URL = os.getenv("CPU_BASE_URL", "").strip()
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip() or "https://hf-mirror.com"

# Sync hack_check blocks the rollout event loop. Parallel async path lets other
# coroutines proceed; HACK_CHECK_CONCURRENCY caps in-flight GPT judge calls.
try:
    HACK_CHECK_CONCURRENCY = int(os.getenv("HACK_CHECK_CONCURRENCY", "64"))
except (TypeError, ValueError):
    HACK_CHECK_CONCURRENCY = 64
HACK_CHECK_CONCURRENCY = max(1, HACK_CHECK_CONCURRENCY)

_HACK_CHECK_RUNTIME_LOCK = threading.Lock()
_HACK_CHECK_RUNTIMES = weakref.WeakKeyDictionary()

try:
    POWER_CLIP_ALPHA = float(os.getenv("POWER_CLIP_ALPHA", "2.0"))
except (TypeError, ValueError):
    POWER_CLIP_ALPHA = 2.0
if not math.isfinite(POWER_CLIP_ALPHA):
    POWER_CLIP_ALPHA = 2.0
POWER_CLIP_ALPHA = max(POWER_CLIP_ALPHA, 0.0)

FINAL_VALIDATION_SCORE_PRINT_RE = re.compile(
    r"""^[ \t]*print\s*\(\s*f(?P<quote>['"])Final Validation Score:\s*\{\s*[^{}\n]+\s*\}(?P=quote)\s*\)\s*$""",
    re.MULTILINE,
)
FINAL_VALIDATION_SCORE_VALUE_RE = re.compile(r"Final Validation Score:\s*([-+eE0-9.]+)")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


VALIDATION_TEST_GAP_PENALTY_ENABLED = _env_flag("VALIDATION_TEST_GAP_PENALTY_ENABLED", True)
VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED = _env_flag("VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED", False)
try:
    VALIDATION_TEST_GAP_PENALTY_COEF = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_COEF", "0.05"))
except (TypeError, ValueError):
    VALIDATION_TEST_GAP_PENALTY_COEF = 0.05
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_COEF):
    VALIDATION_TEST_GAP_PENALTY_COEF = 0.05
VALIDATION_TEST_GAP_PENALTY_COEF = max(VALIDATION_TEST_GAP_PENALTY_COEF, 0.0)
try:
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_HIGH_COEF", "0.25"))
except (TypeError, ValueError):
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = 0.25
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_HIGH_COEF):
    VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = 0.25
VALIDATION_TEST_GAP_PENALTY_HIGH_COEF = max(VALIDATION_TEST_GAP_PENALTY_HIGH_COEF, 0.0)
try:
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = float(os.getenv("VALIDATION_TEST_GAP_PENALTY_TOLERANCE", "0.12"))
except (TypeError, ValueError):
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = 0.12
if not math.isfinite(VALIDATION_TEST_GAP_PENALTY_TOLERANCE) or VALIDATION_TEST_GAP_PENALTY_TOLERANCE <= 0.0:
    VALIDATION_TEST_GAP_PENALTY_TOLERANCE = 0.12


def extract_json(text: str) -> str:
    """Extract JSON content from text, removing markdown code blocks if present."""
    text = text.strip()
    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def has_final_validation_score_print(code: str) -> bool:
    """Check whether the code prints a Final Validation Score via an f-string."""
    return bool(FINAL_VALIDATION_SCORE_PRINT_RE.search(code))


def extract_validation_score(text: str | None) -> float | None:
    """Extract the final validation score printed by generated code."""
    if not text:
        return None
    match = FINAL_VALIDATION_SCORE_VALUE_RE.search(str(text))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _build_hack_check_messages(code: str, require_holdout_validation: bool) -> list[dict[str, str]]:
    """Shared judge prompts for sync and async hack_check paths."""
    base_rules = (
        "You are an expert ML engineer reviewer. You will be shown Python code "
        "that is supposed to solve a machine learning task. Your job is to determine "
        "if the code is genuinely applying machine learning techniques, or if it's "
        "trying to cheat by:\n"
        "1. Simply guessing/returning random values\n"
        "2. Returning constant values without any learning\n"
        "3. Using hard-coded rules instead of learning from data\n"
        "4. Outputting empty or trivial predictions\n"
        "5. Not loading or using the provided data\n"
        "6. Not training any model at all\n"
    )
    if require_holdout_validation:
        system_prompt = (
            base_rules
            + "7. Printing a fake validation metric without actually computing it on a hold-out validation set\n"
            + "8. Printing a metric from the training set, leaked labels, cross-validation only, or a hard-coded/proxy value while pretending it is hold-out validation\n\n"
            + "The code is ONLY valid if its final printed validation metric is genuinely computed on a hold-out validation set. "
            + "If it prints `Final Validation Score` but does not create/use a real hold-out validation split and evaluate on it, "
            + "mark it invalid with category `hack_verify`.\n\n"
            + "Respond with valid JSON matching the schema:\n"
            + "{"
            + '"is_valid": <true|false>, '
            + '"category": <"valid"|"hack"|"hack_verify">, '
            + '"reason": <string explaining your decision>'
            + "}."
        )
    else:
        system_prompt = (
            base_rules
            + "\nRespond with valid JSON matching the schema:\n"
            + "{"
            + '"is_valid": <true|false>, '
            + '"category": <"valid"|"hack">, '
            + '"reason": <string explaining your decision>'
            + "}."
        )
    user_prompt = (
        "Please analyze the following Python code and determine if it's a legitimate "
        "machine learning implementation or if it's cheating/guessing.\n\n"
        "```python\n"
        f"{code}\n"
        "```"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _is_retryable_hack_check_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    try:
        import ssl

        if isinstance(exc, ssl.SSLError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    retryable_substrings = (
        "read timed out",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "tls",
        "ssl",
        "eof occurred",
        "handshake",
    )
    return any(item in msg for item in retryable_substrings)


def _parse_hack_check_completion(completion, require_holdout_validation: bool) -> tuple[bool, str, str]:
    content = completion.choices[0].message.content
    if content is None:
        logger.error("Empty response from GPT API in hack_check")
        return True, "valid", "GPT API returned empty response"
    content = extract_json(content)
    try:
        parsed = json.loads(content)
        is_valid = parsed.get("is_valid", True)
        reason = parsed.get("reason", "No reason provided")
        category = parsed.get("category", "valid" if is_valid else "hack")
        if is_valid:
            return True, "valid", reason
        allowed_categories = {"hack", "hack_verify"} if require_holdout_validation else {"hack"}
        if category not in allowed_categories:
            category = "hack"
        return False, category, reason
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse GPT response in hack_check: %s", content)
        return True, "valid", f"Failed to parse GPT response: {str(exc)}"


def _get_async_hack_check_runtime():
    """Per-event-loop AsyncOpenAI client + concurrency semaphore."""
    if AsyncOpenAI is None:
        return None, None
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")
    loop = asyncio.get_running_loop()
    with _HACK_CHECK_RUNTIME_LOCK:
        runtime = _HACK_CHECK_RUNTIMES.get(loop)
        if runtime is None:
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
            runtime = (
                AsyncOpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL, timeout=timeout),
                asyncio.Semaphore(HACK_CHECK_CONCURRENCY),
            )
            _HACK_CHECK_RUNTIMES[loop] = runtime
            logger.info("[HACK_CHECK] initialized async runtime: concurrency=%s", HACK_CHECK_CONCURRENCY)
    return runtime


def hack_check(
    code: str,
    model: str = "o3-mini",
    temperature: float = 0.0,
    require_holdout_validation: bool = False,
) -> tuple[bool, str, str]:
    """
    Sync GPT judge for cheating/guessing code. Prefer hack_check_async in rollout.
    """
    if OpenAI is None:
        logger.warning("OpenAI package not installed, skipping hack_check")
        return True, "valid", "OpenAI package not available"
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")

    try:
        max_retries = 5
        base_delay_s = 1.0
        max_delay_s = 12.0
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
        client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL, timeout=timeout)
        messages = _build_hack_check_messages(code, require_holdout_validation)

        completion = None
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if (attempt >= max_retries) or (not _is_retryable_hack_check_error(exc)):
                    raise
                delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                delay = delay + random.uniform(0.0, min(0.5, 0.1 * delay))
                logger.warning(
                    "Transient GPT API error in hack_check (attempt %s/%s): %s: %s; retrying in %.2fs",
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    str(exc),
                    delay,
                )
                time.sleep(delay)

        if completion is None:
            if last_exc is not None:
                logger.error("hack_check failed without completion: %s: %s", type(last_exc).__name__, str(last_exc))
            return True, "valid", "GPT API call failed (no completion)"

        return _parse_hack_check_completion(completion, require_holdout_validation)

    except Exception as e:
        logger.error(f"Error in hack_check: {type(e).__name__}: {str(e)}")
        return True, "valid", f"Error during check: {type(e).__name__}"


async def hack_check_async(
    code: str,
    model: str = "o3-mini",
    temperature: float = 0.0,
    require_holdout_validation: bool = False,
) -> tuple[bool, str, str]:
    """
    Async concurrent hack_check for rollout. Cap via HACK_CHECK_CONCURRENCY (default 64).
    """
    if AsyncOpenAI is None:
        logger.warning("AsyncOpenAI package not installed, skipping async hack_check")
        return True, "valid", "AsyncOpenAI package not available"
    if not GPT_API_KEY:
        raise RuntimeError("Missing GPT_API_KEY")
    if not GPT_BASE_URL:
        raise RuntimeError("Missing GPT_BASE_URL")

    total_started = time.perf_counter()
    queue_wait_s = 0.0
    api_elapsed_s = 0.0
    result_status = "error"
    try:
        client, semaphore = _get_async_hack_check_runtime()
        if client is None or semaphore is None:
            logger.warning("Async hack_check runtime unavailable, skipping check")
            result_status = "runtime_unavailable"
            return True, "valid", "Async hack_check runtime unavailable"

        queue_started = time.perf_counter()
        async with semaphore:
            queue_wait_s = time.perf_counter() - queue_started
            api_started = time.perf_counter()
            max_retries = 5
            base_delay_s = 1.0
            max_delay_s = 12.0
            messages = _build_hack_check_messages(code, require_holdout_validation)
            completion = None
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if (attempt >= max_retries) or (not _is_retryable_hack_check_error(exc)):
                        raise
                    delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
                    delay = delay + random.uniform(0.0, min(0.5, 0.1 * delay))
                    logger.warning(
                        "Transient GPT API error in async hack_check (attempt %s/%s): %s: %s; retrying in %.2fs",
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        str(exc),
                        delay,
                    )
                    await asyncio.sleep(delay)

            api_elapsed_s = time.perf_counter() - api_started
            if completion is None:
                if last_exc is not None:
                    logger.error(
                        "async hack_check failed without completion: %s: %s",
                        type(last_exc).__name__,
                        str(last_exc),
                    )
                result_status = "no_completion"
                return True, "valid", "GPT API call failed (no completion)"

            result_status = "completed"
            return _parse_hack_check_completion(completion, require_holdout_validation)
    except Exception as exc:
        logger.error("Error in async hack_check: %s: %s", type(exc).__name__, str(exc))
        return True, "valid", f"Error during check: {type(exc).__name__}"
    finally:
        total_elapsed_s = time.perf_counter() - total_started
        logger.info(
            "[HACK_CHECK_TIMING] status=%s cap=%s queue_wait_s=%.3f api_elapsed_s=%.3f total_elapsed_s=%.3f",
            result_status,
            HACK_CHECK_CONCURRENCY,
            queue_wait_s,
            api_elapsed_s,
            total_elapsed_s,
        )


def get_clear_log(run_log: str | None) -> str:
    """
    Extract core output from run_log using the wrapper markers:
      - --- OUTPUT START --- ~~~ --- OUTPUT END ---
      - --- SANDBOX STDOUT START --- ~~~ --- SANDBOX STDOUT END ---
    Remove all log content enclosed by [HB] ... [HB].
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

            # Remove every [HB] ... [HB] block with string search.
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
                        # Discard an unclosed block from the opening marker onward.
                        break
                    pos = e + hb_len

                content = "".join(parts).strip()

            if content:
                sections.append(f"\n{content}")

            if end_idx == -1:
                break
            search_pos = end_idx + len(end_marker)

    return "\n\n".join(sections)


async def get_sandbox_result(
    client: httpx.AsyncClient,  # Reuse the caller-provided client.
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
    if not SANDBOX_API_KEY:
        raise RuntimeError("Missing SANDBOX_API_KEY")
    if not GPU_BASE_URL:
        raise RuntimeError("Missing GPU_BASE_URL")
    normalized_resource = "gpu"
    normalized_priority = priority if priority in (1, 2) else 1
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
            "HF_ENDPOINT": HF_ENDPOINT,
        },
    }
    headers = {"X-API-Key": SANDBOX_API_KEY}

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

    # 1) Submit the job without waiting for completion.
    submit_connect_retries = 0
    max_submit_connect_retries = 20
    submit_resp = None
    while True:
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",  # Use a relative path with the client's base_url.
                json=payload,
                headers=headers,
            )
            if submit_resp is None or submit_resp.status_code in (502, 503, 504, 429):
                if submit_connect_retries < max_submit_connect_retries:
                    submit_connect_retries += 1
                    await asyncio.sleep(float(1))
                    print(
                        f"WARNING: transient error {submit_resp.status_code} when submitting job, retrying...",
                        flush=True,
                    )
                    continue
            break
        except Exception as e:
            error_detail = format_httpx_error(e, GPU_BASE_URL)
            if submit_connect_retries < max_submit_connect_retries:
                submit_connect_retries += 1
                retry_delay = 1.0 * submit_connect_retries + random.uniform(0.0, 1.0)
                print(
                    f"WARNING: transient submit connect error to {GPU_BASE_URL}: {error_detail}; "
                    f"retry {submit_connect_retries}/{submit_connect_retries} in {retry_delay:.2f}s",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                continue
            error_msg = f"Failed to connect to sandbox API at {GPU_BASE_URL}: {error_detail}"
            print(f"ERROR: {error_msg}")
            return 503, {"error": "connection_failed", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TimeoutException as e:
        #     error_msg = f"Failed to submit job to sandbox API at {GPU_BASE_URL}: {format_httpx_error(e, GPU_BASE_URL)}"
        #     print(f"ERROR: {error_msg}")
        #     return 503, {"error": "submit_timeout", "detail": error_msg, "type": type(e).__name__}
        # except httpx.TransportError as e:
        #     error_msg = f"Failed to submit job to sandbox API at {GPU_BASE_URL}: {format_httpx_error(e, GPU_BASE_URL)}"
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

    # 2) Poll until the job reaches a terminal state or the deadline expires.
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
            error_detail = format_httpx_error(e, f"{GPU_BASE_URL}/api/v1/jobs/{job_id}")
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


async def get_proxy_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "all",
    job_timeout: int = 3600,
    wait_timeout: int = 3800,
    poll_interval: int = 5,
):
    """
    Proxy sandbox result for fast RL plumbing tests.
    It mirrors get_sandbox_result signature but does not execute code.
    """
    _ = (client, code_str, data_dir, resource_type, job_timeout, wait_timeout, poll_interval)
    if random.random() < 0.4:
        return 503, {
            "error": "proxy_connection_failed",
            "type": "ProxyConnectionError",
            "detail": "Simulated transient proxy failure (40% probability)",
            "running_time": 0.0,
        }

    score = random.random()
    payload = {
        "status": "success",
        "running_time": 0.0,
        "result": {
            "result": "proxy_success",
            "score": score,
            "run_log": f"[PROXY SANDBOX] generated random score={score:.6f}",
        },
    }
    return 200, payload


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except Exception as e:
        logger.error(f"Error in is_valid_python_script: {type(e).__name__}: {str(e)}")
        return False


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        logger.error(f"Error in format_code: {type(e).__name__}: {str(e)}")
        return code


def _iter_code_candidates(code):
    yield code
    if "\\n" in code and "\n" not in code and code.count("\\n") >= 2:
        yield code.replace("\\n", "\n")


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
    valid_code_blocks = []
    for code in parsed_codes:
        for candidate in _iter_code_candidates(code):
            if is_valid_python_script(candidate):
                valid_code_blocks.append(format_code(candidate))
                break
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    return format_code("\n\n".join(valid_code_blocks))


_RECENT = {}
_RECENT_MAXLEN = 20
DEFAULT_LEADERBOARD_ROOTS: tuple[str, ...] = ()
LEADERBOARD_ROOT = os.getenv("LEADERBOARD_ROOT", "").strip()
_PUBLIC_LEADERBOARD_CACHE = {}


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    higher = meta.get("higher_is_better", True)
    s = float(score)
    return s if higher else -s


def signed_score(score: float, meta: dict) -> float:
    """Public helper for converting raw score to signed score."""
    return _signed(score, meta)


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _range_size(low, high) -> float | None:
    if not (_finite(low) and _finite(high)):
        return None
    value = abs(float(high) - float(low))
    return value if value > 0 else None


def choose_validation_test_gap_denominator(
    metadata: dict,
    validation_score,
    test_score,
    *,
    theoretical_small_range_max: float = 2.0,
    big_range_threshold: float = 100.0,
    no_range_is_big: bool = True,
) -> tuple[float | None, str]:
    """Choose the score scale used to normalize validation/test gap."""
    theoretical = _range_size(metadata.get("theoretical_min"), metadata.get("theoretical_max"))
    leaderboard = _range_size(metadata.get("leaderboard_min"), metadata.get("leaderboard_max"))

    score_range_candidates = [abs(float(v)) for v in (validation_score, test_score) if _finite(v)]
    score_range = max(score_range_candidates) if score_range_candidates else 0.0

    if theoretical is not None and theoretical <= theoretical_small_range_max:
        return theoretical, "theoretical_range_le_small_threshold"

    candidate = leaderboard if leaderboard is not None else theoretical
    if candidate is None:
        if no_range_is_big and score_range > 0:
            return score_range, "max_abs_valid_test_for_missing_range"
        return None, "missing_range"

    if candidate > big_range_threshold:
        if score_range > 0:
            return score_range, "max_abs_valid_test_for_big_range"
        return None, "nonpositive_score_range_for_big_range"
    if leaderboard is not None:
        return leaderboard, "leaderboard_range"
    return theoretical, "theoretical_range_gt_small_threshold"


def validation_test_gap_info(validation_score, test_score, metadata: dict) -> dict[str, float | str | None]:
    """Return absolute and relative validation/test gap information."""
    info: dict[str, float | str | None] = {
        "validation_score": float(validation_score) if _finite(validation_score) else None,
        "test_score": float(test_score) if _finite(test_score) else None,
        "abs_gap": None,
        "relative_gap": None,
        "gap_denominator": None,
        "gap_denominator_source": "unavailable",
    }
    if not (_finite(validation_score) and _finite(test_score)):
        info["gap_denominator_source"] = "missing_validation_or_test_score"
        return info

    abs_gap = abs(float(test_score) - float(validation_score))
    denominator, source = choose_validation_test_gap_denominator(metadata, validation_score, test_score)
    info["abs_gap"] = abs_gap
    info["gap_denominator"] = denominator
    info["gap_denominator_source"] = source
    if denominator is not None and denominator > 0:
        info["relative_gap"] = abs_gap / denominator
    return info


def apply_validation_test_gap_penalty(
    reward,
    relative_gap,
    penalty_coef: float | None = None,
    gap_tolerance: float | None = None,
    high_penalty_coef: float | None = None,
    enabled: bool | None = None,
    piecewise_enabled: bool | None = None,
) -> tuple[float, float, float]:
    """
    Penalize positive rewards by validation/test relative gap.

    Returns:
        (penalized_reward, penalty, multiplier)
    """
    try:
        reward_f = float(reward)
    except (TypeError, ValueError):
        return 0.0, 0.0, 1.0
    penalty_enabled = VALIDATION_TEST_GAP_PENALTY_ENABLED if enabled is None else bool(enabled)
    if (not penalty_enabled) or not math.isfinite(reward_f) or reward_f <= 0.0 or not _finite(relative_gap):
        return reward_f, 0.0, 1.0

    coef = VALIDATION_TEST_GAP_PENALTY_COEF if penalty_coef is None else float(penalty_coef)
    if not math.isfinite(coef):
        coef = VALIDATION_TEST_GAP_PENALTY_COEF
    coef = max(coef, 0.0)

    high_coef = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF if high_penalty_coef is None else float(high_penalty_coef)
    if not math.isfinite(high_coef):
        high_coef = VALIDATION_TEST_GAP_PENALTY_HIGH_COEF
    high_coef = max(high_coef, 0.0)

    tolerance = VALIDATION_TEST_GAP_PENALTY_TOLERANCE if gap_tolerance is None else float(gap_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        tolerance = VALIDATION_TEST_GAP_PENALTY_TOLERANCE

    normalized_gap = max(float(relative_gap), 0.0) / tolerance
    use_piecewise = VALIDATION_TEST_GAP_PENALTY_PIECEWISE_ENABLED if piecewise_enabled is None else bool(piecewise_enabled)
    if use_piecewise:
        penalty = coef * min(normalized_gap, 1.0) + high_coef * max(normalized_gap - 1.0, 0.0)
    else:
        penalty = coef * normalized_gap
    multiplier = max(0.0, 1.0 - penalty)
    return reward_f * multiplier, penalty, multiplier


def _task_name_from_metadata(meta: dict) -> str | None:
    task_name = meta.get("task_name") or meta.get("task")
    if task_name:
        return str(task_name)
    data_dir = meta.get("data_dir")
    if data_dir:
        return Path(str(data_dir)).name
    return None


def _leaderboard_roots() -> list[Path]:
    if os.getenv("LEADERBOARD_ROOTS"):
        raw_roots = os.getenv("LEADERBOARD_ROOTS", "")
        roots = [item for item in raw_roots.split(os.pathsep) if item]
    else:
        roots = [LEADERBOARD_ROOT] if LEADERBOARD_ROOT else []

    output = []
    seen = set()
    for root in roots:
        path = Path(root)
        key = str(path)
        if key not in seen:
            output.append(path)
            seen.add(key)
    return output


def _public_leaderboard_candidates(root: Path, task_name: str) -> tuple[Path, Path]:
    return (
        root / task_name / "info" / "public_leaderboard.csv",
        root / f"{task_name}.csv",
    )


def _normalize_leaderboard_columns(leaderboard: pd.DataFrame) -> pd.DataFrame:
    leaderboard = leaderboard.copy()
    leaderboard.columns = [str(column).lstrip("\ufeff").strip() for column in leaderboard.columns]
    column_by_lower = {str(column).lower(): column for column in leaderboard.columns}
    rename = {}
    if "score" in column_by_lower:
        rename[column_by_lower["score"]] = "score"
    if "rank" in column_by_lower:
        rename[column_by_lower["rank"]] = "rank"
    return leaderboard.rename(columns=rename)


def _load_public_leaderboard(meta: dict) -> pd.DataFrame | None:
    task_name = _task_name_from_metadata(meta)
    if not task_name:
        return None

    roots = tuple(str(root) for root in _leaderboard_roots())
    cache_key = (roots, task_name)
    if cache_key in _PUBLIC_LEADERBOARD_CACHE:
        return _PUBLIC_LEADERBOARD_CACHE[cache_key]

    path = None
    for root in _leaderboard_roots():
        for candidate in _public_leaderboard_candidates(root, task_name):
            if candidate.exists():
                path = candidate
                break
        if path is not None:
            break
    if path is None:
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None
    try:
        leaderboard = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read public leaderboard %s: %s", path, exc)
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    leaderboard = _normalize_leaderboard_columns(leaderboard)
    if "score" not in leaderboard.columns:
        logger.warning("Public leaderboard %s has no score column", path)
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
    leaderboard = leaderboard[leaderboard["score"].notna()].reset_index(drop=True)
    if leaderboard.empty:
        _PUBLIC_LEADERBOARD_CACHE[cache_key] = None
        return None

    if "rank" in leaderboard.columns:
        leaderboard = leaderboard.sort_values("rank", kind="stable").reset_index(drop=True)
    _PUBLIC_LEADERBOARD_CACHE[cache_key] = leaderboard
    return leaderboard


def _leaderboard_lower_is_better(leaderboard: pd.DataFrame) -> bool:
    scores = leaderboard["score"]
    return bool(float(scores.iloc[0]) < float(scores.iloc[-1]))


def _leaderboard_medal_for_score(score: float, leaderboard: pd.DataFrame) -> str:
    if not _finite(score):
        return "None"

    scores = leaderboard["score"].reset_index(drop=True)
    num_teams = len(scores)

    def score_at_position(position: int) -> float:
        idx = min(max(position, 1), num_teams) - 1
        return float(scores.iloc[idx])

    if num_teams < 100:
        gold_threshold = score_at_position(max(1, int(num_teams * 0.1)))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif num_teams < 250:
        gold_threshold = score_at_position(10)
        silver_threshold = score_at_position(max(1, int(num_teams * 0.2)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.4)))
    elif num_teams < 1000:
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(50)
        bronze_threshold = score_at_position(100)
    else:
        gold_threshold = score_at_position(10 + int(num_teams * 0.002))
        silver_threshold = score_at_position(max(1, int(num_teams * 0.05)))
        bronze_threshold = score_at_position(max(1, int(num_teams * 0.1)))

    score = float(score)
    if _leaderboard_lower_is_better(leaderboard):
        if score <= gold_threshold:
            return "Gold"
        if score <= silver_threshold:
            return "Silver"
        if score <= bronze_threshold:
            return "Bronze"
    else:
        if score >= gold_threshold:
            return "Gold"
        if score >= silver_threshold:
            return "Silver"
        if score >= bronze_threshold:
            return "Bronze"
    return "None"


def leaderboard_medal_binary_reward(score, metadata: dict) -> float:
    leaderboard = _load_public_leaderboard(metadata)
    if leaderboard is None or not _finite(score):
        return 0.0
    return 1.0 if _leaderboard_medal_for_score(float(score), leaderboard) != "None" else 0.0


def leaderboard_rank_reward(score, metadata: dict) -> float:
    leaderboard = _load_public_leaderboard(metadata)
    if leaderboard is None or not _finite(score):
        return 0.0

    scores = leaderboard["score"].astype(float).tolist()
    lower_better = _leaderboard_lower_is_better(leaderboard)
    signed_scores = [(-value if lower_better else value) for value in scores]
    signed_score = -float(score) if lower_better else float(score)

    num_teams = len(signed_scores)
    if num_teams <= 1:
        return 1.0

    better_count = sum(value > signed_score for value in signed_scores)
    reward = 1.0 - (better_count / (num_teams - 1))
    return max(0.0, min(1.0, float(reward)))


def _static_bound_limits_signed(meta: dict):
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

    if higher:
        best_cands = [tmax, lmax]
        worst_cands = [tmin, lmin]
    else:
        best_cands = [tmin, lmin]
        worst_cands = [tmax, lmax]

    best_cands = [x for x in best_cands if x is not None]
    worst_cands = [x for x in worst_cands if x is not None]

    best = max(best_cands) if best_cands else None
    worst = min(worst_cands) if worst_cands else None
    return best, worst


def get_bound_limits_signed(meta: dict):
    """Return one-sided signed-space metadata limits: (best_limit, worst_limit)."""
    return _static_bound_limits_signed(meta)


def _apply_max_range(best, worst, fallback_best=None, fallback_worst=None):
    if not (_finite(best) and _finite(worst)):
        return best, worst
    best = float(best)
    worst = float(worst)

    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        if _finite(fallback_best) and _finite(fallback_worst) and (float(fallback_best) - float(fallback_worst)) <= MAX_RANGE:
            best, worst = float(fallback_best), float(fallback_worst)
        else:
            worst = best - MAX_RANGE

    return best, worst


def _bounds_signed(meta: dict):
    static_best, static_worst = _static_bound_limits_signed(meta)

    dynamic_best = meta.get("dynamic_bound_best_signed")
    dynamic_worst = meta.get("dynamic_bound_worst_signed")
    has_dynamic = _finite(dynamic_best) and _finite(dynamic_worst)
    if has_dynamic:
        dynamic_best = float(dynamic_best)
        dynamic_worst = float(dynamic_worst)

        if _finite(static_best):
            dynamic_best = min(dynamic_best, float(static_best))
        if _finite(static_worst):
            dynamic_worst = max(dynamic_worst, float(static_worst))

        if dynamic_best > dynamic_worst:
            return _apply_max_range(dynamic_best, dynamic_worst, static_best, static_worst)

    if not (_finite(static_best) and _finite(static_worst)):
        return None, None

    return _apply_max_range(static_best, static_worst)


def _preferred_static_bounds_signed(
    meta: dict,
    priority: str = "leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
):
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

    priority = (priority or "leaderboard").strip().lower()

    if priority == "theoretical":
        primary_best, secondary_best = (tmax, lmax) if higher else (tmin, lmin)
        primary_worst, secondary_worst = (tmin, lmin) if higher else (tmax, lmax)
    else:
        primary_best, secondary_best = (lmax, tmax) if higher else (lmin, tmin)
        primary_worst, secondary_worst = (lmin, tmin) if higher else (lmax, tmax)

    best = primary_best if primary_best is not None else secondary_best
    worst = primary_worst if primary_worst is not None else secondary_worst

    if not (_finite(best) and _finite(worst)):
        if _finite(fallback_best) and _finite(fallback_worst):
            return _apply_max_range(float(fallback_best), float(fallback_worst))
        return None, None

    return _apply_max_range(best, worst)


def has_static_bounds_with_priority(
    metadata: dict,
    priority: str = "leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
) -> bool:
    best, worst = _preferred_static_bounds_signed(
        metadata,
        priority=priority,
        fallback_best=fallback_best,
        fallback_worst=fallback_worst,
    )
    return _finite(best) and _finite(worst)


def score2reward_with_static_priority(
    score,
    metadata,
    mode="logistic",
    priority="leaderboard",
    fallback_best: float | None = None,
    fallback_worst: float | None = None,
    power_alpha: float | None = None,
):
    """
    Map score to reward using static bounds with an explicit source priority.

    priority="leaderboard" means:
    - prefer leaderboard_min/max
    - fallback to theoretical_min/max
    """
    if not _finite(score):
        return 0.0

    if mode == "leaderboard_medal_binary":
        return leaderboard_medal_binary_reward(score, metadata)

    if mode == "leaderboard_rank":
        return leaderboard_rank_reward(score, metadata)

    score = float(score)
    signed = _signed(score, metadata)

    if mode == "power_clip":
        best, worst = _preferred_static_bounds_signed(
            metadata,
            priority=priority,
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
        if best is None:
            return 0.0

        rng = max(best - worst, 1e-9)
        s_prime = (signed - worst) / rng
        s_prime = 0.0 if s_prime < 0 else (1.0 if s_prime > 1 else s_prime)
        alpha = POWER_CLIP_ALPHA if power_alpha is None else max(float(power_alpha), 0.0)
        return s_prime**alpha

    if mode == "margin_tanh":
        best, worst = _preferred_static_bounds_signed(
            metadata,
            priority=priority,
            fallback_best=fallback_best,
            fallback_worst=fallback_worst,
        )
        if best is None:
            return 0.0

        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)
        m = (signed - center) / scale
        r = 0.5 + 0.5 * math.tanh(m / 0.50)
        return r**2

    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    if priority == "theoretical":
        first_min, second_min = theoretical_min, leaderboard_min
        first_max, second_max = theoretical_max, leaderboard_max
    else:
        first_min, second_min = leaderboard_min, theoretical_min
        first_max, second_max = leaderboard_max, theoretical_max

    if higher_is_better:
        score_range_best_score = first_max if first_max is not None else second_max
        score_range_worst_score = first_min if first_min is not None else second_min
    else:
        score_range_best_score = first_min if first_min is not None else second_min
        score_range_worst_score = first_max if first_max is not None else second_max

    if score_range_best_score is None or score_range_worst_score is None:
        return 0.0

    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        return 0.0

    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    return reward


def score2reward(score, metadata, mode="logistic", power_alpha: float | None = None):
    """
    Map a metric score to a reward.
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
        if abs(float(score)) < 1e-7:
            return -100.0

        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)

    if mode == "leaderboard_medal_binary":
        return leaderboard_medal_binary_reward(score, metadata)

    if mode == "leaderboard_rank":
        return leaderboard_rank_reward(score, metadata)

    s = _signed(score, metadata)

    # A: Power Clip on MinMax
    if mode == "power_sigmoid":
        raise NotImplementedError('mode="power_sigmoid" is not implemented; use mode="power_clip" instead.')

    if mode == "power_clip":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return _stable_sigmoid(s)

        rng = max(best - worst, 1e-9)
        s_prime = (s - worst) / rng
        s_prime = 0.0 if s_prime < 0 else (1.0 if s_prime > 1 else s_prime)
        alpha = POWER_CLIP_ALPHA if power_alpha is None else max(float(power_alpha), 0.0)
        return s_prime**alpha

    # B: Tanh margin shaping
    if mode == "margin_tanh":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)  # Fallback when no bounds are available.

        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        m = (s - center) / scale

        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y  # (0,1)

        gamma = 2.0
        return r**gamma

    # C: Online percentile
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
        score_range_best_score = theoretical_max if theoretical_max is not None else leaderboard_max

        # worst score
        score_range_worst_score = safe_max(leaderboard_min, theoretical_min, -100)

    else:
        # best score
        score_range_best_score = theoretical_min if theoretical_min is not None else leaderboard_min

        # worst score
        score_range_worst_score = safe_min(leaderboard_max, theoretical_max, 100)

    # Return a neutral reward when the score interval is still unknown.
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


def test_score2reward(base_dir: str):
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
            with open(metadata_path, "r") as f:
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
                rewards.append(float("nan"))

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


def test_hack_check(code_list: list[str], labels: list[str] = None):
    """
    Test the hack_check function on a list of code samples.

    Args:
        code_list: List of code strings to test
        labels: Optional list of descriptive labels for each code sample
    """
    if labels is None:
        labels = [f"Code {i+1}" for i in range(len(code_list))]

    if len(labels) != len(code_list):
        raise ValueError(f"Length of labels ({len(labels)}) must match length of code_list ({len(code_list)})")

    print("=" * 100)
    print("Testing hack_check function")
    print("=" * 100)

    results = []
    for i, (code, label) in enumerate(zip(code_list, labels)):
        print(f"\n[{i+1}/{len(code_list)}] Testing: {label}")
        print("-" * 100)

        is_valid, category, reason = hack_check(code)

        result = {
            "label": label,
            "is_valid": is_valid,
            "category": category,
            "reason": reason,
            "code_preview": code[:200] + "..." if len(code) > 200 else code,
        }
        results.append(result)

        status = "✅ VALID" if is_valid else "❌ INVALID (HACK DETECTED)"
        print(f"Status: {status}")
        print(f"Category: {category}")
        print(f"Reason: {reason}")
        print(f"Code preview: {result['code_preview']}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    valid_count = sum(1 for r in results if r["is_valid"])
    invalid_count = len(results) - valid_count
    print(f"Total: {len(results)}")
    print(f"Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")

    print("\nDetailed Results:")
    for i, r in enumerate(results):
        status = "✅" if r["is_valid"] else "❌"
        print(f"{status} {r['label']}: {r['reason']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate score-to-reward conversion on a leaderboard directory.")
    parser.add_argument("base_dir", help="Directory containing the task folders to inspect.")
    args = parser.parse_args()
    test_score2reward(args.base_dir)
