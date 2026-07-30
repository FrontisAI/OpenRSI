from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .common import atomic_write_json


def _dispatch(operation: str, payload: dict[str, Any]) -> Any:
    if operation == "build":
        import asyncio

        from .build import _run_one_local

        return asyncio.run(_run_one_local(**payload))
    if operation == "overview":
        from .overview import _analyze_task_full

        return _analyze_task_full(
            Path(payload["task_dir"]),
            skip_llm=bool(payload.get("skip_llm", False)),
            metric_execution_mode=payload.get("metric_execution_mode", "process"),
            metric_timeout=float(payload.get("metric_timeout", 120)),
        )
    if operation == "evaluate":
        from .local_evaluator import _evaluate_task

        return _evaluate_task(**payload)
    if operation == "metric":
        from .metric_validation import evaluate_sample_submission

        return evaluate_sample_submission(payload["task_dir"])
    if operation == "prepare":
        from .sandbox_exec import execute_prepare

        return execute_prepare(**payload)
    raise ValueError(f"Unsupported task operation: {operation}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(
            "usage: python -m openmle_gym.task_worker OPERATION REQUEST_JSON RESULT_JSON",
            file=sys.stderr,
        )
        return 2
    operation, request_name, result_name = args
    request_path = Path(request_name)
    result_path = Path(result_name)
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Task request must be a JSON object")
        result = _dispatch(operation, payload)
        atomic_write_json(result_path, {"ok": True, "result": result})
        return 0
    except BaseException as exc:
        try:
            atomic_write_json(
                result_path,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except BaseException:
            pass
        print(f"Task worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
