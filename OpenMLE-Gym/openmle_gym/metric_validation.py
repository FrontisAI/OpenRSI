from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .common import json_safe, validate_task_name
from .process_runner import run_task_process


def _load_module(module_name: str, file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"Metric file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_metric_class(task_dir: str | Path) -> type:
    task_path = Path(task_dir)
    metric_path = task_path / "utils" / "metric.py"
    module_name = f"openmle_metric_{task_path.name.replace('-', '_')}"
    module = _load_module(module_name, metric_path)
    for _, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and obj.__name__.endswith("Metrics")
            and obj.__module__ == module.__name__
        ):
            return obj
    raise LookupError(f"No '*Metrics' class found in {metric_path}")


def evaluate_sample_submission(task_dir: str | Path) -> dict[str, Any]:
    task_path = Path(task_dir)
    public_dir = task_path / "data" / "public"
    private_dir = task_path / "data" / "private"
    submission_path = public_dir / "sample_submission.csv"
    answer_path = private_dir / "test_answer.csv"

    if not submission_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {submission_path}")
    if not answer_path.is_file():
        raise FileNotFoundError(f"Private answer file not found: {answer_path}")

    metric_cls = load_metric_class(task_path)
    metric = metric_cls()
    y_pred = pd.read_csv(submission_path)
    y_true = pd.read_csv(answer_path)
    if hasattr(metric, "validate_submission"):
        metric.validate_submission(y_pred, y_true)
    score = json_safe(metric.evaluate(y_true=y_true, y_pred=y_pred))
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        raise ValueError(
            "Metric evaluate() must return a finite numeric score, "
            f"got {score!r}"
        )
    return {
        "task_dir": str(task_path),
        "metric_class": metric_cls.__name__,
        "score": score,
        "submission_rows": len(y_pred),
        "answer_rows": len(y_true),
    }


def evaluate_sample_submission_process(
    task_dir: str | Path,
    *,
    execution_mode: str = "process",
    task_timeout: float = 120,
) -> dict[str, Any]:
    task_path = Path(task_dir).resolve()
    validate_task_name(task_path.name)
    outcome = run_task_process(
        "metric",
        {"task_dir": str(task_path)},
        timeout=task_timeout,
        execution_mode=execution_mode,
        readonly_paths=(task_path,),
    )
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)
    if (
        not outcome.ok
        or not isinstance(outcome.result, dict)
        or "score" not in outcome.result
    ):
        raise RuntimeError(outcome.error or "Metric task worker returned an invalid result")
    return outcome.result
