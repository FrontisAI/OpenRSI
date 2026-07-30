import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

if __package__:
    from .common import merge_task_values
else:
    from common import merge_task_values

def _load_module_from_path(module_name: str, file_path: Path) -> Any | None:
    """Dynamically load a Python module directly from its file path."""
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            print(f"Error: Could not create module spec for: {file_path}", file=sys.stderr)
            return None
        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules before execution for potential relative imports within the module.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {module_name} from {file_path}: {e}", file=sys.stderr)
        if module_name in sys.modules:
            del sys.modules[module_name]
        return None


def get_metric(competition_name: str, data_dir: str | Path) -> type | None:
    """Get the metric class for a specific competition."""
    internal_module_name = f"{competition_name.replace('-', '_')}.utils.metric"
    metric_dir = Path(data_dir).parent / "utils" / "metric.py"
    module = _load_module_from_path(internal_module_name, metric_dir)
    if module:
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and name.endswith("Metrics")
                and obj.__module__ == module.__name__
            ):
                return obj
        print(f"Warning: No suitable '*Metrics' class found in {internal_module_name}", file=sys.stderr)
    return None


def evaluate_submission(metric, submission_path: Path, data_path: Path) -> dict:
    submission_df = pd.read_csv(submission_path)
    data_df = pd.read_csv(data_path)
    score = metric.evaluate(y_true=data_df, y_pred=submission_df)
    return {"score": score}


def validate_sample_submission(task_dir: str | Path) -> tuple[Any, str | None]:
    task_path = Path(task_dir)
    file_path = task_path / "data" / "private" / "test_answer.csv"
    private_dir = task_path / "data" / "private"
    public_dir = task_path / "data" / "public"
    data_dir = task_path / "data"
    utils_dir = task_path / "utils"

    print(f"\nProcessing file: {file_path}")
    try:
        pd.read_csv(private_dir / "test_answer.csv")
    except Exception as e:
        print(f"Error reading test_answer.csv: {e}")
        return e, f"Error reading test_answer.csv: {e}"

    metric = get_metric(
        competition_name=task_path.name,
        data_dir=data_dir,
    )
    try:
        if metric is None:
            raise ImportError("Metric class could not be loaded.")
    except ImportError as e:
        print(f"Error: {e}")
        return e, str(e)

    try:
        eval_result = evaluate_submission(
            metric=metric(),
            submission_path=public_dir / "sample_submission.csv",
            data_path=private_dir / "test_answer.csv",
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return e, str(e)

    if eval_result["score"] is None:
        print("Error: Evaluation returned None score.")
        return "Evaluation returned None score.", "Evaluation returned None score."

    report = f"@Evaluation result: {eval_result}"
    print(f"Evaluation result: {eval_result}")
    return report, None


def validate_sample_submissions(folders: list | None = None, csv_name: str | None = None) -> None:
    error_dirs = {}
    report = []

    for folder in tqdm(folders or [], desc="Processing files"):
        result, error = validate_sample_submission(folder)
        report.append(result)
        if error is not None:
            error_dirs[str(Path(folder) / "utils")] = error

    csv = pd.read_csv(csv_name)
    csv = merge_task_values(csv, folders or [], {"Metric": report})
    csv.to_csv(csv_name, index=False)

    if error_dirs:
        print("\nDirectories with errors during metric loading or evaluation:")
        for dir_path, error_msg in error_dirs.items():
            print(f"{dir_path}: {error_msg}")
    else:
        print("\nAll directories processed successfully.")
