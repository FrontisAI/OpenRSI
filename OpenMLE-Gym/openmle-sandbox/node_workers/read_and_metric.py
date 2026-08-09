# This script should be deployed to /mnt/pubdatasets2/mlsandbox/workdir/evaluation/.
import argparse
import importlib.util
import inspect
import os
import sys
from typing import Type

import pandas as pd


def load_metric_module(data_dir: str):
    """
    从指定的 data_dir 动态加载 utils/metric.py 模块。
    """
    metric_file_path = os.path.join(data_dir, "utils", "metric.py")
    if not os.path.exists(metric_file_path):
        print(f"Error: Metric file not found at: {metric_file_path}")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("utils.metric", metric_file_path)
        metric_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None  # 为类型检查提供保障
        spec.loader.exec_module(metric_module)
        return metric_module
    except Exception as exc:
        print(f"Error: Failed to load metric module from {metric_file_path}")
        print(f"   Details: {exc}")
        sys.exit(1)


def find_metrics_class(metric_module) -> Type:
    """
    自动寻找同时实现 evaluate 与 validate_submission 的评分类。
    """
    candidates = []
    for name, obj in vars(metric_module).items():
        if not inspect.isclass(obj):
            continue
        if getattr(obj, "__module__", None) != metric_module.__name__:
            continue
        if all(callable(getattr(obj, attr, None)) for attr in ("evaluate", "validate_submission")):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError("未能在 metric.py 中找到满足要求的评分类。")

    candidates.sort(key=lambda cls: (not cls.__name__.lower().endswith("metrics"), cls.__name__))
    return candidates[0]


def find_invalid_submission_error(metric_module) -> Type[Exception]:
    """
    获取 metric 模块中的 InvalidSubmissionError，若不存在则提供默认实现。
    """
    candidate = getattr(metric_module, "InvalidSubmissionError", None)
    if inspect.isclass(candidate):
        return candidate

    class DefaultInvalidSubmissionError(Exception):
        """兜底的提交验证异常类型。"""

    return DefaultInvalidSubmissionError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and score a submission file using dataset-specific metrics."
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir",
        required=True,
        help="Path to the data directory containing 'utils/metric.py'.",
    )
    parser.add_argument(
        "-a",
        "--answer",
        dest="answer_path",
        required=True,
        help="Path to the ground-truth CSV file.",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        dest="prediction_path",
        required=True,
        help="Path to the user's submission CSV file.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    import time
    time.sleep(3)

    print(f"Loading metrics from: {os.path.join(args.data_dir, 'utils', 'metric.py')}")
    metric_module = load_metric_module(args.data_dir)

    try:
        MetricsClass = find_metrics_class(metric_module)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    InvalidSubmissionError = find_invalid_submission_error(metric_module)
    metrics_calculator = MetricsClass()
    print(f"--- {MetricsClass.__name__} Submission Scorer ---")

    try:
        # print(f"📖 Reading answer file: {args.answer_path}")
        ground_truth_df = pd.read_csv(args.answer_path)

        # print(f"📖 Reading prediction file: {args.prediction_path}")
        print(f"Reading prediction file...")
        submission_df = pd.read_csv(args.prediction_path)

        print("\nValidating submission format...")
        validation_message = metrics_calculator.validate_submission(
            submission=submission_df, ground_truth=ground_truth_df
        )
        if validation_message:
            print(f"{validation_message}")

        print("Calculating score...")
        score = metrics_calculator.evaluate(y_true=ground_truth_df, y_pred=submission_df)

        print("\n---------------------------------")
        print(f"Final Score: {score}")
        print("---------------------------------")
        # 输出特定格式以便任务调度器提取分数
        print(f"prefix ##SCORE##{score} suffix")
    except FileNotFoundError as exc:
        print("\nError: File not found.")
        print(f"   Details: {exc}")
        sys.exit(1)
    except InvalidSubmissionError as exc:
        print("\nError: Submission validation failed.")
        print(f"   Reason: {exc}")
        sys.exit(1)
    except Exception as exc:
        print("\nAn unexpected error occurred.")
        print(f"   Details: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
