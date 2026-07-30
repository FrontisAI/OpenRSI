from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build import build_tasks
from .leaderboard import download_leaderboards
from .local_evaluator import evaluation_has_task_failures, evaluate_tasks
from .metric_validation import evaluate_sample_submission_process
from .overview import generate_overview, overview_has_task_failures


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _add_execute_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run external-service or account-mutating operations. Default is dry-run.",
    )


def _add_partial_success_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-partial-success",
        action="store_true",
        help="Return exit code 0 even when one or more batch items failed.",
    )


def _add_execution_flags(parser: argparse.ArgumentParser, timeout: float) -> None:
    parser.add_argument(
        "--execution-mode",
        choices=("process", "isolated"),
        default="process",
        help="Run each task in a child process, or in a configured isolated container.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=timeout,
        help="Maximum seconds allowed for one task worker.",
    )


def _has_batch_failures(data: Any) -> bool:
    if isinstance(data, dict):
        summary = data.get("summary")
        if isinstance(summary, dict) and int(summary.get("failed") or 0) > 0:
            return True
        return (
            data.get("success") is False
            or data.get("status") == "failed"
            or _has_batch_failures(data.get("results", []))
        )
    if isinstance(data, list):
        return any(_has_batch_failures(item) for item in data)
    return False


def _print_batch_json(data: Any, allow_partial_success: bool) -> int:
    _print_json(data)
    if not allow_partial_success and _has_batch_failures(data):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openmle-task",
        description="Build, inspect, and evaluate standardized Kaggle-style ML task packages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build task packages from Kaggle competition slugs.")
    build.add_argument("--slugs-file", required=True, help="Text file with one Kaggle slug or URL per line.")
    build.add_argument("--output-root", default="artifacts/builds", help="Directory where build batches are written.")
    build.add_argument("--info-csv", default="builder_core/info.csv", help="Meta Kaggle info CSV used by the legacy builder.")
    build.add_argument("--batch-name", default=None, help="Optional run/batch name.")
    build.add_argument("--retry", type=int, default=2, help="Per-competition retry count.")
    build.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent competition builds.")
    build.add_argument("--limit", type=int, default=None, help="Optional maximum number of slugs to process.")
    build.add_argument("--include-mlebench", action="store_true", help="Do not skip known MLE-Bench slugs.")
    build.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete each task raw/ directory after build and write raw.txt from forge/fileinfo.txt.",
    )
    _add_execute_flag(build)
    _add_partial_success_flag(build)
    _add_execution_flags(build, 3600)

    overview = subparsers.add_parser("overview", help="Generate local task package overview CSV.")
    overview.add_argument("--tasks-root", required=True, help="Directory containing task packages.")
    overview.add_argument("--output-csv", required=True, help="CSV path to write.")
    overview.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent chunk workers for package inspection and metadata stages.",
    )
    _add_execution_flags(overview, 600)
    overview.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-only category and CPU/GPU classification.",
    )

    metric = subparsers.add_parser("metric-check", help="Evaluate sample_submission.csv with utils/metric.py.")
    metric.add_argument("--task-dir", required=True, help="Single task package directory.")
    _add_execution_flags(metric, 120)

    evaluator = subparsers.add_parser("evaluate", help="Run task-structure and metadata pipeline evaluation.")
    evaluator.add_argument("--root-dir", required=True, help="Root directory containing task packages.")
    evaluator.add_argument("--task-list", required=True, help="Text file with task directory names.")
    evaluator.add_argument("--output-dir", required=True, help="Directory for JSON and CSV results.")
    evaluator.add_argument("--overview-csv", default=None, help="Optional existing overview CSV to reuse.")
    evaluator.add_argument("--workers", type=int, default=1, help="Number of concurrent chunk workers.")
    evaluator.add_argument(
        "--skip-llm",
        action="store_true",
        help=(
            "Skip LLM metadata and quality evaluation; emit deterministic "
            "validation without quality scores."
        ),
    )
    evaluator.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Run deterministic structure and metric checks without producing "
            "quality scores (compatibility alias for --skip-llm)."
        ),
    )
    _add_execution_flags(evaluator, 600)

    leaderboard = subparsers.add_parser("leaderboard", help="Download Kaggle public leaderboards.")
    leaderboard.add_argument("--slugs-file", required=True, help="Text file with one Kaggle slug or URL per line.")
    leaderboard.add_argument("--out-dir", default="artifacts/leaderboards", help="Output directory for CSV files.")
    leaderboard.add_argument("--kaggle-json", default=None, help="Optional Kaggle JSON credential file.")
    leaderboard.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    leaderboard.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between requests.")
    _add_execute_flag(leaderboard)
    _add_partial_success_flag(leaderboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        return _print_batch_json(
            build_tasks(
                slugs_file=args.slugs_file,
                output_root=args.output_root,
                info_csv=args.info_csv,
                batch_name=args.batch_name,
                retry=args.retry,
                max_concurrency=args.max_concurrency,
                limit=args.limit,
                skip_mlebench=not args.include_mlebench,
                delete_raw=args.delete_raw,
                execute=args.execute,
                execution_mode=args.execution_mode,
                task_timeout=args.task_timeout,
            ),
            allow_partial_success=args.allow_partial_success,
        )

    if args.command == "overview":
        output = generate_overview(
            tasks_root=args.tasks_root,
            output_csv=args.output_csv,
            workers=args.workers,
            skip_llm=args.skip_llm,
            execution_mode=args.execution_mode,
            task_timeout=args.task_timeout,
        )
        has_failures = overview_has_task_failures(output)
        _print_json(
            {
                "status": "partial_success" if has_failures else "ok",
                "output_csv": str(output),
            }
        )
        return 1 if has_failures else 0

    if args.command == "metric-check":
        _print_json(
            evaluate_sample_submission_process(
                args.task_dir,
                execution_mode=args.execution_mode,
                task_timeout=args.task_timeout,
            )
        )
        return 0

    if args.command == "evaluate":
        paths = evaluate_tasks(
            root_dir=args.root_dir,
            task_list=args.task_list,
            output_dir=args.output_dir,
            local_only=args.local_only,
            skip_llm=args.skip_llm,
            overview_csv=args.overview_csv,
            workers=args.workers,
            execution_mode=args.execution_mode,
            task_timeout=args.task_timeout,
        )
        has_failures = evaluation_has_task_failures(paths)
        _print_json(
            {
                "status": "partial_success" if has_failures else "ok",
                **{key: str(path) for key, path in paths.items()},
            }
        )
        return 1 if has_failures else 0

    if args.command == "leaderboard":
        return _print_batch_json(
            download_leaderboards(
                slugs_file=args.slugs_file,
                out_dir=args.out_dir,
                kaggle_json=args.kaggle_json,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                execute=args.execute,
            ),
            allow_partial_success=args.allow_partial_success,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
