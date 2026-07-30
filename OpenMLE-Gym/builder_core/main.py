import os
import argparse
from pathlib import Path

from datetime import datetime

from openmle_gym.build import build_tasks

os.environ['OPENBLAS_NUM_THREADS'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="OpenMLE builder-core Kaggle task-package builder.")
    parser.add_argument("--slugs-file", required=True, help="Text file with one Kaggle competition slug per line.")
    parser.add_argument("--output-root", default="artifacts/builds", help="Directory where build batches are written.")
    parser.add_argument("--info-csv", default="info.csv", help="Meta Kaggle info CSV.")
    parser.add_argument("--batch-name", default=None, help="Optional batch name.")
    parser.add_argument("--max-concurrency", type=int, default=64, help="Maximum concurrent builds.")
    parser.add_argument("--retry", type=int, default=2, help="Retry count per task.")
    parser.add_argument("--include-mlebench", action="store_true", help="Do not skip known MLE-Bench slugs.")
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete raw/ after build and write raw.txt from forge/fileinfo.txt.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("process", "isolated"),
        default="process",
        help="Use process isolation, optionally sandboxing generated prepare code.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=3600,
        help="Maximum seconds allowed for one task worker.",
    )
    return parser.parse_args()


def summarize(tasks, output_root, batch_name):
    """
    Generate final summary of all processed competitions.

    Args:
        tasks: task result dictionaries.
        executor: NodeExecutor used by the legacy tool functions.
    """

    print("\n" + "="*60)
    print("FINAL SUMMARY PHASE")
    print("="*60)

    normalized = []
    for index, task in enumerate(tasks):
        if isinstance(task, BaseException):
            normalized.append(
                {
                    "name": f"unknown-task-{index}",
                    "success": False,
                    "errors": [f"{type(task).__name__}: {task}"],
                }
            )
        elif isinstance(task, dict):
            normalized.append(task)
        else:
            normalized.append(
                {
                    "name": f"unknown-task-{index}",
                    "success": False,
                    "errors": [f"Unexpected task result: {task!r}"],
                }
            )

    successful = [task for task in normalized if task.get("success") is True]
    failed = [task for task in normalized if task.get("success") is not True]

    summary_dir = Path(output_root) / batch_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.txt", "w") as file:
        file.write(f"END TIME: {datetime.now().strftime('%Y%m%d-%H%M%S')}\n\n")
        print(f"\nSuccessful competitions ({len(successful)}):")
        file.write(f"\nSuccessful competitions ({len(successful)}):\n")

        for task in successful:
            print(f"  - {task['name']}")
            file.write(f"  - {task['name']}\n")

        print(f"\nFailed competitions ({len(failed)}):")
        file.write(f"\nFailed competitions ({len(failed)}):\n")

        for task in failed:

            errors = ", ".join(task["errors"]) if task["errors"] else "Unknown error"

            print(f"  - {task['name']}: {errors}")
            file.write(f"  - {task['name']}: {errors}\n")
    return len(failed)
        
def main(args):
    batch_name = args.batch_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-openmle"
    batch = build_tasks(
        slugs_file=args.slugs_file,
        output_root=args.output_root,
        info_csv=args.info_csv,
        batch_name=batch_name,
        retry=args.retry,
        max_concurrency=args.max_concurrency,
        skip_mlebench=not args.include_mlebench,
        delete_raw=args.delete_raw,
        execute=True,
        execution_mode=args.execution_mode,
        task_timeout=args.task_timeout,
    )
    failed_count = summarize(batch["results"], args.output_root, batch_name)
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
