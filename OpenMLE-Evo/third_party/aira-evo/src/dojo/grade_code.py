import json
import sys
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
from dojo.config_dataclasses.interpreter.jupyter import JupyterInterpreterConfig
from dojo.config_dataclasses.task import TASK_MAP
from dojo.config_dataclasses.task.mlebench import MLEBenchTaskConfig
from dojo.utils.config import build
from dojo.utils.environment import get_mlebench_data_dir

SUPERIMAGE_VERSION = "2025-05-02v2"


def execute_code(code_file_path: str, task_name: str, timeout_hours: float = 4.0) -> Dict[str, Any]:
    code_path = Path(code_file_path)
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found: {code_file_path}")

    code_content = code_path.read_text()
    cache_dir = get_mlebench_data_dir()
    tmp_dir = tempfile.mkdtemp(prefix="rad_results_")

    task_config = MLEBenchTaskConfig(
        name=str(task_name),
        benchmark="mlebench",
        cache_dir=str(cache_dir),
        results_output_dir=str(tmp_dir),
        public_dir=str(f"{cache_dir}/{task_name}/prepared/public"),
        private_dir=str(f"{cache_dir}/{task_name}/prepared/private"),
        data_dir=str(f"{cache_dir}/{task_name}/prepared/public/"),
        submission_fname="submission.csv",
    )

    interpreter_config = JupyterInterpreterConfig(
        superimage_version=SUPERIMAGE_VERSION, timeout=int(timeout_hours * 60 * 60), working_dir=tmp_dir
    )

    try:
        task = build(task_config, TASK_MAP)
        solver_interpreter = build(interpreter_config, INTERPRETER_MAP, data_dir=task_config.data_dir)

        state, _ = task.prepare(solver_interpreter=solver_interpreter, eval_interpreter=None)

        _, eval_result = task.step_task(state, code_content)

        from dojo.core.tasks.constants import EXECUTION_OUTPUT, TEST_FITNESS, VALID_SOLUTION, VALIDATION_FITNESS

        exec_output = eval_result.get(EXECUTION_OUTPUT)

        summary = {
            "success": False,
            "exit_code": None,
            "timed_out": False,
            "exec_time": None,
            "valid_solution": eval_result.get(VALID_SOLUTION, False),
            "validation_fitness": eval_result.get(VALIDATION_FITNESS),
            "test_fitness": eval_result.get(TEST_FITNESS),
            "error_output": [],
            "stdout": [],
        }

        if exec_output is not None:
            summary["success"] = exec_output.exit_code == 0 and not exec_output.timed_out
            summary["exit_code"] = exec_output.exit_code
            summary["timed_out"] = exec_output.timed_out
            summary["exec_time"] = exec_output.exec_time
            summary["error_output"] = exec_output.term_err if hasattr(exec_output, "term_err") else []
            summary["stdout"] = exec_output.term_out if hasattr(exec_output, "term_out") else []

        grading_report_path = Path(tmp_dir) / "grading_report.json"
        if grading_report_path.exists():
            with open(grading_report_path, "r") as f:
                grading_report = json.load(f)
            summary["grading_report"] = grading_report

        return summary

    except Exception as e:
        return {"error": str(e), "success": False}

    finally:
        if "task" in locals() and "state" in locals():
            task.close(state)
        if tmp_dir and Path(tmp_dir).exists():
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temporary directory {tmp_dir}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Grade code submissions for MLEBench")
    parser.add_argument("code_file_path", type=str, help="Path to the Python code file to grade")
    parser.add_argument("task_name", type=str, help="Name of the MLEBench task")
    parser.add_argument("output_path", type=str, help="Path where the grading results will be saved")
    parser.add_argument(
        "--timeout", type=float, default=4.0, help="Timeout in hours for code execution (default: %(default)s hours)"
    )

    args = parser.parse_args()

    # Validate inputs before starting
    if not Path(args.code_file_path).is_file():
        parser.error(f"Code file not found: {args.code_file_path}")

    try:
        cache_dir = get_mlebench_data_dir()
        if not cache_dir:
            parser.error("MLEBench data directory not found. Please ensure MLEBench is properly configured.")

        task_dir = Path(cache_dir) / args.task_name / "prepared"
        if not task_dir.exists():
            parser.error(
                f"Task '{args.task_name}' not found in MLEBench data directory.\n"
                f"Expected to find: {task_dir}\n"
                f"Available tasks: {', '.join(d.name for d in Path(cache_dir).iterdir() if (d / 'prepared').exists())}"
            )
    except Exception as e:
        parser.error(f"Error accessing MLEBench: {e}")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Grading {args.code_file_path} for task {args.task_name}...")
    result = execute_code(args.code_file_path, args.task_name, args.timeout)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    if result.get("error"):
        print(f"\nError during grading: {result['error']}")

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
