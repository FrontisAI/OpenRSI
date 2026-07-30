# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import os
import queue
import signal
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import humanize
from omegaconf import OmegaConf

from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.utils.logger import CollectiveLogger, LogEvent, get_logger
from dojo.core.interpreters.utils import copy_contents

from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig

import logging

log = logging.getLogger(__name__)


def exception_summary(
    e: BaseException,
    working_dir: Path,
    exec_file_name: str,
    format_tb_ipython: bool,
) -> tuple[str, str, dict, list[tuple]]:
    """Generates a string that summarizes an exception and its stack trace."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join([l for l in tb_lines if "dojo/" not in l and "importlib" not in l])
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    """A file-like object that writes to a multiprocessing Queue."""

    def __init__(self, queue: Queue, timeout: float = 5) -> None:
        self.queue = queue
        self.timeout = timeout

    def write(self, msg: str) -> None:
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logging.warning("RedirectQueue write timed out")

    def flush(self) -> None:
        pass


class PythonInterpreter(Interpreter):
    local = True
    factory = False

    def __init__(
        self,
        cfg: PythonInterpreterConfig,
        data_dir: Path | None = None,
    ) -> None:
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): Working directory of the agent
            timeout (int, optional): Timeout for each code execution step in seconds. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
        """
        self.logger = get_logger()
        self.working_dir = Path(cfg.working_dir).resolve()

        if not self.working_dir.exists():
            log.info(f"Working directory {self.working_dir} does not exist -- creating it")
            self.working_dir.mkdir(parents=True, exist_ok=True)

        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
            # assume the data_dir is already created and ready
            data_link = self.working_dir / "data"

            if cfg.use_symlinks:
                if data_link.is_symlink():
                    data_link.unlink()
                data_link.symlink_to(self.data_dir)
            else:
                copy_contents(source=self.data_dir, destination=data_link, use_symlinks=False)
        else:
            self.data_dir = self.working_dir / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            log.info("WARNING: A data directory was not passed, so it was created at {str(self.data_dir)}")

        self.timeout = cfg.timeout
        self.format_tb_ipython = cfg.format_tb_ipython
        self.process: Process | None = None  # type: ignore

    def child_proc_setup(self, result_outq: Queue) -> None:
        """
        Pre-execution setup in the child process:
        - Changes directory
        - Disables warnings
        - Redirects stdout/stderr to a queue
        """
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        # this helps python find modules in the current working dir
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self,
        code_inq: Queue,
        result_outq: Queue,
        event_outq: Queue,
    ) -> None:
        """
        Main loop running in the child process.
        Waits for (code, file_name, persist_file, execute_code) to arrive in `code_inq`,
        writes it to `file_name`, executes it, and reports results back.
        """
        self.child_proc_setup(result_outq)

        global_scope: dict[str, Any] = {}
        while True:
            # Here, we expect a tuple: (code, file_name, persist_file).
            data = code_inq.get()
            # If you had a sentinel-based exit, you could do:
            # if data == "quit":
            #     break

            code, agent_file_name, persist_file, execute_code = data
            os.chdir(str(self.working_dir))

            # Write code to the chosen file name
            with open(agent_file_name, "w") as f:
                f.write(code)

            if execute_code:
                event_outq.put(("state:ready",))
                try:
                    exec(compile(code, agent_file_name, "exec"), global_scope)
                    # We retrieve the user-supplied result (if any) from __result__
                    return_value = global_scope.get("__result__", None)

                    event_outq.put(("state:finished", None, None, None, return_value))
                except BaseException as e:
                    tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                        e,
                        self.working_dir,
                        agent_file_name,
                        self.format_tb_ipython,
                    )
                    result_outq.put(tb_str)
                    if e_cls_name == "KeyboardInterrupt":
                        e_cls_name = "TimeoutError"

                    # In case of an exception, we pass None for return_value
                    event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack, None))
            else:
                event_outq.put(("state:ready",))
                event_outq.put(("state:finished", None, None, None, None))

            # Remove the file only if it is NOT meant to persist
            if not persist_file:
                os.remove(agent_file_name)

            # put EOF marker to indicate that we're done capturing output
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        """
        Spawns the child process that will run Python code in an isolated environment.
        """
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        self.code_inq = Queue()
        self.result_outq = Queue()
        self.event_outq = Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self) -> None:
        """
        Terminate the child process if it's still running, with escalation (terminate -> kill -> sigkill).
        """
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.join(timeout=2)

            if self.process.exitcode is None:
                self.logger.warning("Process failed to terminate, killing immediately", LogEvent.INTERPRETER)
                self.process.kill()
                self.process.join()

                if self.process.exitcode is None:
                    self.logger.error("Process refuses to die, using SIGKILL", LogEvent.INTERPRETER)
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception as e:
            self.logger.error(f"Error during process cleanup: {e}", LogEvent.INTERPRETER)
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def fetch_file(
        self,
        path: str,
    ) -> str | None:
        if not Path(path).exists():
            return None
        return path

    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
        include_exec_time: bool = True,
    ) -> ExecutionResult:
        """
        Execute the provided Python code in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session
                before executing the code. Defaults to True.
            persist_file (bool, optional): If True, do not delete the file after execution.
                Defaults to False.
            file_name (str, optional): Name of the file into which the code is written
                before execution. Defaults to "runfile.py".
            execute_code: (bool): Whether to execute the code or not. If False, this
                function simply writes the code to a file.

        Returns:
            ExecutionResult: Object containing the output, metadata, and any exception info.
        """

        self.logger.debug(f"REPL is executing code (reset_session={reset_session})", LogEvent.INTERPRETER)
        log.info(f"Executing code:\n```\npython\n{code}\n```")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session must be True on first exec
            assert self.process is not None, "Process not started. Try reset_session=True first."

        assert self.process.is_alive()

        # Send the tuple (code, file_name, persist_file, execute_code) to the child
        self.code_inq.put((code, file_name, persist_file, execute_code))

        # wait for child to actually start execution
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            self.logger.critical(msg, LogEvent.INTERPRETER)
            queue_dump = ""
            while not self.result_outq.empty():
                queue_dump = self.result_outq.get()
                self.logger.error(f"REPL output queue dump: {queue_dump[:1000]}", LogEvent.INTERPRETER)
            self.cleanup_session()
            return ExecutionResult(term_out=[msg, queue_dump], exec_time=0, exit_code=1)
        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False  # indicates if we've exceeded time limit

        while True:
            try:
                # wait for state:finished from child
                state = self.event_outq.get(timeout=1)
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # no message yet, check if child is alive
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    self.logger.critical(msg, LogEvent.INTERPRETER)
                    queue_dump = ""
                    while not self.result_outq.empty():
                        queue_dump = self.result_outq.get()
                        self.logger.error(f"REPL output queue dump: {queue_dump[:1000]}", LogEvent.INTERPRETER)
                    self.cleanup_session()
                    return ExecutionResult(term_out=[msg, queue_dump], exec_time=0, exit_code=1)

                # child is alive, check timeout
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    self.logger.warning(f"Execution exceeded timeout of {self.timeout}s", LogEvent.INTERPRETER)
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    if running_time > self.timeout + 60:
                        self.logger.warning("Child failed to terminate, killing it..", LogEvent.INTERPRETER)
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [], None)
                        exec_time = self.timeout
                        break

        # we now unpack a 5-tuple from the child
        # state: ("state:finished", exc_type, exc_info, exc_stack, eval_return)
        e_cls_name, exc_info, exc_stack, eval_return = state[1:]

        # collect all output lines from child up to the EOF marker
        output: list[str] = []
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                if time.time() - start_collect > 10:
                    self.logger.warning("Output collection timed out", LogEvent.INTERPRETER)
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        # remove the EOF marker if present
        if output and output[-1] == "<|EOF|>":
            output.pop()

        # if we timed out, show that in the output
        if e_cls_name == "TimeoutError":
            output.append(f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}")
        elif include_exec_time:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        log.info(f"Code execution finished.")

        return ExecutionResult(
            term_out=output,
            exec_time=exec_time,
            exit_code=0 if e_cls_name is None else 1,
            eval_return=eval_return,
        )


def main():
    # Instantiate our dummy logger.
    cfg = {}
    cfg["logger"] = logger_config = {
        "base_exp_path": "results",  # Base path for logging.
        "use_console": True,  # Whether to log to stdout.
        "use_wandb": False,  # Whether to log to wandb.ai."
    }
    cfg = OmegaConf.create(cfg)
    logger = CollectiveLogger(cfg)

    # Use the current working directory (must exist).
    working_dir = Path(".").resolve()

    # Create an instance of PythonInterpreter with a 10-second timeout.
    interpreter = PythonInterpreter(logger, working_dir, timeout=10, format_tb_ipython=False)

    # Define a simple Python code snippet to execute.
    code_snippet = """
print("Hello from PythonInterpreter!")
for i in range(5):
    print("Count:", i)
__result__ = "Execution Completed Successfully"
"""

    # Execute the code.
    result = interpreter.run(
        code=code_snippet,
        reset_session=True,
        persist_file=False,
        file_name="demo_run.py",
        execute_code=True,
    )

    # Print the captured terminal output and other execution details.
    print("=== Terminal Output ===")
    for line in result.term_out:
        print(line)
    print("\nExecution Time:", result.exec_time)
    print("Return Value:", result.eval_return)

    # Clean up the interpreter process.
    interpreter.cleanup_session()


if __name__ == "__main__":
    main()
