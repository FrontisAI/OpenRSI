# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from pathlib import Path

import humanize

from dojo.config_dataclasses.interpreter.jupyter import JupyterInterpreterConfig
from dojo.core.interpreters.base import ExecutionResult, Interpreter

from .apptainer_jupyter_server import ApptainerJupyterServer
from .jupyter_code_executor import JupyterCodeExecutor

log = logging.getLogger(__name__)


class JupyterInterpreter(Interpreter):
    local = False
    factory = False

    def __init__(
        self,
        cfg: JupyterInterpreterConfig,
        data_dir: Path | None = None,
    ) -> None:
        self.timeout = cfg.timeout
        self.strip_ansi = cfg.strip_ansi
        self.working_dir = Path(cfg.working_dir).resolve()
        # make sure the working directory ends with a slash
        self.superimage_directory = os.path.join(cfg.superimage_directory, "")
        self.superimage_version = cfg.superimage_version
        self.read_only_overlays = cfg.read_only_overlays or []
        self.read_only_binds = cfg.read_only_binds or {}
        self.env = cfg.env or {}

        if not self.working_dir.exists():
            log.info(f"Working directory {self.working_dir} does not exist -- creating it")
            self.working_dir.mkdir(parents=True, exist_ok=True)

        if data_dir is not None:
            # assume the data_dir is ready to be binded to the container
            self.data_dir = Path(data_dir).resolve()
        else:
            # otherwise create it within the local folder of the run mimicing the working directory of the agent
            self.data_dir = self.working_dir / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.jupyter_server = ApptainerJupyterServer(
            bind_inputs_dir=self.data_dir,
            superimage_directory=self.superimage_directory,
            superimage_version=self.superimage_version,
            read_only_overlays=self.read_only_overlays,
            read_only_binds=self.read_only_binds,
            env=self.env,
        )
        self.code_executor = None

    def create_process(self) -> None:
        self.cleanup_session()
        self.code_executor = JupyterCodeExecutor(self.jupyter_server, timeout=self.timeout)

    def fetch_file(
        self,
        path: str,
    ) -> str | None:
        # first we find what would be the relative path to the file
        # from the working directory
        ppath = Path(path).resolve()
        relative_path = ppath.relative_to(self.working_dir)
        try:
            log.info(f"Fetching the file {ppath}")
            assert isinstance(self.code_executor, JupyterCodeExecutor)
            contents = self.code_executor.fetch_file(relative_path)
            log.info("File successfully fetched.")
        except FileNotFoundError:
            log.warning(f"File {ppath} not found in the working directory.")
            return None
        except TimeoutError:
            log.warning(f"Kernel timed out while fetching the file.")
            return None

        with open(ppath, "wb") as f:
            f.write(contents)

        return ppath.as_posix()

    def cleanup_line(self, line: str) -> str:
        if not self.strip_ansi:
            return line
        return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", line)

    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
        include_exec_time: bool = True,
    ) -> ExecutionResult:
        if reset_session or self.code_executor is None:
            self.create_process()

        assert isinstance(self.code_executor, JupyterCodeExecutor)
        if persist_file:
            # use Jupyter's magic command to write the code to a file
            results = self.code_executor.execute_code(f"%%writefile {file_name}\n{code}")

        if not execute_code:
            return results

        log.info(f"Executing code:\n```\npython\n{code}\n```")
        results = self.code_executor.execute_code(code)
        log.info(f"Code execution finished.")

        outputs = [self.cleanup_line(line) for line in results.term_out.copy()]
        # if we timed out, show that in the output
        if results.timed_out:
            outputs.append(f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}")
        elif include_exec_time:
            outputs.append(
                f"Execution time: {humanize.naturaldelta(results.exec_time)} (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        return ExecutionResult(
            term_out=outputs,
            exit_code=results.exit_code,
            exec_time=results.exec_time,
            eval_return=results.eval_return,
            timed_out=results.timed_out,
        )

    def cleanup_session(self) -> None:
        if self.code_executor is None:
            return
        code_executor = self.code_executor
        self.code_executor = None
        code_executor.stop()

    def close(self):
        self.cleanup_session()
        self.jupyter_server.stop()


class JupyterInterpreterFactory(Interpreter):
    local = False
    factory = True

    def __init__(
        self,
        cfg: JupyterInterpreterConfig,
        data_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg

        self.working_dir = cfg.working_dir
        if data_dir is not None:
            # assume the data_dir is ready to be binded to the container
            self.data_dir = Path(data_dir).resolve()
        else:
            # otherwise create it within the local folder of the run mimicing the working directory of the agent
            self.data_dir = Path(self.working_dir) / "data"
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = cfg.timeout
        self.strip_ansi = cfg.strip_ansi
        self.suerimage_directory = cfg.superimage_directory
        self.superimage_version = cfg.superimage_version
        self.read_only_overlays = cfg.read_only_overlays
        self.read_only_binds = cfg.read_only_binds
        self.env = cfg.env

        self._instance = None

    def reset_session(self):
        instance = self._instance
        self._instance = None
        if instance is None:
            return
        instance.close()

    @property
    def instance(self):
        if self._instance is None:
            self._instance = JupyterInterpreter(self.cfg, data_dir=self.data_dir)
        return self._instance

    def fetch_file(
        self,
        path: str,
    ) -> str | None:
        return self.instance.fetch_file(path)

    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
        include_exec_time: bool = True,
    ) -> ExecutionResult:
        if reset_session:
            self.reset_session()

        return self.instance.run(
            code,
            reset_session=False,
            persist_file=persist_file,
            file_name=file_name,
            execute_code=execute_code,
            include_exec_time=include_exec_time,
        )

    def cleanup_session(self) -> None:
        self.reset_session()

    def close(self):
        self.reset_session()
