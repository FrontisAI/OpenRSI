# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataclasses_json import DataClassJsonMixin


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exit_code: int | None = None
    eval_return: Any | None = None
    timed_out: bool = False

    @staticmethod
    def get_empty():
        return ExecutionResult(
            term_out=[" "],
            exec_time=0,
            exit_code=0,
            eval_return=None,
            timed_out=False,
        )


class Interpreter(ABC):
    """
    An abstract base class defining the interface for a code interpreter that:
      - Creates an isolated environment or process to run code,
      - Executes code of any language, depending on implementation.
      - Captures and returns execution results (stdout, stderr, exceptions),
      - Cleans up resources (terminates processes, closes streams, etc.).
    """

    @abstractmethod
    def __init__(
        self,
        working_dir: Path | str,
        data_dir: Path | str = None,
        timeout: int = 3600,
    ) -> None:
        """
        Initializes the interpreter with a working directory, execution timeout,
        and optional exception formatting preferences.

        Args:
            working_dir (Path | str): The working directory in which code will be executed.
            timeout (int, optional): The maximum allowed runtime for a code snippet, in seconds.
        """
        raise NotImplementedError("Subclasses must implement __init__()")

    @abstractmethod
    def run(
        self,
        code: str,
        reset_session: bool = True,
        persist_file: bool = False,
        file_name: str = "runfile.py",
        execute_code: bool = True,
    ) -> ExecutionResult:
        """
        Execute the provided code in the environment managed by this interpreter.

        Args:
            code (str): The code to be executed.
            reset_session (bool, optional): Whether to reset the interpreter session
                before executing the code (e.g., restart the subprocess).
            persist_file (bool, optional): If True, do not delete the file with the code
                after execution. (Implementation detail may vary.)
            file_name (str, optional): Name of the file to which code is written before
                execution.
            execute_code (bool): Whether to execute the code or not. If False, this
                function simply writes the code to a file.

        Returns:
            ExecutionResult: The result of execution, including stdout, stderr, exceptions,
                             execution time, etc.
        """
        raise NotImplementedError("Subclasses must implement run()")

    @abstractmethod
    def cleanup_session(self) -> None:
        """
        Cleans up resources, such as terminating the subprocess or closing open handles.
        """
        raise NotImplementedError("Subclasses must implement cleanup_session()")
