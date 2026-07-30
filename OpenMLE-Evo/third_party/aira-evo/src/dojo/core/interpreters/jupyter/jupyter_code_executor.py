# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Microsoft Corporation
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/microsoft/autogen/blob/main/LICENSE-CODE

import base64
import json
import os
import sys
import uuid
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Union

from optuna import artifacts
from regex import B, D

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .base import CodeBlock, CodeExecutor, CodeExtractor, IPythonCodeResult, CodeResult

# from ..markdown_code_extractor import MarkdownCodeExtractor
# from ..utils import silence_pip
from .base import JupyterConnectable, JupyterConnectionInfo
from .jupyter_client import JupyterClient
from ..base import ExecutionResult

import logging

log = logging.getLogger(__name__)


class JupyterCodeExecutor(CodeExecutor):
    def __init__(
        self,
        jupyter_server: Union[JupyterConnectable, JupyterConnectionInfo],
        kernel_name: str | None = None,
        timeout: int = 60,
        output_dir: Union[Path, str] = Path(),
    ):
        """(Experimental) A code executor class that executes code statefully using
        a Jupyter server supplied to this class.

        Each execution is stateful and can access variables created from previous
        executions in the same session.

        Args:
            jupyter_server (Union[JupyterConnectable, JupyterConnectionInfo]): The Jupyter server to use.
            timeout (int): The timeout for code execution, by default 60.
            kernel_name (str | None): The kernel name to use. Make sure it is installed.
                By default, it choses default from kernelspecs.
            output_dir (str): The directory to save output files, by default ".".
        """
        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.exists():
            raise ValueError(f"Output directory {output_dir} does not exist.")

        if isinstance(jupyter_server, JupyterConnectable):
            self._connection_info = jupyter_server.connection_info
        elif isinstance(jupyter_server, JupyterConnectionInfo):
            self._connection_info = jupyter_server
        else:
            raise ValueError("jupyter_server must be a JupyterConnectable or JupyterConnectionInfo.")

        self._jupyter_client = JupyterClient(self._connection_info)
        available_kernels = self._jupyter_client.list_kernel_specs()
        if kernel_name is None:
            kernel_name = available_kernels["default"]
        if kernel_name not in available_kernels["kernelspecs"]:
            raise ValueError(f"Kernel {kernel_name} is not installed.")

        log.warning(f"Starting kernel {kernel_name}")
        self._kernel_id = self._jupyter_client.start_kernel(kernel_name)
        log.warning(f"Kernel {kernel_name} started with id {self._kernel_id}")
        self._kernel_name = kernel_name
        log.warning(f"Getting kernel client")
        self._jupyter_kernel_client = self._jupyter_client.get_kernel_client(self._kernel_id)
        log.warning(f"Kernel client {self._jupyter_kernel_client} created")
        self._timeout = timeout
        self._wait_timeout = 120
        self._fetch_file_timeout = 1800
        self._output_dir = output_dir

    def execute_code(self, code: str) -> ExecutionResult:
        start_time = time.monotonic()

        log.warning(f"Waiting for ready")
        ready = self._jupyter_kernel_client.wait_for_ready(timeout_seconds=self._wait_timeout)
        if not ready:
            log.warning("Kernel did not become ready in time.")
            return ExecutionResult(
                term_out=["ERROR:", "Kernel did not become ready in time."],
                exit_code=1,
                exec_time=time.monotonic() - start_time,
                timed_out=True,
            )
        log.warning(f"Ready")
        output_lines = []
        # output_file = None

        log.warning(f"Executing code")
        result = self._jupyter_kernel_client.execute(code, timeout_seconds=self._timeout)
        log.warning(f"Done Executing code")
        elapsed_time = time.monotonic() - start_time
        log.warning(f"Execution time: {elapsed_time:.2f} seconds")

        if result.timed_out:
            log.warning("Execution timed out. Interrupting the kernel.")
            self._jupyter_client.interrupt_kernel(self._kernel_id)
            log.warning("Kernel interrupt signal sent successfully. This does not guarantee the kernel will stop")

        if not result.is_ok:
            log.warning("Execution failed.")
            return ExecutionResult(
                term_out=["ERROR:", *result.output],
                exit_code=1,
                exec_time=elapsed_time,
                timed_out=result.timed_out,
            )

        output_lines = result.output

        for data in result.data_items:
            log.warning(f"Data item detected: {data.mime_type}")
            if data.mime_type == "application/octet-stream":
                output_lines = [data.data]
            else:
                output_lines.append(json.dumps(data.data))

        log.warning("All good, returning output, with exit code 0")
        return ExecutionResult(
            term_out=output_lines,
            exit_code=0,
            exec_time=elapsed_time,
            timed_out=result.timed_out,
        )

    def fetch_file(
        self,
        filename: str,
    ) -> bytes:
        code = """
        def publish_blob(filename):
            import base64
            from IPython.display import publish_display_data

            with open(filename, "rb") as f:
                file_data = f.read()

            b64_data = base64.b64encode(file_data).decode("utf-8")

            publish_display_data(
                data={"application/octet-stream": b64_data},
                metadata={"application/octet-stream": {"fileName": filename}}
            )"""
        code += f"""
        publish_blob('{filename}')
        """
        log.warning(f"Waiting kernel to be ready")
        ready = self._jupyter_kernel_client.wait_for_ready(timeout_seconds=self._wait_timeout)
        if not ready:
            log.warning("Kernel did not become ready in time.")
            raise TimeoutError("Kernel did not become ready in time.")

        log.warning(f"Kernel is ready, executing code")
        result = self._jupyter_kernel_client.execute(code, timeout_seconds=self._fetch_file_timeout)

        if result.timed_out:
            log.warning("Execution timed out. Interrupting the kernel.")
            self._jupyter_client.interrupt_kernel(self._kernel_id)
            log.warning("Kernel interrupted successfully.")
            raise TimeoutError("Kernel did not become ready in time.")

        log.warning(
            f"Done executing code; result.is_ok: {result.is_ok}, len(result.data_items): {len(result.data_items)}"
        )
        if not result.is_ok or len(result.data_items) != 1:
            raise FileNotFoundError(f"File {filename} not found.")

        data_item = result.data_items[0]
        buffer = base64.b64decode(data_item.data)
        log.warning(f"Size of decoded data: {len(buffer)}")
        return buffer

    def restart(self) -> None:
        """(Experimental) Restart a new session."""
        log.warning(f"Restarting kernel {self._kernel_id}")
        self._jupyter_client.restart_kernel(self._kernel_id)
        log.warning(f"Kernel {self._kernel_id} restarted")
        self._jupyter_kernel_client = self._jupyter_client.get_kernel_client(self._kernel_id)

    def stop(self) -> None:
        """Stop the kernel."""
        log.warning(f"Stopping kernel {self._kernel_id}")

        log.warning(f"Stopping kernel client {self._jupyter_kernel_client}")
        self._jupyter_kernel_client.stop()
        log.warning(f"Deleting kernel {self._kernel_id}")
        self._jupyter_client.delete_kernel(self._kernel_id)
        log.warning(f"Kernel {self._kernel_id} stopped")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) The code extractor used by this code executor."""
        raise NotImplementedError("code_extractor is not implemented")

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CodeResult:
        """(Experimental) Execute code blocks and return the result.

        This method should be implemented by the code executor.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CodeResult: The result of the code execution.
        """
        raise NotImplementedError("execute_code_blocks is not implemented")
