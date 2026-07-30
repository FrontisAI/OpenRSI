# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Microsoft Corporation
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/microsoft/autogen/blob/main/LICENSE-CODE

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Mapping
from typing import Any, Literal, Optional, Protocol, TypedDict, Union, runtime_checkable

from pydantic import BaseModel, Field


class CodeBlock(BaseModel):
    """(Experimental) A class that represents a code block."""

    code: str = Field(description="The code to execute.")

    language: str = Field(description="The language of the code.")


class CodeResult(BaseModel):
    """(Experimental) A class that represents the result of a code execution."""

    exit_code: int = Field(description="The exit code of the code execution.")

    output: str = Field(description="The output of the code execution.")


class CodeExtractor(Protocol):
    """(Experimental) A code extractor class that extracts code blocks from a message."""

    def extract_code_blocks(self, message: str | list[Any] | None) -> list[CodeBlock]:
        """(Experimental) Extract code blocks from a message.

        Args:
            message (str): The message to extract code blocks from.

        Returns:
            List[CodeBlock]: The extracted code blocks.
        """
        ...  # pragma: no cover


@runtime_checkable
class CodeExecutor(Protocol):
    """(Experimental) A code executor class that executes code blocks and returns the result."""

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) The code extractor used by this code executor."""
        ...  # pragma: no cover

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CodeResult:
        """(Experimental) Execute code blocks and return the result.

        This method should be implemented by the code executor.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CodeResult: The result of the code execution.
        """
        ...  # pragma: no cover

    def restart(self) -> None:
        """(Experimental) Restart the code executor.

        This method should be implemented by the code executor.

        This method is called when the agent is reset.
        """
        ...  # pragma: no cover


class IPythonCodeResult(CodeResult):
    """(Experimental) A code result class for IPython code executor."""

    output_files: list[str] = Field(
        default_factory=list,
        description="The list of files that the executed code blocks generated.",
    )


CodeExecutionConfig = TypedDict(
    "CodeExecutionConfig",
    {
        "executor": Union[Literal["ipython-embedded", "commandline-local"], CodeExecutor],
        "last_n_messages": Union[int, Literal["auto"]],
        "timeout": int,
        "use_docker": Union[bool, str, list[str]],
        "work_dir": str,
        "ipython-embedded": Mapping[str, Any],
        "commandline-local": Mapping[str, Any],
    },
    total=False,
)


@dataclass
class JupyterConnectionInfo:
    """(Experimental)"""

    host: str
    """`str` - Host of the Jupyter gateway server"""
    use_https: bool
    """`bool` - Whether to use HTTPS"""
    port: Optional[int] = None
    """`Optional[int]` - Port of the Jupyter gateway server. If None, the default port is used"""
    token: Optional[str] = None
    """`Optional[str]` - Token for authentication. If None, no token is used"""


@runtime_checkable
class JupyterConnectable(Protocol):
    """(Experimental)"""

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        """Return the connection information for this connectable."""
        pass
