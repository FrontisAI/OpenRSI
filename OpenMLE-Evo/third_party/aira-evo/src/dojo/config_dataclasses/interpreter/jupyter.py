# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI

from dojo.config_dataclasses.interpreter.base import InterpreterConfig
from dojo.utils.environment import get_superimage_dir


@dataclass
class JupyterInterpreterConfig(InterpreterConfig):
    superimage_directory: str = field(
        default=get_superimage_dir(),
        metadata={
            "help": "Directory where the superimage is stored. This should be a path to a directory containing the superimage files."
        },
    )
    superimage_version: str = field(
        default="2025-03-25",
        metadata={"help": "Use symlinks to avoid copying files. This is useful for large files."},
    )
    strip_ansi: bool = field(
        default=True,
        metadata={"help": "Strip ANSI escape codes from the output."},
    )
    read_only_overlays: list[str] = field(
        default="",
        metadata={"help": "Read-only overlays to mount in the container."},
    )
    read_only_binds: dict = field(
        default_factory=dict,
        metadata={
            "help": "Read-only binds to mount in the container. Example: {'/path/on/host': '/path/in/container'}"
        },
    )
    env: dict = field(
        default_factory=dict,
        metadata={
            "help": "Environment variables to set in the container. Example: {'HF_HUB_OFFLINE': '1', 'NLTK_DATA': '/root/.nltk_data'}"
        },
    )

    def validate(self) -> None:
        super().validate()
