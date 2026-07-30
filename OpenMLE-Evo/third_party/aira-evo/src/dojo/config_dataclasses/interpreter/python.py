# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI

from dojo.config_dataclasses.interpreter.base import InterpreterConfig


@dataclass
class PythonInterpreterConfig(InterpreterConfig):
    use_symlinks: bool = field(
        default=True,
        metadata={"help": "Use symlinks to avoid copying files. This is useful for large files."},
    )
    format_tb_ipython: bool = field(
        default=False,
        metadata={
            "help": "Format traceback using IPython style. This is useful for debugging.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
