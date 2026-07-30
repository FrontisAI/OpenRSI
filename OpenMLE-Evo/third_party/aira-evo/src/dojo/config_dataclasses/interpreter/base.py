# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI

from aira_core.config.base import BaseConfig


@dataclass
class InterpreterConfig(BaseConfig):
    working_dir: str = field(
        default=SI("${logger.output_dir}/workspace_agent/"),
        metadata={
            "help": "Working directory for the interpreter.",
            "exclude_from_hash": True,
        },
    )
    timeout: int = field(
        default=SI("${solver.execution_timeout}"),
        metadata={"help": "Timeout for the interpreter."},
    )

    def validate(self) -> None:
        super().validate()
