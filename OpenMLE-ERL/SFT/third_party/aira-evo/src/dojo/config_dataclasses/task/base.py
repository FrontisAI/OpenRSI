# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig


@dataclass
class TaskConfig(BaseConfig):
    name: str = field(
        default=MISSING,
        metadata={
            "help": "Task name, e.g., 'random-acts-of-pizza'.",
        },
    )
    benchmark: str = field(
        default=MISSING,
        metadata={
            "help": "Overaching benchmark for the task, e.g., 'mlebench', 'AIRS-H'.",
        },
    )
    data_dir: str = field(
        default=SI("${hydra:runtime.cwd}/data"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
