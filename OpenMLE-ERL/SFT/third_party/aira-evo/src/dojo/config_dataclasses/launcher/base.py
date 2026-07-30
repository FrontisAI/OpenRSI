# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig


@dataclass
class LauncherConfig(BaseConfig):
    await_completion: bool = field(
        default=True,
        metadata={
            "help": "Whether to await completion of the job. If False, the job will be submitted and the function will return immediately.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    debug: bool = field(
        default=False,
        metadata={
            "help": "Whether to run in debug mode. If True, the job will not be submitted and the commands will be printed instead.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    monitor_jobs: bool = field(
        default=True,
        metadata={
            "help": "Whether to monitor the jobs.",
            "exclude_from_hash": True,
            "exclude_from_executor": True,
        },
    )

    def validate(self) -> None:
        super().validate()
