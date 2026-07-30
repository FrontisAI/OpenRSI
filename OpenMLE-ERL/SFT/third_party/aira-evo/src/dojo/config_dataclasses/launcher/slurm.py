# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from dojo.config_dataclasses.launcher.base import LauncherConfig
from dojo.utils.environment import (
    get_default_slurm_partition,
    get_default_slurm_account,
    get_default_slurm_qos
)


@dataclass
class SlurmConfig(LauncherConfig):
    account: str = field(
        default=get_default_slurm_account(),
        metadata={
            "help": "Slurm account name",
            "exclude_from_hash": True,
        },
    )
    qos: str = field(
        default=get_default_slurm_qos(),
        metadata={
            "help": "Quality of service level",
            "exclude_from_hash": True,
        },
    )
    nodes: int = field(
        default=1,
        metadata={
            "help": "Number of nodes to allocate",
        },
    )
    ntasks_per_node: int = field(
        default=1,
        metadata={
            "help": "Number of tasks per node",
            "exclude_from_hash": True,
        },
    )
    gpus_per_node: int = field(
        default=1,
        metadata={
            "help": "Number of GPUs per node",
        },
    )
    cpus_per_task: int = field(
        default=24,
        metadata={
            "help": "Number of CPU cores per task",
            "exclude_from_hash": True,
        },
    )
    partition: str = field(
        default=get_default_slurm_partition(),
        metadata={
            "help": "Slurm partition to use",
            "exclude_from_hash": True,
        },
    )
    time: str = field(
        default="24:00:00",
        metadata={
            "help": "Maximum job runtime in format HH:MM:SS",
            "exclude_from_hash": True,
        },
    )
    mem: str = field(
        default="200G",
        metadata={
            "help": "Memory allocation per node",
            "exclude_from_hash": True,
        },
    )
    # requeue: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to requeue the job if it fails.",
    #         "exclude_from_hash": False,
    #     },
    # ) # Not used anywhere

    array_parallelism: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of parallel jobs to run.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
