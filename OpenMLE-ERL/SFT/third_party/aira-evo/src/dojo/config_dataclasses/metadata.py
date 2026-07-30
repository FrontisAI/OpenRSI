# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import II

from aira_core.config.base import BaseConfig


@dataclass
class MetadataConfig(BaseConfig):
    script_id: str = field(
        default="",
        metadata={"help": "ID of the script used to launch the experiment."},
    )
    git_issue_id: str = field(
        default="",
        metadata={
            "help": "GitHub issue ID for experiment tracking. An issue ID typically corresponds to multiple related experiments (e.g. a hyperparameter sweep)"
        },
    )
    description: str = field(
        default="",
        metadata={"help": "Description of the experiment."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for the experiment."})

    # Tracking information (generated automatically through OmegaConf interpolations)
    launch_time: str = field(
        default=II("get_current_time:"),
        metadata={
            "help": "Date and time when the experiment was launched.",
            "exclude_from_hash": True,
        },
    )
    base_path: str = field(
        default=II("oc.env:PWD"),
        metadata={"help": "Base path for code used to run the experiment.", "exclude_from_hash": True},
    )
    user: str = field(
        default=II("oc.env:USER"),
        metadata={
            "help": "User who launched the experiment.",
            "exclude_from_hash": True,
        },
    )

    # Useful debugging information (generated automatically through OmegaConf interpolations)
    git_commit_id: str = field(
        default=II("get_git_commit_id:"),
        metadata={
            "help": "(Last) Commit id for this run. For Arrival experiments, this is generated using omegaconf interpolations.",
            "exclude_from_hash": True,
        },
    )
    torch_version: str = field(
        default=II("get_torch_version:"),
        metadata={
            "help": "Version of PyTorch that the experiment was run with. For Arrival experiments, this is generated using omegaconf interpolations."
        },
    )
    slurm_id: str = field(
        default="",
        metadata={
            "help": "Slurm id for the experiment.",
            "exclude_from_hash": True,
        },
    )
