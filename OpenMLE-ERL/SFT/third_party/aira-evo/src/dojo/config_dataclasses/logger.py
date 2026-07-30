# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import os
from omegaconf import SI

from aira_core.config.base import BaseConfig
from dojo.utils.environment import get_log_dir


@dataclass
class LoggerConfig(BaseConfig):
    output_dir: str = field(
        default=os.path.join(
            get_log_dir(), SI("aira-dojo/user_${metadata.user}_issue_${metadata.git_issue_id}/${id}")
        ),
        metadata={
            "help": "The output directory where experiment artifacts (e.g. logs, checkpoints) will be written.",
            "exclude_from_hash": True,
        },
    )
    use_console: bool = field(
        default=True,
        metadata={
            "help": "Whether to log to stdout.",
            "exclude_from_hash": True,
        },
    )
    use_wandb: bool = field(
        default=True,
        metadata={
            "help": "Whether to log to wandb.ai.",
            "exclude_from_hash": True,
        },
    )
    use_json: bool = field(
        default=True,
        metadata={
            "help": "Whether to save files locally in JSON format",
            "exclude_from_hash": True,
        },
    )
    wandb_entity: str | None = field(
        default="aira-dojo",
        metadata={
            "help": "entity name in wandb.ai",
            "exclude_from_hash": True,
        },
    )
    wandb_project_name: str | None = field(
        default="hillclimbing-mlebench",
        metadata={
            "help": "Project name in wandb.ai.",
            "exclude_from_hash": True,
        },
    )

    tags: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "Tags to add to the experiment.",
            "exclude_from_hash": True,
        },
    )

    detailed_logging: bool = field(
        default=False,
        metadata={
            "help": "Having mean/std/min/max can clutter wandb so we make it optional.",
            "exclude_from_hash": True,
        },
    )

    print_config: bool = field(
        default=True,
        metadata={
            "help": "Whether to print the config to stdout.",
            "exclude_from_hash": True,
        },
    )
    write_env_vars: bool = field(
        default=True,
        metadata={
            "help": "Whether to write the environment variables to the config file.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
