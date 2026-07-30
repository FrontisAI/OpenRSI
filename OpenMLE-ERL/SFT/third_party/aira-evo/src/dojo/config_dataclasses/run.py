# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, field

from omegaconf import II, MISSING, OmegaConf

from aira_core.config.base import BaseConfig
from dojo.config_dataclasses.utils import dataclass_from_dict
from dojo.config_dataclasses.logger import LoggerConfig
from dojo.config_dataclasses.metadata import MetadataConfig
from dojo.config_dataclasses.task.base import TaskConfig
from dojo.config_dataclasses.solver.base import SolverConfig
from dojo.config_dataclasses.interpreter.base import InterpreterConfig


@dataclass
class RunConfig(BaseConfig):
    id: str = field(
        default=II("generate_id: "),
        metadata={"help": "Unique ID to use for an experiment.", "exclude_from_hash": True},
    )
    meta_id: str = field(
        default="",
        metadata={
            "help": """
            Unique ID of the Runner job that this Run is part of. Aggregating all runs
            under a single meta_id allows to get the full results of a single experiment.
            """,
            "exclude_from_hash": True,
        },
    )
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)

    def save(self) -> None:
        """
        For distributed training, be careful to only call this function on the main process.
        """
        config_dict = OmegaConf.to_container(
            OmegaConf.structured(self),
            enum_to_str=True,
        )
        with open(f"{self.logger.output_dir}/dojo_config.json", "w") as file:
            json.dump(config_dict, file)

    @classmethod
    def load_from_json(cls, json_path: str) -> "RunConfig":
        """
        Load a RunConfig from a JSON file.
        """
        with open(json_path, "r") as file:
            config_data = json.load(file)
        return RunConfig.from_dict(config_data)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return dataclass_from_dict(cls, data)
