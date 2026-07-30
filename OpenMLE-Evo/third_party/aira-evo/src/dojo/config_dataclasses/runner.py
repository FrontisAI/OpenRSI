# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import copy
from dataclasses import dataclass, field

from omegaconf import II, MISSING, OmegaConf

from aira_core.config.base import BaseConfig
from dojo.config_dataclasses.logger import LoggerConfig
from dojo.config_dataclasses.metadata import MetadataConfig
from dojo.config_dataclasses.benchmark import BenchmarkConfig
from dojo.config_dataclasses.solver.base import SolverConfig
from dojo.config_dataclasses.interpreter.base import InterpreterConfig
from dojo.config_dataclasses.launcher.base import LauncherConfig


@dataclass
class RunnerConfig(BaseConfig):
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    vars: dict[str, list] = field(
        default_factory=dict,
        metadata={
            "help": "Override variables to sweep over.",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()

        # Check if there is any duplicated values in the vars dictionary
        for key, values in self.vars.items():
            if not values:
                raise ValueError(f"Value for {key} is None or empty")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicated values found in vars for key: {key}")

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

    @property
    def id(self) -> str:
        """
        Generate a unique ID for the runner based on an unresolved config.

        This is not a robust hash at all, it will be missing some fields
        due to the fact that some fields are unresolved, but it is a
        simple way to aggregate the runs with the same config together
        at run time because we know that the fields overwritten in `vars`
        will be present here.
        """
        config = copy.deepcopy(self)
        fields_to_exclude = RunnerConfig.fields_to_exclude_from_hash()
        # Accessing fields of config will resolve them
        for field, field_type in fields_to_exclude:
            if field_type in (int, float):
                exec(f"config.{field} = 0")
            elif field_type == bool:
                exec(f"config.{field} = False")
            else:
                exec(f'config.{field} = ""')

        return config.hash()
