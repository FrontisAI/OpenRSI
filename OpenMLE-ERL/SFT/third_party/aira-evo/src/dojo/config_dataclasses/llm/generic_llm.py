# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig
from dojo.config_dataclasses.client.base import ClientConfig


@dataclass
class GenericLLMConfig(BaseConfig):
    client: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={
            "description": "Client configuration for the solver.",
            "example": {},
        },
    )

    generation_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "LLM generation arguments."},
    )

    def validate(self) -> None:
        super().validate()
