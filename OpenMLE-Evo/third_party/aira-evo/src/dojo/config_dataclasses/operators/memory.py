# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig


@dataclass
class MemoryOpConfig(BaseConfig):
    memory_processor: str = field(
        default="simple_memory",
        metadata={
            "help": "The memory processor to use. Options: simple_memory, no_memory.",
            "choices": ["simple_memory", "no_memory"],
        },
    )
    memory_op_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "Additional arguments for the memory processor.",
            "example": {"max_memory_size": 100},
        },
    )

    def validate(self) -> None:
        super().validate()
