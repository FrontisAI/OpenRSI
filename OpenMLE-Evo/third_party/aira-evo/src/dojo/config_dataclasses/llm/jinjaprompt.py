# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig


@dataclass
class JinjaPromptConfig(BaseConfig):
    template: str = field(
        default="",
        metadata={"help": "The Jinja template to be used for the prompt."},
    )

    input_variables: list[str] = field(
        default_factory=list,
        metadata={"help": "List of input variables for the Jinja template."},
    )

    partial_variables: dict = field(
        default_factory=dict,
        metadata={"help": "Dictionary of partial variables for the Jinja template."},
    )

    def validate(self) -> None:
        super().validate()
