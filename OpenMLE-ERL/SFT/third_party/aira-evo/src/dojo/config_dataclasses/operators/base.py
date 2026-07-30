# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig
from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig
from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig


@dataclass
class OperatorConfig(BaseConfig):
    llm: GenericLLMConfig = field(
        default_factory=GenericLLMConfig,
        metadata={
            "description": "LLM configuration for the operator.",
            "example": {},
        },
    )

    system_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)
    init_user_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)
    user_message_prompt_template: JinjaPromptConfig = field(default_factory=JinjaPromptConfig)

    def validate(self) -> None:
        super().validate()
