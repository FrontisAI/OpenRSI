# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig


@dataclass
class ClientConfig(BaseConfig):
    api: str = field(
        default="litellm",
        metadata={
            "description": "API endpoint for the LLM client.",
            "example": "litellm",
            "exclude_from_hash": True,
        },
    )

    model_id: str = field(
        default="gpt-4o",
        metadata={
            "description": "The identifier for the model to use.",
            "example": "deepseek-ai/DeepSeek-R1",
        },
    )

    base_url: str = field(
        default="https://azure-services-endpoint-here.azure-api.net",
        metadata={
            "description": "Base URL for the API endpoint.",
            "example": "http://self-hosted-endpoint:20000/v1",
            "exclude_from_hash": True,
        },
    )

    use_azure_client: bool = field(
        default=True,
        metadata={
            "description": "Whether to use Azure client for API calls.",
            "exclude_from_hash": True,
        },
    )

    provider: str = field(
        default="openai",
        metadata={
            "description": "The provider of the LLM service.",
            "example": "selfhosted",
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
