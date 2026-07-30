# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 John Yang, Carlos E. Jimenez,
# Alexander Wettig, Shunyu Yao, Karthik Narasimhan, Ofir Press
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/SWE-agent/SWE-agent/blob/main/LICENSE

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ModelArguments:
    """Arguments configuring the model and it's behavior."""

    # Name of the model to use
    model_name: str
    # Cost limit for every task
    per_instance_cost_limit: float = 0.0
    # Total cost limit
    total_cost_limit: float = 0.0
    # Sampling temperature
    temperature: float = 0.8
    # Sampling top_p
    top_p: float = 1.0
    # Path to replay file when using the replay model
    replay_path: str | None = None
    # Host URL when using Ollama model
    host_url: str = "localhost:11434"


@dataclass
class APIStats:
    total_cost: float = 0.0
    task_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: APIStats):
        if not isinstance(other, APIStats):
            msg = "Can only add APIStats with APIStats, got type {type(other)}"
            raise TypeError(msg)

        return APIStats(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)},
        )

    def replace(self, other):
        if not isinstance(other, APIStats):
            msg = "Can only replace APIStats with APIStats"
            raise TypeError(msg)

        return APIStats(**{field.name: getattr(other, field.name) for field in fields(self)})


class ContextWindowExceededError(Exception):
    pass


class CostLimitExceededError(Exception):
    pass


class APIError(Exception):
    pass


class RateLimitExceededError(Exception):
    pass
