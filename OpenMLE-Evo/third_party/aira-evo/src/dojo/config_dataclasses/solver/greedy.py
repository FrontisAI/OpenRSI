# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from dojo.config_dataclasses.solver.base import SolverConfig


@dataclass
class GreedySolverConfig(SolverConfig):
    # --- Search Configuration ---
    improvement_steps: int = field(
        default=5,
        metadata={
            "description": "Number of improvement iterations to perform.",
            "example": 3,
        },
    )

    data_preview: bool = field(
        default=False,
        metadata={
            "description": "Whether to provide the agent with a preview of the data before execution.",
            "example": True,
        },
    )

    # --- Debugging Configuration ---
    max_debug_depth: int = field(
        default=3,
        metadata={
            "description": "Maximum depth of debugging analysis.",
            "example": 2,
        },
    )

    debug_prob: float = field(
        default=0.5,
        metadata={
            "description": "Probability of running a debug step in the process.",
            "example": 0.3,
        },
    )

    # --- Drafting Configuration ---
    num_drafts: int = field(
        default=5,
        metadata={
            "description": "Number of draft outputs to generate for selection.",
            "example": 3,
        },
    )

    def validate(self) -> None:
        super().validate()
