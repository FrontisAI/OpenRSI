# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

from aira_core.config.base import BaseConfig
from dojo.config_dataclasses.operators.base import OperatorConfig

from dojo.config_dataclasses.operators.memory import MemoryOpConfig


@dataclass
class SolverConfig(BaseConfig):
    step_limit: int = field(
        default=500,
        metadata={
            "description": "Maximum number of steps allowed in the search process.",
            "example": 300,
        },
    )

    available_packages: list[str] = field(
        default_factory=lambda: [
            "numpy",  # Numerical computing
            "pandas",  # Data manipulation and analysis
            "scikit-learn",  # Machine learning library
            "statsmodels",  # Statistical modeling and econometrics
            "xgboost",  # Gradient boosting for structured data
            "lightgbm",  # Efficient gradient boosting framework
            "torch",  # PyTorch deep learning framework
            "torchvision",  # Image processing utilities for PyTorch
            "torch-geometric",  # Graph-based deep learning with PyTorch
            "bayesian-optimization",  # Bayesian optimization library
            "timm",  # PyTorch image models collection
        ],
        metadata={
            "description": "List of available packages for the solver.",
            "example": ["numpy", "pandas"],
        },
    )
    operators: dict[str, OperatorConfig] = field(
        default_factory={},
        metadata={
            "description": "List of operator names for the solver.",
            "example": {},
        },
    )
    memory: MemoryOpConfig = field(
        default_factory=MemoryOpConfig,
        metadata={
            "description": "Memory configuration for the solver.",
            "example": {},
        },
    )
    debug_memory: MemoryOpConfig = field(
        default_factory=MemoryOpConfig,
        metadata={
            "description": "Memory configuration for debug cycle of the solver.",
            "example": {},
        },
    )

    # Experiment name variable used in path configuration.
    exp_name: str = field(
        default=SI("${id}"),
        metadata={
            "description": "Experiment name variable used in path configuration.",
            "example": "user_...",
            "exclude_from_hash": True,
        },
    )

    execution_timeout: int = field(
        default=14400,
        metadata={
            "help": "The timeout for the interpreter in seconds.",
        },
    )

    time_limit_secs: int = field(
        default=86400,
        metadata={
            "help": "The time limit for the task in seconds.",
        },
    )

    export_search_results: bool = field(
        default=True,
        metadata={
            "description": "Whether to export search results after execution",
            "example": True,
            "exclude_from_hash": True,
        },
    )

    checkpoint_path: str = field(
        default=SI("${logger.output_dir}/checkpoint"),
        metadata={
            "description": "Path to the checkpoint file.",
            "example": "path/to/checkpoint",
            "exclude_from_hash": True,
        },
    )

    use_test_score: bool = field(
        default=False,
        metadata={
            "description": "Whether to use the test score for evaluation",
            "example": True,
        },
    )

    use_complexity: bool = field(
        default=False,
        metadata={
            "description": "Whether to consider complexity differences in prompts - only works with certain operators",
            "example": True,
        },
    )

    max_llm_call_retries: int = field(
        default=3,
        metadata={
            "description": "Maximum number of retries for failed LLM API calls.",
            "example": 2,
            "exclude_from_hash": True,
        },
    )

    def validate(self) -> None:
        super().validate()
