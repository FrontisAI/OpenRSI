# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from dojo.config_dataclasses.solver.base import SolverConfig


@dataclass
class EvolutionarySolverConfig(SolverConfig):
    # Hyperparameters for the evolutionary process.
    num_islands: int = field(default=MISSING, metadata={"help": "Number of islands (sub-populations) to initialize."})
    max_island_size: int = field(
        default=MISSING, metadata={"help": "Maximum number of samples allowed in each island."}
    )
    crossover_prob: float = field(
        default=MISSING, metadata={"help": "Probability of choosing a crossover operation over mutation."}
    )
    migration_prob: float = field(
        default=MISSING,
        metadata={"help": "Probability of refreshing weaker islands with seeds from stronger islands."},
    )
    initial_temp: float = field(default=MISSING, metadata={"help": "Starting temperature for exploration."})
    final_temp: float = field(
        default=MISSING, metadata={"help": "Final temperature value, shifting towards exploitation."}
    )
    num_generations_till_migration: int = field(
        default=MISSING, metadata={"help": "Number of generations before migration can occur."}
    )
    num_generations_till_crossover: int = field(
        default=MISSING, metadata={"help": "Number of generations before crossover can occur."}
    )

    # Few-shot prompting configuration for different operations.
    few_shot: dict = field(
        default_factory=lambda: {"improve": 1, "crossover": 2},
        metadata={"help": "Few-shot prompting configuration for different operations."},
    )

    # Evolution settings for the search process.
    num_generations: int = field(default=MISSING, metadata={"help": "Total number of generations to run."})
    individuals_per_generation: int = field(
        default=MISSING, metadata={"help": "Number of individuals per generation."}
    )

    max_debug_time: float = field(
        default=MISSING, metadata={"description": "Maximum time allowed for debugging analysis"}
    )
    max_debug_depth: int = field(default=MISSING, metadata={"description": "Maximum depth of debugging analysis"})
    fresh_draft_prob: float = field(
        default=0.0,
        metadata={
            "description": "Probability of starting a fresh draft instead of mutating an existing solution."
        },
    )
    max_wall_time_secs: float = field(
        default=0.0,
        metadata={
            "description": "Optional wall-clock safety cap; zero disables the cap."
        },
    )

    # --- Agent Configuration ---
    data_preview: bool = field(
        default=MISSING,
        metadata={"description": "Whether to provide the agent with a preview of the data before execution"},
    )
    experience: dict = field(
        default_factory=dict,
        metadata={"description": "Deterministic experience-card and parent-selection settings."},
    )
    execution_mode: str = field(
        default="generation",
        metadata={"description": "Search scheduler: generation or async_steady_state."},
    )
    async_workers: Any = field(
        default=1,
        metadata={"description": "Number of async steady-state workers, or auto to use visible GPU count."},
    )
    async_sandbox_urls: list[str] = field(
        default_factory=list,
        metadata={"description": "Sandbox/router base URLs used by async workers."},
    )
    async_sandbox_assignment: str = field(
        default="round_robin",
        metadata={"description": "Sandbox URL assignment policy for async workers."},
    )
    async_checkpoint_every_commits: int = field(
        default=1,
        metadata={"description": "Save checkpoint every N async commits."},
    )
    async_worker_max_retries: int = field(
        default=3,
        metadata={
            "description": "Retries for one async attempt before consuming another attempt id."
        },
    )
    async_worker_retry_backoff_secs: float = field(
        default=1.0,
        metadata={
            "description": "Initial exponential backoff for transient async worker failures."
        },
    )

    def validate(self) -> None:
        super().validate()
