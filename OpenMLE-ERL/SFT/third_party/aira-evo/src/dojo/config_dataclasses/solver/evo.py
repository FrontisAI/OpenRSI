# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from omegaconf import SI, MISSING

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

    # --- Agent Configuration ---
    data_preview: bool = field(
        default=MISSING,
        metadata={"description": "Whether to provide the agent with a preview of the data before execution"},
    )

    def validate(self) -> None:
        super().validate()
