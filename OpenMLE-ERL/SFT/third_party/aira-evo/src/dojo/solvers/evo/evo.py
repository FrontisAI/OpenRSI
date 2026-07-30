# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Rishi Hazra, Alkis Sygkounas
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/RishiHazra/Revolve/blob/main/LICENSE

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2023 Google DeepMind
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/google-deepmind/funsearch/blob/main/LICENSE

import json
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import hydra
import numpy
import time

import dojo.core.solvers.utils.search_utils as utils
from dojo.core.solvers.base import Solver
from dojo.core.solvers.operators.analyze import analyze_op
from dojo.core.solvers.operators.core import execute_op_plan_code
from dojo.core.solvers.operators.crossover import crossover_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.utils import data_preview
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.core.solvers.utils.search_exporter import export_search_results
from dojo.solvers.utils import get_complextiy_level
from dojo.utils.logger import CollectiveLogger, LogEvent
from dojo.utils.code_parsing import parse_json_output
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig
from dojo.utils.state import EvolutionaryState
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM


class Island:
    """A population of solutions (Nodes) in an island."""

    def __init__(
        self,
        island_id: int,
        initial_solution_nodes: List[Node],
        lower_is_better: bool,
        logger: CollectiveLogger,
    ):
        self.island_id = island_id
        self.lower_is_better = lower_is_better
        self.logger = logger
        self.nodes: List[Node] = initial_solution_nodes

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def fitness_scores(self) -> List[float]:
        """Returns fitness scores, using +/- inf for None metrics."""
        scores = []
        for node in self.nodes:
            fitness = node.metric.value
            if fitness is None:
                # Use +/- inf for comparison purposes
                scores.append(float("inf") if self.lower_is_better else float("-inf"))
            else:
                scores.append(fitness)
        return scores

    @property
    def best_fitness_score(self) -> float:
        """Returns the best fitness score in the island."""
        scores = self.fitness_scores
        if not scores:
            # Return worst possible score if island is empty
            return float("inf") if self.lower_is_better else float("-inf")
        op = numpy.min if self.lower_is_better else numpy.max
        return op(scores)

    @property
    def average_fitness_score(self):
        """Returns the average fitness score."""
        scores = self.fitness_scores
        if not scores:
            # Avoid numpy warning on empty list, return worst score
            return float("inf") if self.lower_is_better else float("-inf")
        # numpy.mean handles inf correctly (inf + finite = inf, inf + inf = inf, inf - inf = nan)
        # However, if all scores are inf/-inf, mean might be inf/-inf.
        # Filter out inf/-inf before calculating mean for a more representative average?
        # Or rely on the fact that inf/-inf indicates poor solutions.
        finite_scores = [s for s in scores if s != float("inf") and s != float("-inf")]
        if not finite_scores:
            # If only inf/-inf scores, return that score
            return scores[0] if scores else (float("inf") if self.lower_is_better else float("-inf"))
        return numpy.mean(finite_scores)

    @property
    def fittest_individual(self) -> Optional[Node]:  # Renamed to fittest_node
        """Returns the node with the best fitness score."""
        if not self.nodes:
            return None
        scores = self.fitness_scores
        op = numpy.argmin if self.lower_is_better else numpy.argmax
        fittest_idx = op(scores)
        return self.nodes[fittest_idx]

    @property
    def solution_nodes(self) -> List[Node]:  # Keep this property
        return self.nodes

    def register_node_in_island(
        self,
        solution_node: Node,
    ):
        """
        Add a solution Node to the island population.
        """
        # IDs are stored within the node if needed elsewhere (e.g., node.id)
        self.logger.info(
            f"Registering Node {solution_node.id} in Island {self.island_id}. Metric: {solution_node.metric.value}",
            LogEvent.SOLVER,
        )
        self.nodes.append(solution_node)

    def remove_lowest(self):
        """
        Removes the node with the worst fitness score in the island.
        """
        if not self.nodes:
            return  # Nothing to remove

        scores = self.fitness_scores
        op = numpy.argmax if self.lower_is_better else numpy.argmin
        lowest_score_index = op(scores)
        weakest_node = self.nodes.pop(lowest_score_index)
        self.logger.info(
            f"Removed weakest node {weakest_node.id} (Score: {scores[lowest_score_index]}) from island {self.island_id}",
            LogEvent.SOLVER,
        )

    def remove_node(self, to_remove_node: Node):  # Renamed from remove_individual
        """
        Remove a specific node from the island.
        """
        initial_size = len(self.nodes)
        self.nodes = [node for node in self.nodes if node.id != to_remove_node.id]
        if len(self.nodes) < initial_size:
            self.logger.info(f"Removed node {to_remove_node.id} from island {self.island_id}", LogEvent.SOLVER)
        else:
            self.logger.warning(
                f"Attempted to remove node {to_remove_node.id} from island {self.island_id}, but it was not found.",
                LogEvent.SOLVER,
            )

    def only_keep_best(self):
        """
        Remove all nodes except the single best one.
        Handles ties by keeping only one of the best.
        """
        best_node = self.fittest_individual  # Renamed property
        if best_node:
            self.logger.info(
                f"Island {self.island_id}: Keeping only best node {best_node.id} (Score: {best_node.metric.value})",
                LogEvent.SOLVER,
            )
            self.nodes = [best_node]
        else:
            self.logger.info(f"Island {self.island_id}: only_keep_best called on empty island.", LogEvent.SOLVER)
            self.nodes = []

    def migrate_node(
        self,
        founder_node: Node,  # Changed type to Node
    ):
        """
        Migrate a node from a founder island to this island.
        """
        self.logger.info(f"Migrating node {founder_node.id} to Island {self.island_id}", LogEvent.SOLVER)
        self.register_node_in_island(founder_node)


class SolutionsDatabase:
    """
    Maintains and updates a Database of all solutions (Nodes).

    Adapted from Fun Search: https://github.com/google-deepmind/funsearch/blob/main
    and from REvolve: https://arxiv.org/pdf/2406.01309
    """

    def __init__(
        self,
        num_islands: int,
        max_size: int,
        lower_is_better: bool,
        logger: CollectiveLogger,
    ):
        self.num_islands = num_islands
        self.max_size = max_size
        self.lower_is_better = lower_is_better
        self.logger = logger
        self._islands: List[Island] = []
        self.global_min_fitness = float("inf")
        self.global_max_fitness = float("-inf")

        # Initialize empty islands.
        self._islands = [
            Island(island_id, [], self.lower_is_better, self.logger) for island_id in range(self.num_islands)
        ]

    def state_dict(self) -> dict:
        return {
            "num_islands": self.num_islands,
            "max_size": self.max_size,
            "lower_is_better": self.lower_is_better,
            "global_min_fitness": self.global_min_fitness,
            "global_max_fitness": self.global_max_fitness,
            "island_populations": [
                [str(node.id) for node in island.nodes]
                for island in self._islands
            ],
        }

    def seed_islands_with_nodes(
        self,
        solution_nodes: List[Node],
        island_ids: List[int],
    ):
        """
        Initialize islands with the first generation of Nodes.
        """
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            if 0 <= island_id < len(self._islands):
                self._islands[island_id].register_node_in_island(
                    solution_node,
                )
            else:
                self.logger.info(f"Invalid island_id {island_id} during seeding. Max index: {len(self._islands) - 1}")
            # Seed initial global fitness range if nodes have scores
            fitness_score = solution_node.metric.value
            if fitness_score is not None and numpy.isfinite(fitness_score):
                self._update_global_fitness_range(fitness_score)

    def _update_global_fitness_range(self, score: float):
        """Updates the global min and max fitness scores seen so far."""
        if numpy.isfinite(score):
            self.global_min_fitness = min(self.global_min_fitness, score)
            self.global_max_fitness = max(self.global_max_fitness, score)
            self.logger.info(
                f"Updated global fitness range: min={self.global_min_fitness}, max={self.global_max_fitness}",
                LogEvent.SOLVER,
            )

    def get_normalized_score(self, score: Optional[float]) -> float:
        """
        Normalizes a raw fitness score to the range [0, 1], where 1.0 is always best.
        Handles None scores by returning 0.0 (worst).
        """
        if score is None or not numpy.isfinite(score):
            return 0.0  # Worst normalized score for None or non-finite scores

        # If global range isn't established or is a single point, return neutral 0.5
        if (
            not numpy.isfinite(self.global_min_fitness)
            or not numpy.isfinite(self.global_max_fitness)
            or self.global_min_fitness == self.global_max_fitness
        ):
            return 0.5

        # Normalize to [0, 1]
        if self.lower_is_better:
            # For lower_is_better, higher values are worse.
            # (global_max - score) / (global_max - global_min)
            # If score is min_fitness, result is 1. If score is max_fitness, result is 0.
            normalized = (self.global_max_fitness - score) / (self.global_max_fitness - self.global_min_fitness)
        else:
            # For higher_is_better, higher values are better.
            # (score - global_min) / (global_max - global_min)
            # If score is min_fitness, result is 0. If score is max_fitness, result is 1.
            normalized = (score - self.global_min_fitness) / (self.global_max_fitness - self.global_min_fitness)

        # Clamp to [0, 1] to handle scores outside the current global range robustly
        return float(numpy.clip(normalized, 0.0, 1.0))

    def add_nodes_to_islands(
        self,
        solution_nodes: List[Node],
        island_ids: List[int],
        migration_prob: float,
    ):
        """
        Add evaluated Nodes to appropriate islands based on fitness improvement.
        Manages island size and triggers migration/reset.
        """
        for solution_node, island_id in zip(
            solution_nodes,
            island_ids,
        ):
            if not (0 <= island_id < len(self._islands)):
                self.logger.error(f"Invalid island_id {island_id} when adding node {solution_node.id}. Skipping.")
                continue

            fitness_score = solution_node.metric.value  # This can be None or a float
            if fitness_score is not None and numpy.isfinite(fitness_score):  # Update global range
                self._update_global_fitness_range(fitness_score)

            current_island = self._islands[island_id]
            island_avg_fitness_score = current_island.average_fitness_score

            # Determine if the new node improves the island
            if current_island.size == 0:
                improvement_condition = True  # Always add to an empty island
            elif fitness_score is None:
                improvement_condition = False  # A solution with None fitness never improves an island
            else:
                improvement_condition = (self.lower_is_better and fitness_score <= island_avg_fitness_score) or (
                    not self.lower_is_better and fitness_score >= island_avg_fitness_score
                )

            # check if individual is adding any value to the island
            if improvement_condition:
                current_island.register_node_in_island(
                    solution_node,
                )
                self.logger.info(
                    f"Node {solution_node.id} added to island {island_id}. New avg score: {current_island.average_fitness_score:.4f}",
                    LogEvent.SOLVER,
                )

            # if island size exceeds max size, discard individual with the lowest score
            if current_island.size > self.max_size:
                self.logger.info(
                    f"Exceeded maximum size ({self.max_size}) on island {island_id}, discarding weakest node(s)",
                    LogEvent.SOLVER,
                )
                while current_island.size > self.max_size:
                    current_island.remove_lowest()

        # Migration / Reset Logic
        if len(self._islands) > 1 and random.random() <= migration_prob:
            self.reset_islands()

    def reset_islands(self):
        """
        Resets the weaker half of islands and seeds them
        with nodes migrated from fitter islands.
        """
        if len(self._islands) < 2:  # Cannot reset if less than 2 islands
            self.logger.warning("Reset islands called with less than 2 islands. Skipping.", LogEvent.SOLVER)
            return

        self.logger.info("============ Resetting Islands ============", LogEvent.SOLVER)

        # Get island scores (best score per island)
        island_scores = []
        for island in self._islands:
            best_score = island.best_fitness_score  # Uses +/- inf for empty/None
            island_scores.append(best_score)

        # Add small noise to break ties during sorting
        noisy_scores = numpy.array(island_scores) + numpy.random.randn(len(self._islands)) * 1e-9

        # Sort islands by score (ascending for lower_is_better, descending otherwise)
        indices_sorted_by_score = numpy.argsort(noisy_scores)
        if not self.lower_is_better:
            indices_sorted_by_score = indices_sorted_by_score[::-1]

        num_islands_to_reset = len(self._islands) // 2
        if num_islands_to_reset == 0:  # Ensure at least one island is kept
            self.logger.info("Not enough islands to perform reset.", LogEvent.SOLVER)
            return

        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]

        self.logger.info(f"Resetting islands: {reset_islands_ids}", LogEvent.SOLVER)
        self.logger.info(f"Keeping islands: {keep_islands_ids}", LogEvent.SOLVER)

        for reset_island_id in reset_islands_ids:
            reset_island = self._islands[reset_island_id]
            # Keep only the single best node in the island being reset
            reset_island.only_keep_best()

            # Find a suitable founder island (must have > 1 node to donate)
            possible_founder_ids = [idx for idx in keep_islands_ids if self._islands[idx].size > 1]
            if not possible_founder_ids:
                self.logger.warning(
                    f"No suitable founder island with >1 node found for resetting island {reset_island_id}. Skipping migration.",
                    LogEvent.SOLVER,
                )
                continue  # Skip to next island to reset

            founder_island_id = numpy.random.choice(possible_founder_ids)
            founder_island = self._islands[founder_island_id]

            # Sample a node from the founder island (weighted by fitness, excluding the absolute best)
            candidates = []
            candidate_scores = []
            best_founder_score = founder_island.best_fitness_score
            for node in founder_island.nodes:
                score = node.metric.value
                # Handle None scores - treat as worst possible for sampling
                numeric_score = (
                    score if score is not None else (float("inf") if self.lower_is_better else float("-inf"))
                )
                # Exclude the absolute best node(s) from being migrated
                if numeric_score != best_founder_score:
                    candidates.append(node)
                    # Use the numeric score for weighting the sampling
                    candidate_scores.append(numeric_score)

            if not candidates:
                self.logger.warning(
                    f"Founder island {founder_island_id} has no non-best nodes to migrate. Skipping migration for island {reset_island_id}.",
                    LogEvent.SOLVER,
                )
                continue  # Skip to next island to reset

            # Perform weighted sampling (using normalized utility)
            # Note: utils.normalized expects scores where higher is better for probability
            normalized_candidate_scores = [self.get_normalized_score(s) for s in candidate_scores]
            sampling_weights = utils.normalized(
                normalized_candidate_scores, temp=1.0
            )  # Normalized scores are always higher_is_better
            if sum(sampling_weights) == 0:  # Avoid error if all weights are zero
                # Fallback to uniform sampling if weights are zero
                founder_node_to_migrate = random.choice(candidates)
                self.logger.warning(
                    "Sampling weights were all zero, falling back to uniform selection for migration.",
                    LogEvent.SOLVER,
                )
            else:
                founder_node_to_migrate = random.choices(candidates, weights=sampling_weights, k=1)[0]

            # Migrate the selected node
            self.logger.info(
                f"Migrating node {founder_node_to_migrate.id} from Island {founder_island_id} to Island {reset_island_id}",
                LogEvent.SOLVER,
            )
            reset_island.migrate_node(founder_node_to_migrate)

            # Remove the migrated node from the founder island
            founder_island.remove_node(founder_node_to_migrate)

    def sample_in_context(
        self, num_samples: Dict, temperature: float, crossover_prob: float
    ) -> Tuple[List[Node], int, str]:
        """
        Samples nodes for the next generation, selecting islands and then nodes based on fitness.
        Returns sampled nodes, the island they came from, and the selected operator ('improve' or 'crossover').
        """
        if not any(island.size > 0 for island in self._islands):
            self.logger.warning(
                "Sample in context called, but all islands are empty. Cannot sample. Back to drafting", LogEvent.SOLVER
            )
            return [], 0, "draft"

        # Calculate average fitness scores for island sampling
        # Use a default worst score for empty islands to give them zero probability
        island_avg_scores = []
        for island in self._islands:
            if island.size > 0:
                island_avg_scores.append(island.average_fitness_score)
            else:
                island_avg_scores.append(float("inf") if self.lower_is_better else float("-inf"))

        # Normalize scores for sampling probabilities (higher score = higher probability)
        # Since get_normalized_score handles lower_is_better and returns higher_is_better output,
        # we pass lower_is_better=False to utils.normalized
        normalized_island_avg_scores = [self.get_normalized_score(s) for s in island_avg_scores]
        island_sampling_weights = utils.normalized(
            normalized_island_avg_scores,
            temp=temperature,  # Apply temperature to avg scores for island selection
            # Normalized scores are always higher_is_better
        )

        # Ensure weights sum to 1 (handle potential all-zero case)
        if sum(island_sampling_weights) == 0:
            self.logger.warning("Island sampling weights are all zero. Falling back to uniform island selection.")
            island_sampling_weights = [1.0 / len(self._islands)] * len(self._islands)

        self.logger.debug(f"Island sampling weights: {island_sampling_weights}", LogEvent.SOLVER)

        # Determine operator and required number of samples
        operator = "improve" if random.random() >= crossover_prob else "crossover"
        num_in_context_samples = num_samples.get(operator, 1)  # Default to 1 if operator key missing
        if not any(island.size >= num_in_context_samples for island in self._islands) and operator == "crossover":
            fallback_samples = num_samples.get("improve", 1)
            if any(island.size >= fallback_samples for island in self._islands):
                self.logger.warning(
                    "Crossover was selected, but no island has enough nodes. Falling back to improve.",
                    LogEvent.SOLVER,
                )
                operator = "improve"
                num_in_context_samples = fallback_samples

        # Loop until a suitable island and samples are found
        sampled_island_id = -1
        sampled_island = None
        in_context_nodes = []
        attempts = 0
        max_attempts = len(self._islands) * 2  # Heuristic limit to prevent infinite loops

        while attempts < max_attempts:
            attempts += 1
            # STEP 1: Sample an island based on average fitness
            sampled_island_id = random.choices(range(len(self._islands)), weights=island_sampling_weights, k=1)[0]
            sampled_island = self._islands[sampled_island_id]

            # Check if the sampled island has enough nodes for the operator
            if sampled_island.size < num_in_context_samples:
                self.logger.debug(
                    f"Sampled island {sampled_island_id} size {sampled_island.size} < required {num_in_context_samples}. Resampling island.",
                    LogEvent.SOLVER,
                )
                continue  # Resample island

            # STEP 2: Sample nodes within the island (weighted by individual fitness)
            island_node_scores = sampled_island.fitness_scores
            normalized_node_scores = [self.get_normalized_score(s) for s in island_node_scores]
            node_sampling_weights = utils.normalized(
                normalized_node_scores, temperature
            )  # Normalized scores are always higher_is_better

            if sum(node_sampling_weights) == 0:
                self.logger.warning(
                    f"Node sampling weights on island {sampled_island_id} are zero. Falling back to uniform node selection."
                )
                # Use uniform sampling if weights are zero
                indices = numpy.random.choice(range(sampled_island.size), size=num_in_context_samples, replace=False)
            else:
                try:
                    indices = numpy.random.choice(
                        range(sampled_island.size),
                        p=node_sampling_weights,
                        size=num_in_context_samples,
                        replace=False,
                    )
                except ValueError as e:
                    # This might happen if weights don't sum to 1 due to float precision
                    self.logger.error(
                        f"Error sampling nodes on island {sampled_island_id}: {e}. Weights: {node_sampling_weights}. Falling back to uniform."
                    )
                    indices = numpy.random.choice(
                        range(sampled_island.size), size=num_in_context_samples, replace=False
                    )

            in_context_nodes = [sampled_island.nodes[i] for i in indices]
            self.logger.info(
                f"{operator.capitalize()} | Sampled island: {sampled_island_id}. Nodes: {[n.id for n in in_context_nodes]}",
                LogEvent.SOLVER,
            )
            return in_context_nodes, sampled_island_id, operator

        # If loop finishes without returning, something went wrong
        self.logger.error(f"Failed to sample context after {max_attempts} attempts. All islands might be too small.")
        raise RuntimeError(
            "Failed to sample context nodes after multiple attempts. Check island sizes or sampling logic."
        )


class Evolutionary(Solver):
    def __init__(self, cfg: EvolutionarySolverConfig, task_info):
        super().__init__(cfg, task_info=task_info)
        self.journal = Journal()
        self.data_preview: str | None = None

        self.task_desc = task_info[TASK_DESCRIPTION]
        self.prompt_context = {
            "task_description": str(task_info.get("task_description") or self.task_desc),
            "data_description": str(task_info.get("data_description") or ""),
            "public_system_prompt": str(task_info.get("public_system_prompt") or ""),
            "public_user_prompt": str(task_info.get("public_user_prompt") or ""),
        }
        self.lower_is_better = task_info.get("lower_is_better", None)
        assert self.lower_is_better is not None  # Ensure lower_is_better is set

        self.setup_operators()

        self.state = EvolutionaryState()

    def save_checkpoint(self):
        super().save_checkpoint()

        # Write the journal to a jsonl file
        journal_sd = self.journal.node_list()
        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        with open(journal_path, "w") as f:
            for node in journal_sd:
                f.write(json.dumps(node) + "\n")
        self.logger.info(f"Checkpoint saved to {journal_path}")

    def load_checkpoint(self):
        super().load_checkpoint()

        journal_path = Path(self.cfg.checkpoint_path) / "journal.jsonl"
        if not journal_path.exists():
            assert self.state.current_step == 0, (
                f"No journal found at {journal_path}, but the state was found. This is unexpected."
            )
            return

        self.logger.info(f"Found journal at {journal_path}. Loading...")
        # Load the journal
        with open(journal_path, "r") as f:
            journal_export = [json.loads(line) for line in f]
        self.journal = Journal.from_export_data({"nodes": journal_export})

    def setup_operators(self):
        """Setup operator LLMs."""

        # First we set up the LLMs
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        crossover_llm = GenericLLM(self.cfg.operators["crossover"])
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        self.draft_fn = partial(
            draft_op, draft_llm, self.cfg, self.memory_op, self.prompt_context
        )
        self.improve_fn = partial(
            improve_op, improve_llm, self.cfg, self.memory_op, self.prompt_context
        )
        self.debug_fn = partial(
            debug_op, debug_llm, self.cfg, self.debug_memory_op, self.prompt_context
        )
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)
        self.crossover_fn = partial(
            crossover_op, crossover_llm, self.cfg, self.memory_op, self.prompt_context
        )

    def _draft(self) -> Node:
        """
        Generate a new solution from scratch using the draft LLM operator.

        Uses the draft operator to create a new code solution based on the task description.
        The resulting code is packaged into a new Node object with relevant metadata.

        Returns:
            Node: A new node containing the drafted solution
        """
        plan, code, metrics = execute_op_plan_code(
            self.draft_fn,
            self.task_desc,
            self.journal,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            get_complextiy_level(self.root_node) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(
            plan=plan, code=code, parents=[self.root_node], operators_used=["draft"], operators_metrics=[metrics]
        )
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        self.logger.info(f"Draft Code: {code}")
        return node

    def _improve(self, parent_node: Node) -> Node:
        """
        Improve an existing solution using the improve LLM operator.

        Takes a parent node with a working solution and attempts to enhance it
        using the improve operator.

        Args:
            parent_node: The node containing the solution to improve

        Returns:
            Node: A new node containing the improved solution
        """
        plan, code, metrics = execute_op_plan_code(
            self.improve_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            get_complextiy_level(parent_node) if self.cfg.use_complexity else None,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        self.logger.info(f"Improve Code: {code}")
        return node

    def _debug(self, parent_node: Node) -> Node:
        """
        Debug a buggy solution using the debug LLM operator.

        Takes a parent node with a buggy solution and attempts to fix it
        using the debug operator, with access to the execution output/error.

        Args:
            parent_node: The node containing the buggy solution to debug

        Returns:
            Node: A new node containing the debugged solution
        """
        plan, code, metrics = execute_op_plan_code(
            self.debug_fn,
            self.task_desc,
            self.journal,
            parent_node,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics])
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        self.logger.info(f"Debug Code: {code}")
        return node

    def _analyze(self, node: Node) -> Union[str, dict]:
        """
        Analyze a node's execution results using the analyze LLM operator.

        Processes the task description, code, and execution output to determine
        if the solution is buggy and to extract metrics when available.

        Args:
            node: The node to analyze

        Returns:
            Union[str, dict]: Analysis results, either as a string or dictionary
        """
        analysis, metrics = self.analyze_fn(self.task_desc, node)
        node.operators_used.append("analysis")
        node.operators_metrics.append(metrics)
        self.logger.info(f"Node Analysis Performed - Metrics: {metrics}")
        return analysis

    def _crossover(self, parent_node1: Node, parent_node2: Node) -> Node:
        plan, code, metrics = execute_op_plan_code(
            self.crossover_fn,
            self.task_desc,
            self.journal,
            parent_node1,
            parent_node2,
            self.state.current_step,
            self.cfg.time_limit_secs - self.state.running_time,
            self.data_preview,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = Node(
            plan=plan,
            code=code,
            parents=[parent_node1, parent_node2],
            operators_used=["crossover"],
            operators_metrics=[metrics],
        )
        self.logger.info(f"Crossover Node Created - Metrics: {metrics}")
        self.logger.info(f"Crossover Code: {code}")
        return node

    def update_data_preview(self, state):
        """
        Generate a data preview to provide context for the LLM operators.

        Creates a small preview of the data (head, shapes, etc.) that can be used
        to help the LLM understand the data structure when generating solutions.

        Args:
            state: The current solver state containing the interpreter
        """
        assert "solver_interpreter" in state, (
            "For generating data previews, the solver needs access to an interpreter."
        )

        if not self.cfg.data_preview:
            self.data_preview = ""
            return

        self.logger.debug("Generating data preview")
        if state["solver_interpreter"].local:
            self.data_preview = data_preview.generate(state["solver_interpreter"].working_dir)
        else:
            import inspect

            path = Path(inspect.getsourcefile(data_preview))
            script = path.read_text()
            code = f"{script}\nprint(generate(Path('.').resolve()))"
            exec_result = state["solver_interpreter"].run(code, include_exec_time=False)
            self.data_preview = "\n".join(exec_result.term_out)
        self.logger.debug("Data preview generated")

    def linear_decay(self, iteration: int):
        """defines a temperature schedule for sampling of islands and individuals"""
        initial_temp = self.cfg.initial_temp
        final_temp = self.cfg.final_temp
        num_generations = self.cfg.num_generations
        return initial_temp - (initial_temp - final_temp) * iteration / num_generations

    def debug_cycle(self, state, task, buggy_node: Node):
        current_debug_node = buggy_node  # Start with the initial buggy node
        debug_path = [current_debug_node]  # Path includes the initial buggy node
        debug_depth = self.cfg.max_debug_depth
        fixed_metric = None
        # We set the initial debug cycle time to the execution time of the first buggy node
        total_debug_cycle_time = current_debug_node.exec_time if current_debug_node.exec_time is not None else 0

        # We run the debug cycle for a number of times
        # or until time runs out, whichever comes first
        for _ in range(debug_depth):
            # Create the debugged node
            fixed_node_attempt = self._debug(current_debug_node)
            # Evaluate the attempt
            try:
                state, eval_result = task.step_task(state, extract_code(fixed_node_attempt.code))
                self.parse_eval_result(node=fixed_node_attempt, eval_result=eval_result)
                debug_path.append(fixed_node_attempt)
                current_debug_node = fixed_node_attempt  # Update the node for the next iteration
            except Exception as e:
                self.logger.error(f"Error during debug step execution/parsing: {e}", LogEvent.SOLVER)
                break  # Break the debug cycle on execution/parsing error

            # Break if we have a fixed metric - i.e. the solution is no longer buggy
            if not current_debug_node.is_buggy:
                fixed_metric = current_debug_node.metric.value
                self.logger.info(
                    f"Debug cycle successful for node {buggy_node.id}, final node {current_debug_node.id} has metric: {fixed_metric}",
                    LogEvent.SOLVER,
                )
                break

            # Add exec time if available
            if current_debug_node.exec_time is not None:
                total_debug_cycle_time += current_debug_node.exec_time

            # or if the debug time is reached we break
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                self.logger.info(
                    f"Debug cycle time exceeded: {total_debug_cycle_time} seconds for initial node {buggy_node.id}",
                    LogEvent.SOLVER,
                )
                break

        # Return state, the full path, and the metric of the final node (or None if not fixed)
        return state, debug_path, fixed_metric

    def parse_eval_result(self, node: Node, eval_result: Dict[str, Any]):
        """
        Parse evaluation results and update the node accordingly.

        Processes the execution output, extracts metrics, determines if the solution
        is buggy, and updates the node with this information. Also applies the analysis
        operator to get additional insights about the solution.

        Args:
            node: The node to update with evaluation results
            eval_result: Dictionary containing evaluation results from task execution
        """
        self.logger.debug(f"Parsing execution results for node {node.id}")

        # Safely ensure we have eval_result
        if isinstance(eval_result, dict):
            assert EXECUTION_OUTPUT in eval_result
        else:
            raise ValueError(f"Unexpected eval_result type: {type(eval_result)}")

        # Absorb the execution output into the node
        node.absorb_exec_result(eval_result[EXECUTION_OUTPUT])

        # Safely perform the analyze operation
        try:
            response = self._analyze(node)
        except Exception as e:
            self.logger.error(f"Error during analysis operator: {str(e)}")
            response = {}

        # Parse response to dictionary
        # If the response is a string, we try to parse it into a dictionary
        response = parse_json_output(response)

        # Validate it's actually a dictionary
        if not isinstance(response, dict):
            self.logger.warning(f"Parsed response is not a dictionary: {type(response)}")
            response = {}

        # If the response is empty, we return a default dictionary
        if len(response) == 0:
            response = {"metric": None, "summary": "", "is_bug": True}
        else:
            if "metric" not in response:
                response["metric"] = None
            if "summary" not in response:
                response["summary"] = ""
            if "is_bug" not in response:
                if response["metric"] is not None:
                    response["is_bug"] = False
                else:
                    response["is_bug"] = True

        # If the metric isn't a float or int then fill the metric with the worst metric
        if not isinstance(response["metric"], (float, int)):
            response["metric"] = None

        # If a validation fitness value is provided (not test fitness) from the task
        # we replace the validation metric with it.
        if eval_result.get(VALIDATION_FITNESS, None) is not None:
            response["metric"] = float(eval_result[VALIDATION_FITNESS])
            response["is_bug"] = False

        # Store the analysis summary
        node.analysis = str(response["summary"]).strip()

        # Extract potential auxliary evaluation information to store
        # in the node. This is more for logging purposes.
        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {})

        if self.cfg.use_test_score:
            test_score = aux_eval_info.get("score", None)
            aux_eval_info["validation_score"] = response["metric"]
            response["metric"] = test_score
            self.logger.info(f"Using Test score: {test_score}")

        # Determine if solution is valid
        # If the task does not return this key, we assume the solution is valid
        valid_solution = eval_result.get(VALID_SOLUTION, True)
        validity_feedback = eval_result.get(VALID_SOLUTION_FEEDBACK, None)
        if validity_feedback is not None:
            aux_eval_info["validity_feedback"] = validity_feedback
            validity_feedback = f"\n\n submission.csv Grader Feedback: {validity_feedback}"
            node._term_out.append(validity_feedback)
        else:
            aux_eval_info["validity_feedback"] = "submission grader feedback not available"

        node.is_buggy = (
            response["is_bug"] or (not node.exit_code == 0) or (response["metric"] is None) or (not valid_solution)
        )

        if node.is_buggy:
            node.metric = WorstMetricValue(info=aux_eval_info)
            self.logger.debug(f"Node {node.id} marked as buggy")
        else:
            node.metric = MetricValue(response["metric"], maximize=not self.lower_is_better, info=aux_eval_info)
            self.logger.debug(f"Node {node.id} metric: {response['metric']}")

    def create_root_node(self):
        self.root_node = Node(
            code="",
            plan="",
            analysis="",
            metric=WorstMetricValue(maximize=not self.lower_is_better),
            is_buggy=True,
        )
        self.root_node.absorb_exec_result(None)
        self.journal.append(self.root_node)
        self.log_journal()
        self.state.current_step += 1

    def _build_solution_database_from_journal(self) -> SolutionsDatabase:
        solution_database = SolutionsDatabase(
            num_islands=self.cfg.num_islands,
            max_size=self.cfg.max_island_size,
            lower_is_better=self.lower_is_better,
            logger=self.logger,
        )
        solution_nodes = [
            node
            for node in self.journal.nodes
            if not self.journal.is_root_node(node)
            and not node.is_buggy
            and node.metric is not None
            and node.metric.value is not None
        ]
        if not solution_nodes:
            return solution_database

        # Historical SFT AIRA Evo runs use a single island. For multi-island
        # checkpoints that predate database persistence, place recovered nodes
        # on island 0 rather than discarding the searchable context.
        solution_database.seed_islands_with_nodes(
            solution_nodes,
            [0 for _ in solution_nodes],
        )
        self.logger.info(
            f"Recovered {len(solution_nodes)} solution nodes from checkpoint journal",
            LogEvent.SOLVER,
        )
        return solution_database

    def search(self, task, state) -> Optional[str]:
        if not self.journal.nodes:
            self.create_root_node()
        else:
            self.root_node = self.journal.nodes[0]
        self.logger.info("Starting evolutionary search", LogEvent.SOLVER)

        # database of solutions: load all islands if it > 0, else initialize empty islands
        solution_database = self._build_solution_database_from_journal()

        # define a schedule for temperature of sampling
        temp_scheduler = self.linear_decay

        for generation_id in range(self.state.current_generation, self.cfg.num_generations):
            # Measure state time of the generation
            start_time = time.monotonic()

            # fix the temperature for sampling
            temperature = temp_scheduler(iteration=generation_id)
            self.logger.info(
                f"\n========= Generation {generation_id} | temperature: {round(temperature, 2)} ==========",
                LogEvent.SOLVER,
            )

            solution_nodes = []
            island_ids = []
            counter_ids = []
            for counter_id in range(self.cfg.individuals_per_generation):
                if generation_id == 0:  # initially, uniformly populate the islands
                    island_id = random.choice(range(solution_database.num_islands))
                    create_node_fn = self._draft
                    in_context_nodes = []
                else:  # gen_id > 0: start the evolutionary process
                    in_context_nodes, island_id, operator = solution_database.sample_in_context(
                        self.cfg.few_shot,
                        temperature,
                        (0 if generation_id < self.cfg.num_generations_till_crossover else self.cfg.crossover_prob),
                    )  # weighted sampling of islands and corresponding individuals
                    if operator == "improve":
                        create_node_fn = self._improve
                    elif operator == "draft":
                        island_id = random.choice(range(solution_database.num_islands))
                        create_node_fn = self._draft
                        in_context_nodes = []
                    else:
                        create_node_fn = self._crossover

                island_ids.append(island_id)

                self.logger.info(
                    f"Creating node for individual {counter_id} in generation {generation_id}", LogEvent.SOLVER
                )

                child_node = create_node_fn(*in_context_nodes)
                state, eval_result = task.step_task(state, extract_code(child_node.code))
                self.parse_eval_result(child_node, eval_result)
                # if the node is buggy, we run a debug cycle
                # and add the fixed node to the generation
                # if the node is not buggy, we add it to the generation
                if not child_node.is_buggy:
                    self.journal.append(child_node)
                    self.log_journal()
                    self.state.current_step += 1
                    solution_nodes.append(child_node)
                    counter_ids.append(counter_id)
                else:
                    self.logger.info(f"Node {child_node.id} was buggy, entering debug cycle.", LogEvent.SOLVER)
                    state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                    # Add the debug path to the journal
                    for n in debug_path:
                        self.journal.append(n)
                        self.log_journal()
                        self.state.current_step += 1
                    if fixed_metric is not None:
                        fixed_node = debug_path[-1]  # Get the last node (the fixed one)
                        self.logger.info(
                            f"Node {child_node.id} was fixed by node {fixed_node.id}, adding fixed node to generation.",
                            LogEvent.SOLVER,
                        )
                        # Add only the final fixed node to the generation
                        solution_nodes.append(fixed_node)
                        counter_ids.append(counter_id)
                    else:
                        self.logger.info(
                            f"Node {child_node.id} could not be fixed by debug cycle, discarding individual {counter_id}.",
                            LogEvent.SOLVER,
                        )

            # store individuals solutions only if it improves overall island fitness
            # for initialization, we don't use this step
            if generation_id > 0:
                solution_database.add_nodes_to_islands(
                    solution_nodes,
                    island_ids,
                    (0 if generation_id < self.cfg.num_generations_till_migration else self.cfg.migration_prob),
                )
            else:  # initialization step (generation = 0)
                solution_database.seed_islands_with_nodes(
                    solution_nodes,
                    island_ids,
                )

            self.state.running_time += time.monotonic() - start_time
            self.logger.info(
                f"Step {self.state.current_step} | Generation {self.state.current_generation}: Time taken for generation: {self.state.running_time:.3f} seconds"
            )

            # Update the state with the current generation
            self.state.current_generation += 1

            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs or self.state.current_step >= self.cfg.step_limit:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        return state, self.journal.get_best_node().code

    def log_journal(self):
        # Get the current best node in the tree.
        best_node = self.journal.get_best_node()
        best_node_step = 0 if best_node is None else best_node.step

        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

    def __call__(self, task, state):
        # Possibly generate data preview first
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        state, solution = self.search(task, state)

        # Export results at the end of the search process
        export_search_results(self.cfg, self.journal, self.logger, "EVO")

        # Get the best node for returning
        best_node = self.journal.get_best_node()

        # Return the best node
        if best_node:
            return state, best_node.code, best_node
        else:
            self.logger.info("No suitable code found after all generations.", LogEvent.SOLVER)
            return state, None, None
