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

from __future__ import annotations

import asyncio
import json
import random
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy

import dojo.core.solvers.utils.search_utils as utils
from dojo.core.solvers.base import Solver
from dojo.core.solvers.operators.analyze import analyze_op
from dojo.core.solvers.operators.core import execute_op_plan_code
from dojo.core.solvers.operators.crossover import crossover_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.rich_memory_summary import rich_memory_summary_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.utils import data_preview
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt
from dojo.core.solvers.utils.search_exporter import export_search_results
from dojo.solvers.utils import get_complextiy_level
from dojo.solvers.evo.experience import (
    collect_operator_memory_nodes,
    compute_parent_utilities,
    prompt_memory_enabled,
    prompt_memory_max_related_cards,
    prompt_memory_operator_k,
    prompt_memory_sibling_rank_weights,
)
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
from tts_search.airaevo_async_resources import (
    WorkerSpec,
    build_worker_specs,
    fetch_router_idle_worker_count,
    resolve_async_sandbox_urls,
    resolve_async_worker_count,
    resolve_execution_mode,
)


@dataclass(frozen=True)
class AsyncWorkItem:
    attempt_id: int
    generation_id: int
    temperature: float
    island_id: int
    operator: str
    parent_nodes: list[Node]
    parent_selection_trace: dict[str, Any] | None
    worker: WorkerSpec


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
        experience_config: Optional[dict] = None,
    ):
        self.num_islands = num_islands
        self.max_size = max_size
        self.lower_is_better = lower_is_better
        self.logger = logger
        self.experience_config = dict(experience_config or {})
        self.last_parent_selection: Optional[dict[str, Any]] = None
        self._islands: List[Island] = []
        self.global_min_fitness = float("inf")
        self.global_max_fitness = float("-inf")

        # Initialize empty islands.
        self._islands = [
            Island(island_id, [], self.lower_is_better, self.logger) for island_id in range(self.num_islands)
        ]

    @property
    def has_nodes(self) -> bool:
        return any(island.size > 0 for island in self._islands)

    def has_island_with_size(self, size: int) -> bool:
        return any(island.size >= size for island in self._islands)

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

    def request_fresh_draft(self, reason: str) -> None:
        """Make the next non-initial generation explore a clean candidate."""
        self._forced_fresh_draft_reason = str(reason)

    def sample_in_context(
        self,
        num_samples: Dict,
        temperature: float,
        crossover_prob: float,
        fresh_draft_prob: float = 0.0,
    ) -> Tuple[List[Node], int, str]:
        """
        Samples nodes for the next generation, selecting islands and then nodes based on fitness.
        Returns sampled nodes, the island they came from, and the selected operator ('improve' or 'crossover').
        """
        if not any(island.size > 0 for island in self._islands):
            self.logger.warning(
                "Sample in context called, but all islands are empty. Cannot sample. Back to drafting", LogEvent.SOLVER
            )
            self.last_parent_selection = None
            return [], 0, "draft"

        forced_reason = getattr(self, "_forced_fresh_draft_reason", None)
        fresh_probability = min(1.0, max(0.0, float(fresh_draft_prob)))
        if forced_reason or (fresh_probability > 0 and random.random() < fresh_probability):
            reason = str(forced_reason or "fresh_draft_probability")
            self._forced_fresh_draft_reason = None
            island_id = random.randrange(len(self._islands))
            self.last_parent_selection = {
                "enabled": False,
                "operator": "draft",
                "island_id": island_id,
                "reason": reason,
                "fresh_draft_prob": fresh_probability,
            }
            self.logger.info(
                f"Starting a fresh draft on island {island_id}: {reason}",
                LogEvent.SOLVER,
            )
            return [], island_id, "draft"

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
        operator_fallback_reason = None
        eligible_island_ids = [
            island_id
            for island_id, island in enumerate(self._islands)
            if island.size >= num_in_context_samples
        ]
        if not eligible_island_ids and operator == "crossover":
            improve_samples = num_samples.get("improve", 1)
            improve_island_ids = [
                island_id
                for island_id, island in enumerate(self._islands)
                if island.size >= improve_samples
            ]
            if improve_island_ids:
                self.logger.info(
                    "No island has enough parents for crossover; falling back to improve.",
                    LogEvent.SOLVER,
                )
                operator = "improve"
                num_in_context_samples = improve_samples
                eligible_island_ids = improve_island_ids
                operator_fallback_reason = "crossover_parent_shortage"
        if not eligible_island_ids:
            island_id = random.randrange(len(self._islands))
            self.last_parent_selection = {
                "enabled": False,
                "operator": "draft",
                "island_id": island_id,
                "reason": "no_eligible_parent_context",
            }
            self.logger.info(
                "No island has enough eligible parents; starting a fresh draft.",
                LogEvent.SOLVER,
            )
            return [], island_id, "draft"
        self.last_parent_selection = None

        # Loop until a suitable island and samples are found
        sampled_island_id = -1
        sampled_island = None
        in_context_nodes = []
        attempts = 0
        max_attempts = len(self._islands) * 2  # Heuristic limit to prevent infinite loops

        while attempts < max_attempts:
            attempts += 1
            # STEP 1: Sample an island based on average fitness
            eligible_weights = [island_sampling_weights[island_id] for island_id in eligible_island_ids]
            if sum(eligible_weights) == 0:
                eligible_weights = [1.0 / len(eligible_island_ids)] * len(eligible_island_ids)
            sampled_island_id = random.choices(
                eligible_island_ids,
                weights=eligible_weights,
                k=1,
            )[0]
            sampled_island = self._islands[sampled_island_id]

            # Check if the sampled island has enough nodes for the operator
            if sampled_island.size < num_in_context_samples:
                self.logger.debug(
                    f"Sampled island {sampled_island_id} size {sampled_island.size} < required {num_in_context_samples}. Resampling island.",
                    LogEvent.SOLVER,
                )
                continue  # Resample island

            # STEP 2: Sample nodes within the island (weighted by individual fitness)
            parent_selection_trace: dict[str, Any] = {
                "enabled": False,
                "operator": operator,
                "island_id": sampled_island_id,
                "temperature": float(temperature),
                "num_in_context_samples": int(num_in_context_samples),
                "candidates": [],
                "selected_node_ids": [],
            }
            if operator_fallback_reason:
                parent_selection_trace["operator_fallback_reason"] = operator_fallback_reason
            parent_selection_cfg = dict(self.experience_config.get("parent_selection") or {})
            component_normalization_cfg = dict(parent_selection_cfg.get("component_normalization") or {})
            experience_parent_selection_enabled = bool(self.experience_config.get("enabled", False)) and bool(
                parent_selection_cfg.get("enabled", True)
            )
            if experience_parent_selection_enabled:
                previous_cards = [
                    card
                    for island in self._islands
                    for node in island.nodes
                    if isinstance((card := getattr(node, "experience_card", None)), dict)
                ]
                utility_items = compute_parent_utilities(
                    sampled_island.nodes,
                    lower_is_better=self.lower_is_better,
                    previous_cards=previous_cards,
                    weights=dict(parent_selection_cfg.get("weights") or {}),
                    component_normalization=component_normalization_cfg,
                    temperature=temperature,
                )
                node_sampling_weights = [item["probability"] for item in utility_items]
                parent_selection_trace.update(
                    {
                        "enabled": True,
                        "weights": dict(parent_selection_cfg.get("weights") or {}),
                        "component_normalization": component_normalization_cfg,
                        "candidates": utility_items,
                    }
                )
                self.logger.debug(
                    f"Experience parent sampling weights on island {sampled_island_id}: {node_sampling_weights}",
                    LogEvent.SOLVER,
                )
            else:
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
            parent_selection_trace["selected_node_ids"] = [node.id for node in in_context_nodes]
            self.last_parent_selection = parent_selection_trace
            self.logger.info(
                f"{operator.capitalize()} | Sampled island: {sampled_island_id}. Nodes: {[n.id for n in in_context_nodes]}",
                LogEvent.SOLVER,
            )
            return in_context_nodes, sampled_island_id, operator

        # This should be unreachable after sampling only eligible islands, but a
        # fresh draft preserves the search if a concurrent mutation changes state.
        island_id = random.randrange(len(self._islands))
        self.last_parent_selection = {
            "enabled": False,
            "operator": "draft",
            "island_id": island_id,
            "reason": "parent_sampling_exhausted",
        }
        self.logger.warning(
            f"Parent sampling exhausted after {max_attempts} attempts; starting a fresh draft.",
            LogEvent.SOLVER,
        )
        return [], island_id, "draft"

    def sample_in_context_with_trace(
        self,
        num_samples: Dict,
        temperature: float,
        crossover_prob: float,
        fresh_draft_prob: float = 0.0,
    ) -> tuple[list[Node], int, str, dict[str, Any] | None]:
        nodes, island_id, operator = self.sample_in_context(
            num_samples,
            temperature,
            crossover_prob,
            fresh_draft_prob,
        )
        return nodes, island_id, operator, self.last_parent_selection


class Evolutionary(Solver):
    def __init__(self, cfg: EvolutionarySolverConfig, task_info):
        super().__init__(cfg, task_info=task_info)
        self.journal = Journal()
        self.data_preview: str | None = None

        self.task_desc = task_info[TASK_DESCRIPTION]
        raw_task_description = str(
            task_info.get("naturebench_raw_task_description") or self.task_desc
        )
        task_family_guidance = str(task_info.get("task_family_guidance") or "").strip()
        if task_family_guidance:
            raw_task_description = (
                f"{raw_task_description}\n\nTask-Family Execution Guidance:\n"
                f"- {task_family_guidance}"
            )
        self.prompt_context = {
            "task_description": raw_task_description,
            "data_description": str(task_info.get("data_description") or ""),
            "visible_data_analysis": str(
                task_info.get("visible_data_analysis") or ""
            ),
            "public_system_prompt": str(task_info.get("public_system_prompt") or ""),
            "public_user_prompt": str(task_info.get("public_user_prompt") or ""),
        }
        self.lower_is_better = task_info.get("lower_is_better", None)
        assert self.lower_is_better is not None  # Ensure lower_is_better is set
        self.prompt_context["lower_is_better"] = bool(self.lower_is_better)

        self.setup_operators()

        self.state = EvolutionaryState()
        self._rich_summary_locks: dict[str, threading.Lock] = {}
        self._rich_summary_locks_guard = threading.Lock()

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
        rich_memory_summary_llm = (
            GenericLLM(self.cfg.operators["rich_memory_summary"])
            if prompt_memory_enabled(self.cfg)
            and "rich_memory_summary" in self.cfg.operators
            else None
        )

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
        self.rich_memory_summary_fn = (
            partial(rich_memory_summary_op, rich_memory_summary_llm, self.cfg)
            if rich_memory_summary_llm is not None
            else None
        )
        self.crossover_fn = partial(
            crossover_op, crossover_llm, self.cfg, self.memory_op, self.prompt_context
        )

    def _rich_memory_cache_dir(self) -> Path:
        checkpoint_path = Path(str(self.cfg.checkpoint_path))
        return checkpoint_path.parent / "rich_summaries"

    def _rich_memory_cache_path(self, node: Node) -> Path:
        return self._rich_memory_cache_dir() / f"{node.id}.json"

    def _rich_summary_lock_for_node(self, node: Node) -> threading.Lock:
        node_id = str(getattr(node, "id", "") or id(node))
        with self._rich_summary_locks_guard:
            lock = self._rich_summary_locks.get(node_id)
            if lock is None:
                lock = threading.Lock()
                self._rich_summary_locks[node_id] = lock
            return lock

    def _load_cached_rich_summary(self, node: Node) -> dict[str, str] | None:
        cache_path = self._rich_memory_cache_path(node)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        summary = payload.get("rich_summary") if isinstance(payload, dict) else None
        if isinstance(summary, dict) and summary.get("method_overview") and summary.get("parent_comparison_experience"):
            return {
                "method_overview": str(summary["method_overview"]),
                "parent_comparison_experience": str(summary["parent_comparison_experience"]),
            }
        return None

    def _store_rich_summary(self, node: Node, summary: dict[str, str]) -> None:
        node.rich_summary = summary
        card = getattr(node, "experience_card", None)
        if isinstance(card, dict):
            card["rich_summary"] = summary
        cache_path = self._rich_memory_cache_path(node)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "node_id": node.id,
                        "parent_node_ids": [parent.id for parent in list(node.parents or [])],
                        "rich_summary": summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            self.logger.warning(f"Failed to write rich memory cache for node {node.id}: {exc}")

    def _normalize_rich_summary(self, response: Any) -> dict[str, str] | None:
        parsed = parse_json_output(response)
        if not isinstance(parsed, dict):
            return None
        method_overview = str(parsed.get("method_overview") or "").strip()
        parent_experience = str(parsed.get("parent_comparison_experience") or "").strip()
        if not method_overview or not parent_experience:
            return None
        return {
            "method_overview": method_overview,
            "parent_comparison_experience": parent_experience,
        }

    def _ensure_node_rich_summary(self, node: Node) -> None:
        if node is None:
            return
        with self._rich_summary_lock_for_node(node):
            self._ensure_node_rich_summary_unlocked(node)

    def _ensure_node_rich_summary_unlocked(self, node: Node) -> None:
        existing = getattr(node, "rich_summary", None)
        if isinstance(existing, dict) and existing.get("method_overview") and existing.get("parent_comparison_experience"):
            card = getattr(node, "experience_card", None)
            if isinstance(card, dict):
                card["rich_summary"] = existing
            return

        card = getattr(node, "experience_card", None)
        if isinstance(card, dict):
            card_summary = card.get("rich_summary")
            if (
                isinstance(card_summary, dict)
                and card_summary.get("method_overview")
                and card_summary.get("parent_comparison_experience")
            ):
                node.rich_summary = card_summary
                return

        cached_summary = self._load_cached_rich_summary(node)
        if cached_summary is not None:
            self._store_rich_summary(node, cached_summary)
            return

        if self.rich_memory_summary_fn is None:
            return

        parent_node = list(node.parents or [None])[0]
        try:
            result = self.rich_memory_summary_fn(self.task_desc, node, parent_node)
            if isinstance(result, tuple) and len(result) == 2:
                response, metrics = result
            else:
                response, metrics = result, {}
            node.operators_used.append("rich_memory_summary")
            node.operators_metrics.append(metrics)
        except Exception as exc:
            self.logger.warning(f"Rich memory summary failed for node {node.id}: {exc}")
            return

        summary = self._normalize_rich_summary(response)
        if summary is None:
            self.logger.warning(f"Rich memory summary returned invalid JSON for node {node.id}")
            return
        self._store_rich_summary(node, summary)

    def _prepare_operator_rich_memory(
        self,
        operator: str,
        parent_nodes: list[Node],
        *,
        current_node: Node | None = None,
    ) -> None:
        if not prompt_memory_enabled(self.cfg):
            return
        sections = collect_operator_memory_nodes(
            operator,
            parent_nodes,
            journal=self.journal,
            lower_is_better=bool(self.lower_is_better),
            current_node=current_node,
            ancestor_k=prompt_memory_operator_k(
                self.cfg,
                operator,
                "ancestor_k",
                3 if str(operator).lower() == "improve" else 2,
            ),
            sibling_k=prompt_memory_operator_k(
                self.cfg,
                operator,
                "sibling_k",
                3 if str(operator).lower() == "improve" else 2,
            ),
            sibling_rank_weights=prompt_memory_sibling_rank_weights(self.cfg),
            max_related_cards=prompt_memory_max_related_cards(self.cfg, operator),
        )
        memory_nodes = []
        for key in [
            "primary",
            "vertical",
            "horizontal",
            "parent_1_vertical",
            "parent_1_horizontal",
            "parent_2_vertical",
            "parent_2_horizontal",
            "debug_related",
        ]:
            memory_nodes.extend(sections.get(key) or [])
        seen_node_ids = set()
        deduped_memory_nodes = []
        for memory_node in memory_nodes:
            node_id = str(getattr(memory_node, "id", "") or "")
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            deduped_memory_nodes.append(memory_node)
        for memory_node in deduped_memory_nodes:
            self._ensure_node_rich_summary(memory_node)

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
        self._prepare_operator_rich_memory("improve", [parent_node])
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
        self._prepare_operator_rich_memory("debug", [parent_node], current_node=parent_node)
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
        self._prepare_operator_rich_memory("crossover", [parent_node1, parent_node2])
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
        temperature = (
            initial_temp
            - (initial_temp - final_temp) * max(0, iteration) / num_generations
        )
        return min(
            max(temperature, min(initial_temp, final_temp)),
            max(initial_temp, final_temp),
        )

    def _experience_enabled(self) -> bool:
        experience_cfg = getattr(self.cfg, "experience", {}) or {}
        if hasattr(experience_cfg, "get"):
            return bool(experience_cfg.get("enabled", False))
        return bool(getattr(experience_cfg, "enabled", False))

    def _append_evaluated_node(self, node: Node) -> None:
        """Append an evaluated node and immediately mirror/log it if not already present."""
        node_id = str(getattr(node, "id", "") or "")
        if any(str(getattr(existing, "id", "") or "") == node_id for existing in self.journal.nodes):
            return
        self.journal.append(node)
        self.log_journal()
        self.state.current_step += 1

    @staticmethod
    def _task_budget_exempt_seconds(task: Any) -> float:
        getter = getattr(task, "get_budget_exempt_wait_seconds", None)
        if not callable(getter):
            return 0.0
        try:
            value = float(getter())
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    @staticmethod
    def _repeated_debug_failure_reason(debug_path: List[Node]) -> str | None:
        """Identify repeated failures where another repair is unlikely to add information."""
        if len(debug_path) < 2:
            return None

        def failure_signature(node: Node) -> str | None:
            metric = getattr(node, "metric", None)
            info = dict(getattr(metric, "info", None) or {})
            preflight_category = str(info.get("preflight_category") or "").strip()
            if preflight_category:
                return f"preflight:{preflight_category}"
            status = str(info.get("status") or "").strip().lower()
            status_code = info.get("status_code")
            if status == "timeout" or status_code == 504:
                return "timeout"
            feedback = str(
                info.get("feedback")
                or info.get("validity_feedback")
                or ""
            ).lower()
            if "modulenotfounderror" in feedback or "importerror" in feedback:
                return "import_error"
            if "out of memory" in feedback or "memoryerror" in feedback:
                return "memory_error"
            return None

        previous = failure_signature(debug_path[-2])
        current = failure_signature(debug_path[-1])
        if previous and previous == current:
            return f"repeated_{current}"
        return None

    def debug_cycle(self, state, task, buggy_node: Node):
        if self._experience_enabled():
            self._append_evaluated_node(buggy_node)
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
                if self._experience_enabled():
                    self._append_evaluated_node(fixed_node_attempt)
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

    def _new_solution_database(self) -> SolutionsDatabase:
        return SolutionsDatabase(
            num_islands=self.cfg.num_islands,
            max_size=self.cfg.max_island_size,
            lower_is_better=self.lower_is_better,
            logger=self.logger,
            experience_config=dict(getattr(self.cfg, "experience", {}) or {}),
        )

    def _seed_solution_database_from_journal(self, solution_database: SolutionsDatabase) -> None:
        good_nodes = [
            node
            for node in self.journal.nodes
            if not self.journal.is_root_node(node) and not bool(getattr(node, "is_buggy", True))
        ]
        if not good_nodes:
            return
        island_ids = [
            index % solution_database.num_islands
            for index, _ in enumerate(good_nodes)
        ]
        solution_database.add_nodes_to_islands(good_nodes, island_ids, migration_prob=0.0)

    def _restore_solution_database_from_journal(self) -> SolutionsDatabase:
        """Create the islands database and restore all resumable journal nodes."""
        solution_database = self._new_solution_database()
        self.solution_database = solution_database
        self._seed_solution_database_from_journal(solution_database)
        return solution_database

    def _async_worker_count(self) -> int:
        return resolve_async_worker_count(getattr(self.cfg, "async_workers", 1))

    def _execution_mode(self) -> str:
        return resolve_execution_mode(getattr(self.cfg, "execution_mode", "generation"))

    def _async_attempt_limit(self) -> int:
        return max(
            1,
            int(self.cfg.num_generations)
            * max(1, int(self.cfg.individuals_per_generation)),
        )

    def _async_sandbox_urls(self, task) -> list[str]:
        sandbox_cfg = dict(getattr(task, "cfg", {}).get("sandbox", {}) or {})
        fallback_url = str(sandbox_cfg.get("base_url") or "")
        return resolve_async_sandbox_urls(
            getattr(self.cfg, "async_sandbox_urls", None),
            fallback_url=fallback_url,
        )

    def _should_stop_async_search(
        self,
        task,
        *,
        include_attempt_limit: bool = True,
    ) -> bool:
        if getattr(task, "stop_requested", False):
            return True
        wall_elapsed = max(0.0, time.monotonic() - self.start_time)
        exempt_wait = self._task_budget_exempt_seconds(task)
        effective_elapsed = max(0.0, wall_elapsed - exempt_wait)
        if effective_elapsed >= float(self.cfg.time_limit_secs):
            return True
        max_wall_time = float(
            getattr(self.cfg, "max_wall_time_secs", 0.0) or 0.0
        )
        if max_wall_time > 0 and wall_elapsed >= max_wall_time:
            return True
        if int(self.state.current_step) >= int(self.cfg.step_limit):
            return True
        return (
            include_attempt_limit
            and int(self.state.current_generation) >= self._async_attempt_limit()
        )

    def _next_async_work_item(
        self,
        *,
        solution_database: SolutionsDatabase,
        worker: WorkerSpec,
    ) -> AsyncWorkItem:
        attempt_id = int(self.state.current_generation)
        self.state.current_generation += 1
        generation_id = attempt_id // max(
            1,
            int(self.cfg.individuals_per_generation),
        )
        temperature = self.linear_decay(iteration=generation_id)

        if not solution_database.has_nodes:
            return AsyncWorkItem(
                attempt_id=attempt_id,
                generation_id=generation_id,
                temperature=temperature,
                island_id=random.choice(range(solution_database.num_islands)),
                operator="draft",
                parent_nodes=[],
                parent_selection_trace=None,
                worker=worker,
            )

        crossover_prob = 0 if generation_id < self.cfg.num_generations_till_crossover else self.cfg.crossover_prob
        crossover_parent_count = int(dict(self.cfg.few_shot).get("crossover", 2))
        if not solution_database.has_island_with_size(crossover_parent_count):
            crossover_prob = 0

        try:
            parent_nodes, island_id, operator, parent_selection_trace = solution_database.sample_in_context_with_trace(
                self.cfg.few_shot,
                temperature,
                crossover_prob,
                float(getattr(self.cfg, "fresh_draft_prob", 0.0) or 0.0),
            )
        except RuntimeError:
            parent_nodes, island_id, operator, parent_selection_trace = (
                [],
                random.choice(range(solution_database.num_islands)),
                "draft",
                None,
            )

        if operator == "draft":
            island_id = random.choice(range(solution_database.num_islands))
            parent_nodes = []

        return AsyncWorkItem(
            attempt_id=attempt_id,
            generation_id=generation_id,
            temperature=temperature,
            island_id=island_id,
            operator=operator,
            parent_nodes=list(parent_nodes),
            parent_selection_trace=parent_selection_trace,
            worker=worker,
        )

    def _create_node_from_work_item(self, work_item: AsyncWorkItem) -> Node:
        if work_item.operator == "improve":
            node = self._improve(work_item.parent_nodes[0])
        elif work_item.operator == "crossover":
            node = self._crossover(work_item.parent_nodes[0], work_item.parent_nodes[1])
        else:
            node = self._draft()
        node.async_work_metadata = {
            "execution_mode": "async_steady_state",
            "attempt_id": work_item.attempt_id,
            "generation_id": work_item.generation_id,
            "temperature": work_item.temperature,
            "worker_id": work_item.worker.worker_id,
            "gpu_index": work_item.worker.gpu_index,
            "sandbox_url": work_item.worker.sandbox_url,
        }
        if work_item.parent_selection_trace is not None:
            node.experience_parent_selection = work_item.parent_selection_trace
        return node

    async def _step_task_async(self, task, state, code: str, sandbox_url: str):
        if hasattr(task, "step_task_async"):
            return await task.step_task_async(
                dict(state),
                code,
                sandbox_base_url=sandbox_url,
            )
        return await asyncio.to_thread(task.step_task, dict(state), code)

    async def _commit_async_node(
        self,
        *,
        task,
        node: Node,
        solution_database: SolutionsDatabase,
        work_item: AsyncWorkItem,
        commit_lock: asyncio.Lock,
        add_to_database: bool,
        commit_count: list[int],
    ) -> bool:
        async with commit_lock:
            if self._should_stop_async_search(
                task=task,
                include_attempt_limit=False,
            ):
                return False
            self._append_evaluated_node(node)
            self.state.running_time = time.monotonic() - self.start_time
            if add_to_database and not bool(node.is_buggy):
                migration_prob = (
                    0
                    if work_item.generation_id < self.cfg.num_generations_till_migration
                    else self.cfg.migration_prob
                )
                solution_database.add_nodes_to_islands(
                    [node],
                    [work_item.island_id],
                    migration_prob,
                )
            commit_count[0] += 1
            checkpoint_every = max(1, int(getattr(self.cfg, "async_checkpoint_every_commits", 1)))
            if commit_count[0] % checkpoint_every == 0:
                self.save_checkpoint()
            return True

    async def _debug_cycle_async(
        self,
        *,
        state,
        task,
        buggy_node: Node,
        solution_database: SolutionsDatabase,
        work_item: AsyncWorkItem,
        commit_lock: asyncio.Lock,
        commit_count: list[int],
    ) -> None:
        await self._commit_async_node(
            task=task,
            node=buggy_node,
            solution_database=solution_database,
            work_item=work_item,
            commit_lock=commit_lock,
            add_to_database=False,
            commit_count=commit_count,
        )
        current_debug_node = buggy_node
        debug_path = [buggy_node]
        total_debug_cycle_time = current_debug_node.exec_time if current_debug_node.exec_time is not None else 0

        for debug_attempt_index in range(self.cfg.max_debug_depth):
            if self._should_stop_async_search(
                task,
                include_attempt_limit=False,
            ):
                break
            try:
                fixed_node_attempt = await asyncio.to_thread(self._debug, current_debug_node)
                fixed_node_attempt.async_work_metadata = {
                    **dict(getattr(buggy_node, "async_work_metadata", {}) or {}),
                    "debug_attempt_index": debug_attempt_index,
                }
                _, eval_result = await self._step_task_async(
                    task,
                    state,
                    extract_code(fixed_node_attempt.code),
                    work_item.worker.sandbox_url,
                )
                self.parse_eval_result(node=fixed_node_attempt, eval_result=eval_result)
                await self._commit_async_node(
                    task=task,
                    node=fixed_node_attempt,
                    solution_database=solution_database,
                    work_item=work_item,
                    commit_lock=commit_lock,
                    add_to_database=not bool(fixed_node_attempt.is_buggy),
                    commit_count=commit_count,
                )
                current_debug_node = fixed_node_attempt
                debug_path.append(fixed_node_attempt)
            except Exception as exc:
                if exc.__class__.__name__ == "StopSearch":
                    raise
                self.logger.error(f"Error during async debug step: {exc}", LogEvent.SOLVER)
                break

            if not current_debug_node.is_buggy:
                self.logger.info(
                    f"Async debug cycle fixed node {buggy_node.id} with node {current_debug_node.id}",
                    LogEvent.SOLVER,
                )
                break
            if current_debug_node.exec_time is not None:
                total_debug_cycle_time += current_debug_node.exec_time
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                break
        if current_debug_node.is_buggy:
            repeated_failure_reason = self._repeated_debug_failure_reason(debug_path)
            if repeated_failure_reason:
                solution_database.request_fresh_draft(repeated_failure_reason)

    async def _async_worker_loop(
        self,
        *,
        worker: WorkerSpec,
        state,
        task,
        solution_database: SolutionsDatabase,
        sample_lock: asyncio.Lock,
        commit_lock: asyncio.Lock,
        stop_event: asyncio.Event,
        commit_count: list[int],
    ) -> None:
        while not stop_event.is_set():
            if self._should_stop_async_search(task):
                stop_event.set()
                break

            async with sample_lock:
                if self._should_stop_async_search(task):
                    stop_event.set()
                    break
                work_item = self._next_async_work_item(
                    solution_database=solution_database,
                    worker=worker,
                )

            max_retries = max(
                0,
                int(getattr(self.cfg, "async_worker_max_retries", 3) or 0),
            )
            retry_backoff = max(
                0.0,
                float(
                    getattr(
                        self.cfg,
                        "async_worker_retry_backoff_secs",
                        1.0,
                    )
                    or 0.0
                ),
            )
            failure_index = 0
            while not stop_event.is_set():
                try:
                    child_node = await asyncio.to_thread(
                        self._create_node_from_work_item,
                        work_item,
                    )
                    _, eval_result = await self._step_task_async(
                        task,
                        state,
                        extract_code(child_node.code),
                        work_item.worker.sandbox_url,
                    )
                    self.parse_eval_result(child_node, eval_result)
                    if child_node.is_buggy:
                        await self._debug_cycle_async(
                            state=state,
                            task=task,
                            buggy_node=child_node,
                            solution_database=solution_database,
                            work_item=work_item,
                            commit_lock=commit_lock,
                            commit_count=commit_count,
                        )
                    else:
                        await self._commit_async_node(
                            task=task,
                            node=child_node,
                            solution_database=solution_database,
                            work_item=work_item,
                            commit_lock=commit_lock,
                            add_to_database=True,
                            commit_count=commit_count,
                        )
                    break
                except Exception as exc:
                    if exc.__class__.__name__ == "StopSearch":
                        stop_event.set()
                        break
                    if failure_index >= max_retries:
                        self.logger.error(
                            f"Async worker {worker.worker_id} exhausted retries "
                            f"for attempt {work_item.attempt_id}: {exc}",
                            LogEvent.SOLVER,
                        )
                        break
                    delay = retry_backoff * (2**failure_index)
                    failure_index += 1
                    self.logger.warning(
                        f"Async worker {worker.worker_id} failed attempt "
                        f"{work_item.attempt_id}; retry {failure_index}/"
                        f"{max_retries} in {delay:.2f}s: {exc}",
                        LogEvent.SOLVER,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

    async def search_async_steady_state(self, task, state, *, worker_count: int) -> Optional[str]:
        if self.journal.nodes:
            self.root_node = self.journal.nodes[0]
            if self.state.current_step != len(self.journal):
                self.state.current_step = len(self.journal)
        else:
            self.create_root_node()

        solution_database = self._restore_solution_database_from_journal()

        sandbox_urls = self._async_sandbox_urls(task)
        worker_specs = build_worker_specs(
            worker_count=worker_count,
            sandbox_urls=sandbox_urls,
            assignment=str(
                getattr(
                    self.cfg,
                    "async_sandbox_assignment",
                    "round_robin",
                )
            ),
        )
        for url in sandbox_urls:
            idle_count = await fetch_router_idle_worker_count(url)
            if idle_count is not None:
                self.logger.info(f"Sandbox router {url} idle workers: {idle_count}", LogEvent.SOLVER)
        self.logger.info(
            f"Starting async steady-state evolutionary search with {worker_count} workers and sandbox URLs: {sandbox_urls}",
            LogEvent.SOLVER,
        )

        sample_lock = asyncio.Lock()
        commit_lock = asyncio.Lock()
        stop_event = asyncio.Event()
        commit_count = [0]
        workers = [
            asyncio.create_task(
                self._async_worker_loop(
                    worker=worker,
                    state=state,
                    task=task,
                    solution_database=solution_database,
                    sample_lock=sample_lock,
                    commit_lock=commit_lock,
                    stop_event=stop_event,
                    commit_count=commit_count,
                )
            )
            for worker in worker_specs
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            stop_event.set()
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self.save_checkpoint()

        best_node = self.journal.get_best_node()
        return state, best_node.code if best_node is not None else None

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

        # Extract potential auxliary evaluation information to store
        # in the node. This is more for logging purposes.
        aux_eval_info = eval_result.get(AUX_EVAL_INFO, {}) or {}
        if not isinstance(aux_eval_info, dict):
            aux_eval_info = {}

        def coerce_float(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not numpy.isfinite(numeric):
                return None
            return numeric

        def deterministic_failure() -> bool:
            failure_statuses = {
                "buggy",
                "error",
                "failed",
                "failure",
                "invalid",
                "sandbox_error",
                "scoring_failed",
                "submission_missing",
                "timeout",
                "unknown",
            }
            status = str(aux_eval_info.get("status") or "").strip().lower()
            if status in failure_statuses:
                return True
            status_code = aux_eval_info.get("status_code")
            try:
                return status_code is not None and int(status_code) >= 400
            except (TypeError, ValueError):
                return False

        def run_legacy_analyze() -> dict:
            try:
                analyze_response = self._analyze(node)
            except Exception as e:
                self.logger.error(f"Error during analysis operator: {str(e)}")
                analyze_response = {}

            parsed_response = parse_json_output(analyze_response)
            if not isinstance(parsed_response, dict):
                self.logger.warning(f"Parsed response is not a dictionary: {type(parsed_response)}")
                parsed_response = {}

            if len(parsed_response) == 0:
                return {"metric": None, "summary": "", "is_bug": True}

            if "metric" not in parsed_response:
                parsed_response["metric"] = None
            if "summary" not in parsed_response:
                parsed_response["summary"] = ""
            if "is_bug" not in parsed_response:
                parsed_response["is_bug"] = parsed_response["metric"] is None
            return parsed_response

        validation_fitness = coerce_float(eval_result.get(VALIDATION_FITNESS))
        sanitize_prompt_scores = prompt_score_sanitization_enabled(self.cfg)
        experience_enabled = self._experience_enabled()

        if not experience_enabled:
            # Preserve the inference AIRA-Evo path: call analyze for every node,
            # then let deterministic validation fitness override the metric.
            response = run_legacy_analyze()
            if validation_fitness is not None:
                response["metric"] = validation_fitness
                response["is_bug"] = False
        elif validation_fitness is not None:
            response = {
                "metric": validation_fitness,
                "summary": sanitize_execution_output_for_prompt(
                    aux_eval_info.get("feedback") or "Deterministic validation fitness available.",
                    enabled=sanitize_prompt_scores,
                ),
                "is_bug": False,
            }
        elif deterministic_failure() or node.exit_code != 0:
            response = {
                "metric": None,
                "summary": sanitize_execution_output_for_prompt(
                    aux_eval_info.get("feedback") or aux_eval_info.get("clear_run_log") or "",
                    enabled=sanitize_prompt_scores,
                ),
                "is_bug": True,
            }
        else:
            # Fall back to the legacy analyze operator only when sandbox metadata
            # cannot deterministically provide a metric or failure state.
            response = run_legacy_analyze()

        if not isinstance(response["metric"], (float, int)):
            response["metric"] = None

        node.analysis = response["summary"]

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
            validity_feedback = f"\n\n Evaluator Feedback: {validity_feedback}"
            node._term_out.append(validity_feedback)
        else:
            aux_eval_info["validity_feedback"] = "evaluator feedback not available"

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

    def search(self, task, state) -> Optional[str]:
        execution_mode = self._execution_mode()
        worker_count = self._async_worker_count()
        if execution_mode == "async_steady_state":
            return asyncio.run(
                self.search_async_steady_state(
                    task,
                    state,
                    worker_count=worker_count,
                )
            )
        if worker_count != 1:
            raise ValueError(
                "generation mode requires exactly one worker; "
                "use execution=multi_gpu for async steady-state search"
            )

        if self.journal.nodes:
            self.root_node = self.journal.nodes[0]
            if self.state.current_step != len(self.journal):
                self.logger.warning(
                    f"Checkpoint step {self.state.current_step} does not match journal length {len(self.journal)}. "
                    "Using journal length for resume.",
                    LogEvent.SOLVER,
                )
                self.state.current_step = len(self.journal)
        else:
            self.create_root_node()
        self.logger.info("Starting evolutionary search", LogEvent.SOLVER)

        # Restore resumable solutions into the same islands database used by async mode.
        solution_database = self._restore_solution_database_from_journal()

        # define a schedule for temperature of sampling
        temp_scheduler = self.linear_decay

        for generation_id in range(self.state.current_generation, self.cfg.num_generations):
            # Measure state time of the generation
            start_time = time.monotonic()
            generation_exempt_time_start = self._task_budget_exempt_seconds(task)

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
                        self.cfg.fresh_draft_prob,
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
                if generation_id > 0 and solution_database.last_parent_selection is not None:
                    child_node.experience_parent_selection = solution_database.last_parent_selection
                state, eval_result = task.step_task(state, extract_code(child_node.code))
                self.parse_eval_result(child_node, eval_result)
                # if the node is buggy, we run a debug cycle
                # and add the fixed node to the generation
                # if the node is not buggy, we add it to the generation
                if not child_node.is_buggy:
                    self._append_evaluated_node(child_node)
                    solution_nodes.append(child_node)
                    counter_ids.append(counter_id)
                else:
                    self.logger.info(f"Node {child_node.id} was buggy, entering debug cycle.", LogEvent.SOLVER)
                    state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                    if not self._experience_enabled():
                        for n in debug_path:
                            self._append_evaluated_node(n)
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
                        repeated_failure_reason = self._repeated_debug_failure_reason(debug_path)
                        if repeated_failure_reason:
                            solution_database.request_fresh_draft(repeated_failure_reason)

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

            generation_wall_time = time.monotonic() - start_time
            generation_exempt_time = max(
                0.0,
                self._task_budget_exempt_seconds(task) - generation_exempt_time_start,
            )
            generation_effective_time = max(0.0, generation_wall_time - generation_exempt_time)
            self.state.running_time += generation_effective_time
            self.logger.info(
                f"Step {self.state.current_step} | Generation {self.state.current_generation}: "
                f"effective={generation_effective_time:.3f}s, gpu_wait={generation_exempt_time:.3f}s, "
                f"budget_used={self.state.running_time:.3f}s"
            )

            # Update the state with the current generation
            self.state.current_generation += 1

            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs or self.state.current_step >= self.cfg.step_limit:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        best_node = self.journal.get_best_node()
        return state, best_node.code if best_node is not None else None

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
