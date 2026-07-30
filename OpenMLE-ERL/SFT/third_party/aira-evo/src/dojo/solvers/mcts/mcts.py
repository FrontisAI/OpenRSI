# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import hydra
from dojo.core.solvers.base import Solver
from dojo.core.solvers.operators.analyze import analyze_op
from dojo.core.solvers.operators.core import execute_op_plan_code
from dojo.core.solvers.operators.debug import debug_op
from dojo.core.solvers.operators.draft import draft_op
from dojo.core.solvers.operators.improve import improve_op
from dojo.core.solvers.operators.memory import create_memory_op
from dojo.core.solvers.utils import data_preview
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import extract_code
from dojo.solvers.utils import get_complextiy_level
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.utils.code_parsing import parse_json_output
from dojo.core.solvers.utils.search_exporter import (
    export_search_results,
)
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
from dojo.config_dataclasses.solver.mcts import MCTSSolverConfig
from dojo.utils.state import MCTSState


def normalise_q_value(q_value: float, global_max_q_val: float, global_min_q_val: float) -> float:
    # Avoid division by zero in case global_max and global_min are equal.
    if global_max_q_val == global_min_q_val:
        return 0.5  # Neutral value in the [0,1] range.
    return (q_value - global_min_q_val) / (global_max_q_val - global_min_q_val)


def uct_value(
    q_value: float,
    explore_count: int,
    parent_explore_count: int,
    uct_c: float,
    global_max_q_val: float,
    global_min_q_val: float,
):
    if explore_count == 0:
        return -1e8

    norm_q = normalise_q_value(q_value, global_max_q_val, global_min_q_val)

    exploration = math.sqrt(math.log(parent_explore_count) / explore_count)
    return norm_q + uct_c * exploration


class MCTSNode(Node):
    explore_count: int = 0
    node_value: float = 0

    def q_value(self, lower_is_better: bool = False):
        if self.explore_count == 0:
            return -1e8

        q_value = self.node_value / self.explore_count

        if lower_is_better:
            q_value = -q_value

        return q_value

    def set_value(self, value: float) -> None:
        """Set the node's value."""
        self.node_value = value

    def add_value(self, value: float) -> None:
        """
        Add to the node's cumulative value.

        Args:
            value: Value to add to node's cumulative value
        """
        if value is not None:
            self.node_value += value

    def increment_explore_count(self):
        self.explore_count += 1

    def extra_metrics_to_log(self):
        if self.explore_count == 0:
            v_div_ec = None
        else:
            v_div_ec = self.node_value / self.explore_count
        return {"node_value": self.node_value, "explore_count": self.explore_count, "v_div_ec": v_div_ec}


class MCTS(Solver):
    def __init__(self, cfg: MCTSSolverConfig, task_info):
        """
        Initialize the MCTS solver.

        Args:
            task_info: Dictionary containing task information including description and optimization direction
            **cfg: Configuration dictionary with solver parameters
        """
        super().__init__(cfg, task_info=task_info)
        self.journal = Journal()
        self.data_preview: str | None = None

        self.task_desc = task_info[TASK_DESCRIPTION]
        self.lower_is_better = task_info.get("lower_is_better", None)

        assert self.lower_is_better is not None

        self.state = MCTSState()

        self.setup_operators()

        self.global_max_q_val = -1e8
        self.global_min_q_val = 1e8

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
        """
        Initialize and configure the LLM operators used in the MCTS solver.

        This method instantiates the draft, improve, debug, and analyze LLMs
        and creates partial functions for each operator with the appropriate parameters.
        """

        # First we set up the LLMs
        draft_llm = GenericLLM(self.cfg.operators["draft"])
        improve_llm = GenericLLM(self.cfg.operators["improve"])
        debug_llm = GenericLLM(self.cfg.operators["debug"])
        analyze_llm = GenericLLM(self.cfg.operators["analyze"])

        # Create the memory for operators
        self.memory_op = create_memory_op(self.cfg.memory)
        self.debug_memory_op = create_memory_op(self.cfg.debug_memory)

        # Then we create the operators
        self.draft_fn = partial(draft_op, draft_llm, self.cfg, self.memory_op)
        self.improve_fn = partial(improve_op, improve_llm, self.cfg, self.memory_op)
        self.debug_fn = partial(debug_op, debug_llm, self.cfg, self.debug_memory_op)
        self.analyze_fn = partial(analyze_op, analyze_llm, self.cfg)

    def create_root_node(self):
        self.root_node = MCTSNode(
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

    @property
    def remaining_steps(self):
        return self.cfg.step_limit - self.state.current_step

    def __call__(self, task, state):
        """
        Run the MCTS solver for a specified number of iterations.

        Executes the search process for the configured number of steps, tracking the best
        solution found. After all iterations, exports search results and returns the best code.

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and the best code solution (or None if no valid solution found)
        """
        self.logger.info("Starting MCTS search")

        # Create a blank root node to start.
        self.create_root_node()

        # Run the search
        while self.state.current_step <= self.cfg.step_limit:
            start_time = time.monotonic()
            state = self.step(task, state)
            self.state.running_time += time.monotonic() - start_time
            self.logger.info(
                f"Step {self.state.current_step}: Time taken for step: {self.state.running_time:.3f} seconds"
            )

            self.logger.info(f"Step {self.state.current_step}: Saving checkpoint")
            self.save_checkpoint()

            if self.state.running_time >= self.cfg.time_limit_secs:
                self.logger.info("Maximum runtime reached, stopping search")
                break

        # Get the best node
        best_node = self.journal.get_best_node()

        # Export the search results
        try:
            export_search_results(self.cfg, self.journal, self.logger, "MCTS")
        except Exception as e:
            self.logger.error(f"Error exporting search results: {e}")

        # Return the best node
        if best_node:
            return state, best_node.code, best_node
        else:
            self.logger.info("No suitable code found after all iterations.")
            return state, None, None

    def search_policy(self, root_node: MCTSNode) -> List[MCTSNode]:
        """
        Traverse the tree from root to leaf using UCT selection.

        Args:
            root_node: Starting node for path selection

        Returns:
            List of nodes representing path from root to selected leaf
        """
        path: List[MCTSNode] = []
        current_node = root_node

        while True:
            path.append(current_node)
            if not current_node.children:
                # It's a leaf
                return path

            current_node = max(
                current_node.children,
                key=lambda c: uct_value(
                    q_value=c.q_value(self.lower_is_better),
                    explore_count=c.explore_count,
                    parent_explore_count=current_node.explore_count,
                    uct_c=self.cfg.uct_c,
                    global_max_q_val=self.global_max_q_val,
                    global_min_q_val=self.global_min_q_val,
                ),
            )

    def _draft(self, parent: Optional[MCTSNode] = None) -> MCTSNode:
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
            get_complextiy_level(parent) if self.cfg.use_complexity else None,
            self.root_node,
            max_operator_tries=self.cfg.max_llm_call_retries,
        )
        node = MCTSNode(plan=plan, code=code, parents=[parent], operators_used=["draft"], operators_metrics=[metrics])
        self.logger.info(f"Draft Node Created - Metrics: {metrics}")
        self.logger.info(f"Draft Code: {code}")
        return node

    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
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
        node = MCTSNode(
            plan=plan, code=code, parents=[parent_node], operators_used=["improve"], operators_metrics=[metrics]
        )
        self.logger.info(f"Improve Node Created - Metrics: {metrics}")
        self.logger.info(f"Improve Code: {code}")
        return node

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
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
        node = MCTSNode(
            plan=plan, code=code, parents=[parent_node], operators_used=["debug"], operators_metrics=[metrics]
        )
        self.logger.info(f"Debug Node Created - Metrics: {metrics}")
        self.logger.info(f"Debug Code: {code}")
        return node

    def _analyze(self, node: MCTSNode) -> Union[str, dict]:
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

    def step(self, task, state):
        """
        Execute a single iteration of the MCTS solver process.

        This method implements the core MCTS algorithm:
        1. Select a node to work on using UCT.
        2. Apply the appropriate operator to generate new code
        3. Evaluate the code using the task
        4. Parse the results and update the journal

        Args:
            task: The task object that provides evaluation capabilities
            state: The current solver state

        Returns:
            tuple: Updated state and evaluation results
        """
        self.logger.info(f"Step {self.state.current_step}: Starting iteration")

        # Possibly generate data preview first
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview(state)

        # 1) Selection: Find a path from root → leaf using UCT
        path = self.search_policy(self.root_node)

        # 2) Expansion and Backprop
        state = self._expand_leaf_and_backprop(path, state, task)

        return state

    def _backprop_step(self, path: List[MCTSNode], value_estimate: float):
        """
        Backpropagates the reward/score up the path (increasing explore_count
        and adjusting the metrics).
        """
        for node in path:
            node.increment_explore_count()
            node.add_value(value_estimate)

    def log_journal(self):
        # Get the current best node in the tree.
        best_node = self.journal.get_best_node()
        best_node_step = 0 if best_node is None else best_node.step

        self.logger.log(
            self.journal.get_node_data(self.state.current_step) | {"current_best_node": best_node_step},
            "JOURNAL",
            step=self.state.current_step,
        )

    def set_global_q_values(self, new_metric_value):
        if new_metric_value is not None:
            self.global_max_q_val = max(self.global_max_q_val, new_metric_value)
            self.global_min_q_val = min(self.global_min_q_val, new_metric_value)

    def _expand_leaf_and_backprop(self, path: List[MCTSNode], state: Any, task: Any) -> Tuple[Any, int]:
        """
        Expand a leaf node and backpropagate results.

        Args:
            path: List of nodes from root to leaf
            state: Current state object
            task: Task object for evaluation

        Returns:
            Tuple of (updated state, number of trials performed)
        """
        leaf_node = path[-1]
        num_children_to_create = min(self.cfg.num_children, self.remaining_steps)
        for _ in range(num_children_to_create):
            # If root node, we draft otherwise we improve
            if not leaf_node.parents:
                child_node = self._draft(leaf_node)
            else:
                child_node = self._improve(leaf_node)

            # Evaluate the code
            self.logger.debug(f"Step {self.state.current_step}: Executing generated code")
            state, eval_result = task.step_task(state, extract_code(child_node.code))
            self.parse_eval_result(node=child_node, eval_result=eval_result)

            # Add the child to the journal
            self.journal.append(child_node)
            self.log_journal()
            self.state.current_step += 1

            # If the child node is not buggy, we backpropagate
            if not child_node.is_buggy:
                self._backprop_step(path=path + [child_node], value_estimate=child_node.metric.value)
                self.set_global_q_values(child_node.metric.value)
            else:
                # Execute debug cycle
                state, debug_path, fixed_metric = self.debug_cycle(state, task, child_node)
                # We now exclude the child node from the path and backprop the fixed metric up the rest of the tree
                if fixed_metric is not None:
                    self._backprop_step(path=path + debug_path, value_estimate=fixed_metric)
                    self.set_global_q_values(fixed_metric)

            # If we have used up all the steps we break
            if self.state.current_step > self.cfg.step_limit:
                self.logger.info(f"Step limit reached: {self.state.current_step} steps")
                break

        return state

    def debug_cycle(self, state, task, buggy_node: MCTSNode):
        debug_path = [buggy_node]
        debug_depth = min(self.remaining_steps, self.cfg.max_debug_depth)
        fixed_metric = None
        # We set the initial debug cycle time to the execution time of the first buggy node
        total_debug_cycle_time = buggy_node.exec_time
        # We run the debug cycle for a number of times
        # or until time runs out, whichever comes first
        for _ in range(debug_depth):
            buggy_node = self._debug(buggy_node)
            state, eval_result = task.step_task(state, extract_code(buggy_node.code))
            self.parse_eval_result(node=buggy_node, eval_result=eval_result)
            self.journal.append(buggy_node)
            self.log_journal()
            self.state.current_step += 1
            debug_path.append(buggy_node)
            # Break if we have a fixed metric - i.e. the solution is no longer buggy
            if buggy_node.metric.value is not None:
                fixed_metric = buggy_node.metric.value
                break
            elif self.state.current_step > self.cfg.step_limit:
                self.logger.info(f"Step limit reached: {self.state.current_step} steps")
                break
            # or if the debug depth is reached we break
            if buggy_node.debug_depth >= self.cfg.max_debug_depth:
                break
            # or if the debug time is reached we break
            total_debug_cycle_time += buggy_node.exec_time
            if total_debug_cycle_time >= self.cfg.max_debug_time:
                self.logger.info(f"Debug cycle time exceeded: {total_debug_cycle_time} seconds")
                break

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
        node.analysis = response["summary"]

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
