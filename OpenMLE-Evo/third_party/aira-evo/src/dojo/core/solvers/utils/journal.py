# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

"""
The journal is the core datastructure in Greedy that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import time
import uuid
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Dict, List, Literal, Optional

from dataclasses_json import DataClassJsonMixin
from omegaconf import OmegaConf

from dojo.core.interpreters.base import ExecutionResult
from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
from dojo.core.solvers.utils.response import trim_long_string

import logging

log = logging.getLogger(__name__)


@total_ordering
@dataclass
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str = field(compare=False)
    plan: str = field(default=None, kw_only=True, compare=False)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True, compare=False)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True, compare=False)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True, compare=False)
    parents: List["Node"] = field(default_factory=list, kw_only=True, compare=False)
    children: set["Node"] = field(default_factory=set, kw_only=True, compare=False)

    # ---- operator info ----
    operators_used: List[str] = field(default_factory=list, kw_only=True, compare=False)
    operators_metrics: List[Dict[str, Any]] = field(default_factory=list, kw_only=True, compare=False)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True, compare=False)  # type: ignore
    exec_time: float = field(default=None, kw_only=True, compare=False)  # type: ignore
    exit_code: int | None = field(default=None, kw_only=True, compare=False)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True, compare=False)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True, compare=False)  # type: ignore

    def __post_init__(self) -> None:
        if self.parents is None:
            return

        for parent in self.parents:
            if parent is None:
                raise ValueError("Parent node is None")

    def attach_to_parents(self) -> None:
        if not self.parents:
            return

        for parent in self.parents:
            parent.children.add(self)

    def remove_child(self, child: "Node"):
        if child and self.children:
            self.children.discard(child)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parents is None or len(self.parents) == 0:
            return "draft"
        return "debug" if list(self.parents)[0].is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        if exec_result is None:
            exec_result = ExecutionResult.get_empty()
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exit_code = exec_result.exit_code

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        if self._term_out:
            return trim_long_string("".join(self._term_out))
        return ""

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __gt__(self, other: "Node") -> bool:
        return self.metric > other.metric  # type: ignore

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0

        if not self.parents or len(self.parents) > 1:
            raise ValueError("Debug node must have exactly one parent.")
        (parent,) = self.parents
        return parent.debug_depth + 1  # type: ignore

    def extra_metrics_to_log(self):
        return {}


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)
        # Link the node only after it has a stable step inside the session.
        node.attach_to_parents()

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step + 1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step + 1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step + 2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)
        # Link the node only after it has a stable step inside the journal.
        node.attach_to_parents()

    def is_root_node(self, node: Node) -> bool:
        """Check if the node is a root node (no parents)."""
        first_check = node.parents is None or len(node.parents) == 0
        second_check = node.code == "" and node.plan == "" and node.analysis == ""
        third_check = isinstance(node.metric, WorstMetricValue)
        fourth_check = node.step == 0
        return first_check and second_check and third_check and fourth_check

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [
            n
            for n in self.nodes
            if ((n.parents is None) or (len(n.parents) == 0) or self.is_root_node(n.parents[0]))
            and not self.is_root_node(n)
        ]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy and not self.is_root_node(n)]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """Return the node with the best validation metric, or ``None`` if none exist.

        If *only_good* is True, search only in ``self.good_nodes``; otherwise search
        the full ``self.nodes`` list.
        """
        pool = self.good_nodes if only_good else self.nodes
        if not pool:
            return None

        # NB: keep metrics whose value is None – they are still orderable.
        return max(
            [n for n in pool if isinstance(n.metric, MetricValue)],
            key=lambda n: n.metric,
            default=None,  # empty after filtering
        )

    def generate_summary(
        self, include_code: bool = False, include_buggy_nodes: bool = False, only_plans: bool = False
    ) -> str:
        """Generate a summary of the journal for the agent."""
        log.warning("generate_summary is deprecated. Use memory operator instead.")

        summary = []
        nodes = self.nodes if include_buggy_nodes else self.good_nodes
        for n in nodes:
            summary_part = f"Design: {n.plan}\n"
            if not only_plans:
                if include_code:
                    summary_part += f"Code: {n.code}\n"
                if n.is_buggy:
                    summary_part += f"Node was Buggy: {n.term_out}\n"
                else:
                    summary_part += f"Results: {n.analysis}\n"
                    summary_part += f"Validation Metric: {n.metric.value}\n"

            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def get_node_data(self, idx: int):
        node = self.nodes[idx]
        # If node.metric is a MetricValue, use node.metric.value, else None
        if isinstance(node.metric, MetricValue):
            metric_value = node.metric.value
        elif isinstance(node.metric, WorstMetricValue):
            metric_value = None  # or some sentinel for "worst"
        else:
            metric_value = None

        children = [c.step for c in node.children if c is not None and c.step is not None] if node.children else []
        parents = [p.step for p in node.parents if p is not None and p.step is not None] if node.parents else []
        try:
            node_data = {
                "step": node.step,
                "id": node.id,
                "plan": node.plan,
                "code": node.code,
                "metric": metric_value,
                "metric_info": node.metric.info if node.metric else None,
                "metric_maximize": node.metric.maximize if node.metric else None,
                "is_buggy": node.is_buggy,
                "analysis": node.analysis,
                "operators_metrics": node.operators_metrics,
                "children": children,
                "parents": parents,
                "creation_time": node.ctime,
                "term_out": node.term_out,
                "operators_used": node.operators_used,
                "exec_time": node.exec_time,
                "exit_code": node.exit_code,
                "_term_out": node._term_out,
            }
        except:
            # Legacy data support
            node_data = {
                "step": node.step,
                "id": node.id,
                "plan": node.plan,
                "code": node.code,
                "metric": metric_value,
                "metric_info": node.metric.info if node.metric else None,
                "metric_maximize": node.metric.maximize if node.metric else None,
                "is_buggy": node.is_buggy,
                "analysis": node.analysis,
                "children": children,
                "parents": parents,
                "creation_time": node.ctime,
                "term_out": node.term_out,
                "exec_time": node.exec_time,
                "exit_code": node.exit_code,
                "_term_out": node._term_out,
            }

        return node_data

    def export_data(self) -> dict:
        """Gather data into a dictionary structure."""

        best_node = self.get_best_node()
        if best_node:
            best_solution_code = best_node.code
        else:
            best_solution_code = ""

        node_list = []
        for i in range(len(self.nodes)):
            node_data = self.get_node_data(i)
            node_list.append(node_data)

        search_data = {
            "nodes": node_list,
            "solution": best_solution_code,
        }

        return search_data

    @classmethod
    def from_export_data(cls, export_data: dict) -> "Journal":
        """
        Reconstruct a Journal object from exported search data.

        Args:
            export_data: Dictionary containing the exported search data

        Returns:
            Journal: Reconstructed journal with all nodes and their relationships
        """
        journal = cls()

        # First create all nodes without setting relationships
        step_to_node: Dict[int, Node] = {}

        for node_data in export_data["nodes"]:
            # Create metric value if it exists
            metric = None
            if node_data.get("metric", None) is not None:
                # find keys that start with "metric_info/*"
                metric_info = {
                    k[len("metric_info/") :]: v for k, v in node_data.items() if k.startswith("metric_info/")
                }
                metric = MetricValue(
                    value=node_data["metric"],
                    info=metric_info,
                    maximize=node_data.get("metric_maximize", True),
                )
            else:
                metric = WorstMetricValue()

            # Create the node
            try:
                node = Node(
                    code=node_data["code"],
                    plan=node_data["plan"],
                    step=node_data["step"],
                    id=node_data["id"],
                    ctime=node_data["creation_time"],
                    operators_metrics=node_data["operators_metrics"],
                    operators_used=node_data["operators_used"],
                    _term_out=node_data["_term_out"],
                    exec_time=node_data.get("exec_time"),
                    exit_code=node_data.get("exit_code"),
                    analysis=node_data["analysis"],
                    metric=metric,
                    is_buggy=node_data["is_buggy"],
                    # Initialize relationships as empty
                    parents=[],
                    children=set(),
                )
            except:
                # Deprecated legacy data support.
                node = Node(
                    code=node_data["code"],
                    plan=node_data["plan"],
                    step=node_data["step"],
                    id=node_data["id"],
                    ctime=node_data["creation_time"],
                    _term_out=node_data["_term_out"],
                    exec_time=node_data.get("exec_time"),
                    exit_code=node_data.get("exit_code"),
                    analysis=node_data["analysis"],
                    metric=metric,
                    is_buggy=node_data["is_buggy"],
                    # Initialize relationships as empty
                    parents=[],
                    children=set(),
                )

            step_to_node[node.step] = node

        # Second pass to set up parent-child relationships
        for node_data in export_data["nodes"]:
            node = step_to_node[node_data["step"]]

            # Set parents if they exist
            if node_data["parents"]:
                node.parents = [step_to_node[p_step] for p_step in node_data["parents"]]

            # Set children if they exist
            if node_data["children"] and not any([child is None for child in node_data["children"]]):
                node.children = {step_to_node[c_step] for c_step in node_data["children"]}

            journal.append(Node(**node.__dict__))

        return journal

    def node_list(self) -> dict:
        """Gather data into a dictionary structure."""

        node_list = []
        for i in range(len(self.nodes)):
            node_data = self.get_node_data(i)
            node_list.append(node_data)

        return node_list
