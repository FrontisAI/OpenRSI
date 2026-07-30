# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Callable, Optional
from xml.etree.ElementInclude import include

import humanize
import hydra
from omegaconf import DictConfig, OmegaConf

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import wrap_code
from dojo.config_dataclasses.operators.memory import MemoryOpConfig

import logging

log = logging.getLogger(__name__)


# Memory Op Factory
############################################
def create_memory_op(
    cfg: Optional[MemoryOpConfig],
) -> Callable[[Journal, Optional[Node]], str]:
    if cfg is None:
        return None

    memory_processor = MEM_OPS[cfg.memory_processor]
    memory_op_kwargs = cfg.memory_op_kwargs

    def memory_op(journal: Journal, node: Optional[Node] = None) -> str:
        memory = memory_processor(
            journal=journal,
            node=node,
            **memory_op_kwargs,
        )

        if not memory:
            memory = "(No memory available.)"

        return memory

    return memory_op


# Memory Processors
############################################


def truncate_string(s: str, max_prefix: int = 100, max_suffix: int = 100) -> str:
    """
    Truncate a string to 200 characters and add ellipsis.
    """
    if len(s) <= 200:
        return s
    return s[:max_prefix] + "...(truncated)..." + s[-max_suffix:]


def get_node_summary(
    node: Node,
    include_code: bool = False,
    only_plans: bool = False,
) -> str:
    """
    Generate a summary of the node for the agent.
    """
    summary = ""

    # If the node has no plan and no code, return an empty summary
    if not (node.plan and node.code):
        return ""

    if "debug" in node.operators_used:
        summary += f"Debug plan: {node.plan}\n"
    else:
        summary += f"Design: {node.plan}\n"

    if only_plans:
        return summary

    if include_code:
        summary += f"Code: ```{node.code}```\n"

    if node.is_buggy:
        pass
    else:
        summary += f"Validation Metric: {node.metric.value}\n"

    analysis = str(node.analysis or "").strip()
    if analysis:
        label = "Failure Analysis" if node.is_buggy else "Analysis"
        summary += f"{label}: {analysis}\n"

    return summary


def generate_journal_summary(
    journal: Journal,
    include_code: bool = False,
    include_buggy_nodes: bool = False,
    only_plans: bool = False,
) -> str:
    """
    Generate a summary of the journal for the agent.
    """
    separator = "\n-------------------------------\n"
    nodes = journal.nodes if include_buggy_nodes else journal.good_nodes
    nodes = [node for node in nodes if node.plan and node.code]
    summary = [
        get_node_summary(
            node,
            include_code=include_code,
            only_plans=only_plans,
        )
        for node in nodes
    ]
    summary = separator.join(summary)
    if not summary:
        summary = "(No memory available.)"
    return summary


def generate_ancestral_summary(
    node: Optional[Node],
    include_code: bool = False,
    include_buggy_nodes: bool = True,
    only_plans: bool = False,
    until_successful_parent: bool = True,
) -> str:
    """
    Generate a summary of the ancestral journal for the agent.
    """
    separator = "\n-------------------------------\n"
    summaries_in_reverse = []

    if not include_buggy_nodes:
        log.warning("It's not recommended to use ancestral memory without buggy nodes.")

    while node is not None:
        # now if this was a successful parent node, we can stop
        if until_successful_parent and not node.is_buggy:
            break

        if include_buggy_nodes or not node.is_buggy:
            node_summary = get_node_summary(node, include_code=include_code, only_plans=only_plans)
            if node_summary:
                summaries_in_reverse.append(node_summary)

        parents = node.parents
        if not parents:
            node = None
            break

        # if len(parents) > 1:
        #     raise ValueError("AncestralMemory supports maximum one parent.")

        node = parents[0]  # unpack the first parent

    summaries = reversed(summaries_in_reverse)
    return separator.join(summaries)


def simple_memory(
    journal: Journal,
    node: Optional[Node] = None,
    max_length: Optional[int] = None,
    **kwargs,
) -> str:
    summary = generate_journal_summary(journal, **kwargs)

    if max_length and summary:
        if len(summary) > max_length:
            summary = f"...(truncated) {summary[-max_length:]}"

    return summary


def ancestral_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    return generate_ancestral_summary(node, **kwargs)


def no_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    return ""


def get_sibling_summary(
    parent_node: Optional[Node],
    include_code: bool = False,
    include_buggy_nodes: bool = True,
    only_plans: bool = True,
):
    separator = "\n-------------------------------\n"
    previous_siblings = [node for node in parent_node.children]
    if not include_buggy_nodes:
        log.warning("It's not recommended to use sibling memory without buggy nodes.")

    summaries = []
    for node in previous_siblings:
        if include_buggy_nodes or not node.is_buggy:
            node_summary = get_node_summary(node, include_code=include_code, only_plans=only_plans)
            if node_summary:
                summaries.append(node_summary)

    if len(previous_siblings) == 0 or previous_siblings is None:
        return "(No memory available.)"

    summary = separator.join(summaries)
    if not summary:
        summary = "(No memory available.)"
    return summary


def sibling_memory(
    journal: Journal,
    node: Optional[Node] = None,
    **kwargs,
) -> str:
    if node is None:
        raise ValueError("Sibling memory requires a node to be provided.")

    return get_sibling_summary(node, **kwargs)


MEM_OPS = {
    "simple_memory": simple_memory,
    "no_memory": no_memory,
    "sibling_memory": sibling_memory,
    "ancestral_memory": ancestral_memory,
}
