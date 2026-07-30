# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import Callable, Optional

import humanize
from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt
from dojo.core.solvers.utils.response import wrap_code
from dojo.solvers.evo.experience import (
    append_experience_memory,
    build_operator_experience_memory,
    prompt_memory_enabled,
    prompt_memory_max_related_cards,
    prompt_memory_operator_k,
    prompt_memory_sibling_rank_weights,
    prompt_memory_use_base_memory,
)
from dojo.solvers.utils import Complexity

_EMPTY_DATA_PREVIEWS = {
    "data (missing)",
    "no data preview available",
    "(no data preview available)",
}


def _clean_data_preview(data_preview: Optional[str]) -> Optional[str]:
    if data_preview is None:
        return None
    data_preview = data_preview.strip()
    if not data_preview or data_preview.lower() in _EMPTY_DATA_PREVIEWS:
        return None
    return data_preview


def crossover_op(
    crossover_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    prompt_context: dict[str, str],
    task_description: str,
    journal: Journal,
    input_node1: Node,
    input_node2: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    pkgs = cfg.available_packages
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    steps_remaining = cfg.step_limit - step_count
    use_targeted_memory = prompt_memory_enabled(cfg)
    sanitize_prompt_scores = prompt_score_sanitization_enabled(cfg)
    if memory_op is not None and (
        not use_targeted_memory or prompt_memory_use_base_memory(cfg)
    ):
        memory = memory_op(journal, input_node1)
    else:
        memory = ""
    if use_targeted_memory:
        memory = append_experience_memory(
            memory,
            build_operator_experience_memory(
                "crossover",
                [input_node1, input_node2],
                journal=journal,
                lower_is_better=bool(prompt_context.get("lower_is_better", False)),
                max_related_cards=prompt_memory_max_related_cards(cfg),
                ancestor_k=prompt_memory_operator_k(cfg, "crossover", "ancestor_k", 2),
                sibling_k=prompt_memory_operator_k(cfg, "crossover", "sibling_k", 2),
                sibling_rank_weights=prompt_memory_sibling_rank_weights(cfg),
            ),
        )

    improve_data = {
        "task_desc": task_description,
        "task_description": prompt_context.get("task_description") or task_description,
        "data_description": prompt_context.get("data_description", ""),
        "visible_data_analysis": prompt_context.get("visible_data_analysis", ""),
        "public_system_prompt": prompt_context.get("public_system_prompt", ""),
        "public_user_prompt": prompt_context.get("public_user_prompt", ""),
        "prev_code1": wrap_code(input_node1.code),
        "prev_terminal_output1": wrap_code(
            sanitize_execution_output_for_prompt(
                input_node1.term_out,
                enabled=sanitize_prompt_scores,
            ),
            lang="",
        ),
        "prev_code2": wrap_code(input_node2.code),
        "prev_terminal_output2": wrap_code(
            sanitize_execution_output_for_prompt(
                input_node2.term_out,
                enabled=sanitize_prompt_scores,
            ),
            lang="",
        ),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "improve_complexity": None,
        "memory": None,
        "data_overview": None,
        "packages": pkg_str,
    }

    if memory:
        improve_data["memory"] = memory

    if cfg.data_preview:
        improve_data["data_overview"] = _clean_data_preview(data_preview)

    return crossover_llm(query_data=improve_data)
