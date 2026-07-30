# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import Optional, Callable

import humanize
from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import wrap_code


def debug_op(
    debug_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    prompt_context: dict[str, str],
    task_description: str,
    journal: Journal,
    input_node: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    pkgs = cfg.available_packages
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    if memory_op is not None:
        memory = memory_op(journal, input_node)
    else:
        memory = None

    prev_buggy_code = input_node.code
    execution_output = input_node.term_out

    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    steps_remaining = cfg.step_limit - step_count

    debug_data = {
        "task_desc": task_description,
        "task_description": prompt_context.get("task_description") or task_description,
        "data_description": prompt_context.get("data_description", ""),
        "public_system_prompt": prompt_context.get("public_system_prompt", ""),
        "public_user_prompt": prompt_context.get("public_user_prompt", ""),
        "prev_buggy_code": wrap_code(prev_buggy_code),
        "execution_output": wrap_code(execution_output, lang=""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "packages": pkg_str,
        "memory": None,
        "data_overview": None,
    }

    if memory:
        debug_data["memory"] = memory

    if cfg.data_preview and data_preview is not None:
        debug_data["data_overview"] = data_preview
    else:
        debug_data["data_overview"] = "(No data preview available)"

    return debug_llm(query_data=debug_data)
