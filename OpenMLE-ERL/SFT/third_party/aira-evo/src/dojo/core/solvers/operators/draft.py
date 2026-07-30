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
from dojo.solvers.utils import Complexity


def draft_op(
    draft_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    prompt_context: dict[str, str],
    task_description: str,
    journal: Journal,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
    complexity: Optional[Complexity] = None,
    parent_node: Optional[Node] = None,
) -> str:
    pkgs = cfg.available_packages
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    if memory_op is not None:
        memory = memory_op(journal, parent_node)
    else:
        memory = ""

    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    steps_remaining = cfg.step_limit - step_count

    draft_data = {
        "task_desc": task_description,
        "task_description": prompt_context.get("task_description") or task_description,
        "data_description": prompt_context.get("data_description", ""),
        "public_system_prompt": prompt_context.get("public_system_prompt", ""),
        "public_user_prompt": prompt_context.get("public_user_prompt", ""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "packages": pkg_str,
        "other_remarks": None,
        "draft_complexity": None,
        "memory": None,
        "data_overview": None,
    }

    if memory:
        draft_data["memory"] = memory

    if cfg.data_preview and data_preview is not None:
        draft_data["data_overview"] = data_preview
    else:
        draft_data["data_overview"] = "(No data preview available)"

    if complexity is not None:
        draft_data["draft_complexity"] = complexity.value

    return draft_llm(query_data=draft_data)
