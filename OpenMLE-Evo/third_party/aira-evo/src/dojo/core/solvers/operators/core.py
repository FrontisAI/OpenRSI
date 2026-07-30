# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from dojo.core.solvers.utils.response import extract_code, extract_text_up_to_code, parse_thinking_tags


def execute_op_plan_code(
    operator_fn: Callable, *operator_args, max_operator_tries: int, requires_plan: bool = False
) -> tuple[str, str, str]:
    """Executes an operator function with the given arguments, attempts to extract the generated plan/code from the output
    and retries if the extraction fails."""
    completion_text = None
    text_without_thinking = None
    for _ in range(max_operator_tries):
        completion_text, metrics = operator_fn(*operator_args)
        thinking_text, text_without_thinking = parse_thinking_tags(completion_text)
        code = extract_code(text_without_thinking)
        plan = extract_text_up_to_code(text_without_thinking)

        if code:
            if requires_plan and not plan:
                print("Plan extraction failed, retrying...")
                continue
            return plan, code, metrics

        print("Retrying Extraction...")

    return "", text_without_thinking, metrics
