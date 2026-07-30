# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt
from dojo.core.solvers.utils.response import wrap_code
from dojo.core.solvers.utils.journal import Node

analyze_schema_without_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "description": "Required. If there is a bug, summarize the root cause and propose a fix. Otherwise, write a short summary (2-3 sentences) of the empirical findings."
        }
    },
    "required": ["is_bug", "summary"],
    "additionalProperties": false
}"""

analyze_schema_with_eval = """{
    "type": "object",
    "properties": {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false."
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "description": "Required. If there is a bug, summarize the root cause and propose a fix. Otherwise, write a short summary (2-3 sentences) of the empirical findings. DO NOT suggest fixes or improvements for successful runs."
        },
        "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
            }
    },
    "required": ["is_bug", "summary", "metric"],
    "additionalProperties": false
}"""


def analyze_op(
    analyze_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    input_node: Node,
    fetch_metric: bool = True,
) -> str:
    code = input_node.code
    execution_output = sanitize_execution_output_for_prompt(
        input_node.term_out,
        enabled=prompt_score_sanitization_enabled(cfg),
    )

    if fetch_metric:
        output_format = (
            "Return exactly one valid JSON object and nothing else. "
            "Use exactly these keys: is_bug, summary, metric. "
            "Do not add keys such as bugs, empirical_findings, validation_auc, "
            "analysis, result, or explanation. "
            "The JSON shape must be: "
            '{"is_bug": false, "summary": "non-empty summary", "metric": 0.0}. '
            "Set is_bug to true only when execution failed or the output shows a bug. "
            "If execution failed, still return the same JSON shape with "
            "is_bug=true, metric=null, and a summary that explains the error. "
            "Set metric to the numeric validation metric if it is reported; otherwise set metric to null. "
            "The summary field is required and must be non-empty. "
            "Never return an empty object or an empty code block. "
            "Do not return markdown, code fences, headings, or prose outside the JSON object."
        )
    else:
        output_format = (
            "Return exactly one valid JSON object and nothing else. "
            "Use exactly these keys: is_bug, summary. "
            "Do not add keys such as bugs, empirical_findings, analysis, result, or explanation. "
            "The JSON shape must be: "
            '{"is_bug": false, "summary": "non-empty summary"}. '
            "If execution failed, still return the same JSON shape with "
            "is_bug=true and a summary that explains the error. "
            "The summary field is required and must be non-empty. "
            "Never return an empty object or an empty code block. "
            "Do not return markdown, code fences, headings, or prose outside the JSON object."
        )

    analyze_data = {
        "task_desc": task_description,
        "code": wrap_code(code),
        "execution_output": wrap_code(execution_output, lang=""),
        "output_format": output_format,
    }

    schema = analyze_schema_with_eval if fetch_metric else analyze_schema_without_eval

    return analyze_llm(
        query_data=analyze_data,
        json_schema=schema,
        function_name="submit_review",
        function_description="Submit a review evaluating the output of the training script.",
        no_user_message=True,
    )
