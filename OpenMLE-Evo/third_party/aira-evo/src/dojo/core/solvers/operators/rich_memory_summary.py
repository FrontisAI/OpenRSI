from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Node
from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt
from dojo.core.solvers.utils.response import wrap_code

rich_memory_summary_schema = """{
    "type": "object",
    "properties": {
        "method_overview": {
            "type": "string",
            "minLength": 1,
            "description": "2-5 concise sentences describing the concrete method used by the current node."
        },
        "parent_comparison_experience": {
            "type": "string",
            "minLength": 1,
            "description": "2-5 concise sentences comparing the current node against its parent and extracting reusable success or failure experience."
        }
    },
    "required": ["method_overview", "parent_comparison_experience"],
    "additionalProperties": false
}"""


def _metric_info(node: Node | None) -> dict[str, Any]:
    if node is None:
        return {}
    info = getattr(getattr(node, "metric", None), "info", {}) or {}
    return dict(info) if isinstance(info, dict) else {}


def _card(node: Node | None) -> dict[str, Any]:
    if node is None:
        return {}
    card = getattr(node, "experience_card", None)
    return dict(card) if isinstance(card, dict) else {}


def _value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def rich_memory_summary_op(
    rich_memory_summary_llm: GenericLLM,
    cfg: DictConfig,
    task_description: str,
    input_node: Node,
    parent_node: Node | None = None,
) -> str:
    current_card = _card(input_node)
    parent_card = _card(parent_node)
    current_info = _metric_info(input_node)
    parent_info = _metric_info(parent_node)
    sanitize_prompt_scores = prompt_score_sanitization_enabled(cfg)

    query_data = {
        "task_description": task_description,
        "node_id": getattr(input_node, "id", ""),
        "operator": (list(getattr(input_node, "operators_used", []) or [""])[0] or ""),
        "method_family": current_card.get("method_family_auto") or "unknown",
        "status": _value(
            current_card.get("status"), current_info.get("status"), "unknown"
        ),
        "score": _value(
            current_card.get("fitness"),
            getattr(getattr(input_node, "metric", None), "value", None),
        ),
        "delta_vs_parent": current_card.get("delta_vs_parent"),
        "runtime": _value(
            current_card.get("sandbox_time_used"),
            getattr(input_node, "exec_time", None),
        ),
        "error_signature": current_card.get("error_signature"),
        "parent_node_id": (
            getattr(parent_node, "id", "") if parent_node is not None else "none"
        ),
        "parent_method_family": parent_card.get("method_family_auto") or "unknown",
        "parent_status": _value(
            parent_card.get("status"), parent_info.get("status"), "unknown"
        ),
        "parent_score": _value(
            parent_card.get("fitness"),
            getattr(getattr(parent_node, "metric", None), "value", None),
        ),
        "parent_runtime": _value(
            parent_card.get("sandbox_time_used"),
            getattr(parent_node, "exec_time", None),
        ),
        "current_plan": getattr(input_node, "plan", "") or "",
        "parent_plan": (
            getattr(parent_node, "plan", "") if parent_node is not None else ""
        ),
        "current_code": wrap_code(getattr(input_node, "code", "") or ""),
        "current_execution_output": wrap_code(
            sanitize_execution_output_for_prompt(
                getattr(input_node, "term_out", "") or "",
                enabled=sanitize_prompt_scores,
            ),
            lang="",
        ),
        "parent_code": wrap_code(
            getattr(parent_node, "code", "") if parent_node is not None else ""
        ),
        "parent_execution_output": wrap_code(
            sanitize_execution_output_for_prompt(
                getattr(parent_node, "term_out", "") if parent_node is not None else "",
                enabled=sanitize_prompt_scores,
            ),
            lang="",
        ),
    }

    return rich_memory_summary_llm(
        query_data=query_data,
        json_schema=rich_memory_summary_schema,
        function_name="write_rich_memory_summary",
        function_description="Write reusable memory for one completed ML experiment node.",
    )
