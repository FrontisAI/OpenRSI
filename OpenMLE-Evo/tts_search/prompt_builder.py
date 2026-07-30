"""Prompt building utilities aligned with prompt_alignment/.

This module builds the *final* (system_prompt, user_prompt) pair for the LLM.
It supports draft / improve(debug) / crossover modes.
"""

from __future__ import annotations

from typing import Optional

from tts_search.program_database import Program


def _safe_text(text: str | None) -> str:
    return (text or "").strip()


def split_legacy_public_prompts(
    public_system_prompt: str, public_user_prompt: str
) -> tuple[str, str, str, str]:
    """
    Split legacy fully-assembled prompts into the 4 fragments expected by the aligned templates:

    - task_description: between "Task Description:" and "Data Preview:"
    - data_description: between "Data Preview:" and "Output Instructions:"
    - public_user_prompt: starting from "Output Instructions:" (inclusive)
    - public_system_prompt: the legacy system prompt with the first line removed
      (to avoid duplicating the template's first line)

    If the legacy markers are not found, returns empty descriptions and keeps the prompts unchanged.
    """

    user_text = (public_user_prompt or "").strip()
    task_marker = "Task Description:"
    data_marker = "Data Preview:"
    output_marker = "Output Instructions:"

    task_index = user_text.find(task_marker)
    data_index = user_text.find(data_marker)
    output_index = user_text.find(output_marker)
    if (
        task_index == -1
        or data_index == -1
        or output_index == -1
        or not (task_index < data_index < output_index)
    ):
        return "", "", (public_system_prompt or "").strip(), user_text

    task_description = user_text[task_index + len(task_marker) : data_index].strip()
    data_description = user_text[data_index + len(data_marker) : output_index].strip()
    extracted_public_user_prompt = user_text[output_index:].strip()

    system_text = (public_system_prompt or "").strip()
    legacy_prefix = "You are a Kaggle Grandmaster attending a high-stakes competition."
    if system_text.startswith(legacy_prefix):
        _, _, system_text = system_text.partition("\n")

    return task_description, data_description, system_text, extracted_public_user_prompt


def build_draft_system_prompt(public_system_prompt: str) -> str:
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_draft_prompt(task_description: str, data_description: str, public_user_prompt: str) -> str:
    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Draft Guidance:\n\n"
        "Please focus on proposing an advanced idea addressing this task.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_improve_system_prompt(public_system_prompt: str) -> str:
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python, "
        "that improves upon an existing solution to the task.\n"
        f"{_safe_text(public_system_prompt)}"
    )

def build_debug_system_prompt(public_system_prompt: str) -> str:
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python, "
        "that debugs and fixes an existing solution to the task.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_improve_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    feedback = parent_program.feedback
    parent_code = _safe_text(parent_program.code)

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "A previous solution was attempted with the following results:\n\n"
        "Code:\n\n"
        f"{parent_code}\n\n"
        "Feedback:\n\n"
        f"{feedback}\n\n"
        "Improve Guidance:\n\n"
        "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )

def build_debug_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    feedback = parent_program.feedback
    parent_code = _safe_text(parent_program.code)

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "A previous solution was attempted with the following results:\n\n"
        "Code:\n\n"
        f"{parent_code}\n\n"
        "Feedback:\n\n"
        f"{feedback}\n\n"
        "Debug Guidance:\n\n"
        "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_crossover_system_prompt(public_system_prompt: str) -> str:
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "Your goal is to synthesize the strengths of two provided solutions, neutralize their respective weaknesses, "
        'and propose novel improvements that neither original solution possessed. A simple "merge" of existing code won\'t suffice to win. '
        "You must identify the hidden synergies between previous attempts and engineering a third-generation solution that transcends its predecessors.\n"
        f"{_safe_text(public_system_prompt)}"
    )


def build_crossover_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    secondary_parent_program: Program,
    public_user_prompt: str,
) -> str:
    feedback_a = parent_program.feedback
    feedback_b = secondary_parent_program.feedback
    parent_code_a = parent_program.code
    parent_code_b = secondary_parent_program.code

    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Two previous solutions were attempted with the following results:\n\n"
        "Code 1:\n\n"
        f"{_safe_text(parent_code_a)}\n\n"
        "Feedback 1:\n\n"
        f"{feedback_a}\n\n"
        "Code 2:\n\n"
        f"{_safe_text(parent_code_b)}\n\n"
        "Feedback 2:\n\n"
        f"{feedback_b}\n\n"
        "Crossover Guidance:\n\n"
        "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n\n"
        f"{_safe_text(public_user_prompt)}"
    )


def build_prompt(
    mode: str,
    task_description: str,
    data_description: str,
    public_system_prompt: str,
    public_user_prompt: str,
    virtual_data_dir: str,
    parent_program: Optional[Program] = None,
    secondary_parent_program: Optional[Program] = None,
    max_steps: int = 1,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) based on mode.

    Note: `virtual_data_dir` and `max_steps` are kept for compatibility.
    """

    _ = virtual_data_dir
    _ = max_steps

    # Legacy fallback: some datasets already provide fully assembled prompts in `prompt`.
    # In that case metadata may not contain task/data descriptions; we pass through for draft,
    # and append improve/crossover guidance on top for later steps.
    if _safe_text(task_description) == "" and _safe_text(data_description) == "":
        system_prompt = _safe_text(public_system_prompt)
        user_prompt = _safe_text(public_user_prompt)
        if mode == "draft":
            return system_prompt, user_prompt
        if mode in ("improve", "debug"):
            if parent_program is None:
                raise ValueError("parent_program is required for improve/debug mode")
            parent_code = _safe_text(parent_program.code)
            feedback = parent_program.feedback
            guidance_title = "Improve Guidance"
            guidance_text = (
                "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n"
            )
            if mode == "debug":
                guidance_title = "Debug Guidance"
                guidance_text = (
                    "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n"
                )
            appended = (
                "A previous solution was attempted with the following results:\n\n"
                "Code:\n\n"
                f"{parent_code}\n\n"
                "Feedback:\n\n"
                f"{feedback}\n\n"
                f"{guidance_title}:\n\n"
                f"{guidance_text}"
            )
            return system_prompt, f"{user_prompt}\n\n{appended}"
        if mode == "crossover":
            if parent_program is None or secondary_parent_program is None:
                raise ValueError(
                    "Both parent_program and secondary_parent_program are required for crossover mode"
                )
            parent_code_a = _safe_text(parent_program.code)
            parent_code_b = _safe_text(secondary_parent_program.code)
            feedback_a = parent_program.feedback
            feedback_b = secondary_parent_program.feedback
            appended = (
                "Two previous solutions were attempted with the following results:\n\n"
                "Code 1:\n\n"
                f"{parent_code_a}\n\n"
                "Feedback 1:\n\n"
                f"{feedback_a}\n\n"
                "Code 2:\n\n"
                f"{parent_code_b}\n\n"
                "Feedback 2:\n\n"
                f"{feedback_b}\n\n"
                "Crossover Guidance:\n\n"
                "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n"
            )
            return system_prompt, f"{user_prompt}\n\n{appended}"

    if mode == "draft":
        return (
            build_draft_system_prompt(public_system_prompt),
            build_draft_prompt(task_description, data_description, public_user_prompt),
        )
    if mode in ("improve", "debug"):
        if parent_program is None:
            raise ValueError("parent_program is required for improve/debug mode")
        if mode == "debug":
            return (
                build_debug_system_prompt(public_system_prompt),
                build_debug_prompt(
                    task_description=task_description,
                    data_description=data_description,
                    parent_program=parent_program,
                    public_user_prompt=public_user_prompt,
                ),
            )
        return (
            build_improve_system_prompt(public_system_prompt),
            build_improve_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    if mode == "crossover":
        if parent_program is None or secondary_parent_program is None:
            raise ValueError(
                "Both parent_program and secondary_parent_program are required for crossover mode"
            )
        return (
            build_crossover_system_prompt(public_system_prompt),
            build_crossover_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                secondary_parent_program=secondary_parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    raise ValueError(f"Unknown mode: {mode}. Must be one of draft/improve/debug/crossover")
