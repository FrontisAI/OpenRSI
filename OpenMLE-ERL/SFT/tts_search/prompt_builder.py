"""
Prompt building utilities for draft and improve modes.
"""
import os
from typing import Any, Optional, Sequence
from tts_search.program_database import Program
from tts_search.reward_func_utils import get_clear_log

# moved to system prompt
# You are a top-ranked Kaggle grandmaster with extensive competition experience.
# Your objective is to solve a Kaggle competition,
# with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
# You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem,
# the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
# You are likely to train models according to specific competition requirements.
# You have access to a GPU and several CPUs for training DL/ML models.
# Use cuda and PyTorch for faster training if needed.

# moved to user prompt
# You have a total of {max_steps} actions available.
# For a solution and then implement this solution in Python. We will now provide a description of the task.

# Task Description:
# {description}

# Output Instructions:
# The code should be a single-file python program that is self-contained and can be executed as-is.
# No parts of the code should be skipped, don't terminate the script before finishing the code.
# Your response should only contain a single code block.
# All the provided input data is stored in {virtual_public_data_dir} directory.
# **Save test predictions to `submission.csv` in ./ directory as specified in the task description.** This file is critical for evaluation and scoring.
# Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.
# Your goal is to achieve the highest score, so ensure your code produces this file correctly.


def _safe_text(text: str | None) -> str:
    return (text or "").strip()


def split_legacy_public_prompts(
    public_system_prompt: str, public_user_prompt: str
) -> tuple[str, str, str, str]:
    """Split old fully assembled public prompts into task/data/template parts."""

    user_text = _safe_text(public_user_prompt)
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
        return "", "", _safe_text(public_system_prompt), user_text

    task_description = user_text[task_index + len(task_marker) : data_index].strip()
    data_description = user_text[data_index + len(data_marker) : output_index].strip()
    extracted_public_user_prompt = user_text[output_index:].strip()

    system_text = _safe_text(public_system_prompt)
    legacy_prefix = "You are a Kaggle Grandmaster attending a high-stakes competition."
    if system_text.startswith(legacy_prefix):
        _, _, system_text = system_text.partition("\n")

    return task_description, data_description, system_text, extracted_public_user_prompt


def _format_time_limit(time_limit_value: Any) -> str:
    """Format a configured runtime limit for prompt text."""
    if time_limit_value is None:
        return "the allowed time budget"
    if isinstance(time_limit_value, bool):
        return "the allowed time budget"
    if isinstance(time_limit_value, (int, float)):
        return f"{time_limit_value:g} seconds"
    value = _safe_text(str(time_limit_value))
    return value or "the allowed time budget"


def build_pass_k_draft_system_prompt(public_system_prompt: str) -> str:
    """Build the system prompt used by the pass@k draft pipeline."""
    return (
        "You are a Kaggle Grandmaster attending a high-stakes competition. "
        "In order to win this competition, you need to come up with an excellent "
        "and creative plan for a solution and then implement this solution in Python.\n"
        f"{_safe_text(public_system_prompt)}"
    ).strip()


def build_pass_k_draft_user_prompt(
    task_description: str,
    data_description: str,
    public_user_prompt: str,
    time_limit: bool = True,
    time_limit_value: Any = None,
) -> str:
    """Build the user prompt used by the pass@k draft pipeline."""
    if time_limit:
        formatted_time_limit = _format_time_limit(time_limit_value)
        # valid: remind model to be mindful of runtime
        guidance = (
            "Please focus on proposing an advanced idea addressing this task.\n\n"
            f"ML tasks have a strict execution time limit of {formatted_time_limit} seconds. "
            "Treat finishing within this exact limit as a hard requirement, not a soft preference. "
            "Design and implement a solution that reliably completes before the timeout and still writes `submission.csv` correctly.\n\n"
        )
    else:
        # valid without time limit / no-valid
        guidance = "Please focus on proposing an advanced idea addressing this task.\n\n"
    return (
        "We will now provide a description of the task.\n\n"
        "Task Description:\n\n"
        f"{_safe_text(task_description)}\n\n"
        "Data Description:\n\n"
        f"{_safe_text(data_description)}\n\n"
        "Draft Guidance:\n\n"
        f"{guidance}"
        f"{_safe_text(public_user_prompt)}"
    ).strip()


def _extract_public_prompt_components(
    messages: Sequence[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Extract the first public system/user prompts from stored task messages."""
    public_system_prompt = ""
    public_user_prompt = ""
    for message in messages or []:
        role = str(message.get("role", "")).strip().lower()
        content = _safe_text(str(message.get("content", "")))
        if role == "system" and not public_system_prompt:
            public_system_prompt = content
        elif role == "user" and not public_user_prompt:
            public_user_prompt = content
        if public_system_prompt and public_user_prompt:
            break
    return public_system_prompt, public_user_prompt


def build_pass_k_draft_messages(
    metadata: dict[str, Any],
    messages: Sequence[dict[str, Any]] | None,
    time_limit: bool = True,
    time_limit_value: Any = None,
) -> list[dict[str, str]]:
    """
    Rebuild pass@k draft messages from task metadata plus public prompts.

    The parquet rows store task/data descriptions in metadata and public
    system/user instructions in the input prompt. This builder mirrors the
    reference MLE draft prompt structure by combining them into a fresh
    system/user message pair.
    """
    public_system_prompt, public_user_prompt = _extract_public_prompt_components(
        messages
    )
    return [
        {
            "role": "system",
            "content": build_pass_k_draft_system_prompt(public_system_prompt),
        },
        {
            "role": "user",
            "content": build_pass_k_draft_user_prompt(
                task_description=str(metadata.get("task_description", "")),
                data_description=str(metadata.get("data_description", "")),
                time_limit=time_limit,
                time_limit_value=time_limit_value,
                public_user_prompt=public_user_prompt,
            ),
        },
    ]


def build_draft_prompt(description: str, virtual_data_dir: str, max_steps: int = 1) -> str:
    """
    Build prompt for draft mode (generating code from scratch).

    Args:
        description: Task description text
        virtual_data_dir: Virtual task data path (for example, <TASK_DATA_ROOT>/task_name).
        max_steps: Maximum number of steps (not used in single-turn, kept for compatibility)

    Returns:
        Description text
    """

    return description

# moved to system prompt
# You are a top-ranked Kaggle grandmaster with extensive competition experience.
# Your objective is to solve a Kaggle competition,
# with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
# You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem,
# the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
# You are likely to train models according to specific competition requirements.
# You have access to a GPU and several CPUs for training DL/ML models.
# Use cuda and PyTorch for faster training if needed.

# moved to user prompt
# You have a total of {max_steps} actions available.
# For a solution and then implement this solution in Python. We will now provide a description of the task.

# Task Description:
# {description}

# Output Instructions:
# The code should be a single-file python program that is self-contained and can be executed as-is.
# No parts of the code should be skipped, don't terminate the script before finishing the code.
# Your response should only contain a single code block.
# All the provided input data is stored in {virtual_public_data_dir} directory.
# **Save test predictions to `submission.csv` in ./ directory as specified in the task description.** This file is critical for evaluation and scoring.
# Your metric score (position score) depends on generating a valid `submission.csv` file at the correct location.
# Your goal is to achieve the highest score, so ensure your code produces this file correctly.

def build_improve_prompt(description: str, virtual_data_dir: str, parent_program: Program, max_steps: int = 1) -> str:
    """
    Build prompt for improve mode (improving existing code).

    Args:
        description: Task description text (already formatted with <|im_start|> and <|im_end|> tags)
        virtual_data_dir: Virtual data directory path
        parent_program: Parent program to improve upon
        payload: Payload from sandbox execution
        max_steps: Maximum number of steps (not used in single-turn, kept for compatibility)

    Returns:
        Complete prompt for improve mode with parent info inserted into user section
    """

    # Format parent program information
    feedback = parent_program.feedback
    parent_code = parent_program.code

    return f"""
{description}
Previous Attempt:
A previous solution was attempted with the following results:

Code:
```python
{parent_code}
```

{feedback}

Your Task:
Based on the previous attempt, its score, and execution output, please improve the solution.

Analyze what worked well and what didn't. Consider:
1. Were there any errors or issues in the execution?
2. Can the model/approach be improved?
3. Are there better feature engineering techniques?
4. Can hyperparameters be tuned?
5. Should a different model or ensemble be tried?

Please improve your code based on this feedback. Your goal is to achieve a BETTER performance than the previous attempt, so ensure your improvements are effective.
"""


def build_prompt(mode: str, description: str, virtual_data_dir: str,
                parent_program: Optional[Program] = None, max_steps: int = 1) -> str:
    """
    Build prompt based on mode.

    Args:
        mode: Either 'draft' or 'improve'
        description: Task description text
        virtual_data_dir: Virtual data directory path
        parent_program: Parent program (required for improve mode)
        payload: Payload from sandbox execution
        max_steps: Maximum number of steps

    Returns:
        Complete prompt
    """
    if mode == 'draft':
        return build_draft_prompt(description, virtual_data_dir, max_steps)
    elif mode == 'improve':
        if parent_program is None:
            raise ValueError("parent_program is required for improve mode")
        return build_improve_prompt(description, virtual_data_dir, parent_program, max_steps)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'draft' or 'improve'")
