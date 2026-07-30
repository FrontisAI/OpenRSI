"""Prompt building utilities for draft/improve/crossover modes."""

from typing import Optional
from program_database import Program
from reward_func_utils import get_clear_log


def _safe_text(text: Optional[str]) -> str:
    return (text or "").strip()


def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    if status_code == 200:
        result = payload.get("result") or {}
        status = payload.get("status", "unknown")
        run_log = get_clear_log(result.get("run_log"))
        run_result = result.get("result", "")
        score = result.get("score")

        feedback = f"## Execution Result\n"
        feedback += f"**Status**: {status}\n\n"

        if score is not None:
            feedback += f"**Score**: {score}\n\n"

        if run_result:
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        feedback = f"## Connection Error\n"
        feedback += f"**Type**: {error_type}\n"
        feedback += f"**Detail**: {error_detail}\n\n"
    else:
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        feedback = f"## Other Error\n"
        feedback += f"**Error**: {error_msg}\n"
        if isinstance(detail, dict):
            if detail.get("status"):
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                feedback += f"**Message**: {detail.get('message')}\n"

    return feedback


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


def build_improve_prompt(
    task_description: str,
    data_description: str,
    parent_program: Program,
    public_user_prompt: str,
) -> str:
    status_code = parent_program.status_code
    payload = parent_program.payload
    feedback = format_sandbox_feedback(status_code, payload)
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
    status_code_a = parent_program.status_code
    status_code_b = secondary_parent_program.status_code
    feedback_a = format_sandbox_feedback(status_code_a, parent_program.payload)
    feedback_b = format_sandbox_feedback(status_code_b, secondary_parent_program.payload)
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



def _airaevo_memory_text(memory: Optional[str], empty_text: str) -> str:
    memory = _safe_text(memory)
    if memory:
        return memory
    return empty_text


def _airaevo_memory_block(
    memory: Optional[str],
    label: str,
    empty_suffix: str,
    post_instructions: tuple[str, ...] = (),
) -> str:
    memory = _safe_text(memory)
    if not memory:
        return f"{label}: {empty_suffix}\n"
    block = f"{label}:\n```markdown\n{memory}\n```\n"
    if post_instructions:
        block += "\n".join(post_instructions) + "\n"
    return block


def _airaevo_data_overview_block(data_overview: Optional[str]) -> str:
    data_overview = _safe_text(data_overview)
    if not data_overview:
        return ""
    lowered = data_overview.lower()
    if lowered in {"data (missing)", "no data preview available", "(no data preview available)"}:
        return ""
    return f"\nRuntime Data Overview:\n```\n{data_overview}\n```\n"


def _airaevo_execution_block(execution_timeout: str | int | float | None) -> str:
    timeout = _safe_text(str(execution_timeout or "")) or "the sandbox timeout"
    if timeout.replace(".", "", 1).isdigit():
        timeout = f"{timeout}s"
    return (
        "Execution Budget:\n\n"
        f"Be aware of the running time of the code, it should complete within {timeout}.\n"
        f"Your code will be executed in a sandbox after generation.\n"
        f"A single sandbox run must finish within {timeout}.\n"
        "Always choose a validation/training strategy that leaves enough time to write ./submission.csv.\n"
    )


def _airaevo_submission_check() -> str:
    return (
        "Submission Check:\n\n"
        "Before finishing, verify that ./submission.csv exists, has the same row count and required columns/order as sample_submission.csv when available, preserves the sample id/order, and contains no NaN/inf values.\n"
        "When building features, align train/test columns explicitly and handle missing values before model fitting or prediction.\n"
    )


def _airaevo_feedback(program: Program) -> str:
    return format_sandbox_feedback(program.status_code, program.payload)


def build_airaevo_prompt(
    mode: str,
    task_description: str,
    data_description: str,
    public_system_prompt: str,
    public_user_prompt: str,
    virtual_data_dir: str,
    parent_program: Optional[Program] = None,
    secondary_parent_program: Optional[Program] = None,
    max_steps: int = 1,
    memory: Optional[str] = None,
    data_overview: Optional[str] = None,
    execution_timeout: str | int | float | None = None,
) -> tuple[str, str]:
    """Build prompts aligned with the latest inference AIRA-Evo operator templates."""
    _ = virtual_data_dir
    _ = max_steps
    mode = str(mode or "draft").lower()
    public_system_prompt = _safe_text(public_system_prompt)
    public_user_prompt = _safe_text(public_user_prompt)
    task_description = _safe_text(task_description)
    data_description = _safe_text(data_description)
    data_block = _airaevo_data_overview_block(data_overview)
    execution_block = _airaevo_execution_block(execution_timeout)
    submission_block = _airaevo_submission_check()

    if mode == "draft":
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python.\n"
            f"{public_system_prompt}"
        )
        memory_block = _airaevo_memory_block(memory, "Previously explored ideas", "none yet.")
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "Draft Guidance:\n\n"
            "Please focus on proposing an advanced idea addressing this task.\n"
            "Propose exactly one new solution idea that is different from the previously explored ideas.\n"
            "Keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        return system_prompt, user_prompt

    if mode == "improve":
        if parent_program is None:
            raise ValueError("parent_program is required for AIRA improve mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python, that improves upon an existing solution to the task.\n"
            f"{public_system_prompt}"
        )
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted memory for this improvement",
            "none available.",
            (
                "Use the selected parent memory to preserve what already works.",
                "Use vertical ancestor memory to understand the recent evolution path and which changes helped or hurt.",
                "Use horizontal sibling memory to avoid repeating nearby failed alternatives and to borrow useful details.",
                "Use board stats only as weak global guidance; prioritize evidence from the selected parent branch.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "A previous solution was attempted with the following results:\n\n"
            "Code:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Improve Guidance:\n\n"
            "Based on the previous attempt, its score, and execution output, please analyze what worked well and what didn't, then improve the solution.\n"
            "Propose exactly one improvement idea that is different from the previously explored improvement ideas.\n"
            "Keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        return system_prompt, user_prompt

    if mode == "debug":
        if parent_program is None:
            raise ValueError("parent_program is required for AIRA debug mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, you need to come up with an excellent and creative plan for a solution and then implement this solution in Python, that debugs and fixes an existing solution to the task.\n"
            f"{public_system_prompt}"
        )
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted debug memory",
            "none available.",
            (
                "Use same-error memories first to avoid repeating known failure modes.",
                "Use recent related memories to preserve working modeling logic while fixing the bug.",
                "Make the smallest code change that resolves the failure before attempting extra improvements.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "A previous solution was attempted with the following results:\n\n"
            "Code:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Debug Guidance:\n\n"
            "Based on the previous attempt, its score, and execution output, please identify the root cause of the failure and fix the code so it runs successfully and follows the output requirements. After it runs, improve the solution if possible.\n"
            "Preserve the current core idea unless the feedback shows that it cannot be made valid.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}"
            "If the previous attempt timed out, reduce compute first: fewer epochs/folds, smaller input size, smaller model, cached features, or simpler ensembling.\n\n"
            f"{submission_block}\n\n{public_user_prompt}"
        )
        return system_prompt, user_prompt

    if mode == "crossover":
        if parent_program is None or secondary_parent_program is None:
            raise ValueError("Both parent_program and secondary_parent_program are required for AIRA crossover mode")
        system_prompt = (
            "You are a Kaggle Grandmaster attending a high-stakes competition. Your goal is to synthesize the strengths of two provided solutions, neutralize their respective weaknesses, and propose novel improvements that neither original solution possessed. A simple \"merge\" of existing code won't suffice to win. You must identify the hidden synergies between previous attempts and engineering a third-generation solution that transcends its predecessors.\n"
            f"{public_system_prompt}"
        )
        memory_block = _airaevo_memory_block(
            memory,
            "Targeted memory for this crossover",
            "none available.",
            (
                "Use parent memories to identify each branch's strongest reusable components.",
                "Use vertical ancestor memory to understand how each branch evolved.",
                "Use horizontal sibling memory to avoid combining changes that already failed nearby.",
                "Prefer a coherent hybrid with clear synergy over mechanically concatenating both codebases.",
            ),
        )
        user_prompt = (
            "We will now provide a description of the task.\n\n"
            "Task Description:\n\n"
            f"{task_description}\n\n"
            "Data Description:\n\n"
            f"{data_description}\n\n"
            "Two previous solutions were attempted with the following results:\n\n"
            "Code 1:\n\n"
            f"{_safe_text(parent_program.code)}\n\n"
            "Feedback 1:\n\n"
            f"{_airaevo_feedback(parent_program)}\n\n"
            "Code 2:\n\n"
            f"{_safe_text(secondary_parent_program.code)}\n\n"
            "Feedback 2:\n\n"
            f"{_airaevo_feedback(secondary_parent_program)}\n\n"
            "Crossover Guidance:\n\n"
            "Based on the previous attempts, their scores, and execution outputs, please synthesize the winning components of both solutions into a superior hybrid while introducing novel improvements to address shared weaknesses.\n"
            "Propose exactly one crossover plan and keep the evaluation method consistent across iterations.\n\n"
            "AIRA Evo Search Context:\n\n"
            f"{memory_block}"
            f"{data_block}\n"
            f"{execution_block}\n\n{submission_block}\n\n{public_user_prompt}"
        )
        return system_prompt, user_prompt

    raise ValueError(f"Unknown AIRA mode: {mode}")


def build_airaevo_rich_memory_summary_prompt(
    task_description: str,
    current_program: Program,
    parent_program: Optional[Program],
    current_card: dict,
) -> str:
    parent_card = parent_program.metadata.get("airaevo_experience_card", {}) if parent_program and isinstance(parent_program.metadata, dict) else {}
    return (
        "You are an expert machine learning experiment analyst.\n"
        "Your job is to distill one completed experiment node into reusable memory.\n\n"
        "Return exactly one valid JSON object and nothing else. Use exactly these keys:\n"
        "{\"method_overview\": string, \"parent_comparison_experience\": string}\n\n"
        "# Task Description\n"
        f"{_safe_text(task_description)}\n\n"
        "# Current Node Metadata\n"
        f"node_id: {current_program.id}\n"
        f"operator_that_created_this_node: {current_program.generation_mode}\n"
        f"method_family: {current_card.get('method_family_auto') or 'unknown'}\n"
        f"status: {current_card.get('status') or 'unknown'}\n"
        f"score: {current_card.get('fitness')}\n"
        f"delta_vs_parent: {current_card.get('delta_vs_parent')}\n"
        f"runtime_seconds: {current_card.get('sandbox_time_used')}\n"
        f"error_signature: {current_card.get('error_signature')}\n\n"
        "# Parent Node Metadata\n"
        f"parent_node_id: {parent_program.id if parent_program else 'none'}\n"
        f"parent_method_family: {parent_card.get('method_family_auto') or 'unknown'}\n"
        f"parent_score: {parent_card.get('fitness')}\n\n"
        "# Current Node Plan / Raw Response\n"
        f"{_safe_text(current_program.raw_text)[:5000]}\n\n"
        "# Current Node Code\n"
        "```python\n"
        f"{_safe_text(current_program.code)[:6000]}\n"
        "```\n\n"
        "# Current Node Execution Output\n"
        "```\n"
        f"{format_sandbox_feedback(current_program.status_code, current_program.payload)[:4000]}\n"
        "```\n\n"
        "# Parent Code\n"
        "```python\n"
        f"{_safe_text(parent_program.code)[:4000] if parent_program else ''}\n"
        "```\n\n"
        "# Instructions\n"
        "Write method_overview in 2-5 concise sentences describing the concrete method used by the current node.\n"
        "Write parent_comparison_experience in 2-5 concise sentences comparing current node vs parent and extracting reusable success or failure experience.\n"
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
    """
    Build prompt based on mode.

    Args:
        mode: One of 'draft' | 'improve' | 'debug' | 'crossover'
        task_description: Task description text
        data_description: Data description text
        public_system_prompt: Existing system prompt to append
        public_user_prompt: Existing user prompt to append
        virtual_data_dir: Virtual data directory path (kept for compatibility)
        parent_program: Parent program (required for improve mode)
        max_steps: Maximum number of steps (kept for compatibility)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    _ = virtual_data_dir
    _ = max_steps
    if mode == "draft":
        return (
            build_draft_system_prompt(public_system_prompt),
            build_draft_prompt(task_description, data_description, public_user_prompt),
        )
    elif mode in ("improve", "debug"):
        if parent_program is None:
            raise ValueError("parent_program is required for improve/debug mode")
        return (
            build_improve_system_prompt(public_system_prompt),
            build_improve_prompt(
                task_description=task_description,
                data_description=data_description,
                parent_program=parent_program,
                public_user_prompt=public_user_prompt,
            ),
        )
    elif mode == "crossover":
        if parent_program is None or secondary_parent_program is None:
            raise ValueError("Both parent_program and secondary_parent_program are required for crossover mode")
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
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of draft/improve/debug/crossover")
