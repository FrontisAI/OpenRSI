# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

import json
import re

import black

OFFICIAL_SCORE_REDACTION = (
    "[Official sandbox score redacted; use Final Validation Score for search.]"
)

_OFFICIAL_FINAL_SCORE_RE = re.compile(
    r"(?<!Validation )\bFinal Score:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_OFFICIAL_SCORE_MARKER_RE = re.compile(
    r"##SCORE##\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_GRADER_MARKDOWN_SCORE_RE = re.compile(
    r"^\s*\*\*Score\*\*:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)
_GRADER_PLAIN_SCORE_RE = re.compile(
    r"^\s*Score:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def _config_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def prompt_score_sanitization_enabled(cfg) -> bool:
    """Return whether prompt logs should hide official sandbox scores.

    The legacy AIRA-Evo path has no experience config and should therefore
    preserve the original logs exactly. Memory mode enables sanitization by
    default because those prompts are part of search-time decision making.
    """
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    if not bool(_config_get(experience_cfg, "enabled", False)):
        return False
    sanitizer_cfg = (
        _config_get(experience_cfg, "prompt_score_sanitization", {}) or {}
    )
    return bool(_config_get(sanitizer_cfg, "enabled", True))


def sanitize_execution_output_for_prompt(output, *, enabled: bool = True) -> str:
    """Remove official sandbox scores from execution output before prompting.

    The sandbox logs contain two conceptually different scores:
    - ``Final Validation Score`` is produced by the submitted code from visible
      train/CV data and is allowed in search prompts.
    - ``Final Score`` / ``##SCORE##`` / grader ``Score`` are official sandbox
      submission scores and must not guide search-time LLM calls.
    """
    if output is None:
        return ""
    if isinstance(output, (list, tuple)):
        text = "".join(str(part) for part in output)
    else:
        text = str(output)
    if not text:
        return text
    if not enabled:
        return text

    sanitized_lines: list[str] = []
    in_grader_feedback = False
    redaction_inserted = False

    def insert_redaction(newline: str) -> None:
        nonlocal redaction_inserted
        if not redaction_inserted:
            sanitized_lines.append(OFFICIAL_SCORE_REDACTION + newline)
            redaction_inserted = True

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        stripped = body.strip()

        if "submission.csv Grader Feedback" in body or "## Execution Result" in body:
            in_grader_feedback = True

        if _OFFICIAL_FINAL_SCORE_RE.search(body) or _OFFICIAL_SCORE_MARKER_RE.search(
            body
        ):
            insert_redaction(newline)
            continue

        if in_grader_feedback and (
            _GRADER_MARKDOWN_SCORE_RE.match(stripped)
            or _GRADER_PLAIN_SCORE_RE.match(stripped)
        ):
            insert_redaction(newline)
            continue

        sanitized_lines.append(line)

    return "".join(sanitized_lines)


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def parse_thinking_tags(text):
    """
    Extracts the thinking information from text enclosed within <think>...</think> tags.
    Handles cases where the opening or closing tag is missing.
    If both are missing, treats text up to the first code block as thinking information.

    Parameters:
        text (str): The input text generated by the LLM.

    Returns:
        tuple: (thinking_text, text_without_thinking)
    """
    # Case 1: Both <think> and </think> exist
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        text_without_thinking = text.replace(match.group(0), "").strip()
        return thinking_text, text_without_thinking

    # Case 2: Missing <think> but has </think>
    match = re.search(r"(.*?)</think>", text, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        text_without_thinking = text.replace(match.group(0), "").strip()
        return thinking_text, text_without_thinking

    # Default case: No clear separation, return empty thinking and full text
    return "", text.strip()
