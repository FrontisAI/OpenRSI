# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Weco AI Ltd
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/WecoAI/aideml/blob/main/LICENSE

import re

import black
import logging
import json

log = logging.getLogger(__name__)


def is_valid_python_script(script: str) -> bool:
    """
    Check if the provided script is syntactically valid Python code.

    Args:
        script (str): The Python script to validate.

    Returns:
        bool: True if the script compiles without syntax errors, False otherwise.
    """
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def format_code(code: str) -> str:
    """
    Format Python code using Black.

    Args:
        code (str): The Python code to format.

    Returns:
        str: The formatted code if successful; otherwise, the original code.
    """
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def extract_code(text: str) -> str:
    """
    Extract Python code blocks from the given text.

    Searches for code enclosed in triple backticks (optionally with "python")
    and, if none are found, treats the entire text as code.

    Args:
        text (str): The text containing potential Python code blocks.

    Returns:
        str: A single string with all valid, formatted Python code blocks.
    """
    parsed_codes = []
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    if not parsed_codes:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]
    formatted_code = format_code("\n\n".join(valid_code_blocks))
    if bool(formatted_code) and is_valid_python_script(formatted_code):
        return formatted_code

    raise Exception("Solution is not valid python code.")


def parse_json_output(response_text):
    """
    Attempts to extract and parse JSON from a string that might be wrapped in markdown/code blocks,
    contain a leading "json" prefix, or include trailing commas.
    """
    def _remove_trailing_commas(text: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", text)

    def _strip_think_blocks(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    def _scan_first_json_object(text: str):
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.decoder.JSONDecodeError:
                continue
        return None

    if isinstance(response_text, str):
        original_text = response_text  # Keep the original for logging.
        response_text = _strip_think_blocks(response_text).strip()

        # Try simply load it
        try:
            return json.loads(response_text)
        except json.decoder.JSONDecodeError as e:
            log.info("JSON decode error: " + str(e) + " | Original response: " + original_text)

        # 1. Extract JSON from markdown/code block wrappers (```json ... ```).
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
        match = code_block_pattern.search(response_text)
        if match:
            response_text = match.group(1)

        # 2. Remove any leading "json" prefix (case-insensitive).
        response_text = re.sub(r"^json\s*", "", response_text, flags=re.IGNORECASE)

        # 3. Remove trailing commas before closing objects or arrays.
        response_text = _remove_trailing_commas(response_text)

        try:
            return json.loads(response_text)
        except json.decoder.JSONDecodeError as e:
            log.info("JSON decode error: " + str(e) + " | Original response: " + original_text)
            scanned = _scan_first_json_object(_remove_trailing_commas(_strip_think_blocks(original_text)))
            if scanned is not None:
                return scanned
            return {}
    else:
        try:
            return dict(response_text)
        except Exception as e:
            log.info("Error converting non-string response to dict: " + str(e))
            return {}
