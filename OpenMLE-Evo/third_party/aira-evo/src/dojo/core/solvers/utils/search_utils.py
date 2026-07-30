# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fcntl
import inspect
import math
import multiprocessing
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import black
import numpy


def is_valid_python_script(script):
    """
    Check if a given script is a valid Python script.

    This function attempts to compile the provided script. If the compilation
    is successful, it returns True, indicating that the script is syntactically
    correct Python code. If a SyntaxError is raised during compilation, it
    returns False, indicating invalid Python syntax.

    Args:
        script (str): The Python script to validate.

    Returns:
        bool: True if the script is valid Python code, False otherwise.
    """
    try:
        # Attempt to compile the script in 'exec' mode
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        # If a SyntaxError occurs, the script is invalid
        return False


def format_code(code) -> str:
    """
    Format Python code using the Black code formatter.

    This function uses Black to format the provided Python code string. If the
    code is invalid and cannot be parsed by Black, it returns the original
    unformatted code.

    Args:
        code (str): The Python code to format.

    Returns:
        str: The formatted Python code if successful; otherwise, the original code.
    """
    try:
        # Format the code using Black with default settings
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        # If Black cannot parse the code, return it unchanged
        return code


def extract_code(text):
    """
    Extract Python code blocks from a given text.

    This function searches the input text for Python code blocks enclosed within
    triple backticks (```python ... ```). It extracts these code blocks, formats
    them using Black, and ensures they are valid Python scripts. If no such
    blocks are found, it attempts to extract the entire text as a code block.

    Args:
        text (str): The input text containing potential Python code blocks.

    Returns:
        str: A single string containing all valid, formatted Python code blocks
             extracted from the input text, separated by two newlines.
    """
    parsed_codes = []

    # Regex pattern to find code blocks enclosed in ```python ... ```
    # The (python)? makes the 'python' specifier optional
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # If no code blocks were found with backticks, try extracting the entire text
    if len(parsed_codes) == 0:
        # Regex pattern to match the entire text as a code block, with optional backticks
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # Validate each extracted code block and format it
    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]

    # Combine all valid code blocks into a single string separated by two newlines
    return format_code("\n\n".join(valid_code_blocks))


def opt_messages_to_list(
    system_message: str | None, user_message: str | None, content_key: str
) -> list[dict[str, str]]:
    """Helper function to turn optional system/user messages into a list."""
    messages = []
    if system_message:
        messages.append({"role": "system", content_key: str(system_message)})
    if user_message:
        messages.append({"role": "user", content_key: str(user_message)})
    return messages


def define_function_from_string(
    function_string: str,
) -> Tuple[Optional[Callable], List[str]]:
    """
    Takes a string containing a function definition and returns the defined function.

    Args:
    - function_string (str): The string containing the function definition.

    Returns:
    - function: The defined function.
    """
    namespace = {}
    additional_globals = {
        "math": math,
        "numpy": numpy,
        "Tuple": Tuple,
        "List": List,
        "Callable": Callable,
        "Any": Any,
        "Union": Union,
        "Optional": Optional,
        "Dict": Dict,
        "random": random,
        "partial": partial,
        "copy": copy,
        "deepcopy": deepcopy,
        "multiprocessing": multiprocessing,
        "ProcessPoolExecutor": ProcessPoolExecutor,
        "ThreadPoolExecutor": ThreadPoolExecutor,
    }
    namespace.update(additional_globals)
    exec(function_string, namespace)
    function = next((value for key, value in namespace.items() if key == "heuristic_fn"), None)
    args = inspect.getfullargspec(function).args if function else []
    return function, args


class DataLogger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def log(self, data_sample):
        with open(self.log_file_path, "a", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.write(str(data_sample) + "\n")
            fcntl.flock(file, fcntl.LOCK_UN)


def stable_softmax(x, axis=-1, temp=1.0):
    x = x / temp
    x = x - numpy.max(x, axis=axis, keepdims=True)
    e_x = numpy.exp(x)
    return e_x / numpy.sum(e_x, axis=axis, keepdims=True)


def normalized(x: List[float], temp: float = 1) -> List[float]:
    return list(stable_softmax(numpy.asarray(x), temp=temp))
