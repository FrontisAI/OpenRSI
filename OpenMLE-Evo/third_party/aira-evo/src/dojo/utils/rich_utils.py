# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are Apache 2.0 licensed
# Copyright (c) 2023 Neptune Labs.
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/neptune-ai/neptune-client/blob/master/LICENSE

import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, Union

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from dojo.config_dataclasses.run import RunConfig

# Define the boundaries for 32-bit integers.
MAX_32_BIT_INT = 2147483647
MIN_32_BIT_INT = -2147483648


def is_unsupported_float(value: Any) -> bool:
    """
    Determine if the given value is a float that is either infinite or NaN.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a float and is either infinite or NaN, otherwise False.
    """
    if isinstance(value, float):
        return math.isinf(value) or math.isnan(value)
    return False


class StringifyValue:
    """
    A wrapper class that processes and stores a value ensuring that it is
    safe for string conversion.

    The class performs the following:
      - If the value is an integer outside the 32-bit range, it converts it to a float.
      - If the value is a float that is unsupported (infinite or NaN), it converts it to a string.
    """

    def __init__(self, value: Any):
        """
        Process the input value to ensure it is suitable for stringification.

        Args:
            value (Any): The value to process.
        """
        # Convert integers outside the 32-bit range to floats.
        if isinstance(value, int) and (value > MAX_32_BIT_INT or value < MIN_32_BIT_INT):
            value = float(value)

        # Convert unsupported floats (infinity or NaN) to strings.
        if is_unsupported_float(value):
            value = str(value)

        self.__value = value

    @property
    def value(self) -> Any:
        """
        Retrieve the processed value.

        Returns:
            Any: The processed value.
        """
        return self.__value

    def __str__(self) -> str:
        """
        Return the string representation of the processed value.

        Returns:
            str: The string representation.
        """
        return str(self.__value)

    def __repr__(self) -> str:
        """
        Return the official string representation of the processed value.

        Returns:
            str: The official string representation.
        """
        return repr(self.__value)


def stringify_unsupported(value: Any) -> Union[StringifyValue, Mapping[str, Any]]:
    """
    Recursively process a value, converting unsupported types to a StringifyValue instance.

    If the provided value is a mutable mapping (like a dictionary), the function will
    recursively process each key-value pair, ensuring that:
      - All keys are converted to strings.
      - All values are processed so that unsupported values are wrapped accordingly.

    Args:
        value (Any): A dictionary-like object or a single value.

    Returns:
        Union[StringifyValue, Mapping[str, Any]]:
            - A processed dictionary with stringified keys and values, or
            - A StringifyValue instance wrapping the processed value.
    """
    if isinstance(value, MutableMapping):
        # Recursively process each item in the mapping.
        return {str(k): stringify_unsupported(v) for k, v in value.items()}

    return StringifyValue(value=value)


def print_config_tree(
    cfg: RunConfig,
    print_order: Sequence[str] = [],
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logging.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.logger.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)
