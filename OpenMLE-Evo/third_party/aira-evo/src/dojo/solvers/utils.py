# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from dojo.core.solvers.utils.journal import Node


class Complexity(Enum):
    LOW = "simple"
    MEDIUM = "normal"
    HIGH = "complex"

    def __str__(self):
        return self.value


def get_complextiy_level(node: Node = None, num: int = None) -> Complexity:
    """
    Determine the complexity level based on the number of children.

    Args:
        node: The node to evaluate
        num: The number of children (optional)

    Returns:
        Complexity: The complexity level (LOW, MEDIUM, HIGH)
    """
    if node is not None:
        if len(node.children) < 2:
            return Complexity.LOW
        elif len(node.children) < 4:
            return Complexity.MEDIUM
        else:
            return Complexity.HIGH

    if num is not None:
        if num < 2:
            return Complexity.LOW
        elif num < 4:
            return Complexity.MEDIUM
        else:
            return Complexity.HIGH

    raise ValueError("Either node or num must be provided.")
