# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from omegaconf import OmegaConf

from dojo.utils.logger import get_logger

from dojo.config_dataclasses.task.base import TaskConfig


class Task(ABC):
    """
    Abstract base class representing a Task.

    A task specifies:
      - The execution environment available to the agent (e.g., local resources or a VM with a specific configuration).
      - The interface through which the agent can interact with the environment (e.g., bash shell or Python Interpreter)
        and the observation space.
      - A description of the task (e.g., the goal, the constraints, the reward, a description of the execution environment
        and the interface).
      - The fitness function that evaluates the agent's performance on the task.
    """

    def __init__(self, cfg: TaskConfig) -> None:
        """
        Initialize a Task instance.

        Args:
            **cfg: Arbitrary keyword arguments representing the configuration parameters for the task.
        """
        # Convert the provided configuration dictionary into an OmegaConf object for structured access.
        self.cfg = cfg
        self.logger = get_logger()  # Store the logger for use in other methods if needed.

    @abstractmethod
    def prepare(self, **task_args: Optional[Dict]) -> Dict:
        """
        Prepares everything for the task execution to start (e.g., copies data to the agent's workspace).

        This method should be overridden to perform any necessary setup before the task execution begins.

        Args:
            task_args: The arguments needed to prepare for the task execution.
        Returns:
            Dict[str, Any]: The initial state of the task (potentially including the initial observation)
            Dict[str, Any]: Any task_info useful when instantiating the solver.
        """
        pass

    @abstractmethod
    def step_task(self, state: Dict, action: Any) -> Tuple[Dict, Dict]:
        """
        Execute a single step of the task using the provided action.

        This method should be overridden to define how a task processes an action and produces an outcome.

        Args: sv
            state (Dict): The current state of the task environment
            action (Any): The action to be performed in the task environment.

        Returns:
            Tuple[Dict, Dict]: A tuple containing the new state of the task environment and the outcome of the action.
        """
        pass

    @abstractmethod
    def evaluate_fitness(
        self,
        solution: Optional[Dict] = None,
        state: Optional[Dict] = None,
        interpreter: Optional[Dict] = None,
        aux_info: Dict[str, Any] = None,
    ) -> Any:
        """
        Evaluates how good (or correct) the provided solution is for this task.

        Args:
            solution (Dict, optional): The solution to be evaluated.
            state (Dict, optional): The current state of the task environment.
            interpreter (Dict, optional): The interpreter instance that provides an execution environment for the evaluation.
            aux_info (Dict): Additional information that may be needed during the evaluation.

        Returns:
            Any: An object representing the result of the fitness evaluation (e.g., a score, success/failure status, etc.).
        """
        pass

    @abstractmethod
    def close(self, state: Dict) -> None:
        """
        Perform cleanup and close any resources associated with the task.

        This method should be overridden to properly shut down the task and free resources when the task is completed.

        Args:
            state (Dict): The final state of the task.
        """
        pass
