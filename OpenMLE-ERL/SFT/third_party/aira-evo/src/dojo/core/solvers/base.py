# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from abc import ABC, abstractmethod
from pathlib import Path
import json

from omegaconf import OmegaConf

from dojo.utils.logger import get_logger, LogEvent

from dojo.config_dataclasses.solver.base import SolverConfig
from dojo.utils.state import BaseState


class Solver(ABC):
    def __init__(self, cfg: SolverConfig, task_info=None):
        self.cfg = cfg
        self.logger = get_logger()
        self.task_info = task_info
        self.start_time = time.monotonic()

        # state
        self.state = BaseState()

    @abstractmethod
    def __call__(self, task, state):
        raise NotImplementedError()

    def save_checkpoint(self):
        self.logger.info(f"Saving checkpoint to {self.cfg.checkpoint_path}")
        Path(self.cfg.checkpoint_path).mkdir(parents=True, exist_ok=True)
        state_dict = self.state.state_dict()
        with open(Path(self.cfg.checkpoint_path) / "state.json", "w") as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        state_path = Path(self.cfg.checkpoint_path) / "state.json"
        if not state_path.exists():
            self.logger.warning(f"No checkpoint found at {state_path}. Proceeding without loading.")
            return
        self.logger.info(f"Found state checkpoint at {state_path}. Loading...")
        with open(state_path, "r") as f:
            state_dict = json.load(f)

        self.state.load_state_dict(state_dict)

        # Increment the number of starts
        self.state.num_starts += 1
