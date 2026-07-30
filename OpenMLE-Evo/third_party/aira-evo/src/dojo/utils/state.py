# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class BaseState:
    running_time: float = 0.0
    num_starts: int = 0

    def __init__(self):
        pass

    def state_dict(self):
        return {
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    def load_state_dict(self, state_dict):
        self.running_time = state_dict["running_time"]
        self.num_starts = state_dict.get("num_starts", 0)


class GreedyState:
    current_step: int = 0
    running_time: float = 0.0
    num_starts: int = 0

    def __init__(self):
        pass

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.running_time = state_dict["running_time"]
        self.num_starts = state_dict.get("num_starts", 0)


class MCTSState:
    current_step: int = 0
    running_time: float = 0.0
    num_starts: int = 0

    def __init__(self):
        pass

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.running_time = state_dict["running_time"]
        self.num_starts = state_dict.get("num_starts", 0)


class EvolutionaryState:
    current_step: int = 0
    running_time: float = 0.0
    current_generation: int = 0
    num_starts: int = 0

    def __init__(self):
        pass

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "running_time": self.running_time,
            "num_starts": self.num_starts,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        self.running_time = state_dict["running_time"]
        self.num_starts = state_dict.get("num_starts", 0)
