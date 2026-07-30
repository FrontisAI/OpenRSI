"""Generation loop control for batched rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationLoopConfig:
    """Rules for when to keep generating samples for one task."""

    batch_size: int = 4
    accepted_target: int = 4
    max_generated: int | None = 32

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.accepted_target <= 0:
            raise ValueError("accepted_target must be > 0")
        if self.max_generated is not None and self.max_generated <= 0:
            raise ValueError("max_generated must be > 0")


class GenerationLoopController:
    """Controls per-task generation targets independently of gen/eval services."""

    def __init__(self, config: GenerationLoopConfig):
        self._config = config

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    @property
    def accepted_target(self) -> int:
        return self._config.accepted_target

    def initialize_state(self, state: Any) -> None:
        """Ensure a fresh state starts with one batch target."""

        if getattr(state, "target_count", 0) <= 0:
            state.target_count = self.batch_size
        if getattr(state, "max_target", None) is None:
            state.max_target = self._config.max_generated

    def generation_needed(self, state: Any) -> int:
        """Return how many new generations should be queued for this state."""

        if getattr(state, "done", False):
            return 0
        return max(
            0,
            int(state.target_count)
            - int(state.generated_count)
            - int(state.pending_gen),
        )

    def advance_after_completion(self, state: Any) -> int:
        """Advance target after completed evals and return additional needed gens."""

        while not state.done and state.completed_count >= state.target_count:
            if self._target_met(state) or self._max_target_reached(state):
                state.done = True
                break

            next_target = state.target_count + self.batch_size
            if self._config.max_generated is not None:
                next_target = min(next_target, self._config.max_generated)
            if next_target <= state.target_count:
                state.done = True
                break
            state.target_count = next_target

        return self.generation_needed(state)

    def _target_met(self, state: Any) -> bool:
        accepted_target = getattr(state, "accepted_target", None)
        if accepted_target is None:
            accepted_target = self.accepted_target
        return int(state.accepted_count) >= int(accepted_target)

    def _max_target_reached(self, state: Any) -> bool:
        return (
            self._config.max_generated is not None
            and int(state.target_count) >= self._config.max_generated
        )
