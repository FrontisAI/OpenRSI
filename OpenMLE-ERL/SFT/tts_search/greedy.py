from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from tts_search.program_database import Program, ProgramDatabase, SearchAlgorithm
from tts_search.prompt_builder import build_prompt

logger = logging.getLogger(__name__)


@dataclass
class AlgoConfig:
    num_drafts: int | None = 5
    debug_prob: float = 0.5
    draft_prob: float | None = 0.0


class GreedySearch(SearchAlgorithm):
    def __init__(self, config: AlgoConfig):
        self._config = config
        self._draft_prob = config.draft_prob or 0.0
        assert self._draft_prob + config.debug_prob <= 1.0

    def select(
        self,
        database: ProgramDatabase,
        task_name: str,
        description: str,
        data_dir: str,
        max_steps: int = 1,
    ) -> tuple[str | None, Program | None, str, str | None]:
        # The first solution must always be a draft.
        draft_count = database.count_by_generation_mode(task_name, "draft")
        num_drafts = self._config.num_drafts or 0
        if database.is_empty(task_name):
            mode = "draft"
            parent_program = None
            logger.info(
                "[SEARCH] Using DRAFT mode for task %s (draft_count=%d/%d)",
                task_name,
                draft_count,
                num_drafts,
            )
        else:
            debug_selected = False
            if self._draft_prob > 0.0:
                sample = random.random()
                if sample < self._draft_prob:
                    mode = "draft"
                    parent_program = None
                else:
                    mode = "improve"
                    debug_selected = sample < self._draft_prob + self._config.debug_prob
            elif draft_count < num_drafts:
                mode = "draft"
                parent_program = None
                logger.info(
                    "[SEARCH] Using DRAFT mode for task %s (draft_count=%d/%d)",
                    task_name,
                    draft_count,
                    num_drafts,
                )
            else:
                mode = "improve"
                debug_selected = random.random() < self._config.debug_prob

            if mode == "draft":
                parent_program = None
            else:
                parent_program = None
                if debug_selected:
                    parent_program = database.get_random_by_fitness(task_name, 0.0)
                    if parent_program is not None:
                        logger.info(
                            "[SEARCH] Debugging buggy node for task %s (parent_id=%s)",
                            task_name,
                            parent_program.id,
                        )
                if parent_program is None:
                    parent_program = database.get_best(task_name)
                    logger.info(
                        "[SEARCH] Using IMPROVE mode for task %s (parent_id=%s, parent_fitness=%.4f)",
                        task_name,
                        parent_program.id,
                        float(parent_program.fitness),
                    )

        prompt = build_prompt(
            mode=mode,
            description=description,
            virtual_data_dir=data_dir,
            parent_program=parent_program,
            max_steps=max_steps,
        )
        return prompt, parent_program, mode, None

    def select_best(self, database: ProgramDatabase, task_name: str) -> Program:
        return database.get_best(task_name)
