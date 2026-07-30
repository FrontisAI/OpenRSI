# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from pathlib import Path
import inspect

from dojo.utils.helpers import write_env_variables_to_json
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from dojo.utils import rich_utils
from dojo.utils.logger import config_logger, LogEvent
from omegaconf import OmegaConf
from dojo.config_dataclasses.omegaconf.resolvers import register_new_resolvers

from dojo.config_dataclasses.run import RunConfig
from dojo.config_dataclasses.task import TASK_MAP
from dojo.config_dataclasses.solver import SOLVER_MAP
from dojo.config_dataclasses.interpreter import INTERPRETER_MAP
from dojo.utils.config import build
from dojo.utils.environment import (
    get_hardware,
    check_pytorch_gpu,
    check_tensorflow_gpu,
    format_time,
)
from dojo.utils.slurm import get_slurm_id

load_dotenv()
log = logging.getLogger(__name__)

register_new_resolvers()


def _main(cfg: RunConfig):
    # Create the output directory if it doesn't exist
    log.debug(f"Saving experiment artifacts to: {cfg.logger.output_dir}")
    assert cfg.logger.output_dir is not None, (
        "Path to the directory in which the launch artefacts will be written must be specified."
    )
    log.debug(f"Output dir: {cfg.logger.output_dir}")
    output_dir = Path(cfg.logger.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()

    # Set environment variables
    os.environ["HARDWARE"] = get_hardware()
    os.environ["PYTORCH_GPU"] = check_pytorch_gpu()
    os.environ["TENSORFLOW_GPU"] = check_tensorflow_gpu()
    os.environ["TIME_LIMIT_SECS"] = str(cfg.solver.time_limit_secs)
    os.environ["TIME_LIMIT"] = format_time(int(os.environ["TIME_LIMIT_SECS"]))
    os.environ["STEP_LIMIT"] = str(cfg.solver.step_limit)

    # Store the slurm job ID
    cfg.metadata.slurm_id = get_slurm_id()

    # Sanity checks
    log.info(f"Current working directory: {os.getcwd()}")
    import dojo

    log.info(f"`dojo` package source path: {inspect.getsourcefile(dojo)}")
    import aira_core

    log.info(f"`aira_core` package source path: {inspect.getsourcefile(aira_core)}")
    import mlebench

    log.info(f"`mlebench` package source path: {inspect.getsourcefile(mlebench)}")

    # Create the output directory if it doesn't exist
    log.info(f"Saving experiment artifacts to: {cfg.logger.output_dir}")
    assert cfg.logger.output_dir is not None, (
        "Path to the directory in which the launch artefacts will be written must be specified."
    )
    log.info(f"Output dir: {cfg.logger.output_dir}")
    output_dir = Path(cfg.logger.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()

    if cfg.logger.write_env_vars:
        write_env_variables_to_json(cfg.logger.output_dir)

    logger = config_logger(cfg)

    log.info("Instantiating the task...")
    task = build(cfg.task, TASK_MAP)

    # Allocate resources for the agent's workspace and instantiate an object that lets you reference and use them
    solver_interpreter = build(cfg.interpreter, INTERPRETER_MAP, data_dir=cfg.task.data_dir)

    eval_interpreter = None

    log.info("Preparing the workspaces...")
    state, task_info = task.prepare(solver_interpreter=solver_interpreter, eval_interpreter=eval_interpreter)

    log.info("Instantiating the solver...")
    solver = build(cfg.solver, SOLVER_MAP, task_info=task_info)

    # Load checkpoint state if it exists
    log.info("Loading checkpoints, if any...")
    solver.load_checkpoint()

    log.info("Starting the solver...")
    state, solution, best_node = solver(task, state)

    if solution is None:
        log.error("No valid solution was generated")
    else:
        log.info("Evaluating the final solution...")

        if (
            hasattr(best_node, "metric")
            and hasattr(best_node.metric, "info")
            and "score" in best_node.metric.info
            and best_node.metric.info is not None
        ):
            log.info("We have the evaluation score already computed...")
            fitness = best_node.metric.info["score"]
            log.info(f"Final fitness: {fitness}")
        else:
            raise ValueError("This should not be reached and happening.")

        logger.log(fitness, LogEvent.EVAL)

    log.info("Clean up...")
    task.close(state)
    logger.stop()


@hydra.main(version_base="1.3.2", config_path="configs", config_name="default_run")
def main(_cfg: DictConfig):
    ## Validate and setup config
    # 1) Check structure
    cfg_instantiated: RunConfig = hydra.utils.instantiate(_cfg)
    # 2) Resolve interpolations (e.g. experiment ID, time, etc.)
    cfg_dict_config = OmegaConf.structured(cfg_instantiated)
    OmegaConf.resolve(cfg_dict_config)
    # 3) Convert back to dataclass and validate
    cfg: RunConfig = OmegaConf.to_object(cfg_dict_config)

    cfg.validate()

    # Pretty print the config
    if cfg.logger.print_config:
        rich_utils.print_config_tree(cfg_dict_config)

    _main(cfg)


if __name__ == "__main__":
    main()
