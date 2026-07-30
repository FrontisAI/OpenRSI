# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import itertools
import logging
import os
import copy
import sys
import importlib
import inspect
import glob
import subprocess
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import time

from submitit.helpers import RsyncSnapshot

import hydra
import jinja2
import submitit
from submitit.helpers import monitor_jobs
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from dojo.config_dataclasses.omegaconf.resolvers import register_new_resolvers

from dojo.utils import rich_utils
from dojo.utils.environment import get_log_dir
from dojo.config_dataclasses.run import RunConfig

from dojo.main_run import _main as main_run
from dojo.config_dataclasses.launcher.base import LauncherConfig
from dojo.config_dataclasses.launcher.slurm import SlurmConfig
from dojo.config_dataclasses.runner import RunnerConfig
from dojo.utils.git import make_snapshot_shallow_git, get_git_top_level

load_dotenv()
log = logging.getLogger(__name__)

register_new_resolvers()


def launch_jobs(config_list: list[RunConfig], launcher_cfg: LauncherConfig):
    # Go to the git root directory
    og_path = os.getcwd()
    git_root = Path(get_git_top_level())
    os.chdir(git_root)

    # Make snapshot for the batch of jobs
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    snapshot_path = Path(get_log_dir()) / "aira-dojo" / "snapshots" / f"{date_str}"
    snapshot_path.mkdir(parents=True, exist_ok=False)
    log.debug(f"Snapshotting to {snapshot_path}")

    start_time = time.time()
    with RsyncSnapshot(
        snapshot_dir=snapshot_path,
        with_submodules=True,
        exclude=[
            "*.ipynb",
            "*__pycache__",
            "*.mypy_cache",
        ],
        include=glob.glob(f"./src/dojo/tasks/mlebench/**", recursive=True),
    ):
        # Note 1: This copies all files of git repo at the `snapshot_path,
        # then `cd` into `snapshot_path` so os.getcwd() will return
        # `snapshot_path` and `snapshot_path` is the root of git

        # Note 2: Setting the PYTHONPATH to include the snapshot src folder
        # This is needed to actually use the code from the snapshot
        # otherwise it will use the code from the change-exposed folder
        # because of the editable installation of aira-dojo, aira-core and mle-bench

        # Note 3: PYTHONPATH parses multiple paths given with `:` char
        # So here we are actually bypassing our three editable packages
        # with snapshot/dojo, snapshot/aira_core and mlebench

        log.debug(f"Making snapshot took {time.time() - start_time} seconds")

        # Copy the aira-core code to the snapshot path
        import aira_core

        aira_core_init_file = Path(
            inspect.getsourcefile(aira_core)
        )  # This points to the __init__.py file of aira-core
        aira_core_path = aira_core_init_file.parents[
            1
        ] 

        # Copy the aira-core code to the snapshot path
        subprocess.run(["cp", "-r", str(aira_core_path), str(snapshot_path)])

        os.environ["PYTHONPATH"] = (
            f"{snapshot_path}/src:{snapshot_path}/aira-core/src:{snapshot_path}/src/dojo/tasks/mlebench/mle-bench"  # This will override the packages that are installed in editable mode such that it uses the code from the snapshot
        )

        #####################
        # Create an Executor
        if isinstance(launcher_cfg, SlurmConfig):
            slurm_folder = Path(get_log_dir()) / "slurm_logs" / "%j"
            executor = submitit.SlurmExecutor(folder=slurm_folder)
            executor_kwargs = {
                key: val
                for key, val in asdict(launcher_cfg).items()
                if val and not launcher_cfg.__dataclass_fields__[key].metadata.get("exclude_from_executor", False)
            }
            executor.update_parameters(**executor_kwargs)
        else:
            raise ValueError("Unsupported launcher configuration type")

        #####################
        ## Submit the jobs
        jobs = []
        with executor.batch():
            for run_cfg in config_list:
                job = executor.submit(main_run, run_cfg)
                jobs.append(job)

    # Move back to the original path
    os.chdir(og_path)

    return jobs


def override_config(config, key, value):
    if "." in key:
        # Handle nested keys like 'metadata.seed'
        parts = key.split(".")
        if hasattr(config, parts[0]):
            override_config(getattr(config, parts[0]), ".".join(parts[1:]), value)
        else:
            # Key doesn't exist
            raise ValueError(f"Key '{key}' not found in config.")
    elif hasattr(config, key):
        # Handle top-level keys
        setattr(config, key, value)
    else:
        # Key doesn't exist and isn't nested
        raise ValueError(f"Key '{key}' not found in config.")


def fetch_config(config, key):
    if "." in key:
        # Handle nested keys like 'metadata.seed'
        parts = key.split(".")
        if hasattr(config, parts[0]):
            return fetch_config(getattr(config, parts[0]), ".".join(parts[1:]))
        else:
            # Key doesn't exist
            raise ValueError(f"Key '{key}' not found in config.")
    elif hasattr(config, key):
        # Handle top-level keys
        return getattr(config, key)
    else:
        # Key doesn't exist and isn't nested
        raise ValueError(f"Key '{key}' not found in config.")


async def _main(runner_configs: list[RunnerConfig]):
    run_configs = []
    for runner_cfg in runner_configs:
        for task_cfg in runner_cfg.benchmark.to_cfg_list():
            run_cfg = RunConfig(
                meta_id=runner_cfg.id,
                logger=runner_cfg.logger,
                metadata=runner_cfg.metadata,
                task=task_cfg,
                solver=runner_cfg.solver,
                interpreter=runner_cfg.interpreter,
            )

            # Resolve interpolations (e.g. experiment ID, time, etc.)
            run_cfg = OmegaConf.structured(run_cfg)
            OmegaConf.resolve(run_cfg)
            run_cfg = OmegaConf.to_object(run_cfg)
            run_cfg.validate()  # Make sure everything is valid

            # Add the run config to the list
            run_configs.append(run_cfg)

    if runner_cfg.launcher.debug:
        jobs = []

        swept_keys = list(runner_cfg.vars.keys()) + ["task.name"]
        # Print a summary of the jobs that would be launched
        log.debug("Dry run mode: printing job summary")
        for cfg in run_configs:
            for key in swept_keys:
                print(f"{key}{' ' * (30 - len(key))}{fetch_config(cfg, key)}")
            print("============" * 5)
    else:
        log.debug("Launching jobs...")
        jobs = launch_jobs(run_configs, runner_cfg.launcher)

    return jobs


@hydra.main(version_base="1.3.2", config_path="configs", config_name="default_runner")
def main(_cfg: DictConfig):
    ## Validate and setup config
    # 1) Check structure
    og_cfg: RunnerConfig = hydra.utils.instantiate(_cfg)

    ## Create a list of Runner configs, given the list of override variables
    runner_configs = []
    cmd_vars = dict(og_cfg.vars)
    keys = list(cmd_vars.keys())
    for idx, values in enumerate(itertools.product(*(cmd_vars[key] for key in keys))):
        single_vars_comb = dict(zip(keys, values))

        # 1) Apply override variables to the run config
        runner_cfg: RunnerConfig = copy.deepcopy(og_cfg)
        for k, v in single_vars_comb.items():
            override_config(runner_cfg, k, v)

        # runner_cfg.validate()

        runner_configs.append(runner_cfg)

    jobs = asyncio.run(_main(runner_configs))
    jobs = [
        j for j in jobs if j is not None
    ]  # Filter out the jobs that failed to launch (i.e., didn't pass the dry run test)

    if og_cfg.launcher.monitor_jobs:
        log.info("Monitoring jobs...")
        # Monitor the jobs
        monitor_jobs(jobs)
    else:
        log.info("Jobs launched successfully, but not monitored.")
        job_arrays = ", ".join(sorted(set(str(job.job_id).split("_", 1)[0] for job in jobs)))
        log.info(f"Monitoring {len(jobs)} jobs from job arrays {job_arrays} \n")


if __name__ == "__main__":
    main()
