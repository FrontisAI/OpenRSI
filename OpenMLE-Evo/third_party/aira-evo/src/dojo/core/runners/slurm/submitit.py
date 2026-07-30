# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from omegaconf import OmegaConf

from dojo.core.runners.slurm.slurm_utils import submit_job

log = logging.getLogger(__name__)


class SlurmSubmitit:
    def __init__(self, **cfg) -> None:
        self.cfg = OmegaConf.create(cfg)
        self._counter = 0

    async def launch(self, cmd, job_metadata=None):
        cfg = self.cfg

        job = submit_job(
            command=cmd,
            working_dir=cfg.working_dir,
            log_dir=cfg.log_dir,
            slurm_job_name=f"{cfg.exp_name}_{self._counter}",
            **cfg.params,
        )
        self._counter += 1

        if self.cfg.await_completion:
            log.info(f"[Run {self._counter}] Submitted -- waiting")
            await job.awaitable().result()
        else:
            log.info(f"[Run {self._counter}] Submitted.")

        return job, job_metadata
