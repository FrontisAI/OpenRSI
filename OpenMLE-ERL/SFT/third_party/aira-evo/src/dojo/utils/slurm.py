# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from submitit import JobEnvironment


def get_slurm_id() -> str:
    return (
        os.environ.get("SLURM_JOB_ID", "")
        if os.environ.get("SLURM_ARRAY_JOB_ID", None) is None
        else str(JobEnvironment().job_id)
    )
