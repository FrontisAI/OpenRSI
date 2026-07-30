# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import logging
import os
import subprocess
from enum import Enum
from typing import Callable, Optional, Tuple

import submitit
from jinja2 import Template

log = logging.getLogger(__name__)


class JobStatus(Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class JobResult:
    """Aggregate information about the finished job."""

    job_id: str
    metadata: dict[str, str | int | float, bool]
    status: JobStatus
    log_out: list[str]
    log_err: list[str]


class JobObserver:
    """Manages a pool of asyncio tasks for observing submitit job status."""

    shared = None  # Singleton

    def __init__(self):
        # We keep a list of tasks created by 'observe'
        self._observing_tasks: list[asyncio.Task] = []

    def observe(
        self,
        job: submitit.Job,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        focus_rank: Optional[int] = None,
        poll_interval: int = 10,
    ) -> None:
        """
        Observe the status of a submitit job, and execute a callback when finished.

        Args:
            job (submitit.Job): The Submitit job to watch.
            metadata (dict): Some data you want to associate with this job.
            callback (Callable[[JobResult], None]): Called after job finishes with the JobResult.
            focus_rank (Optional[int]): If set, only fetch logs from that subtask index.
            poll_interval (int): How often (seconds) to poll the job status.
        """
        task = asyncio.create_task(
            self._observe_job(
                job=job, poll_interval=poll_interval, focus_rank=focus_rank, callback=callback, metadata=metadata
            )
        )
        self._observing_tasks.append(task)

    async def wait(self) -> None:
        """Returns only when all observed jobs and their callbacks are complete."""
        if self._observing_tasks:
            await asyncio.gather(*self._observing_tasks)

    async def _observe_job(
        self,
        job: submitit.Job,
        poll_interval=30,
        focus_rank: Optional[int] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
        metadata: Optional[dict[str, str | int | float, bool]] = None,
    ) -> None:
        """
        Loop that periodically checks the job until it's done,
        then calls the user-specified callback with a JobResult.
        """
        while not job.done():
            await asyncio.sleep(poll_interval)

        # Once we exit the loop, the job is done from Submitit's perspective,
        # meaning it's not in [PENDING, RUNNING, REQUEUED, ...].
        slurm_state = job.state.upper() if job.state else "UNKNOWN"
        status = self._map_slurm_state_to_job_status(slurm_state, job)

        # Gather logs (stdout & stderr). Focus rank is optional.
        log_out, log_err = get_logs(job, focus_rank)

        # Create the result object
        result = JobResult(
            job_id=job.job_id,
            metadata=metadata,
            status=status,
            log_out=log_out,
            log_err=log_err,
        )

        if callback is not None:
            callback(result)

    def _map_slurm_state_to_job_status(self, slurm_state: str, job: submitit.Job) -> JobStatus:
        """
        Convert a Slurm or Submitit job state to simpler JobStatus states
        """
        if slurm_state == "COMPLETED":
            return JobStatus.COMPLETED

        elif slurm_state in ("CANCELLED"):
            return JobStatus.CANCELLED

        elif job.exception() is not None:
            return JobStatus.FAILED

        elif slurm_state in ("FAILED", "NODE_FAIL", "TIMEOUT"):
            return JobStatus.FAILED

        return JobStatus.UNKNOWN


JobObserver.shared = JobObserver()  # Initialize the singleton


def get_logs(job: submitit.Job, focus_rank: Optional[int] = None) -> Tuple[list[str], list[str]]:
    """
    Gathers job stdout/stderr logs as lists of strings.

    If focus_rank is given, only logs for that subtask index.
    Otherwise, all tasks in an array job (or the single task in a normal job).
    """
    # If a particular rank is requested, only retrieve logs from that subtask
    if focus_rank is not None:
        subjob = job.task(focus_rank)
        out = subjob.stdout() or ""
        err = subjob.stderr() or ""
        return [out], [err]

    # If the job has no subtasks (single job), gather from itself
    if not job._sub_jobs:
        return [job.stdout() or ""], [job.stderr() or ""]

    # Otherwise, gather from each subtask in the array job
    outs, errs = [], []
    for sub in job._sub_jobs:
        out = sub.stdout() or ""
        err = sub.stderr() or ""
        outs.append(out)
        errs.append(err)
    return outs, errs


def submit_job(
    command: str,
    slurm_job_name: str,
    working_dir: str = ".",
    log_dir="slurm_logs",
    env_vars: Optional[dict[str, str]] = None,
    **slurm_kwargs,
):
    """
    Launches a SLURM job using submitit with the specified arguments.

    Args:
        command (str): Path to the Python file that serves as the job entry point.
        num_nodes (int): Number of nodes for the SLURM job.
        cpus_per_task (int): Number of CPUs per task.
        gpus_per_node (int): Number of GPUs per node.
        tasks_per_node (int): Number of tasks per node.
        timeout_min (int): Timeout for the job in minutes.
        job_name (str): Name of the SLURM job.
        account (str): SLURM account to use for the job.
        qos (str): Quality of Service (QoS) for the job.

    Returns:
        str: Job ID of the submitted SLURM job.
    """

    # Create a Submitit executor
    executor = submitit.AutoExecutor(folder=log_dir)

    slurm_parameters = slurm_kwargs
    slurm_parameters["chdir"] = working_dir

    # SLURM configuration
    executor.update_parameters(slurm_job_name=slurm_job_name, slurm_additional_parameters=slurm_parameters)

    def job_function():
        if env_vars:
            for k, v in env_vars.items():
                os.environ[k] = str(v)

        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)

    # Submit the job
    job = executor.submit(job_function)

    log.info(f"Job submitted with ID: {job.job_id}, {job}")
    return job


async def main():
    ENV_VARS = {}

    slurm_parameters = {
        "account": "<<<YOUR_SLURM_ACCOUNT>>>",  # Replace with your SLURM account
        "qos": "<<<YOUR_SLURM_QOS>>>",  # Replace with your SLURM QoS
        "nodes": 1,
        "ntasks_per_node": 1,
        "gpus-per-node": 0,
        "cpus-per-task": 24,
        "partition": "<<<YOUR_SLURM_PARTITION>>>",
        "time": "00:05:00",
    }

    # Test 1
    # job = submit_job(
    #     # command='python test_slurm.py -i test_input/text_desc.txt',
    #     command='python test_slurm_2.py -x 1 -y 2 -j 1',
    #     working_dir='.',
    #     log_dir='submitit_logs/%j', # %x = job name, %j = job ID
    #     slurm_job_name="ts23",
    #     **slurm_parameters
    # )

    # await job.awaitable().result()
    # log_out, log_err = get_logs(job, focus_rank=None)

    # print("~~~Stdout~~~")
    # print(log_out[0])

    # print("~~~Stderr~~~")
    # print(log_err[0])

    # Test 2
    template_str = """python test_slurm_2.py -x {{ x }} -y {{ y }} -j {{ j }}"""
    template = Template(template_str)

    import random

    random.seed(12345)

    tasks = []
    for j in range(1, 5):
        x = random.randint(1, 4)
        y = random.randint(1, 4)
        job_data_dict = {"x": x, "y": y, "j": j}
        cmd = template.render(**job_data_dict)
        job = submit_job(
            # command='python test_slurm.py -i test_input/text_desc.txt',
            command=cmd,
            working_dir=".",
            log_dir="slurm_logs/%j",  # %x = job name, %j = job ID
            slurm_job_name="ts23",
            **slurm_parameters,
        )

        # Create an asyncio task from the job awaitable.
        async def _wrapper(job, data_dict):
            await job.awaitable().result()
            return job, data_dict

        task = asyncio.create_task(_wrapper(job, job_data_dict))
        tasks.append(task)

    # Process tasks as they complete.
    for completed in asyncio.as_completed(tasks):
        job, job_data = await completed

        print("~~~Job data~~~")
        print(job_data)

        log_out, log_err = get_logs(job, focus_rank=None)

        print("~~~Stdout~~~")
        print(log_out[0])

        print("~~~Stderr~~~")
        print(log_err[0])


if __name__ == "__main__":
    asyncio.run(main())
