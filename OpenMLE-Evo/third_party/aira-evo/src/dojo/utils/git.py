# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess  # noqa: S404
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(command: str, check: bool = True, verbose: bool = False) -> subprocess.CompletedProcess:
    if verbose:
        logger.info(f"Running command: `{command}`")

    return subprocess.run(
        command,
        shell=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def get_repo_name() -> str:
    return os.path.basename(get_git_top_level())


def get_git_top_level() -> str:
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()


def make_snapshot_shallow_git(git_root: Path, snapshot_root: Path) -> None:
    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        for repo in ["."]:  # This is needed if we ever have submodules
            logger.info(
                f"Making snapshot repo {repo if repo != '.' else 'aira-dojo'} of {snapshot_root} as shallow git ..."
            )
            run_command(
                f"git clone --depth 1 file:///{git_root}/{repo} {tmpdir}/{repo}"
            )  # We do not do --recurse-submodules here, because --depth 1 does not work with --recurse-submodules
            run_command(f"cp -r {tmpdir}/{repo}/.git {snapshot_root}/{repo}")

    logger.info(f"Making snapshot repo of {snapshot_root} as shallow git took {time.time() - start_time} seconds")


def has_uncommitted_changes() -> bool:
    """Check if there are uncommited changes.

    Returns:
        bool: wether there are uncommiteed changes.
    """
    command = "git status --porcelain"
    output = subprocess.check_output(command.split()).strip().decode("utf-8")
    return len(output) > 0


def get_current_commit_id(repo: str) -> str:
    """Get current commit id.

    Returns:
        str: current commit id.
    """
    command = "git rev-parse HEAD"
    submodule_repos = ["arrival_data", "fairseq2"]
    repos = submodule_repos + ["arrival_ssl"]
    if repo not in repos:
        raise ValueError(f"Unknown {repo=}. Supported {repos=}")
    if repo in submodule_repos:
        command = f"{command}:{repo}"

    commit_id = (
        subprocess.check_output(command.split()).strip().decode("utf-8")  # noqa: S603
    )
    return commit_id
