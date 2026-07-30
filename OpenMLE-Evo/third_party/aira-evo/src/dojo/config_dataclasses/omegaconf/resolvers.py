# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import subprocess
import copy

import omegaconf
from omegaconf import DictConfig, OmegaConf
import torch
import transformers

from dojo.config_dataclasses.run import RunConfig
from dojo.config_dataclasses.runner import RunnerConfig
from dojo.utils.environment import get_mlebench_data_dir, get_superimage_dir


def generate_id(*, _parent_: DictConfig) -> str:
    """
    Generate a unique ID for an experiment given an experiment config.
    """
    # Some OmegaConf magic: _parent_ gives access to the parent
    # of the field being interpolated. In our case, the interpolation
    # is in the Config.id field so this gives us the original Config.
    unresolved_config = copy.deepcopy(_parent_)
    fields_to_exclude = RunConfig.fields_to_exclude_from_hash()
    # Accessing fields of unresolved_config will resolve them
    user, issue, seed = (
        unresolved_config.metadata.user,
        unresolved_config.metadata.git_issue_id,
        unresolved_config.metadata.seed,
    )
    for field, field_type in fields_to_exclude:
        if field_type in (int, float):
            exec(f"unresolved_config.{field} = 0")
        elif field_type == bool:
            exec(f"unresolved_config.{field} = False")
        else:
            exec(f'unresolved_config.{field} = ""')
    config: RunConfig = OmegaConf.to_object(unresolved_config)  # type: ignore[assignment]
    assert isinstance(config, RunConfig)

    return f"user_{user}_issue_{issue}_seed_{seed}_id_{config.hash()}"


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_torch_version() -> str:
    return str(torch.__version__)


def get_git_commit_id() -> str:
    command = "git rev-parse HEAD"
    commit_id = subprocess.check_output(command.split()).strip().decode("utf-8")
    return commit_id


def register_new_resolvers() -> None:
    for resolver, function in {
        "generate_id": generate_id,
        "get_current_time": get_current_time,
        "get_torch_version": get_torch_version,
        "get_git_commit_id": get_git_commit_id,
        "get_superimage_dir": get_superimage_dir,
        "get_mlebench_data_dir": get_mlebench_data_dir,
    }.items():
        if not omegaconf.OmegaConf._get_resolver(resolver):
            omegaconf.OmegaConf.register_new_resolver(resolver, function)
