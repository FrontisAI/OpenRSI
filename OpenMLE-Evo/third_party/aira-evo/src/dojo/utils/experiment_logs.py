# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Union


def is_experiment(path: Union[Path, str]) -> bool:
    """
    Check if the given file path is an experiment directory.
    An experiment directory contains a 'json' subdirectory.
    """
    # Feel free to make this more robust by checking for specific files or directories.
    path = Path(path)
    if not path.is_dir():
        return False
    path_to_cfg = path / "dojo_config.json"
    if not path_to_cfg.exists():
        return False
    return True


def is_meta_experiment(path: Union[Path, str]) -> bool:
    """
    Check if the given file path is a meta-experiment directory.
    A meta-experiment directory contains multiple experiment directories.
    """
    return any(is_experiment(child) for child in Path(path).iterdir() if child.is_dir())
