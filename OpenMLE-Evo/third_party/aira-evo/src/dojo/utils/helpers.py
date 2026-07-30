# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from pathlib import Path


def write_env_variables_to_json(output_dir):
    """
    Write the environment variables to a JSON file.
    """
    # Path to file
    output_file = Path(output_dir) / "env_variables.json"

    # Convert the environment variables to a dictionary
    env_vars = dict(os.environ)

    # Write the dictionary to a JSON file
    with open(output_file, "w") as file:
        json.dump(env_vars, file, indent=4)

    print("Environment variables written as JSON to env_variables.json")
