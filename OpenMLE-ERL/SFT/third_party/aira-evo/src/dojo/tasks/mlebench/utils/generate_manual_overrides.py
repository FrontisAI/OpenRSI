# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def generate_manual_overrides(split_name):
    file_dir = Path(__file__).resolve().parent.parent
    base_dir = file_dir / "splits"
    split_file = base_dir / (split_name + ".txt")
    split_comp_ids = split_file.read_text()
    split_comp_ids = split_comp_ids.strip().split("\n")

    with open(f"mlebench_{split_name}.yaml", "w") as f:
        f.write("vars:\n\tmanual_overrides: [")
        for comp_id in split_comp_ids:
            f.write(f"competition_id={comp_id},")
        f.write("]")


if __name__ == "__main__":
    generate_manual_overrides("low_dev")
