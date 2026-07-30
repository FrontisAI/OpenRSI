#!/usr/bin/env python3
"""Script wrapper for `openmle-task leaderboard`.

The canonical entrypoint is:

    openmle-task leaderboard --slugs-file examples/leaderboard-slugs.txt

It defaults to dry-run unless `--execute` is passed.
"""

import sys

from openmle_gym.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["leaderboard", *sys.argv[1:]]))
