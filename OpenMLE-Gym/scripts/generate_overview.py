#!/usr/bin/env python3
"""Script wrapper for `openmle-task overview`."""

import sys

from openmle_gym.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["overview", *sys.argv[1:]]))
