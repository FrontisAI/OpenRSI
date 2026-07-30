#!/usr/bin/env python3
"""Script wrapper for task quality evaluation.

Prefer the maintained CLI:

    uv run openmle-task evaluate \
      --root-dir artifacts/runs/<run-id>/build/data \
      --task-list artifacts/runs/<run-id>/build/tasks.txt \
      --overview-csv artifacts/runs/<run-id>/metadata/overview.csv \
      --output-dir artifacts/runs/<run-id>/evaluation
"""

from __future__ import annotations

import sys

from openmle_gym.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
