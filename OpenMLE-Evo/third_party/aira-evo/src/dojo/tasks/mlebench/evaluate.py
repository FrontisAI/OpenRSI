# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 OpenAI
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/openai/mle-bench/blob/main/LICENSE

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from mlebench.data import get_leaderboard, is_dataset_prepared
from mlebench.grade_helpers import CompetitionReport
from mlebench.registry import Competition, Registry, registry
from mlebench.utils import (
    get_logger,
    get_timestamp,
    load_answers,
    purple,
    read_csv,
    read_jsonl,
)

import logging

logger = logging.getLogger(__name__)


def is_lower_better(competition):
    leaderboard_df = pd.read_csv(competition.leaderboard)

    try:
        _ = leaderboard_df["score"]
    except:
        raise Exception("You must run GIT LFS for mlebench.")

    return competition.grader.is_lower_better(leaderboard_df)


def evaluate_submission(submission_path: Path, data_dir: Path, competition_id: Path, results_output_dir: Path):
    # Load competition data
    new_registry = registry.set_data_dir(data_dir)
    competition = new_registry.get_competition(competition_id)

    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError(
            f"Dataset for competition `{competition.id}` is not prepared! "
            f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
        )

    score = None
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".csv"

    if submission_exists:
        submission_df = read_csv(submission_path)
        answers = load_answers(competition.answers)
        score = competition.grader(submission_df, answers)
    else:
        logger.warning(
            f"Invalid submission file: {submission_path}. Please check that the file exists and it is a CSV."
        )

    valid_submission = score is not None
    competition_leaderboard = get_leaderboard(competition)
    rank_info = competition.grader.rank_score(score, competition_leaderboard)
    is_lower_better = competition.grader.is_lower_better(competition_leaderboard)

    report = CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=valid_submission,
        is_lower_better=is_lower_better,
        created_at=datetime.now(),
        submission_path=str(submission_path),
    )

    # Ensure the output directory exists and write the report to disk.
    results_output_dir.mkdir(exist_ok=True)
    save_path = results_output_dir / f"grading_report.json"
    with open(save_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    score = report.score
    score = float(score) if score is not None else None
    return score, report.to_dict()
