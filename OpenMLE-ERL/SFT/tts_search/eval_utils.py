from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

LOW_TASK_LIST: list[str] = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]
MIDDLE_TASK_LIST: list[str] = [
    "AI4Code",
    "alaska2-image-steganalysis",
    "billion-word-imputation",
    "cassava-leaf-disease-classification",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "champs-scalar-coupling",
    "facebook-recruiting-iii-keyword-extraction",
    "freesound-audio-tagging-2019",
    "google-quest-challenge",
    "h-and-m-personalized-fashion-recommendations",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "icecube-neutrinos-in-deep-ice",
    "imet-2020-fgvc7",
    "inaturalist-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "kuzushiji-recognition",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "multi-modal-gesture-recognition",
    "osic-pulmonary-fibrosis-progression",
    "petfinder-pawpularity-score",
    "plant-pathology-2021-fgvc8",
    "seti-breakthrough-listen",
    "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge",
    "tensorflow2-question-answering",
    "tgs-salt-identification-challenge",
    "tweet-sentiment-extraction",
    "us-patent-phrase-to-phrase-matching",
    "uw-madison-gi-tract-image-segmentation",
    "ventilator-pressure-prediction",
    "whale-categorization-playground",
]
HIGH_TASK_LIST: list[str] = [
    "3d-object-detection-for-autonomous-vehicles",
    "bms-molecular-translation",
    "google-research-identify-contrails-reduce-global-warming",
    "hms-harmful-brain-activity-classification",
    "iwildcam-2019-fgvc6",
    "nfl-player-contact-detection",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-covid19-detection",
    "smartphone-decimeter-2022",
    "stanford-covid-vaccine",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
]
ALL_TASK_LIST: list[str] = LOW_TASK_LIST + MIDDLE_TASK_LIST + HIGH_TASK_LIST
TIME_SCALING_SCORE_SECONDS = 12 * 60 * 60


def load_leaderboard(metadata: dict[str, Any]) -> pd.DataFrame | None:
    leaderboard_dir = metadata.get("leaderboard_dir")
    task_name = str(metadata["task_name"])
    leaderboard_paths = (
        [Path(metadata["data_dir"]) / "info" / "public_leaderboard.csv"]
        if leaderboard_dir is None
        else [
            Path(leaderboard_dir) / f"{task_name}.csv",
            Path(leaderboard_dir) / task_name / "info" / "public_leaderboard.csv",
            Path(leaderboard_dir) / task_name / "public_leaderboard.csv",
        ]
    )
    for leaderboard_path in leaderboard_paths:
        if leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path)
            column_map = {
                column: str(column).strip().lower()
                for column in leaderboard.columns
            }
            leaderboard = leaderboard.rename(columns=column_map)
            if "score" not in leaderboard.columns:
                return None
            return leaderboard
    return None


def is_lower_better(leaderboard: pd.DataFrame) -> bool:
    scores = leaderboard["score"].tolist()
    return bool(scores[0] < scores[-1])


def get_grade_for_score(score: float, leaderboard: pd.DataFrame) -> float:
    scores = leaderboard["score"].tolist()
    lower_better = is_lower_better(leaderboard)
    if lower_better:
        better_count = sum(1 for s in scores if s < score)
    else:
        better_count = sum(1 for s in scores if s > score)
    rank = min(better_count + 1, len(scores))
    return rank / len(scores)


def _medal_positions(num_teams: int) -> tuple[int, int, int]:
    if num_teams < 100:
        gold_pos = max(1, int(num_teams * 0.1))
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    elif num_teams < 250:
        gold_pos = 10
        silver_pos = max(1, int(num_teams * 0.2))
        bronze_pos = max(1, int(num_teams * 0.4))
    elif num_teams < 1000:
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = 50
        bronze_pos = 100
    else:
        gold_pos = 10 + int(num_teams * 0.002)
        silver_pos = max(1, int(num_teams * 0.05))
        bronze_pos = max(1, int(num_teams * 0.1))
    return gold_pos, silver_pos, bronze_pos


def get_medal_for_score(score: float, leaderboard: pd.DataFrame) -> str:
    scores = leaderboard["score"].tolist()
    gold_pos, silver_pos, bronze_pos = _medal_positions(len(scores))
    gold_threshold = scores[gold_pos - 1]
    silver_threshold = scores[silver_pos - 1]
    bronze_threshold = scores[bronze_pos - 1]
    lower_better = is_lower_better(leaderboard)
    if lower_better:
        if score <= gold_threshold:
            return "gold"
        if score <= silver_threshold:
            return "silver"
        if score <= bronze_threshold:
            return "bronze"
    else:
        if score >= gold_threshold:
            return "gold"
        if score >= silver_threshold:
            return "silver"
        if score >= bronze_threshold:
            return "bronze"
    return "N/A"


def get_medal_for_grade(grade: float, leaderboard: pd.DataFrame) -> str:
    num_teams = len(leaderboard["score"].tolist())
    gold_pos, silver_pos, bronze_pos = _medal_positions(num_teams)
    gold_grade = gold_pos / num_teams
    silver_grade = silver_pos / num_teams
    bronze_grade = bronze_pos / num_teams
    if grade <= gold_grade:
        return "gold"
    if grade <= silver_grade:
        return "silver"
    if grade <= bronze_grade:
        return "bronze"
    return "N/A"


def build_submit_grade_and_medal(
    submit_score: float | None, leaderboard: pd.DataFrame | None
) -> tuple[float, str]:
    if submit_score is None or leaderboard is None:
        return 1.0, "N/A"
    grade = get_grade_for_score(submit_score, leaderboard)
    medal = get_medal_for_score(submit_score, leaderboard)
    return grade, medal


def write_epoch_stat(epoch_output_dir: Path) -> dict[str, Any]:
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
    }

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    task_names: list[str] = []
    task_medals: dict[str, str] = {}
    task_grades: dict[str, float] = {}

    for task_dir in sorted(
        path for path in epoch_output_dir.iterdir() if path.is_dir()
    ):
        stat_path = task_dir / "stat.json"
        if not stat_path.exists():
            continue
        payload = json.loads(stat_path.read_text(encoding="utf-8"))
        task_name = str(payload.get("task_name") or task_dir.name)
        task_names.append(task_name)
        usage["total_cost"] += float(payload.get("total_cost", 0.0))
        usage["total_prompt_tokens"] += int(payload.get("total_prompt_tokens", 0))
        usage["total_completion_tokens"] += int(
            payload.get("total_completion_tokens", 0)
        )
        usage["total_tokens"] += int(payload.get("total_tokens", 0))
        usage["total_steps"] += int(payload.get("num_steps", 0))

        submit_medal = payload.get("submit_medal")
        task_medals[task_name] = str(submit_medal) if submit_medal else "N/A"
        submit_grade = payload.get("submit_grade")
        if submit_grade is not None:
            task_grades[task_name] = float(submit_grade)

    all_task_list = ALL_TASK_LIST or LOW_TASK_LIST + MIDDLE_TASK_LIST + HIGH_TASK_LIST
    level_tasks = {
        "low": [task for task in task_names if task in LOW_TASK_LIST],
        "middle": [task for task in task_names if task in MIDDLE_TASK_LIST],
        "high": [task for task in task_names if task in HIGH_TASK_LIST],
        "all": [task for task in task_names if task in all_task_list]
        if all_task_list
        else list(task_names),
    }
    task_counts = {
        f"{level}_task_count": len(tasks) for level, tasks in level_tasks.items()
    }

    grade_stats: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        grades = [task_grades[task] for task in tasks if task in task_grades]
        grade_stats[f"{level}_grade_avg@n"] = mean(grades) if grades else None
        grade_stats[f"{level}_grade_std@n"] = std(grades) if grades else None

    medal_counts: dict[str, int] = {}
    for level, tasks in level_tasks.items():
        for medal in ("gold", "silver", "bronze"):
            medal_counts[f"{level}_{medal}_count"] = sum(
                1 for task in tasks if task_medals.get(task) == medal
            )
        medal_counts[f"{level}_any_count"] = sum(
            1 for task in tasks if task_medals.get(task) in {"gold", "silver", "bronze"}
        )

    medal_rates: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        total = len(tasks)
        for medal in ("gold", "silver", "bronze", "any"):
            medal_rates[f"{level}_{medal}_rate"] = (
                medal_counts[f"{level}_{medal}_count"] / total if total else None
            )

    epoch_stat = {
        **usage,
        **task_counts,
        **grade_stats,
        **medal_counts,
        **medal_rates,
    }
    (epoch_output_dir / "stat.json").write_text(
        json.dumps(epoch_stat, indent=2), encoding="utf-8"
    )
    return epoch_stat


def write_summary_csv(
    output_dir: Path,
    task_metadata_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    def epoch_index(path: Path) -> int:
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    task_samples: dict[str, list[dict[str, Any]]] = {}
    task_status_counts: dict[str, dict[str, int]] = {}
    fixed_statuses = [
        "code_execution_error",
        "code_missing",
        "scoring_failed",
        "submission_missing",
        "success",
        "timeout",
        "unknown",
    ]

    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    for epoch_dir in epoch_dirs:
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            stat_path = task_dir / "stat.json"
            if not stat_path.exists():
                continue
            payload = json.loads(stat_path.read_text(encoding="utf-8"))
            task_name = str(payload.get("task_name") or task_dir.name)
            task_samples.setdefault(task_name, []).append(payload)
            status_count = payload.get("status_count") or {}
            task_status = task_status_counts.setdefault(task_name, {})
            for status, count in dict(status_count).items():
                task_status[status] = task_status.get(status, 0) + int(count)

    all_statuses = fixed_statuses
    summary_rows: list[dict[str, Any]] = []
    for task_name in sorted(task_samples):
        samples = task_samples[task_name]
        metadata = task_metadata_map.get(task_name)
        leaderboard = load_leaderboard(metadata) if metadata else None
        valid_scores = [
            float(score)
            for score in (sample.get("submit_score") for sample in samples)
            if score is not None
        ]
        costs = [float(sample.get("total_cost", 0.0)) for sample in samples]

        if valid_scores:
            score_avg = mean(valid_scores)
            score_std = std(valid_scores)
            score_best = (
                min(valid_scores)
                if leaderboard is not None and is_lower_better(leaderboard)
                else max(valid_scores)
            )
        else:
            score_avg = None
            score_std = None
            score_best = None

        if leaderboard is None:
            score_grade_avg = "N/A"
            score_medal_avg = "N/A"
        elif score_avg is not None:
            score_grade_avg = get_grade_for_score(score_avg, leaderboard)
            score_medal_avg = get_medal_for_grade(score_grade_avg, leaderboard)
        else:
            score_grade_avg = None
            score_medal_avg = "N/A"

        submit_grades = [
            float(grade)
            for grade in (sample.get("submit_grade") for sample in samples)
            if grade is not None
        ]
        if submit_grades:
            grade_avg = mean(submit_grades)
            grade_std = std(submit_grades)
            grade_best = min(submit_grades)
        else:
            grade_avg = None
            grade_std = None
            grade_best = None

        if leaderboard is None or grade_avg is None or grade_best is None:
            medal_avg = "N/A"
            medal_best = "N/A"
        else:
            medal_avg = get_medal_for_grade(grade_avg, leaderboard)
            medal_best = get_medal_for_grade(grade_best, leaderboard)

        row = {
            "Task": task_name,
            "score_avg@k": score_avg,
            "score_std@k": score_std,
            "score_best@k": score_best,
            "score_avg@k_grade": score_grade_avg,
            "score_avg@k_medal": score_medal_avg,
            "grade_avg@k": grade_avg,
            "grade_std@k": grade_std,
            "grade_best@k": grade_best,
            "medal_avg@k": medal_avg,
            "medal_best@k": medal_best,
            "cost_avg@k": mean(costs),
            "cost_best@k": min(costs),
            "cost_sum@k": sum(costs),
        }
        status_count = task_status_counts.get(task_name, {})
        status_rollup = {status: 0 for status in all_statuses}
        for status, count in dict(status_count).items():
            if status in status_rollup:
                status_rollup[status] += int(count)
            else:
                # Fold unexpected statuses into unknown for stable columns.
                status_rollup["unknown"] += int(count)
        total_status = sum(status_rollup.values())
        for status in all_statuses:
            row[status] = status_rollup[status]
        row["success_rate"] = (
            status_rollup["success"] / total_status if total_status else 0.0
        )
        summary_rows.append(row)

    summary_columns = [
        "Task",
        "score_avg@k",
        "score_std@k",
        "score_best@k",
        "score_avg@k_grade",
        "score_avg@k_medal",
        "grade_avg@k",
        "grade_std@k",
        "grade_best@k",
        "medal_avg@k",
        "medal_best@k",
        "cost_avg@k",
        "cost_best@k",
        "cost_sum@k",
        *all_statuses,
        "success_rate",
    ]
    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return summary_df


def write_global_stat(
    output_dir: Path,
) -> dict[str, Any]:
    def epoch_index(path: Path) -> int:
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    def aggregate(values: list[float | None]) -> tuple[float | None, float | None]:
        numeric_values = [float(value) for value in values if value is not None]
        if not numeric_values:
            return None, None
        return mean(numeric_values), std(numeric_values)

    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )
    usage_keys = {
        "total_cost",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "total_steps",
    }
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
    }
    metric_values: dict[str, list[float | None]] = {}
    grade_avg_keys = {
        "low_grade_avg@n",
        "middle_grade_avg@n",
        "high_grade_avg@n",
        "all_grade_avg@n",
    }
    grade_std_keys = {
        "low_grade_std@n",
        "middle_grade_std@n",
        "high_grade_std@n",
        "all_grade_std@n",
    }
    grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    for epoch_dir in epoch_dirs:
        epoch_stat_path = epoch_dir / "stat.json"
        epoch_stat = json.loads(epoch_stat_path.read_text(encoding="utf-8"))
        usage["total_cost"] += float(epoch_stat["total_cost"])
        usage["total_prompt_tokens"] += int(epoch_stat["total_prompt_tokens"])
        usage["total_completion_tokens"] += int(epoch_stat["total_completion_tokens"])
        usage["total_tokens"] += int(epoch_stat["total_tokens"])
        usage["total_steps"] += int(epoch_stat["total_steps"])
        # Aggregate per-epoch task counts and medal metrics for avg/std.
        for key, value in epoch_stat.items():
            if key in usage_keys:
                continue
            if key in grade_avg_keys:
                level = key.split("_", 1)[0]
                grade_avg_values[level].append(value)
                continue
            if key in grade_std_keys:
                continue
            metric_values.setdefault(key, []).append(value)

    metric_stats: dict[str, Any] = {}
    for key, values in sorted(metric_values.items()):
        avg, std_value = aggregate(values)
        metric_stats[f"{key}_avg"] = avg
        metric_stats[f"{key}_std"] = std_value

    global_stat = {
        **usage,
        "num_epochs": len(epoch_dirs),
        **metric_stats,
    }
    for level, values in grade_avg_values.items():
        avg, std_value = aggregate(values)
        global_stat[f"{level}_grade_avg@k"] = avg
        global_stat[f"{level}_grade_std@k"] = std_value
    (output_dir / "stat.json").write_text(
        json.dumps(global_stat, indent=2), encoding="utf-8"
    )
    return global_stat
