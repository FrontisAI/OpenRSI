from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

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


def _ordered_unique(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if values is None:
        return None
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Skipping non-mapping JSON file %s", path)
        return None
    return payload


def _level_tasks_for(task_names: list[str]) -> dict[str, list[str]]:
    return {
        "low": [task for task in task_names if task in LOW_TASK_LIST],
        "middle": [task for task in task_names if task in MIDDLE_TASK_LIST],
        "high": [task for task in task_names if task in HIGH_TASK_LIST],
        "all": list(task_names),
    }


def get_total_model_time(payload: dict[str, Any]) -> float:
    return float(payload.get("total_model_time", 0.0))


def get_total_sandbox_time(payload: dict[str, Any]) -> float:
    if payload.get("total_sandbox_time") is not None:
        return float(payload["total_sandbox_time"])
    return float(payload.get("total_time", 0.0))


def get_total_model_plus_sandbox_time(payload: dict[str, Any]) -> float:
    if payload.get("total_model_plus_sandbox_time") is not None:
        return float(payload["total_model_plus_sandbox_time"])
    return get_total_model_time(payload) + get_total_sandbox_time(payload)


def load_leaderboard(metadata: dict[str, Any]) -> pd.DataFrame | None:
    if str(metadata.get("benchmark") or "").strip().lower() == "naturebench":
        return None
    leaderboard_dir = metadata.get("leaderboard_dir")
    task_name = str(metadata["task_name"])
    leaderboard_paths = (
        [Path(metadata["data_dir"]) / "info" / "public_leaderboard.csv"]
        if leaderboard_dir is None
        else [
            Path(leaderboard_dir) / f"{task_name}.csv",
            Path(leaderboard_dir)
            / task_name
            / "info"
            / "public_leaderboard.csv",
            Path(leaderboard_dir) / task_name / "public_leaderboard.csv",
        ]
    )
    for leaderboard_path in leaderboard_paths:
        if not leaderboard_path.exists():
            continue
        leaderboard = pd.read_csv(leaderboard_path)
        leaderboard = leaderboard.rename(
            columns={
                column: str(column).strip().lower()
                for column in leaderboard.columns
            }
        )
        if "score" not in leaderboard.columns:
            logger.warning(
                "Leaderboard %s has no score column",
                leaderboard_path,
            )
            return None
        return leaderboard
    logger.warning(
        "No leaderboard found for task %s in candidates: %s",
        task_name,
        ", ".join(str(path) for path in leaderboard_paths),
    )
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


def write_epoch_stat(
    epoch_output_dir: Path,
    expected_task_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
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
    oracle_task_medals: dict[str, str] = {}
    oracle_task_grades: dict[str, float] = {}
    has_oracle_metrics = False
    method2_oracle_task_medals: dict[str, str] = {}
    method2_oracle_task_grades: dict[str, float] = {}
    has_method2_oracle_metrics = False

    for task_dir in sorted(
        path for path in epoch_output_dir.iterdir() if path.is_dir()
    ):
        stat_path = task_dir / "stat.json"
        if not stat_path.exists():
            continue
        payload = _read_json_file(stat_path)
        if payload is None:
            continue
        task_name = str(payload.get("task_name") or task_dir.name)
        task_names.append(task_name)
        usage["total_cost"] += float(payload.get("total_cost", 0.0))
        usage["total_prompt_tokens"] += int(payload.get("total_prompt_tokens", 0))
        usage["total_completion_tokens"] += int(
            payload.get("total_completion_tokens", 0)
        )
        usage["total_tokens"] += int(payload.get("total_tokens", 0))
        usage["total_steps"] += int(payload.get("num_steps", 0))
        usage["total_model_time"] += get_total_model_time(payload)
        usage["total_sandbox_time"] += get_total_sandbox_time(payload)
        usage["total_model_plus_sandbox_time"] += (
            get_total_model_plus_sandbox_time(payload)
        )

        submit_medal = payload.get("submit_medal")
        task_medals[task_name] = str(submit_medal) if submit_medal else "N/A"
        submit_grade = payload.get("submit_grade")
        task_grades[task_name] = (
            float(submit_grade) if submit_grade is not None else 1.0
        )
        oracle_grade = payload.get("self_valid_oracle_best_grade")
        oracle_medal = payload.get("self_valid_oracle_best_medal")
        if oracle_grade is not None or oracle_medal is not None:
            has_oracle_metrics = True
            oracle_task_grades[task_name] = (
                float(oracle_grade) if oracle_grade is not None else 1.0
            )
            oracle_task_medals[task_name] = (
                str(oracle_medal) if oracle_medal else "N/A"
            )
        method2_oracle_grade = payload.get("method2_oracle_best_grade")
        method2_oracle_medal = payload.get("method2_oracle_best_medal")
        if method2_oracle_grade is not None or method2_oracle_medal is not None:
            has_method2_oracle_metrics = True
            method2_oracle_task_grades[task_name] = (
                float(method2_oracle_grade)
                if method2_oracle_grade is not None
                else 1.0
            )
            method2_oracle_task_medals[task_name] = (
                str(method2_oracle_medal) if method2_oracle_medal else "N/A"
            )

    stat_task_names = _ordered_unique(expected_task_names) or task_names
    for task_name in stat_task_names:
        task_medals.setdefault(task_name, "N/A")
        task_grades.setdefault(task_name, 1.0)
        if has_oracle_metrics:
            oracle_task_medals.setdefault(task_name, "N/A")
            oracle_task_grades.setdefault(task_name, 1.0)
        if has_method2_oracle_metrics:
            method2_oracle_task_medals.setdefault(task_name, "N/A")
            method2_oracle_task_grades.setdefault(task_name, 1.0)

    level_tasks = _level_tasks_for(stat_task_names)
    task_counts = {
        f"{level}_task_count": len(tasks) for level, tasks in level_tasks.items()
    }

    grade_stats: dict[str, float | None] = {}
    for level, tasks in level_tasks.items():
        grades = [task_grades.get(task, 1.0) for task in tasks]
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

    oracle_stats: dict[str, float | int | None] = {}
    if has_oracle_metrics:
        for level, tasks in level_tasks.items():
            grades = [oracle_task_grades.get(task, 1.0) for task in tasks]
            oracle_stats[f"self_valid_oracle_{level}_grade_avg@n"] = (
                mean(grades) if grades else None
            )
            oracle_stats[f"self_valid_oracle_{level}_grade_std@n"] = (
                std(grades) if grades else None
            )
            total = len(tasks)
            for medal in ("gold", "silver", "bronze"):
                count = sum(
                    1 for task in tasks if oracle_task_medals.get(task) == medal
                )
                oracle_stats[f"self_valid_oracle_{level}_{medal}_count"] = count
                oracle_stats[f"self_valid_oracle_{level}_{medal}_rate"] = (
                    count / total if total else None
                )
            any_count = sum(
                1
                for task in tasks
                if oracle_task_medals.get(task) in {"gold", "silver", "bronze"}
            )
            oracle_stats[f"self_valid_oracle_{level}_any_count"] = any_count
            oracle_stats[f"self_valid_oracle_{level}_any_rate"] = (
                any_count / total if total else None
            )

    method2_oracle_stats: dict[str, float | int | None] = {}
    if has_method2_oracle_metrics:
        for level, tasks in level_tasks.items():
            grades = [method2_oracle_task_grades.get(task, 1.0) for task in tasks]
            method2_oracle_stats[f"method2_oracle_{level}_grade_avg@n"] = (
                mean(grades) if grades else None
            )
            method2_oracle_stats[f"method2_oracle_{level}_grade_std@n"] = (
                std(grades) if grades else None
            )
            total = len(tasks)
            for medal in ("gold", "silver", "bronze"):
                count = sum(
                    1
                    for task in tasks
                    if method2_oracle_task_medals.get(task) == medal
                )
                method2_oracle_stats[f"method2_oracle_{level}_{medal}_count"] = count
                method2_oracle_stats[f"method2_oracle_{level}_{medal}_rate"] = (
                    count / total if total else None
                )
            any_count = sum(
                1
                for task in tasks
                if method2_oracle_task_medals.get(task)
                in {"gold", "silver", "bronze"}
            )
            method2_oracle_stats[f"method2_oracle_{level}_any_count"] = any_count
            method2_oracle_stats[f"method2_oracle_{level}_any_rate"] = (
                any_count / total if total else None
            )

    epoch_stat = {
        **usage,
        "sandbox_12h_score": usage["total_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        "model_plus_sandbox_12h_score": usage["total_model_plus_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        **task_counts,
        **grade_stats,
        **medal_counts,
        **medal_rates,
        **oracle_stats,
        **method2_oracle_stats,
    }
    (epoch_output_dir / "stat.json").write_text(
        json.dumps(epoch_stat, indent=2), encoding="utf-8"
    )
    return epoch_stat


def write_summary_csv(
    output_dir: Path,
    task_metadata_map: dict[str, dict[str, Any]],
    expected_task_names: list[str] | tuple[str, ...] | None = None,
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
            payload = _read_json_file(stat_path)
            if payload is None:
                continue
            task_name = str(payload.get("task_name") or task_dir.name)
            task_samples.setdefault(task_name, []).append(payload)
            status_count = payload.get("status_count") or {}
            task_status = task_status_counts.setdefault(task_name, {})
            for status, count in dict(status_count).items():
                task_status[status] = task_status.get(status, 0) + int(count)

    all_statuses = fixed_statuses
    summary_rows: list[dict[str, Any]] = []
    summary_task_names = _ordered_unique(expected_task_names)
    if summary_task_names is None:
        summary_task_names = sorted(task_samples)
    num_epochs = len(epoch_dirs)
    for task_name in summary_task_names:
        samples = task_samples.get(task_name, [])
        metadata = task_metadata_map.get(task_name)
        leaderboard = load_leaderboard(metadata) if metadata else None
        valid_scores = [
            float(score)
            for score in (sample.get("submit_score") for sample in samples)
            if score is not None
        ]
        missing_sample_count = max(0, num_epochs - len(samples))
        costs = [float(sample.get("total_cost", 0.0)) for sample in samples]
        costs.extend([0.0] * missing_sample_count)

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
            float(grade) if grade is not None else 1.0
            for grade in (sample.get("submit_grade") for sample in samples)
        ]
        submit_grades.extend([1.0] * missing_sample_count)
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

        oracle_grades = [
            float(grade) if grade is not None else 1.0
            for grade in (
                sample.get("self_valid_oracle_best_grade") for sample in samples
            )
        ]
        oracle_grades.extend([1.0] * missing_sample_count)
        if oracle_grades:
            oracle_grade_avg = mean(oracle_grades)
            oracle_grade_std = std(oracle_grades)
            oracle_grade_best = min(oracle_grades)
        else:
            oracle_grade_avg = None
            oracle_grade_std = None
            oracle_grade_best = None

        if leaderboard is None or oracle_grade_avg is None or oracle_grade_best is None:
            oracle_medal_avg = "N/A"
            oracle_medal_best = "N/A"
        else:
            oracle_medal_avg = get_medal_for_grade(oracle_grade_avg, leaderboard)
            oracle_medal_best = get_medal_for_grade(oracle_grade_best, leaderboard)

        has_method2_oracle = any(
            "method2_oracle_best_grade" in sample
            or "method2_oracle_best_medal" in sample
            for sample in samples
        )
        if has_method2_oracle:
            method2_oracle_grades = [
                float(grade) if grade is not None else 1.0
                for grade in (
                    sample.get("method2_oracle_best_grade") for sample in samples
                )
            ]
            method2_oracle_grades.extend([1.0] * missing_sample_count)
            method2_oracle_grade_avg = mean(method2_oracle_grades)
            method2_oracle_grade_std = std(method2_oracle_grades)
            method2_oracle_grade_best = min(method2_oracle_grades)
        else:
            method2_oracle_grade_avg = None
            method2_oracle_grade_std = None
            method2_oracle_grade_best = None

        if (
            leaderboard is None
            or method2_oracle_grade_avg is None
            or method2_oracle_grade_best is None
        ):
            method2_oracle_medal_avg = "N/A"
            method2_oracle_medal_best = "N/A"
        else:
            method2_oracle_medal_avg = get_medal_for_grade(
                method2_oracle_grade_avg,
                leaderboard,
            )
            method2_oracle_medal_best = get_medal_for_grade(
                method2_oracle_grade_best,
                leaderboard,
            )

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
            "self_valid_oracle_grade_avg@k": oracle_grade_avg,
            "self_valid_oracle_grade_std@k": oracle_grade_std,
            "self_valid_oracle_grade_best@k": oracle_grade_best,
            "self_valid_oracle_medal_avg@k": oracle_medal_avg,
            "self_valid_oracle_medal_best@k": oracle_medal_best,
            "method2_oracle_grade_avg@k": method2_oracle_grade_avg,
            "method2_oracle_grade_std@k": method2_oracle_grade_std,
            "method2_oracle_grade_best@k": method2_oracle_grade_best,
            "method2_oracle_medal_avg@k": method2_oracle_medal_avg,
            "method2_oracle_medal_best@k": method2_oracle_medal_best,
            "cost_avg@k": mean(costs) if costs else 0.0,
            "cost_best@k": min(costs) if costs else 0.0,
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
        status_rollup["unknown"] += missing_sample_count
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
        "self_valid_oracle_grade_avg@k",
        "self_valid_oracle_grade_std@k",
        "self_valid_oracle_grade_best@k",
        "self_valid_oracle_medal_avg@k",
        "self_valid_oracle_medal_best@k",
        "method2_oracle_grade_avg@k",
        "method2_oracle_grade_std@k",
        "method2_oracle_grade_best@k",
        "method2_oracle_medal_avg@k",
        "method2_oracle_medal_best@k",
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
        "total_model_time",
        "total_sandbox_time",
        "total_model_plus_sandbox_time",
    }
    usage = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_steps": 0,
        "total_model_time": 0.0,
        "total_sandbox_time": 0.0,
        "total_model_plus_sandbox_time": 0.0,
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
    oracle_grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    method2_oracle_grade_avg_values: dict[str, list[float | None]] = {
        "low": [],
        "middle": [],
        "high": [],
        "all": [],
    }
    for epoch_dir in epoch_dirs:
        epoch_stat_path = epoch_dir / "stat.json"
        epoch_stat = _read_json_file(epoch_stat_path)
        if epoch_stat is None:
            continue
        usage["total_cost"] += float(epoch_stat["total_cost"])
        usage["total_prompt_tokens"] += int(epoch_stat["total_prompt_tokens"])
        usage["total_completion_tokens"] += int(epoch_stat["total_completion_tokens"])
        usage["total_tokens"] += int(epoch_stat["total_tokens"])
        usage["total_steps"] += int(epoch_stat["total_steps"])
        usage["total_model_time"] += float(epoch_stat.get("total_model_time", 0.0))
        usage["total_sandbox_time"] += float(
            epoch_stat.get("total_sandbox_time", 0.0)
        )
        usage["total_model_plus_sandbox_time"] += float(
            epoch_stat.get("total_model_plus_sandbox_time", 0.0)
        )
        # Aggregate per-epoch task counts and medal metrics for avg/std.
        for key, value in epoch_stat.items():
            if key in usage_keys:
                continue
            if key in grade_avg_keys:
                level = key.split("_", 1)[0]
                grade_avg_values[level].append(value)
                continue
            if key.startswith("self_valid_oracle_") and key.endswith("_grade_avg@n"):
                level = key.removeprefix("self_valid_oracle_").split("_", 1)[0]
                oracle_grade_avg_values[level].append(value)
                continue
            if key.startswith("method2_oracle_") and key.endswith("_grade_avg@n"):
                level = key.removeprefix("method2_oracle_").split("_", 1)[0]
                method2_oracle_grade_avg_values[level].append(value)
                continue
            if key in grade_std_keys:
                continue
            if key.startswith("self_valid_oracle_") and key.endswith("_grade_std@n"):
                continue
            if key.startswith("method2_oracle_") and key.endswith("_grade_std@n"):
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
        "sandbox_12h_score": usage["total_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        "model_plus_sandbox_12h_score": usage["total_model_plus_sandbox_time"]
        / TIME_SCALING_SCORE_SECONDS,
        **metric_stats,
    }
    for level, values in grade_avg_values.items():
        avg, std_value = aggregate(values)
        global_stat[f"{level}_grade_avg@k"] = avg
        global_stat[f"{level}_grade_std@k"] = std_value
    for level, values in oracle_grade_avg_values.items():
        if not values:
            continue
        avg, std_value = aggregate(values)
        global_stat[f"self_valid_oracle_{level}_grade_avg@k"] = avg
        global_stat[f"self_valid_oracle_{level}_grade_std@k"] = std_value
    for level, values in method2_oracle_grade_avg_values.items():
        if not values:
            continue
        avg, std_value = aggregate(values)
        global_stat[f"method2_oracle_{level}_grade_avg@k"] = avg
        global_stat[f"method2_oracle_{level}_grade_std@k"] = std_value
    (output_dir / "stat.json").write_text(
        json.dumps(global_stat, indent=2), encoding="utf-8"
    )
    return global_stat


def write_time_scaling(output_dir: Path) -> pd.DataFrame:
    def epoch_index(path: Path) -> int:
        suffix = path.name.split("program_ep_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    epoch_dirs = sorted(
        [
            path
            for path in output_dir.iterdir()
            if path.is_dir() and path.name.startswith("program_ep_")
        ],
        key=epoch_index,
    )

    samples: list[dict[str, Any]] = []
    for epoch_dir in epoch_dirs:
        for task_dir in sorted(path for path in epoch_dir.iterdir() if path.is_dir()):
            stat_path = task_dir / "stat.json"
            if not stat_path.exists():
                continue
            payload = _read_json_file(stat_path)
            if payload is None:
                continue
            submit_grade = payload.get("submit_grade")
            samples.append(
                {
                    "task_name": str(payload.get("task_name") or task_dir.name),
                    "submit_grade": (
                        float(submit_grade) if submit_grade is not None else 1.0
                    ),
                    "submit_medal": str(payload.get("submit_medal") or "N/A"),
                    "sandbox_time": get_total_sandbox_time(payload),
                    "model_plus_sandbox_time": get_total_model_plus_sandbox_time(
                        payload
                    ),
                }
            )

    axis_specs = [
        ("model_plus_sandbox_time", "Model + Sandbox Time"),
        ("sandbox_time", "Sandbox Time"),
    ]
    rows: list[dict[str, Any]] = []

    if samples:
        total = len(samples)
        for axis_key, axis_label in axis_specs:
            budgets = sorted(
                {
                    0.0,
                    float(TIME_SCALING_SCORE_SECONDS),
                    *[float(sample[axis_key]) for sample in samples],
                }
            )
            for budget_seconds in budgets:
                covered_samples = [
                    sample
                    for sample in samples
                    if float(sample[axis_key]) <= budget_seconds
                ]
                covered_count = len(covered_samples)
                grades = [
                    float(sample["submit_grade"])
                    if float(sample[axis_key]) <= budget_seconds
                    else 1.0
                    for sample in samples
                ]
                medal_rate = (
                    sum(
                        1
                        for sample in covered_samples
                        if sample["submit_medal"] in {"gold", "silver", "bronze"}
                    )
                    / total
                )
                rows.append(
                    {
                        "time_axis": axis_key,
                        "time_label": axis_label,
                        "budget_seconds": float(budget_seconds),
                        "budget_hours": float(budget_seconds) / 3600.0,
                        "sample_count": total,
                        "covered_sample_count": covered_count,
                        "coverage_rate": covered_count / total,
                        "grade_avg@k": sum(grades) / len(grades),
                        "medal_rate": medal_rate,
                    }
                )

    time_scaling_df = pd.DataFrame(
        rows,
        columns=[
            "time_axis",
            "time_label",
            "budget_seconds",
            "budget_hours",
            "sample_count",
            "covered_sample_count",
            "coverage_rate",
            "grade_avg@k",
            "medal_rate",
        ],
    )
    time_scaling_df.to_csv(output_dir / "time_scaling.csv", index=False)

    if time_scaling_df.empty:
        return time_scaling_df

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for column, (axis_key, axis_label) in enumerate(axis_specs):
        axis_df = time_scaling_df[time_scaling_df["time_axis"] == axis_key]

        medal_ax = axes[0, column]
        medal_ax.plot(
            axis_df["budget_hours"],
            axis_df["medal_rate"],
            color="tab:blue",
            linewidth=2,
        )
        medal_ax.axvline(12.0, color="gray", linestyle="--", linewidth=1)
        medal_ax.set_title(axis_label)
        medal_ax.set_ylabel("Medal Rate")
        medal_ax.set_xlabel("Time Budget (hours)")
        medal_ax.set_ylim(0.0, 1.0)
        medal_ax.grid(alpha=0.3)

        grade_ax = axes[1, column]
        grade_ax.plot(
            axis_df["budget_hours"],
            axis_df["grade_avg@k"],
            color="tab:orange",
            linewidth=2,
        )
        grade_ax.axvline(12.0, color="gray", linestyle="--", linewidth=1)
        grade_ax.set_ylabel("Grade (lower is better)")
        grade_ax.set_xlabel("Time Budget (hours)")
        grade_ax.set_ylim(1.0, 0.0)
        grade_ax.grid(alpha=0.3)

    fig.suptitle("Time Scaling")
    fig.savefig(output_dir / "time_scaling.png", dpi=200)
    plt.close(fig)
    return time_scaling_df
