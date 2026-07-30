"""
Logging utilities for tracking training data and samples.
"""

import csv
import json
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional

EVAL_GROUP_SIZE = 8


class TrainingLogger:
    """Manages logging of training data to CSV files."""

    def __init__(self, log_file_name: str = None, n_samples_per_prompt: int = 8):
        """
        Initialize training logger.

        Args:
            log_file_name: Name of the log file (default: train_log.csv)
            n_samples_per_prompt: Number of samples per prompt group
        """
        self.n_samples_per_prompt = n_samples_per_prompt

        # Setup log directory - use RUN_DIR from environment if available
        run_dir_from_env = os.environ.get("RUN_DIR")
        self.run_dir = run_dir_from_env
        os.makedirs(self.run_dir, exist_ok=True)

        if log_file_name is None:
            log_file_name = os.environ.get("LOG_FILE_NAME", "train_log.csv")

        self.csv_log_file = os.path.join(self.run_dir, log_file_name)
        self.csv_initialized = False

        # Group logging buffer
        self.group_logging_buffer = defaultdict(list)
        self.group_logging_lock = Lock()

    def init_csv(self):
        """Initialize CSV log file with headers."""
        if not self.csv_initialized:
            with open(self.csv_log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = [
                    "group_id",
                    "task_id",
                    "task_name",
                    "higher_is_better",
                    "theoretical_max",
                    "theoretical_min",
                    "leaderboard_max",
                    "leaderboard_min",
                    "prompt_type",
                    # "prompts",
                    "scores",
                    "rewards",
                    "static_base_rewards",
                    "base_rewards",
                    "exploit_coefficients",
                    "explore_coefficients",
                    "cooling_coefficients",
                    "results",
                    "codes",
                    "raw_texts",
                    "parent_ids",
                    "parent_codes",
                    "parent_scores",
                    "parent_rewards",
                    "modes",
                    "hacks",
                    "code_categories",
                    "static_raw_rewards",
                    "static_mercy_raw_rewards",
                    "dynamic_raw_rewards",
                ]
                writer.writerow(headers)
            self.csv_initialized = True
            print(
                f"[LOG] Initialized CSV log file: {self.csv_log_file} with {self.n_samples_per_prompt} samples per prompt"
            )
            print(f"[LOG] Run directory: {self.run_dir}")

    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir

    def log_group_data(self, group_key: str, group_data: List[Dict], group_id: int, group_size: int):
        """Write a complete group's data to CSV."""
        print("[LOG_GROUP_DATA]")

        if len(group_data) != group_size:
            print(f"[LOG WARNING] Group {group_key} has {len(group_data)} samples, expected {group_size}. Skipping.")
            return

        # Extract common metadata from first sample
        first_sample = group_data[0]
        metadata = first_sample["metadata"]

        row = [
            group_id,  # Use sequential group ID as step
            metadata.get("uuid", "N/A"),  # task_id (uuid in metadata)
            metadata.get("task_name", "N/A"),
            metadata.get("higher_is_better", "N/A"),
            metadata.get("theoretical_max", "N/A"),
            metadata.get("theoretical_min", "N/A"),
            metadata.get("leaderboard_max", "N/A"),
            metadata.get("leaderboard_min", "N/A"),
            metadata.get("prompt_type", "N/A"),
        ]

        (
            # prompts,
            scores,
            rewards,
            static_base_rewards,
            base_rewards,
            exploit_coefficients,
            explore_coefficients,
            cooling_coefficients,
            results,
            codes,
            raw_texts,
            parent_ids,
            parent_codes,
            parent_scores,
            parent_rewards,
            modes,
            hacks,
            code_categories,
            static_raw_rewards,
            static_mercy_raw_rewards,
            dynamic_raw_rewards,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample_data in group_data:
            # Collect prompt (serialize if it's a list/dict)
            # prompt = sample_data.get("prompt", "")
            # if isinstance(prompt, (dict, list)):
            #     prompt_serialized = json.dumps(prompt, ensure_ascii=False)
            # else:
            #     prompt_serialized = str(prompt) if prompt else ""
            # prompt_serialized = prompt_serialized.replace("\r", "").replace("\n", "\\n")
            # prompts.append(prompt_serialized)

            scores.append(sample_data["score"])
            rewards.append(sample_data["reward"])
            static_base_rewards.append(sample_data.get("static_base_reward"))
            base_rewards.append(sample_data.get("base_reward", sample_data["reward"]))
            exploit_coefficients.append(sample_data.get("exploit_coefficient"))
            explore_coefficients.append(sample_data.get("explore_coefficient"))
            cooling_coefficients.append(sample_data.get("cooling_coefficient"))

            result_info = sample_data.get("result", "")
            if isinstance(result_info, (dict, list)):
                result_serialized = result_info
            else:
                result_serialized = "" if result_info is None else str(result_info)
                result_serialized = result_serialized.replace("\r", "").replace("\n", "\\n")
            results.append(result_serialized)

            code = sample_data["code"]
            code = code.replace("\r", "").replace("\n", "\\n")
            codes.append(code)

            raw_text = sample_data.get("raw_text", "")
            raw_text = raw_text.replace("\r", "").replace("\n", "\\n")
            raw_texts.append(raw_text)

            parent_ids.append(sample_data.get("parent_id"))

            parent_code = sample_data.get("parent_code", "")
            parent_code = parent_code.replace("\r", "").replace("\n", "\\n")
            parent_codes.append(parent_code)

            parent_scores.append(sample_data.get("parent_score"))
            parent_rewards.append(sample_data.get("parent_reward"))

            modes.append(sample_data.get("mode", "draft"))
            hacks.append(sample_data.get("hack", 0))  # Default to 0 (valid) if not present
            code_categories.append(sample_data.get("code_category", "valid"))
            static_raw_rewards.append(sample_data.get("static_raw_reward"))
            static_mercy_raw_rewards.append(sample_data.get("static_mercy_raw_reward"))
            dynamic_raw_rewards.append(sample_data.get("dynamic_raw_reward"))

        # row.append(json.dumps(prompts, ensure_ascii=False))
        row.append(json.dumps(scores, ensure_ascii=False))
        row.append(json.dumps(rewards, ensure_ascii=False))
        row.append(json.dumps(static_base_rewards, ensure_ascii=False))
        row.append(json.dumps(base_rewards, ensure_ascii=False))
        row.append(json.dumps(exploit_coefficients, ensure_ascii=False))
        row.append(json.dumps(explore_coefficients, ensure_ascii=False))
        row.append(json.dumps(cooling_coefficients, ensure_ascii=False))
        row.append(json.dumps(results, ensure_ascii=False))
        row.append(json.dumps(codes, ensure_ascii=False))
        row.append(json.dumps(raw_texts, ensure_ascii=False))
        row.append(json.dumps(parent_ids, ensure_ascii=False))
        row.append(json.dumps(parent_codes, ensure_ascii=False))
        row.append(json.dumps(parent_scores, ensure_ascii=False))
        row.append(json.dumps(parent_rewards, ensure_ascii=False))
        row.append(json.dumps(modes, ensure_ascii=False))
        row.append(json.dumps(hacks, ensure_ascii=False))
        row.append(json.dumps(code_categories, ensure_ascii=False))
        row.append(json.dumps(static_raw_rewards, ensure_ascii=False))
        row.append(json.dumps(static_mercy_raw_rewards, ensure_ascii=False))
        row.append(json.dumps(dynamic_raw_rewards, ensure_ascii=False))

        # Write to CSV
        with open(self.csv_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[LOG] Wrote group data for group_id={group_id}, task={metadata.get('task_name', 'N/A')}")

    def log_sample(
        self,
        sample,
        score: Optional[float],
        reward: float,
        static_base_reward: Optional[float],
        base_reward: float,
        result_info: Any,
        code: str,
        raw_text: str = "",
        parent_id: Optional[int] = None,
        parent_code: str = "",
        parent_score: Optional[float] = None,
        parent_reward: Optional[float] = None,
        mode: str = "draft",
        hack: int = 0,
        code_category: str = "valid",
        static_raw_reward: Optional[float] = None,
        static_mercy_raw_reward: Optional[float] = None,
        dynamic_raw_reward: Optional[float] = None,
        exploit_coefficient: Optional[float] = None,
        explore_coefficient: Optional[float] = None,
        cooling_coefficient: Optional[float] = None,
    ):
        """
        Collect sample information and write to CSV when a group is complete.

        Uses sample.group_index to uniquely identify each group.
        This ensures that even when the same prompt appears multiple times in a batch,
        each group is tracked separately.

        Args:
            sample: The sample object
            score: The score from sandbox execution
            reward: The final reward used for training (may be diff for improve mode)
            base_reward: The base reward before considering parent
            result_info: Execution result information
            code: The extracted code
            raw_text: Raw response text from model
            parent_id: Parent program ID (for improve mode)
            parent_code: Parent program code (for improve mode)
            parent_score: Parent program score (for improve mode)
            parent_reward: Parent program reward (for improve mode)
            mode: Generation mode ('draft' or 'improve')
            hack: Hack check result (0 = not hacking, valid ML code, 1 = cheating/guessing)
            code_category: Code category ('valid', 'hack_non_empty', 'empty', 'no_valid_code')
        """
        print("[LOG_SAMPLE_TO_GROUP]")

        # Use group_index as the unique identifier for this group
        # group_index is set by the data source and is unique for each group
        group_index = sample.group_index if hasattr(sample, "group_index") and sample.group_index is not None else -1

        if group_index < 0:
            print(f"[LOG WARNING] Sample has no valid group_index, using fallback")
            # Fallback to prompt hash if group_index is not available
            group_key = f"fallback_{sample.metadata.get('uuid', 'N/A')}"
            print(f"[LOG WARNING] Group key: {group_key}")
        else:
            group_key = f"group_{group_index}"

        # Extract prompt from sample
        # prompt = sample.prompt if hasattr(sample, "prompt") else ""

        # Prepare sample data
        sample_data = {
            "metadata": sample.metadata if hasattr(sample, "metadata") else {},
            # "prompt": prompt,
            "score": score,
            "reward": reward,
            "static_base_reward": static_base_reward,
            "base_reward": base_reward,
            "result": result_info,
            "code": code,
            "raw_text": raw_text,
            "sample_index": sample.index if hasattr(sample, "index") else None,
            "parent_id": parent_id,
            "parent_code": parent_code,
            "parent_score": parent_score,
            "parent_reward": parent_reward,
            "mode": mode,
            "hack": hack,
            "code_category": code_category,
            "static_raw_reward": static_raw_reward,
            "static_mercy_raw_reward": static_mercy_raw_reward,
            "dynamic_raw_reward": dynamic_raw_reward,
            "exploit_coefficient": exploit_coefficient,
            "explore_coefficient": explore_coefficient,
            "cooling_coefficient": cooling_coefficient,
        }

        # Thread-safe buffer update
        with self.group_logging_lock:
            self.group_logging_buffer[group_key].append(sample_data)
            group_size = self.n_samples_per_prompt if group_index >= 0 else EVAL_GROUP_SIZE
            print(
                f"[LOG WARNING] Group: {len(self.group_logging_buffer[group_key])}/{group_size} for group_key: {group_key}"
            )
            # When we have n_samples_per_prompt samples (a complete group), write to CSV
            if len(self.group_logging_buffer[group_key]) == group_size:
                group_data = self.group_logging_buffer[group_key]

                # Sort samples by their index to ensure consistent order
                group_data.sort(key=lambda x: x.get("sample_index", 0))

                # Use the group_index directly as the step number (it's already unique and sequential)
                self.log_group_data(group_key, group_data, group_index, group_size)
                # Clear this group from buffer
                del self.group_logging_buffer[group_key]
            elif len(self.group_logging_buffer[group_key]) > group_size:
                # This shouldn't happen, but log a warning if it does
                print(
                    f"[LOG WARNING] Group {group_key} has more than {group_size} samples: {len(self.group_logging_buffer[group_key])}"
                )


# Global logger instance (can be initialized by generate_mle.py)
_global_logger = None


def get_logger() -> TrainingLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger()
        _global_logger.init_csv()
    return _global_logger


def init_logger(log_file_name: str = None, n_samples_per_prompt: int = 8):
    """Initialize the global logger."""
    global _global_logger
    _global_logger = TrainingLogger(log_file_name, n_samples_per_prompt)
    _global_logger.init_csv()
    return _global_logger
