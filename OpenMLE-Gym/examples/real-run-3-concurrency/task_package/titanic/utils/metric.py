from typing import Any
import pandas as pd
import numpy as np


class TitanicMetrics:
    """Metric class for Titanic - Machine Learning from Disaster competition using Accuracy."""

    def __init__(self, value: str = "Survived", higher_is_better: bool = True):
        self.higher_is_better = higher_is_better
        self.value = value
        self.id_column = "PassengerId"

    def evaluate(self, y_true: pd.DataFrame = None, y_pred: pd.DataFrame = None) -> float:
        """Calculate accuracy between the predictions and the true labels."""
        # Merge on PassengerId to ensure correct alignment
        merged = pd.merge(
            y_true[[self.id_column, self.value]],
            y_pred[[self.id_column, self.value]],
            on=self.id_column,
            suffixes=("_true", "_pred"),
        )

        if len(merged) == 0:
            raise Exception("No matching PassengerIds found between submission and ground truth.")

        true_vals = merged[f"{self.value}_true"].astype(int)
        pred_vals = merged[f"{self.value}_pred"].astype(int)

        accuracy = (true_vals == pred_vals).mean()
        return float(accuracy)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise Exception("Submission must be a pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise Exception("Ground truth must be a pandas DataFrame.")

        # Check required columns
        if self.id_column not in submission.columns:
            raise Exception(f"Submission must contain the '{self.id_column}' column.")
        if self.value not in submission.columns:
            raise Exception(f"Submission must contain the '{self.value}' column.")
        if self.id_column not in ground_truth.columns:
            raise Exception(f"Ground truth must contain the '{self.id_column}' column.")
        if self.value not in ground_truth.columns:
            raise Exception(f"Ground truth must contain the '{self.value}' column.")

        # Check number of rows
        if len(submission) != len(ground_truth):
            raise Exception(
                f"Number of rows in submission ({len(submission)}) does not match "
                f"ground truth ({len(ground_truth)})."
            )

        # Check that all PassengerIds in ground truth are present in submission
        gt_ids = set(ground_truth[self.id_column])
        sub_ids = set(submission[self.id_column])
        if gt_ids != sub_ids:
            missing = gt_ids - sub_ids
            extra = sub_ids - gt_ids
            msg = "PassengerId mismatch."
            if missing:
                msg += f" Missing IDs: {sorted(missing)[:10]}..."
            if extra:
                msg += f" Extra IDs: {sorted(extra)[:10]}..."
            raise Exception(msg)

        # Check that Survived values are binary (0 or 1)
        unique_vals = submission[self.value].dropna().unique()
        for v in unique_vals:
            if int(v) not in (0, 1):
                raise Exception(
                    f"Survived column must contain only 0 or 1 values. Found: {v}"
                )

        # Check for NaN in Survived
        if submission[self.value].isna().any():
            raise Exception("Submission contains NaN values in the Survived column.")

        return "Submission is valid."