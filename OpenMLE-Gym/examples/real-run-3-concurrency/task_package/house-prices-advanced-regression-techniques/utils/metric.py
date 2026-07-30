import numpy as np
import pandas as pd
from typing import Any


class AmesHousingMetrics:
    """Metric class for Ames Housing Price Prediction using RMSLE."""

    def __init__(self, value: str = "SalePrice", higher_is_better: bool = False):
        self.higher_is_better = higher_is_better
        self.value = value

    def evaluate(self, y_true: pd.DataFrame = None, y_pred: pd.DataFrame = None) -> float:
        """Calculate RMSE between log of predicted and log of actual SalePrice."""
        # Merge on Id to ensure alignment
        merged = pd.merge(y_true, y_pred, on="Id", suffixes=("_true", "_pred"))

        true_values = merged[f"{self.value}_true"].values.astype(float)
        pred_values = merged[f"{self.value}_pred"].values.astype(float)

        # Clip to avoid log(0) or log(negative)
        true_values = np.clip(true_values, 1e-15, None)
        pred_values = np.clip(pred_values, 1e-15, None)

        log_true = np.log(true_values)
        log_pred = np.log(pred_values)

        rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))

        return float(rmse)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise Exception("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise Exception("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if "Id" not in submission.columns:
            raise Exception("Submission must contain an 'Id' column.")
        if "Id" not in ground_truth.columns:
            raise Exception("Ground truth must contain an 'Id' column.")

        if self.value not in submission.columns:
            raise Exception(f"Submission must contain the '{self.value}' column.")
        if self.value not in ground_truth.columns:
            raise Exception(f"Ground truth must contain the '{self.value}' column.")

        if len(submission) != len(ground_truth):
            raise Exception(
                f"Number of rows in submission ({len(submission)}) does not match "
                f"ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        # Check that all Ids in ground truth are present in submission
        gt_ids = set(ground_truth["Id"].values)
        sub_ids = set(submission["Id"].values)
        if gt_ids != sub_ids:
            raise Exception("The set of Ids in submission does not match the set of Ids in ground truth.")

        # Check for NaN values in submission SalePrice
        if submission[self.value].isnull().any():
            raise Exception(f"Submission contains NaN values in the '{self.value}' column.")

        # Check for non-positive values
        if (submission[self.value] <= 0).any():
            raise Exception(f"Submission contains non-positive values in the '{self.value}' column. All prices must be positive.")

        return "Submission is valid."