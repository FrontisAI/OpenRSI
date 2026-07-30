from typing import Any
import pandas as pd


class SpaceshipTitanicMetrics:
    """Metric class for Spaceship Titanic competition using Classification Accuracy."""

    def __init__(self, value: str = "Transported", higher_is_better: bool = True):
        self.higher_is_better = higher_is_better
        self.value = value

    def _parse_bool(self, series: pd.Series) -> pd.Series:
        """Convert string or boolean values to consistent boolean."""
        def convert(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() == "true"
            # Handle numpy bool etc.
            return bool(val)
        return series.apply(convert)

    def evaluate(self, y_true: pd.DataFrame = None, y_pred: pd.DataFrame = None) -> float:
        """Calculate classification accuracy between predictions and true labels."""
        if self.value not in y_true.columns or self.value not in y_pred.columns:
            raise Exception(f"Both ground truth and submission must contain the '{self.value}' column.")

        # Align by PassengerId if present
        if "PassengerId" in y_true.columns and "PassengerId" in y_pred.columns:
            merged = y_true[["PassengerId", self.value]].merge(
                y_pred[["PassengerId", self.value]],
                on="PassengerId",
                suffixes=("_true", "_pred")
            )
            true_vals = self._parse_bool(merged[f"{self.value}_true"])
            pred_vals = self._parse_bool(merged[f"{self.value}_pred"])
        else:
            true_vals = self._parse_bool(y_true[self.value].reset_index(drop=True))
            pred_vals = self._parse_bool(y_pred[self.value].reset_index(drop=True))

        if len(true_vals) == 0:
            raise Exception("No matching records found for evaluation.")

        accuracy = (true_vals == pred_vals).mean()
        return float(accuracy)

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise Exception("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise Exception("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise Exception(
                f"Number of rows in submission ({len(submission)}) does not match "
                f"ground truth ({len(ground_truth)}). Please ensure both have the same number of rows."
            )

        if self.value not in submission.columns:
            raise Exception(f"Submission must contain the '{self.value}' column.")
        if self.value not in ground_truth.columns:
            raise Exception(f"Ground truth must contain the '{self.value}' column.")

        if "PassengerId" in ground_truth.columns and "PassengerId" not in submission.columns:
            raise Exception("Submission must contain the 'PassengerId' column.")

        if "PassengerId" in ground_truth.columns and "PassengerId" in submission.columns:
            gt_ids = set(ground_truth["PassengerId"])
            sub_ids = set(submission["PassengerId"])
            if gt_ids != sub_ids:
                raise Exception("PassengerId values in submission do not match those in ground truth.")

        # Check for null values in the target column
        if submission[self.value].isnull().any():
            raise Exception(f"Submission contains null values in the '{self.value}' column.")

        return "Submission is valid."