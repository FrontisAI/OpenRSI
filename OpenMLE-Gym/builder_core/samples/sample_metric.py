"""Sample metric implementation used as an LLM prompt reference."""
from typing import Any
import pandas as pd
from sklearn.metrics import log_loss

class KobeBryantShotSelectionMetrics:
    # Metric class for Kobe Bryant Shot Selection competition using Log Loss.
    def __init__(self, value: str = "shot_made_flag", higher_is_better: bool = False):
        self.higher_is_better = higher_is_better
        self.value = value

    def evaluate(self, y_true: pd.DataFrame=None, y_pred: pd.DataFrame=None) -> float:
        # Calculate log loss between the predictions and the true labels.
        if self.value not in y_true.columns or self.value not in y_pred.columns:
            raise Exception(f"Both ground truth and submission must contain the '{self.value}' column.")
        
        # Calculate log loss without sorting or shot_id
        score = log_loss(y_true[self.value], y_pred[self.value])
        return score

    def validate_submission(self, submission: Any, ground_truth: Any) -> str:
        if not isinstance(submission, pd.DataFrame):
            raise Exception("Submission must be a pandas DataFrame. Please provide a valid pandas DataFrame.")
        if not isinstance(ground_truth, pd.DataFrame):
            raise Exception("Ground truth must be a pandas DataFrame. Please provide a valid pandas DataFrame.")

        if len(submission) != len(ground_truth):
            raise Exception(f"Number of rows in submission ({len(submission)}) does not match ground truth ({len(ground_truth)}). Please ensure both have the same number of rows.")

        # Check for required 'value' column in both submission and ground truth
        if self.value not in submission.columns or self.value not in ground_truth.columns:
            raise Exception(f"Both submission and ground truth must contain the '{self.value}' column.")

        # Check if the columns match exactly between submission and ground truth
        if not (submission.columns == ground_truth.columns).all():
            raise Exception("The columns in submission and ground truth must match exactly.")

        return "Submission is valid."
