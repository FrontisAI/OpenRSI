import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    # Load the training data (only file with labels)
    train_df = pd.read_csv(raw / "train.csv")

    # Verify expected structure
    assert "PassengerId" in train_df.columns, "Missing PassengerId column"
    assert "Survived" in train_df.columns, "Missing Survived column"
    assert len(train_df) == 891, f"Expected 891 rows, got {len(train_df)}"
    assert set(train_df["Survived"].unique()).issubset({0, 1}), "Survived should be binary"

    # Stratified 80/20 split with deterministic seed
    train_set, test_set = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["Survived"]
    )

    # Validate split
    assert len(train_set) + len(test_set) == len(train_df), "Split size mismatch"
    assert abs(len(test_set) / len(train_df) - 0.2) < 0.02, "Split ratio off"

    # Sort by PassengerId for consistency
    train_set = train_set.sort_values("PassengerId").reset_index(drop=True)
    test_set = test_set.sort_values("PassengerId").reset_index(drop=True)

    # Save train.csv (with labels) to public
    train_set.to_csv(public / "train.csv", index=False)

    # Save test.csv (without Survived) to public
    test_public = test_set.drop(columns=["Survived"])
    test_public.to_csv(public / "test.csv", index=False)

    # Save test_answer.csv (with labels) to private
    test_answer = test_set[["PassengerId", "Survived"]].copy()
    test_answer.to_csv(private / "test_answer.csv", index=False)

    # Generate sample_submission.csv with random valid labels
    rng = np.random.RandomState(42)
    sample_submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"].values,
        "Survived": rng.randint(0, 2, size=len(test_set))
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Validation checks
    test_answer_check = pd.read_csv(private / "test_answer.csv")
    sample_sub_check = pd.read_csv(public / "sample_submission.csv")
    test_csv_check = pd.read_csv(public / "test.csv")
    train_csv_check = pd.read_csv(public / "train.csv")

    assert list(test_answer_check.columns) == ["PassengerId", "Survived"]
    assert list(sample_sub_check.columns) == ["PassengerId", "Survived"]
    assert len(test_answer_check) == len(sample_sub_check)
    assert set(test_answer_check["PassengerId"]) == set(sample_sub_check["PassengerId"])
    assert set(test_answer_check["PassengerId"]) == set(test_csv_check["PassengerId"])
    assert "Survived" not in test_csv_check.columns, "Label leakage in test.csv!"
    assert "Survived" in train_csv_check.columns, "Train must have labels"
    assert len(set(train_csv_check["PassengerId"]) & set(test_csv_check["PassengerId"])) == 0, "Train/test overlap"
    assert set(sample_sub_check["Survived"].unique()).issubset({0, 1}), "Invalid sample submission labels"

    # Write description
    n_train = len(train_set)
    n_test = len(test_set)

    description = f"""Titanic - Machine Learning from Disaster

The competition is the classic Titanic ML challenge. Use machine learning to build a predictive model that answers the question: "what sorts of people were more likely to survive the Titanic shipwreck?" using passenger data such as name, age, gender, socio-economic class, etc.

The sinking of the RMS Titanic on April 15, 1912, resulted in the death of 1502 out of 2224 passengers and crew due to insufficient lifeboats. While luck played a role, certain groups of people were more likely to survive.

Evaluation
----------
Submissions are evaluated on accuracy — the percentage of passengers correctly predicted (binary classification: survived or not). For each PassengerId in the test set, predict a 0 or 1 value for the Survived variable.

Submission File Format
----------------------
Submit a CSV file with exactly {n_test} entries plus a header row. The file should have exactly 2 columns:
- PassengerId (sorted in any order)
- Survived (contains your binary predictions: 1 for survived, 0 for deceased)

Example:
PassengerId,Survived
1,0
2,1
3,0

Dataset Description
-------------------
The data has been split into two groups: a training set ({n_train} passengers) and a test set ({n_test} passengers). The training set includes the ground truth Survived label and should be used to build models. The test set does not include the Survived label; participants must predict survival outcomes.

Data Dictionary:
| Variable | Definition | Key |
|----------|-----------|-----|
| PassengerId | Unique passenger identifier | |
| Survived | Survival | 0 = No, 1 = Yes |
| Pclass | Ticket class (proxy for socio-economic status) | 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
| Name | Passenger name | |
| Sex | Sex | |
| Age | Age in years | Fractional if < 1; estimated ages in form xx.5 |
| SibSp | # of siblings/spouses aboard | |
| Parch | # of parents/children aboard | |
| Ticket | Ticket number | |
| Fare | Passenger fare | |
| Cabin | Cabin number | |
| Embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

Files
-----
- train.csv: Training set with survival labels ({n_train} passengers). Contains all features plus the Survived column.
- test.csv: Test set without survival labels ({n_test} passengers). Contains all features except Survived.
- sample_submission.csv: A sample submission file in the correct format with random predictions.
"""

    with open(public / "description.txt", "w") as f:
        f.write(description)