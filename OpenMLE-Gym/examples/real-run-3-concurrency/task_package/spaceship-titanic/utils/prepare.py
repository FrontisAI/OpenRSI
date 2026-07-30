import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    # Load training data (the only data with labels)
    train_df = pd.read_csv(raw / "train.csv")

    # Verify expected columns
    assert "PassengerId" in train_df.columns, "Missing PassengerId column"
    assert "Transported" in train_df.columns, "Missing Transported column"
    assert train_df["PassengerId"].nunique() == len(train_df), "PassengerId not unique"

    # Perform stratified 80/20 split with deterministic seed
    train_set, test_set = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["Transported"]
    )

    # Validate split
    assert len(train_set) + len(test_set) == len(train_df), "Split size mismatch"
    assert abs(len(test_set) / len(train_df) - 0.2) < 0.01, "Split ratio off"

    # No overlap
    assert len(set(train_set["PassengerId"]) & set(test_set["PassengerId"])) == 0, "ID overlap"

    # Save train.csv (with labels) to public
    train_set.to_csv(public / "train.csv", index=False)

    # Save test.csv (without labels) to public
    test_public = test_set.drop(columns=["Transported"])
    test_public.to_csv(public / "test.csv", index=False)

    # Save test_answer.csv to private
    test_answer = test_set[["PassengerId", "Transported"]].copy()
    test_answer.to_csv(private / "test_answer.csv", index=False)

    # Create sample_submission.csv with random but valid labels
    rng = np.random.RandomState(42)
    valid_labels = [True, False]
    random_labels = rng.choice(valid_labels, size=len(test_set))
    sample_submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"].values,
        "Transported": random_labels
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Validate alignment between test_answer and sample_submission
    ta = pd.read_csv(private / "test_answer.csv")
    ss = pd.read_csv(public / "sample_submission.csv")
    assert list(ta.columns) == list(ss.columns), "Column mismatch"
    assert len(ta) == len(ss), "Row count mismatch"
    assert set(ta["PassengerId"]) == set(ss["PassengerId"]), "ID mismatch"

    # Validate test.csv and test_answer alignment
    tc = pd.read_csv(public / "test.csv")
    assert len(tc) == len(ta), "Test CSV and answer row count mismatch"
    assert set(tc["PassengerId"]) == set(ta["PassengerId"]), "Test CSV and answer ID mismatch"

    # Ensure no label leakage in public test.csv
    assert "Transported" not in tc.columns, "Label leakage in test.csv"

    # Write description
    description = f"""# Spaceship Titanic - Predict which passengers are transported to an alternate dimension

## Overview

The year is 2912, and the Spaceship Titanic, an interstellar passenger liner, has collided with a spacetime anomaly hidden within a dust cloud. Almost half of the passengers were transported to an alternate dimension. Your task is to predict which passengers were transported using records recovered from the ship's damaged computer system. This is a binary classification task.

## Evaluation

Submissions are evaluated based on **classification accuracy**, the percentage of predicted labels that are correct.

## Submission Format

For each PassengerId in the test set, predict either `True` or `False` for the `Transported` variable. The file should contain a header and have the following format:

```
PassengerId,Transported
0013_01,False
0018_01,False
etc.
```

## Dataset Description

- **train.csv** — Personal records for {len(train_set)} passengers (training data). Contains the target column `Transported` (True/False). Features include:
  - PassengerId: Unique ID in the form gggg_pp (group and number within group)
  - HomePlanet: The planet the passenger departed from
  - CryoSleep: Whether the passenger elected to be put into suspended animation
  - Cabin: Cabin number in format deck/num/side
  - Destination: The planet the passenger is traveling to
  - Age: The age of the passenger
  - VIP: Whether the passenger paid for VIP service
  - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: Amounts billed at luxury amenities
  - Name: First and last name of the passenger
  - Transported: Whether the passenger was transported (target variable)

- **test.csv** — Personal records for {len(test_set)} passengers (test data). Does not contain the `Transported` column.

- **sample_submission.csv** — A sample submission file in the correct format with {len(test_set)} entries.

## Notes

- People in a travel group (same gggg prefix) are often family members, but not always.
- Passengers in cryosleep are confined to their cabins.
- Some fields may contain missing values.
"""
    with open(public / "description.txt", "w") as f:
        f.write(description)