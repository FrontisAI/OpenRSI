import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path):
    # Create output directories
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # Load the training data (only file with labels)
    train_data = pd.read_csv(raw / "train.csv")

    # Verify expected structure
    assert 'Id' in train_data.columns, "Missing 'Id' column"
    assert 'SalePrice' in train_data.columns, "Missing 'SalePrice' column"
    assert len(train_data) == 1460, f"Expected 1460 rows, got {len(train_data)}"
    assert train_data.shape[1] == 81, f"Expected 81 columns, got {train_data.shape[1]}"

    # Deterministic split: 80% train, 20% test
    train_set, test_set = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42
    )

    # Validate split
    assert len(train_set) + len(test_set) == len(train_data), "Split size mismatch"
    assert abs(len(test_set) / len(train_data) - 0.2) < 0.02, "Split ratio off"

    # Reset indices
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    # Save train.csv (with labels)
    train_set.to_csv(public / "train.csv", index=False)

    # Save test.csv (without SalePrice)
    test_public = test_set.drop(columns=['SalePrice'])
    test_public.to_csv(public / "test.csv", index=False)

    # Save test_answer.csv (private)
    test_answer = test_set[['Id', 'SalePrice']].copy()
    test_answer.to_csv(private / "test_answer.csv", index=False)

    # Create sample_submission.csv with random but valid SalePrice values
    rng = np.random.RandomState(42)
    random_prices = rng.choice(train_set['SalePrice'].values, size=len(test_set), replace=True)
    sample_submission = pd.DataFrame({
        'Id': test_set['Id'].values,
        'SalePrice': random_prices.astype(float)
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)

    # Copy data_description.txt as additional metadata
    if (raw / "data_description.txt").exists():
        shutil.copy2(raw / "data_description.txt", public / "data_description.txt")

    # Validate alignment between test.csv, test_answer.csv, and sample_submission.csv
    test_csv_check = pd.read_csv(public / "test.csv")
    test_answer_check = pd.read_csv(private / "test_answer.csv")
    sample_sub_check = pd.read_csv(public / "sample_submission.csv")

    assert list(test_csv_check['Id']) == list(test_answer_check['Id']), "Id mismatch: test vs test_answer"
    assert list(test_csv_check['Id']) == list(sample_sub_check['Id']), "Id mismatch: test vs sample_submission"
    assert len(test_answer_check) == len(sample_sub_check), "Row count mismatch"
    assert list(test_answer_check.columns) == list(sample_sub_check.columns), "Column mismatch"
    assert 'SalePrice' not in test_csv_check.columns, "Label leakage in test.csv!"

    # Verify train.csv has labels
    train_check = pd.read_csv(public / "train.csv")
    assert 'SalePrice' in train_check.columns, "train.csv missing SalePrice"
    assert len(train_check) == len(train_set), "train.csv row count mismatch"

    # Generate description.txt
    description = f"""# Ames Housing Price Prediction

## Overview

The competition challenges participants to predict the final sale price of residential homes in Ames, Iowa, using 79 explanatory variables describing nearly every aspect of the homes. The dataset covers features ranging from lot size, building type, and construction quality to basement conditions, garage details, and porch areas. The competition encourages practice in creative feature engineering and advanced regression techniques like random forest and gradient boosting.

## Evaluation

Submissions are evaluated on **Root-Mean-Squared-Error (RMSE)** between the **logarithm** of the predicted value and the **logarithm** of the observed sales price. Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally. For each Id in the test set, participants must predict the value of the SalePrice variable.

## Submission Format

The submission file should contain a header and have the following format:

```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
etc.
```

## Dataset Description

The Ames Housing dataset was compiled by Dean De Cock for use in data science education. It contains 79 explanatory variables describing residential homes in Ames, Iowa. The target variable is **SalePrice** (the property's sale price in dollars).

Features include: MSSubClass (building class), MSZoning (zoning classification), LotFrontage (linear feet of street connected to property), LotArea (lot size in sq ft), Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1/2 (proximity to roads/railroad), BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl, Exterior1st/2nd, MasVnrType, MasVnrArea, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1/2, BsmtFinSF1/2, BsmtUnfSF, TotalBsmtSF, Heating, HeatingQC, CentralAir, Electrical, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, Bedroom, Kitchen, KitchenQual, TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond, PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, PoolQC, Fence, MiscFeature, MiscVal, MoSold, YrSold, SaleType, and SaleCondition.

## Files

- **train.csv** - the training set with {len(train_set)} rows and 81 columns (Id + 79 features + SalePrice target).
- **test.csv** - the test set with {len(test_set)} rows and 80 columns (Id + 79 features, no SalePrice). Predict SalePrice for each Id.
- **sample_submission.csv** - a sample submission file with {len(test_set)} rows showing the correct format (Id, SalePrice).
- **data_description.txt** - full description of each column including variable names, descriptions, and categorical value mappings for all features.
"""
    with open(public / "description.txt", "w") as f:
        f.write(description)