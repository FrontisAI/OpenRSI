"""Sample prepare implementation used as an LLM prompt reference."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path):
    # Prepare the Aerial Cactus Identification dataset
    
    # This function:
    # 1. Loads the training data with labels
    # 2. Splits it into 80% train, 20% test
    # 3. Copies image files to appropriate directories
    # 4. Creates sample_submission.csv and test_answer.csv
    # 5. Create an updated description.txt
    
    # Load the training labels
    train_labels_path = raw / "train.csv"
    train_labels = pd.read_csv(train_labels_path)
    
    # Verify expected columns
    assert 'id' in train_labels.columns, "Missing 'id' column in train.csv"
    assert 'has_cactus' in train_labels.columns, "Missing 'has_cactus' column in train.csv"
    
    # Perform stratified train-test split (80-20) with deterministic behavior
    train_set, test_set = train_test_split(
        train_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=train_labels['has_cactus']  # Stratify to maintain label distribution
    )
    
    # Validate the split
    assert len(train_set) + len(test_set) == len(train_labels), "Split validation failed"
    assert abs(len(test_set) / len(train_labels) - 0.2) < 0.01, "Split ratio validation failed"
    
    # Create public directories for images
    public_train_dir = public / "train"
    public_test_dir = public / "test"
    public_train_dir.mkdir(parents=True, exist_ok=True)
    public_test_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine source directory for images
    raw_train_dir = raw / "train"
    if not raw_train_dir.exists():
        raw_train_dir = raw
    
    # Copy train images
    for idx, row in train_set.iterrows():
        img_id = row['id']
        src_path = raw_train_dir / img_id
        dst_path = public_train_dir / img_id
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Copy test images
    for idx, row in test_set.iterrows():
        img_id = row['id']
        src_path = raw_train_dir / img_id
        dst_path = public_test_dir / img_id
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Save train.csv (with labels)
    train_set_output = train_set.copy()
    train_set_output.to_csv(public / "train.csv", index=False)
    
    # Create test.csv (without labels - only IDs)
    test_set_public = test_set[['id']].copy()
    test_set_public.to_csv(public / "test.csv", index=False)
    
    # Create test_answer.csv (private - with labels as probabilities)
    test_answer = test_set[['id', 'has_cactus']].copy()
    # The evaluation uses ROC curve, so we keep the binary labels as probabilities (0.0 or 1.0)
    test_answer = test_answer.rename(columns={'has_cactus': 'has_cactus'})
    test_answer['has_cactus'] = test_answer['has_cactus'].astype(float)
    test_answer.to_csv(private / "test_answer.csv", index=False)
    
    # Create sample_submission.csv (baseline prediction of 0.5 for all)
    sample_submission = pd.DataFrame({
        'id': test_set['id'].values,
        'has_cactus': np.full(len(test_set), 0.5)
    })
    sample_submission.to_csv(public / "sample_submission.csv", index=False)
    
    # Validate that test_answer.csv and sample_submission.csv have matching structure
    test_answer_check = pd.read_csv(private / "test_answer.csv")
    sample_submission_check = pd.read_csv(public / "sample_submission.csv")
    
    assert list(test_answer_check.columns) == list(sample_submission_check.columns), \
        "Column mismatch between test_answer.csv and sample_submission.csv"
    assert len(test_answer_check) == len(sample_submission_check), \
        "Row count mismatch between test_answer.csv and sample_submission.csv"
    assert set(test_answer_check['id']) == set(sample_submission_check['id']), \
        "ID mismatch between test_answer.csv and sample_submission.csv"
        
    # Update description.txt
    description = """
(the new description)
"""
    with open(public / "description.txt", "w") as f:
        f.write(description)
