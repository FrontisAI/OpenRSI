"""Sample solution implementation used as an LLM prompt reference."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import re

warnings.filterwarnings('ignore')

def load_data(base_path):
    """Load training and test datasets."""
    train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    test = pd.read_csv(os.path.join(base_path, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
    return train, test, sample_sub

def extract_title(name):
    """Extract title from passenger name."""
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        # Normalize rare titles
        if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
            return 'Rare'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title == 'Mme':
            return 'Mrs'
        return title
    return 'Unknown'

def extract_cabin_deck(cabin):
    """Extract deck letter from cabin."""
    if pd.isna(cabin):
        return 'Unknown'
    cabins = str(cabin).split()
    if len(cabins) > 0:
        return cabins[0][0]
    return 'Unknown'

def extract_ticket_prefix(ticket):
    """Extract ticket prefix."""
    if pd.isna(ticket):
        return 'Unknown'
    ticket = str(ticket)
    if ' ' in ticket:
        return ticket.split()[0].replace('.', '').replace('/', '').upper()
    return 'None'

def engineer_features(df):
    """Create new features from existing data."""
    df = df.copy()
    
    # Title from name
    df['Title'] = df['Name'].apply(extract_title)
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Cabin deck
    df['CabinDeck'] = df['Cabin'].apply(extract_cabin_deck)
    
    # Has cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Ticket prefix
    df['TicketPrefix'] = df['Ticket'].apply(extract_ticket_prefix)
    
    # Age bins (will be filled after imputation)
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Sex binary
    df['SexEncoded'] = (df['Sex'] == 'male').astype(int)
    
    # Pclass as string for encoding
    df['PclassStr'] = df['Pclass'].astype(str)
    
    return df

def impute_missing_values(train, test):
    """Impute missing values in both datasets."""
    train = train.copy()
    test = test.copy()
    
    # Impute Age using median by Title and Pclass
    age_medians = train.groupby(['Title', 'Pclass'])['Age'].median()
    
    def fill_age(row):
        if pd.isna(row['Age']):
            key = (row['Title'], row['Pclass'])
            if key in age_medians:
                return age_medians[key]
            return train['Age'].median()
        return row['Age']
    
    train['Age'] = train.apply(fill_age, axis=1)
    test['Age'] = test.apply(fill_age, axis=1)
    
    # Impute Fare (test set has missing value)
    fare_median = train.groupby(['Pclass', 'Embarked'])['Fare'].median()
    
    def fill_fare(row, df, is_train=True):
        if pd.isna(row['Fare']):
            key = (row['Pclass'], row['Embarked'])
            if key in fare_median:
                return fare_median[key]
            return train['Fare'].median()
        return row['Fare']
    
    test['Fare'] = test.apply(lambda r: fill_fare(r, test, False), axis=1)
    
    # Impute Embarked
    embarked_mode = train['Embarked'].mode()[0]
    train['Embarked'] = train['Embarked'].fillna(embarked_mode)
    test['Embarked'] = test['Embarked'].fillna(embarked_mode)
    
    return train, test

def encode_features(train, test):
    """Encode categorical features."""
    train = train.copy()
    test = test.copy()
    
    # Label encode categorical features
    categorical_cols = ['Sex', 'Embarked', 'Title', 'CabinDeck', 'TicketPrefix', 'AgeBand', 'PclassStr']
    
    for col in categorical_cols:
        if col in train.columns:
            le = LabelEncoder()
            train_vals = train[col].astype(str).fillna('Unknown')
            test_vals = test[col].astype(str).fillna('Unknown')
            combined = pd.concat([train_vals, test_vals], axis=0)
            le.fit(combined)
            train[col + '_Encoded'] = le.transform(train_vals)
            test[col + '_Encoded'] = le.transform(test_vals)
    
    return train, test

def select_features(df):
    """Select features for model training."""
    feature_cols = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'FamilySize', 'IsAlone', 'HasCabin', 'SexEncoded', 'FarePerPerson',
        'Embarked_Encoded', 'Title_Encoded', 'CabinDeck_Encoded', 
        'TicketPrefix_Encoded', 'AgeBand_Encoded', 'PclassStr_Encoded'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    return df[available_features]

def train_model(X_train, y_train):
    """Train ensemble model with cross-validation."""
    
    # Define base models
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Fit on full training data
    ensemble.fit(X_train, y_train)
    
    return ensemble

def main():
    # Get paths
    base_path = os.environ.get('PUBLIC_ROOT', '')
    
    # Load data
    train_df, test_df, sample_sub = load_data(base_path)
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Store passenger IDs
    train_ids = train_df['PassengerId'].values
    test_ids = test_df['PassengerId'].values
    
    # Store target
    y_train = train_df['Survived'].values
    
    # Feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Impute missing values
    train_df, test_df = impute_missing_values(train_df, test_df)
    
    # Re-apply age band after imputation
    train_df['AgeBand'] = pd.cut(train_df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    test_df['AgeBand'] = pd.cut(test_df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
    
    # Encode features
    train_df, test_df = encode_features(train_df, test_df)
    
    # Select features
    X_train = select_features(train_df)
    X_test = select_features(test_df)
    
    print(f"Features used: {X_train.columns.tolist()}")
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions.astype(int)
    })
    
    # Ensure format matches sample submission
    submission = submission[['PassengerId', 'Survived']]
    submission['PassengerId'] = submission['PassengerId'].astype(int)
    submission['Survived'] = submission['Survived'].astype(int)
    
    # Save submission
    save_path = os.environ.get('SUBMISSION_PATH', 'submission.csv')
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction distribution: {submission['Survived'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
