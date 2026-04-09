"""
preprocessing.py
----------------
Loads the fraud dataset, handles any cleaning, and splits into train/test sets.
Also handles class imbalance using class_weight parameter later in modelling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_and_preprocess(filepath="data/fraud_dataset.csv"):
    """
    Load the dataset, clean it, scale numerical features,
    and return train/test splits ready for modelling.
    """

    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop the raw_fraud_score — that was only used for label generation
    # We don't want to "leak" the label construction score into the model
    if "raw_fraud_score" in df.columns:
        df = df.drop(columns=["raw_fraud_score"])

    # Check for missing values (there shouldn't be any in synthetic data, but good practice)
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: missing values found, filling with 0")
        df = df.fillna(0)

    # -------------------------------------------------------
    # Define feature groups
    # -------------------------------------------------------

    # All binary features (already 0/1, no scaling needed)
    binary_features = [
        # Fraud green flags
        "gps_at_address", "registered_device", "registered_ip", "branch_device",
        # Scam green flags
        "paid_before", "close_to_beneficiary", "name_match", "demographic_match", "no_prior_fraud",
        # Fraud red flags
        "new_ip", "new_device", "ekyc_account", "recently_created_account",
        "multiple_device_logins", "fast_traveller", "password_reset",
        "number_change", "account_takeover_alert", "language_change",
        # Scam red flags
        "weak_name_match", "remote_access_session", "crypto_payment", "remitter_payment",
        "vulnerable_customer", "international_payment", "high_risk_bsb", "prior_victim",
        "suspicious_reference", "active_call", "gambling_activity",
        # Fraud concerns
        "login_irregularity", "sim_swap", "identity_theft",
        # Scam concerns
        "phishing", "investment_scam", "romance_scam", "business_email_compromise",
        "goods_services_scam", "remote_access_scam", "hi_mum_scam", "job_scam",
    ]

    # Numerical features that need scaling
    numerical_features = [
        "transaction_amount", "account_age_days", "tx_count_24h", "hour_of_day",
    ]

    # These will be added by feature_engineering.py
    engineered_features = [
        "device_change_count", "ip_change_frequency", "behavioural_risk_score",
        "trust_score", "account_takeover_pattern", "scam_pattern_score",
    ]

    # Build the final feature list from what's available in the dataframe
    available_features = binary_features + numerical_features
    for feat in engineered_features:
        if feat in df.columns:
            available_features.append(feat)

    # Target variable
    target = "is_fraud"

    X = df[available_features].copy()
    y = df[target].copy()

    # Scale the numerical features using StandardScaler
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Scale engineered features if they exist
    eng_to_scale = [f for f in engineered_features if f in X.columns]
    if eng_to_scale:
        X[eng_to_scale] = scaler.fit_transform(X[eng_to_scale])

    # Train/test split - stratify to keep fraud ratio consistent in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set:     {X_test.shape[0]} rows")
    print(f"Fraud rate in train: {y_train.mean():.2%}")
    print(f"Fraud rate in test:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, available_features


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    print(f"\nFeatures used ({len(features)} total):")
    for f in features:
        print(f"  - {f}")
