"""
model.py
--------
Trains a Random Forest classifier on the fraud dataset.
Outputs probability scores (0 to 1), saves the model to disk,
and prints feature importance so we can explain what the model learned.

We use class_weight='balanced' to handle the class imbalance
(fraud cases are much rarer than legitimate ones).
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Local imports
from dataset import generate_dataset
from feature_engineering import engineer_features
from preprocessing import load_and_preprocess


def train_model(use_random_forest=True):
    """
    Full training pipeline:
    1. Check if data exists, generate if not
    2. Engineer features
    3. Preprocess
    4. Train model
    5. Save model + feature list
    6. Print feature importances
    """

    # -------------------------------------------------------
    # Step 1: Generate data if needed
    # -------------------------------------------------------
    data_path = Path("data/fraud_dataset.csv")
    if not data_path.exists():
        print("Dataset not found, generating...")
        Path("data").mkdir(exist_ok=True)
        df = generate_dataset()
        df.to_csv(data_path, index=False)
        print(f"Dataset saved: {data_path}")
    else:
        print(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)

    # -------------------------------------------------------
    # Step 2: Engineer features
    # -------------------------------------------------------
    print("Engineering features...")
    df = engineer_features(df)

    # Save the enriched dataset so preprocessing can use it
    df.to_csv("data/fraud_dataset_enriched.csv", index=False)

    # -------------------------------------------------------
    # Step 3: Preprocess
    # -------------------------------------------------------
    print("Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(
        "data/fraud_dataset_enriched.csv"
    )

    # -------------------------------------------------------
    # Step 4: Train the model
    # -------------------------------------------------------
    if use_random_forest:
        print("\nTraining Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,       # 100 trees
            max_depth=10,           # Limit depth to prevent overfitting
            min_samples_leaf=20,    # Each leaf needs at least 20 samples
            class_weight="balanced", # Handle class imbalance
            random_state=42,
            n_jobs=-1,              # Use all CPU cores
        )
    else:
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )

    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # -------------------------------------------------------
    # Step 5: Save model and feature list
    # -------------------------------------------------------
    Path("models").mkdir(exist_ok=True)

    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Model saved to models/fraud_model.pkl")
    print("Feature names saved to models/feature_names.pkl")

    # -------------------------------------------------------
    # Step 6: Feature importance
    # -------------------------------------------------------
    if use_random_forest:
        importances = model.feature_importances_
    else:
        # For logistic regression, use absolute coefficient values
        importances = np.abs(model.coef_[0])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n=== Top 20 Most Important Features ===")
    print(importance_df.head(20).to_string(index=False))

    # Save feature importance chart
    plt.figure(figsize=(10, 8))
    top20 = importance_df.head(20)
    plt.barh(top20["feature"][::-1], top20["importance"][::-1], color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Top 20 Feature Importances - Fraud Detection Model")
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance chart saved to outputs/feature_importance.png")

    # Save importance CSV for the dashboard
    importance_df.to_csv("outputs/feature_importance.csv", index=False)

    return model, X_train, X_test, y_train, y_test, feature_names


def load_model():
    """
    Loads the saved model and feature names from disk.
    Used by the API and dashboard at prediction time.
    """
    with open("models/fraud_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, feature_names


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test, feature_names = train_model(use_random_forest=True)

    # Quick sanity check on predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\nSample predictions (first 5):")
    for i in range(5):
        print(f"  True: {y_test.iloc[i]} | Predicted probability: {y_prob[i]:.4f}")

    print(f"\nMean predicted probability on fraud cases: {y_prob[y_test == 1].mean():.4f}")
    print(f"Mean predicted probability on legit cases: {y_prob[y_test == 0].mean():.4f}")
