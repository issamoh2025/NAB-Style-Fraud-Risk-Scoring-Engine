"""
evaluate.py
-----------
Evaluates the trained fraud detection model.
Shows confusion matrix, precision, recall, F1, and ROC-AUC.
Also prints plain-English interpretations - great for interviews.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path

from model import train_model, load_model
from preprocessing import load_and_preprocess
from feature_engineering import engineer_features


def evaluate_model():
    """
    Loads or trains the model, runs evaluation, and saves result charts.
    """

    # Try to load existing model, otherwise train
    try:
        model, feature_names = load_model()
        print("Loaded saved model.")
        # Load the enriched dataset
        import pandas as pd
        df = pd.read_csv("data/fraud_dataset_enriched.csv")
        X_train, X_test, y_train, y_test, _ = load_and_preprocess("data/fraud_dataset_enriched.csv")
    except FileNotFoundError:
        print("No saved model found, training now...")
        model, X_train, X_test, y_train, y_test, feature_names = train_model()

    # -------------------------------------------------------
    # Get predictions
    # -------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (legit correctly identified):  {tn:,}")
    print(f"  False Positives (legit flagged as fraud):     {fp:,}")
    print(f"  False Negatives (fraud missed):               {fn:,}")
    print(f"  True Positives (fraud correctly caught):      {tp:,}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"ROC-AUC Score:         {roc_auc:.4f}")
    print(f"Average Precision:     {avg_precision:.4f}")

    # -------------------------------------------------------
    # Plain English interpretation (good for interviews!)
    # -------------------------------------------------------
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print("PLAIN ENGLISH INTERPRETATION")
    print("="*60)

    print(f"""
ROC-AUC = {roc_auc:.4f}
  → The model can distinguish between fraud and legitimate transactions 
    with {roc_auc*100:.1f}% accuracy. A score of 0.5 is random guessing; 
    1.0 is perfect. A value above 0.85 is considered strong for fraud detection.

Precision = {precision:.4f}
  → When the model flags a transaction as fraud, it's correct 
    {precision*100:.1f}% of the time.
  → In fraud detection, low precision means many false alarms, which 
    frustrates customers and increases review workload for the operations team.

Recall = {recall:.4f}
  → The model catches {recall*100:.1f}% of all actual fraud cases.
  → In fraud detection, recall is critical. Missing fraud (false negatives) 
    means real losses for customers and the bank.

F1-Score = {f1:.4f}
  → The harmonic mean of precision and recall. Useful when the classes 
    are imbalanced (as they are here, since fraud is rare).

False Positives: {fp:,}
  → These are legitimate transactions incorrectly blocked or flagged. 
    Too many false positives lead to poor customer experience.

False Negatives: {fn:,}
  → These are fraudulent transactions the model missed. 
    This is the more costly error in a real banking environment.

In production, the model threshold (default 0.5) would be tuned based 
on the bank's risk appetite and acceptable false positive rate.
""")

    # -------------------------------------------------------
    # Save evaluation charts
    # -------------------------------------------------------
    Path("outputs").mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3)

    # --- Confusion Matrix ---
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    tick_marks = [0, 1]
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(["Legitimate", "Fraud"])
    ax1.set_yticklabels(["Legitimate", "Fraud"])
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)

    # --- ROC Curve ---
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random Guess")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax2.legend()

    # --- Precision-Recall Curve ---
    ax3 = fig.add_subplot(gs[2])
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    ax3.plot(rec_vals, prec_vals, color="darkorange", lw=2, label=f"AP = {avg_precision:.3f}")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("outputs/evaluation_charts.png", dpi=150)
    plt.close()
    print("\nEvaluation charts saved to outputs/evaluation_charts.png")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_precision, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }])
    metrics_df.to_csv("outputs/metrics.csv", index=False)
    print("Metrics saved to outputs/metrics.csv")

    return {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    evaluate_model()
