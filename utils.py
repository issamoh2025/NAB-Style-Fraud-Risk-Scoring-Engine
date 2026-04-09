"""
utils.py
--------
Helper functions used across the project.
Handles risk level classification, recommendation mapping,
explanation formatting, and combining ML + rule-based scores.
"""

from typing import Dict, Any, List
import numpy as np


# -------------------------------------------------------
# Risk level thresholds
# These mirror common industry thresholds for fraud scoring
# -------------------------------------------------------
RISK_THRESHOLDS = {
    "LOW":    (0.00, 0.25),
    "MEDIUM": (0.25, 0.50),
    "HIGH":   (0.50, 0.75),
    "CRITICAL": (0.75, 1.00),
}


def classify_risk_level(score: float) -> str:
    """
    Converts a fraud probability score (0-1) to a categorical risk level.

    Args:
        score: float between 0 and 1

    Returns:
        str: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
    """
    if score < 0.25:
        return "LOW"
    elif score < 0.50:
        return "MEDIUM"
    elif score < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"


def map_recommendation(score: float) -> str:
    """
    Maps a combined fraud risk score (0-1) to an action recommendation.

    In a real banking system, these thresholds would be calibrated
    based on the bank's risk appetite, operational capacity, and
    the cost of false positives vs false negatives.

    Args:
        score: float between 0 and 1

    Returns:
        str: "APPROVE", "REVIEW", "ESCALATE", or "BLOCK"
    """
    if score >= 0.70:
        return "BLOCK"
    elif score >= 0.45:
        return "ESCALATE"
    elif score >= 0.25:
        return "REVIEW"
    else:
        return "APPROVE"


def combine_scores(ml_score: float, rule_score: float,
                   ml_weight: float = 0.6, rule_weight: float = 0.4) -> float:
    """
    Combines the machine learning probability score and the
    rule-based score into a single final risk score.

    The ML model gets slightly more weight because it uses
    all features together, but the rules are operationally
    important and shouldn't be ignored.

    Args:
        ml_score:    probability from the ML model (0-1)
        rule_score:  score from the rules engine (0-1)
        ml_weight:   weight for ML score (default 0.6)
        rule_weight: weight for rule score (default 0.4)

    Returns:
        float: combined score clipped to [0, 1]
    """
    combined = (ml_score * ml_weight) + (rule_score * rule_weight)
    return round(float(np.clip(combined, 0.0, 1.0)), 4)


def get_top_risk_factors(feature_names: List[str], feature_values: List[float],
                          model_importances: np.ndarray, top_n: int = 5) -> List[str]:
    """
    Returns the top N features that contributed most to the fraud prediction
    for this specific transaction.

    We multiply feature importance (from the model) by the actual value
    for this transaction, so features that are important AND present rank higher.

    Args:
        feature_names:       list of feature names
        feature_values:      list of actual values for this transaction
        model_importances:   feature importance array from the trained model
        top_n:               how many top factors to return

    Returns:
        list of feature names
    """
    # Multiply importance by absolute value to get "contribution"
    contributions = model_importances * np.abs(feature_values)

    # Sort by contribution descending and return top N names
    top_indices = np.argsort(contributions)[::-1][:top_n]
    return [feature_names[i] for i in top_indices if contributions[i] > 0]


def format_feature_name(raw_name: str) -> str:
    """
    Converts a snake_case feature name to a human-readable label.

    Example:
        "account_takeover_alert" -> "Account Takeover Alert"
    """
    return raw_name.replace("_", " ").title()


def format_risk_badge(risk_level: str) -> str:
    """
    Returns a formatted string for display with emoji indicators.
    Used in the Streamlit dashboard and API response.
    """
    badges = {
        "LOW":      "🟢 LOW",
        "MEDIUM":   "🟡 MEDIUM",
        "HIGH":     "🟠 HIGH",
        "CRITICAL": "🔴 CRITICAL",
    }
    return badges.get(risk_level, risk_level)


def format_recommendation_badge(recommendation: str) -> str:
    """
    Returns a formatted recommendation string with emoji indicators.
    """
    badges = {
        "APPROVE":  "✅ APPROVE",
        "REVIEW":   "🔍 REVIEW",
        "ESCALATE": "⚠️ ESCALATE",
        "BLOCK":    "🚫 BLOCK",
    }
    return badges.get(recommendation, recommendation)


def build_full_prediction_response(
    ml_score: float,
    rule_result: Dict[str, Any],
    top_risk_factors: List[str],
) -> Dict[str, Any]:
    """
    Builds the final prediction response dictionary that is
    returned by the FastAPI endpoint and displayed in the dashboard.

    Args:
        ml_score:         probability from the ML model
        rule_result:      output from the rules engine
        top_risk_factors: list of most important feature names

    Returns:
        Full prediction response dictionary
    """
    rule_score = rule_result["rule_score"]
    combined_score = combine_scores(ml_score, rule_score)
    risk_level = classify_risk_level(combined_score)
    recommendation = map_recommendation(combined_score)

    return {
        "ml_score": round(ml_score, 4),
        "rule_score": rule_score,
        "fraud_score": combined_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "top_risk_factors": [format_feature_name(f) for f in top_risk_factors],
        "triggered_rules": rule_result["triggered_rules"],
        "rule_explanations": rule_result["rule_explanations"],
    }


if __name__ == "__main__":
    # Quick test of utility functions
    print("=== Utils Test ===")
    print(classify_risk_level(0.10))   # LOW
    print(classify_risk_level(0.35))   # MEDIUM
    print(classify_risk_level(0.65))   # HIGH
    print(classify_risk_level(0.90))   # CRITICAL

    print(map_recommendation(0.10))    # APPROVE
    print(map_recommendation(0.30))    # REVIEW
    print(map_recommendation(0.55))    # ESCALATE
    print(map_recommendation(0.80))    # BLOCK

    combined = combine_scores(ml_score=0.85, rule_score=0.70)
    print(f"Combined score: {combined}")  # Should be ~0.79

    print(format_feature_name("account_takeover_alert"))  # Account Takeover Alert
    print(format_risk_badge("HIGH"))                       # 🟠 HIGH
    print(format_recommendation_badge("BLOCK"))            # 🚫 BLOCK
