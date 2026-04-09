"""
feature_engineering.py
-----------------------
Creates derived features from raw binary and numerical features.

KEY CHANGES:
- n2n_payment added to green flags / trust_score
- Fraud concerns (PPF=login_irregularity, PPO=sim_swap, IDTO=identity_theft)
  removed from behavioural_risk_score — they are analyst annotations only.
- Scam concerns (PHI=phishing, investment_scam, romance_scam, bec,
  goods_services_scam, remote_access_scam, hi_mum_scam, job_scam)
  removed from scam_pattern_score — they are analyst annotations only.
- account_takeover_pattern no longer includes sim_swap (concern flag).
"""

import pandas as pd
import numpy as np


# -------------------------------------------------------
# Feature group constants — single source of truth used
# by both batch (DataFrame) and real-time (dict) functions.
# -------------------------------------------------------

# Red flags that drive behavioural_risk_score (concerns excluded)
FRAUD_RED_FLAGS = [
    "new_ip", "new_device", "ekyc_account", "recently_created_account",
    "multiple_device_logins", "fast_traveller", "password_reset",
    "number_change", "account_takeover_alert", "language_change",
]

# Green flags that drive trust_score (includes new n2n_payment)
GREEN_FLAGS = [
    "gps_at_address", "registered_device", "registered_ip", "branch_device",
    "paid_before", "close_to_beneficiary", "name_match", "demographic_match",
    "no_prior_fraud", "n2n_payment",
]

# Scam red flags that drive scam_pattern_score (concerns excluded)
SCAM_RED_FLAGS = [
    "weak_name_match", "remote_access_session", "crypto_payment", "remitter_payment",
    "vulnerable_customer", "international_payment", "high_risk_bsb", "prior_victim",
    "suspicious_reference", "active_call", "gambling_activity",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw dataframe and adds derived composite features.
    Returns the dataframe with new columns added.
    """

    df = df.copy()

    # 1. device_change_count — how many device-change signals are present
    df["device_change_count"] = (
        df["new_device"].astype(int)
        + df["multiple_device_logins"].astype(int)
        + df["fast_traveller"].astype(int)
    )

    # 2. ip_change_frequency — signals pointing to an unusual IP
    df["ip_change_frequency"] = (
        df["new_ip"].astype(int)
        + df["remote_access_session"].astype(int)
        + df["international_payment"].astype(int)
    )

    # 3. behavioural_risk_score — sum of fraud RED flags only.
    #    Concerns (PPF / PPO / IDTO) are analyst annotations and excluded.
    df["behavioural_risk_score"] = df[FRAUD_RED_FLAGS].sum(axis=1)

    # 4. trust_score — sum of ALL green flags including n2n_payment.
    #    Higher trust_score = lower fraud likelihood.
    df["trust_score"] = df[GREEN_FLAGS].sum(axis=1)

    # 5. account_takeover_pattern — red flags only, no concern flags.
    df["account_takeover_pattern"] = (
        df["new_device"].astype(int)
        + df["new_ip"].astype(int)
        + df["password_reset"].astype(int)
        + df["account_takeover_alert"].astype(int)
        + df["number_change"].astype(int)
    )

    # 6. scam_pattern_score — sum of scam RED flags only.
    #    Scam concerns (PHI / investment_scam / romance_scam etc.) are excluded.
    df["scam_pattern_score"] = df[SCAM_RED_FLAGS].sum(axis=1)

    # 7. net_risk_score — quick overall direction of risk
    df["net_risk_score"] = (
        df["behavioural_risk_score"] + df["scam_pattern_score"] - df["trust_score"]
    )

    print("Engineered features added:")
    new_cols = [
        "device_change_count", "ip_change_frequency", "behavioural_risk_score",
        "trust_score", "account_takeover_pattern", "scam_pattern_score", "net_risk_score"
    ]
    for col in new_cols:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

    return df


def engineer_single_transaction(input_dict: dict) -> dict:
    """
    Adds the same engineered features to a single transaction dict.
    Used at prediction time by the API and dashboard.
    """

    def _get(key):
        return int(input_dict.get(key, 0))

    input_dict["device_change_count"] = (
        _get("new_device") + _get("multiple_device_logins") + _get("fast_traveller")
    )

    input_dict["ip_change_frequency"] = (
        _get("new_ip") + _get("remote_access_session") + _get("international_payment")
    )

    # behavioural_risk_score: fraud red flags only (no concerns)
    input_dict["behavioural_risk_score"] = sum(_get(f) for f in FRAUD_RED_FLAGS)

    # trust_score: all green flags including n2n_payment
    input_dict["trust_score"] = sum(_get(f) for f in GREEN_FLAGS)

    # account_takeover_pattern: red flags only
    input_dict["account_takeover_pattern"] = (
        _get("new_device") + _get("new_ip") + _get("password_reset")
        + _get("account_takeover_alert") + _get("number_change")
    )

    # scam_pattern_score: scam red flags only (no concerns)
    input_dict["scam_pattern_score"] = sum(_get(f) for f in SCAM_RED_FLAGS)

    input_dict["net_risk_score"] = (
        input_dict["behavioural_risk_score"]
        + input_dict["scam_pattern_score"]
        - input_dict["trust_score"]
    )

    return input_dict


if __name__ == "__main__":
    from dataset import generate_dataset

    df = generate_dataset()
    df_engineered = engineer_features(df)
    print(f"\nFinal dataframe shape: {df_engineered.shape}")
