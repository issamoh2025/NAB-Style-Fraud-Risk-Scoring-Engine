"""
dataset.py
----------
Generates a synthetic fraud detection dataset with 50,000 rows.

KEY DESIGN DECISIONS (updated):
- n2n_payment added as a Scam Green Flag (name-to-name payment = trust signal)
- Fraud Concerns (PPF/PPO/IDTO) and Scam Concerns (PHI etc.) do NOT contribute
  to the fraud label score — they are analyst annotations only.
  Only red flags and green flags drive the probability.
- close_to_beneficiary renamed to close_to_bsb in comments (key unchanged for
  backwards compatibility with saved models)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

N = 50_000  # Number of transactions to generate

def generate_dataset():
    # -------------------------------------------------------
    # Base probabilities for each feature (how often it's True)
    # -------------------------------------------------------

    data = {}

    # --- Fraud Green Flags (reduce fraud probability) ---
    data["gps_at_address"]    = np.random.binomial(1, 0.65, N)
    data["registered_device"] = np.random.binomial(1, 0.70, N)
    data["registered_ip"]     = np.random.binomial(1, 0.68, N)
    data["branch_device"]     = np.random.binomial(1, 0.20, N)

    # --- Scam Green Flags (reduce scam probability) ---
    data["paid_before"]           = np.random.binomial(1, 0.55, N)
    data["close_to_beneficiary"]  = np.random.binomial(1, 0.35, N)  # also known as Close to BSB
    data["name_match"]            = np.random.binomial(1, 0.60, N)
    data["demographic_match"]     = np.random.binomial(1, 0.50, N)
    data["no_prior_fraud"]        = np.random.binomial(1, 0.80, N)
    data["n2n_payment"]           = np.random.binomial(1, 0.30, N)  # NEW: Name-to-Name payment

    # --- Fraud Red Flags (increase fraud probability) ---
    data["new_ip"]                   = np.random.binomial(1, 0.25, N)
    data["new_device"]               = np.random.binomial(1, 0.20, N)
    data["ekyc_account"]             = np.random.binomial(1, 0.15, N)
    data["recently_created_account"] = np.random.binomial(1, 0.18, N)
    data["multiple_device_logins"]   = np.random.binomial(1, 0.12, N)
    data["fast_traveller"]           = np.random.binomial(1, 0.08, N)
    data["password_reset"]           = np.random.binomial(1, 0.14, N)
    data["number_change"]            = np.random.binomial(1, 0.10, N)
    data["account_takeover_alert"]   = np.random.binomial(1, 0.07, N)
    data["language_change"]          = np.random.binomial(1, 0.09, N)

    # --- Scam Red Flags (increase scam probability) ---
    data["weak_name_match"]        = np.random.binomial(1, 0.18, N)
    data["remote_access_session"]  = np.random.binomial(1, 0.08, N)
    data["crypto_payment"]         = np.random.binomial(1, 0.12, N)
    data["remitter_payment"]       = np.random.binomial(1, 0.14, N)
    data["vulnerable_customer"]    = np.random.binomial(1, 0.12, N)
    data["international_payment"]  = np.random.binomial(1, 0.22, N)
    data["high_risk_bsb"]          = np.random.binomial(1, 0.10, N)
    data["prior_victim"]           = np.random.binomial(1, 0.06, N)
    data["suspicious_reference"]   = np.random.binomial(1, 0.09, N)
    data["active_call"]            = np.random.binomial(1, 0.07, N)
    data["gambling_activity"]      = np.random.binomial(1, 0.10, N)

    # --- Fraud Concerns (PPF / PPO / IDTO) ---
    # These are ANALYST ANNOTATIONS — they do NOT affect the fraud label.
    # PPF = Prohibited Pattern Flag (login irregularity)
    # PPO = Portable Number Order (SIM swap)
    # IDTO = Identity Takeover
    data["login_irregularity"] = np.random.binomial(1, 0.15, N)  # PPF
    data["sim_swap"]           = np.random.binomial(1, 0.05, N)  # PPO
    data["identity_theft"]     = np.random.binomial(1, 0.06, N)  # IDTO

    # --- Scam Concerns ---
    # These are ANALYST ANNOTATIONS — they do NOT affect the fraud label.
    # PHI = Phishing concern
    data["phishing"]                  = np.random.binomial(1, 0.08, N)  # PHI
    data["investment_scam"]           = np.random.binomial(1, 0.07, N)
    data["romance_scam"]              = np.random.binomial(1, 0.05, N)
    data["business_email_compromise"] = np.random.binomial(1, 0.06, N)
    data["goods_services_scam"]       = np.random.binomial(1, 0.07, N)
    data["remote_access_scam"]        = np.random.binomial(1, 0.05, N)
    data["hi_mum_scam"]               = np.random.binomial(1, 0.04, N)
    data["job_scam"]                  = np.random.binomial(1, 0.05, N)

    # --- Numerical features ---
    data["transaction_amount"] = np.random.exponential(scale=500, size=N).clip(1, 50000).round(2)

    # Account opened date stored as days since today (supports 60+ year old accounts)
    # Range: 0 (today) to 36500 (100 years ago)
    data["account_age_days"] = np.random.randint(0, 36500, N)

    data["tx_count_24h"] = np.random.poisson(lam=3, size=N).clip(0, 30)
    data["hour_of_day"]  = np.random.randint(0, 24, N)

    df = pd.DataFrame(data)

    # -------------------------------------------------------
    # Label generation — ONLY red flags and green flags contribute.
    # Concerns are analyst annotations and are excluded here.
    # -------------------------------------------------------

    fraud_score = np.zeros(N)

    # Fraud red flags → raise risk
    fraud_score += df["new_ip"]                   * 0.15
    fraud_score += df["new_device"]               * 0.18
    fraud_score += df["ekyc_account"]             * 0.12
    fraud_score += df["recently_created_account"] * 0.10
    fraud_score += df["multiple_device_logins"]   * 0.12
    fraud_score += df["fast_traveller"]            * 0.10
    fraud_score += df["password_reset"]            * 0.12
    fraud_score += df["number_change"]             * 0.10
    fraud_score += df["account_takeover_alert"]    * 0.25
    fraud_score += df["language_change"]           * 0.08

    # Scam red flags → raise risk
    fraud_score += df["weak_name_match"]        * 0.08
    fraud_score += df["remote_access_session"]  * 0.18
    fraud_score += df["crypto_payment"]         * 0.12
    fraud_score += df["remitter_payment"]       * 0.10
    fraud_score += df["vulnerable_customer"]    * 0.12
    fraud_score += df["international_payment"]  * 0.08
    fraud_score += df["high_risk_bsb"]          * 0.12
    fraud_score += df["prior_victim"]           * 0.18
    fraud_score += df["suspicious_reference"]   * 0.12
    fraud_score += df["active_call"]            * 0.15
    fraud_score += df["gambling_activity"]      * 0.08

    # NOTE: Fraud concerns (PPF/PPO/IDTO) and scam concerns (PHI etc.)
    # do NOT contribute to fraud_score — they are analyst annotations only.

    # Green flags → reduce risk
    fraud_score -= df["gps_at_address"]    * 0.12
    fraud_score -= df["registered_device"] * 0.15
    fraud_score -= df["registered_ip"]     * 0.12
    fraud_score -= df["branch_device"]     * 0.10
    fraud_score -= df["paid_before"]       * 0.10
    fraud_score -= df["name_match"]        * 0.08
    fraud_score -= df["no_prior_fraud"]    * 0.12
    fraud_score -= df["n2n_payment"]       * 0.10  # N2N payment reduces scam risk

    # High-risk combinations (red flags only)
    ato_pattern      = df["new_device"] & df["new_ip"] & df["password_reset"]
    remote_scam      = df["remote_access_session"] & df["active_call"] & df["international_payment"]
    crypto_scam      = df["crypto_payment"] & df["suspicious_reference"] & df["prior_victim"]
    ato_alert_combo  = df["number_change"] & df["account_takeover_alert"]
    bsb_pattern      = df["high_risk_bsb"] & df["remitter_payment"] & df["weak_name_match"]
    travel_pattern   = df["fast_traveller"] & df["multiple_device_logins"]

    fraud_score += ato_pattern     * 0.30
    fraud_score += remote_scam     * 0.35
    fraud_score += crypto_scam     * 0.35
    fraud_score += ato_alert_combo * 0.20
    fraud_score += bsb_pattern     * 0.25
    fraud_score += travel_pattern  * 0.20

    fraud_score += (df["transaction_amount"] > 5000).astype(int) * 0.08
    fraud_score += (df["account_age_days"] < 30).astype(int)     * 0.12

    noise = np.random.normal(0, 0.05, N)
    fraud_score += noise

    fraud_score = fraud_score.clip(0, 1)

    random_draw = np.random.uniform(0, 1, N)
    df["is_fraud"] = (random_draw < fraud_score * 0.35).astype(int)
    df["raw_fraud_score"] = fraud_score

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Total fraud cases: {df['is_fraud'].sum()}")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    output_path = Path("data")
    output_path.mkdir(exist_ok=True)
    df.to_csv("data/fraud_dataset.csv", index=False)
    print("Dataset saved to data/fraud_dataset.csv")
