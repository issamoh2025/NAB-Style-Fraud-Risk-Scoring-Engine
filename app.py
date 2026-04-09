"""
app.py (FastAPI)
----------------
REST API backend for the fraud detection system.

Run with:
    uvicorn app:app --reload --port 8000

KEY CHANGES:
- n2n_payment added as a Scam Green Flag input field
- account_age_days now accepts up to 36500 (100 years) for long-standing accounts
- Fraud Concerns (PPF/PPO/IDTO) and Scam Concerns (PHI etc.) are present as
  input fields but documented as analyst annotations — they do NOT affect the
  ML score or rule_score. They appear in analyst_notes in the response.
- Response includes analyst_notes and analyst_explanations alongside triggered_rules
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import load_model
from rules_engine import run_rules_engine
from feature_engineering import engineer_single_transaction
from utils import (
    combine_scores, classify_risk_level, map_recommendation,
    get_top_risk_factors, build_full_prediction_response,
)


app = FastAPI(
    title="Fraud Prevention and Detection Risk Scoring API",
    description=(
        "Predicts the probability of fraud or scam for a transaction. "
        "Returns an ML-based score, rule-based score, top risk factors, "
        "and a plain-English recommendation. "
        "Concerns (PPF/PPO/IDTO/PHI etc.) are analyst annotations only — "
        "they appear in analyst_notes and do NOT affect any score."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_names = None


@app.on_event("startup")
def load_model_on_startup():
    global model, feature_names
    try:
        model, feature_names = load_model()
        print(f"✅ Model loaded: {len(feature_names)} features")
    except FileNotFoundError:
        print("⚠️  No saved model found. Run model.py first to train the model.")


# -------------------------------------------------------
# Request schema
# -------------------------------------------------------
class TransactionInput(BaseModel):

    # --- Fraud Green Flags ---
    gps_at_address:    int = Field(0, ge=0, le=1, description="GPS location matches registered address")
    registered_device: int = Field(0, ge=0, le=1, description="Device is registered to the account")
    registered_ip:     int = Field(0, ge=0, le=1, description="IP address is known/registered")
    branch_device:     int = Field(0, ge=0, le=1, description="Transaction from a bank branch device")

    # --- Scam Green Flags ---
    paid_before:           int = Field(0, ge=0, le=1, description="Customer has paid this beneficiary before")
    close_to_beneficiary:  int = Field(0, ge=0, le=1, description="Customer is close to beneficiary / Close to BSB")
    name_match:            int = Field(0, ge=0, le=1, description="Payee name matches account name")
    demographic_match:     int = Field(0, ge=0, le=1, description="Demographic details match")
    no_prior_fraud:        int = Field(0, ge=0, le=1, description="No prior fraud history on this account")
    n2n_payment:           int = Field(0, ge=0, le=1, description="Name-to-Name payment — trust signal")

    # --- Fraud Red Flags ---
    new_ip:                   int = Field(0, ge=0, le=1)
    new_device:               int = Field(0, ge=0, le=1)
    ekyc_account:             int = Field(0, ge=0, le=1)
    recently_created_account: int = Field(0, ge=0, le=1)
    multiple_device_logins:   int = Field(0, ge=0, le=1)
    fast_traveller:           int = Field(0, ge=0, le=1)
    password_reset:           int = Field(0, ge=0, le=1)
    number_change:            int = Field(0, ge=0, le=1)
    account_takeover_alert:   int = Field(0, ge=0, le=1)
    language_change:          int = Field(0, ge=0, le=1)

    # --- Scam Red Flags ---
    weak_name_match:       int = Field(0, ge=0, le=1)
    remote_access_session: int = Field(0, ge=0, le=1)
    crypto_payment:        int = Field(0, ge=0, le=1)
    remitter_payment:      int = Field(0, ge=0, le=1)
    vulnerable_customer:   int = Field(0, ge=0, le=1)
    international_payment: int = Field(0, ge=0, le=1)
    high_risk_bsb:         int = Field(0, ge=0, le=1)
    prior_victim:          int = Field(0, ge=0, le=1)
    suspicious_reference:  int = Field(0, ge=0, le=1)
    active_call:           int = Field(0, ge=0, le=1)
    gambling_activity:     int = Field(0, ge=0, le=1)

    # --- Fraud Concerns (PPF / PPO / IDTO) ---
    # ANALYST ANNOTATIONS — present for completeness but do NOT affect any score.
    login_irregularity: int = Field(0, ge=0, le=1, description="PPF — Login irregularity (analyst annotation)")
    sim_swap:           int = Field(0, ge=0, le=1, description="PPO — SIM swap / Portable Number Order (analyst annotation)")
    identity_theft:     int = Field(0, ge=0, le=1, description="IDTO — Identity takeover (analyst annotation)")

    # --- Scam Concerns ---
    # ANALYST ANNOTATIONS — do NOT affect any score.
    phishing:                  int = Field(0, ge=0, le=1, description="PHI — Phishing concern (analyst annotation)")
    investment_scam:           int = Field(0, ge=0, le=1, description="Analyst annotation")
    romance_scam:              int = Field(0, ge=0, le=1, description="Analyst annotation")
    business_email_compromise: int = Field(0, ge=0, le=1, description="Analyst annotation")
    goods_services_scam:       int = Field(0, ge=0, le=1, description="Analyst annotation")
    remote_access_scam:        int = Field(0, ge=0, le=1, description="Analyst annotation")
    hi_mum_scam:               int = Field(0, ge=0, le=1, description="Analyst annotation")
    job_scam:                  int = Field(0, ge=0, le=1, description="Analyst annotation")

    # --- Numerical Features ---
    transaction_amount: float = Field(100.0, ge=0, description="Transaction amount in dollars")
    account_age_days:   int   = Field(365, ge=0, le=36500, description="Account age in days (supports 100+ year old accounts)")
    tx_count_24h:       int   = Field(2, ge=0, description="Number of transactions in last 24 hours")
    hour_of_day:        int   = Field(12, ge=0, le=23, description="Hour of transaction (0-23)")


# -------------------------------------------------------
# Response schema
# -------------------------------------------------------
class PredictionResponse(BaseModel):
    ml_score:             float
    rule_score:           float
    fraud_score:          float
    risk_level:           str
    recommendation:       str
    top_risk_factors:     list
    triggered_rules:      list
    rule_explanations:    list
    analyst_notes:        list  # concern rules — do not affect score
    analyst_explanations: list


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Fraud Prevention and Detection Risk Scoring API v2",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/predict", "/health", "/docs"],
        "note": "Concerns (PPF/PPO/IDTO/PHI) are analyst annotations — they do not affect any score.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(feature_names) if feature_names else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    """
    Returns a full fraud risk assessment.
    Concern flags (PPF/PPO/IDTO/PHI etc.) appear in analyst_notes but
    do NOT change ml_score, rule_score, or fraud_score.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run model.py to train and save the model first.",
        )

    input_dict = transaction.dict()
    input_dict = engineer_single_transaction(input_dict)

    feature_vector = [input_dict.get(feat, 0) for feat in feature_names]
    feature_df     = pd.DataFrame([feature_vector], columns=feature_names)

    ml_score   = float(model.predict_proba(feature_df)[0][1])
    rule_result = run_rules_engine(input_dict)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    top_risk_factors = get_top_risk_factors(
        feature_names=feature_names,
        feature_values=np.array(feature_vector),
        model_importances=importances,
        top_n=5,
    )

    response = build_full_prediction_response(
        ml_score=ml_score,
        rule_result=rule_result,
        top_risk_factors=top_risk_factors,
    )

    # Add analyst notes to the response (score=0 concern rules)
    response["analyst_notes"]        = rule_result.get("analyst_notes", [])
    response["analyst_explanations"] = rule_result.get("analyst_explanations", [])

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
