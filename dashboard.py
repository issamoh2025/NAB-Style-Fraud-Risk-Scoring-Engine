"""
dashboard.py (Streamlit)
------------------------
Interactive fraud detection dashboard.

KEY CHANGES:
- n2n_payment added as Scam Green Flag
- Close to Beneficiary → Close to BSB (label only, key unchanged)
- SIM Swap → PPO, Login Irregularity → PPF, Identity Theft → IDTO, Phishing → PHI
- Fraud/Scam Concerns clearly labelled as "Analyst Annotations (no score impact)"
- account_age_days replaced with a date picker — supports 60+ year old accounts
- Concern flags shown in a separate Analyst Notes section, not the main risk summary

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, datetime

from model import load_model
from rules_engine import run_rules_engine
from feature_engineering import engineer_single_transaction
from utils import (
    combine_scores, classify_risk_level, map_recommendation,
    get_top_risk_factors, format_feature_name,
    format_risk_badge, format_recommendation_badge,
    build_full_prediction_response,
)

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="NAB Fraud Detection Risk Scoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #000000;
        border-bottom: 3px solid #D4001A; padding-bottom: 10px; margin-bottom: 20px;
    }
    .nab-red  { color: #D4001A; }
    .concern-box {
        background: #f7f7f7; border: 1px solid #e0e0e0;
        border-left: 4px solid #6D6E71; border-radius: 6px;
        padding: 10px 14px; margin: 6px 0; font-size: 0.85rem; color: #6D6E71;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Load model (cached)
# -------------------------------------------------------
@st.cache_resource
def load_cached_model():
    try:
        return load_model()
    except FileNotFoundError:
        from model import train_model
        st.info("No saved model found. Training model now for first run...")
        model, _, _, _, _, feature_names = train_model()
        return model, feature_names


model, feature_names = load_cached_model()

if model is None:
    import subprocess
    st.warning("Model not found — training model (one-time setup)...")
    subprocess.run(["python", "model.py"])

    model, feature_names = load_cached_model()


# -------------------------------------------------------
# Helper: calculate account age from a date
# -------------------------------------------------------
def days_from_date(opened: date) -> int:
    return max(0, (date.today() - opened).days)


def format_age(days: int) -> str:
    if days < 30:
        return f"{days} days"
    elif days < 365:
        return f"{days // 30} months ({days} days)"
    else:
        yrs = days // 365
        mo  = (days % 365) // 30
        return f"{yrs} yr{'s' if yrs > 1 else ''}{', ' + str(mo) + ' mo' if mo else ''} ({days:,} days)"


# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('<div class="main-header">🔍 NAB Fraud Prevention & Detection Risk Scoring</div>', unsafe_allow_html=True)
st.markdown("""
> Combines a **machine learning model** (Random Forest, trained on 50,000 synthetic transactions)
> with a **bank-style rules engine**.
> **Fraud/Scam Concerns are analyst annotations** — they do not affect the risk score.
""")

if model is None:
    st.error("⚠️ Model not found. Please run `python model.py` first.")
    st.stop()


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("⚙️ Transaction Features")

st.sidebar.subheader("🟢 Fraud Green Flags")
gps_at_address    = st.sidebar.checkbox("GPS at Address",    value=False)
registered_device = st.sidebar.checkbox("Registered Device", value=True)
registered_ip     = st.sidebar.checkbox("Registered IP",     value=True)
branch_device     = st.sidebar.checkbox("Branch Device",     value=False)

st.sidebar.subheader("🟢 Scam Green Flags")
paid_before          = st.sidebar.checkbox("Paid Before",      value=False)
n2n_payment          = st.sidebar.checkbox("N2N Pmt",          value=False, help="Name-to-Name payment — trust signal")
close_to_beneficiary = st.sidebar.checkbox("Close to BSB",     value=False)
name_match           = st.sidebar.checkbox("Name Match",       value=False)
demographic_match    = st.sidebar.checkbox("Demographic Match",value=False)
no_prior_fraud       = st.sidebar.checkbox("No Prior Fraud",   value=True)

st.sidebar.subheader("🔴 Fraud Red Flags")
new_ip                   = st.sidebar.checkbox("New IP",                   value=False)
new_device               = st.sidebar.checkbox("New Device",               value=False)
ekyc_account             = st.sidebar.checkbox("eKYC Account",             value=False)
recently_created_account = st.sidebar.checkbox("Recently Created Account", value=False)
multiple_device_logins   = st.sidebar.checkbox("Multiple Device Logins",   value=False)
fast_traveller           = st.sidebar.checkbox("Fast Traveller",           value=False)
password_reset           = st.sidebar.checkbox("Password Reset",           value=False)
number_change            = st.sidebar.checkbox("Number Change",            value=False)
account_takeover_alert   = st.sidebar.checkbox("Account Takeover Alert",   value=False)
language_change          = st.sidebar.checkbox("Language Change",          value=False)

st.sidebar.subheader("🔴 Scam Red Flags")
weak_name_match       = st.sidebar.checkbox("Weak Name Match",        value=False)
remote_access_session = st.sidebar.checkbox("Remote Access Session",  value=False)
crypto_payment        = st.sidebar.checkbox("Crypto Payment",         value=False)
remitter_payment      = st.sidebar.checkbox("Remitter Payment",       value=False)
vulnerable_customer   = st.sidebar.checkbox("Vulnerable Customer",    value=False)
international_payment = st.sidebar.checkbox("International Payment",  value=False)
high_risk_bsb         = st.sidebar.checkbox("High Risk BSB",          value=False)
prior_victim          = st.sidebar.checkbox("Prior Victim",           value=False)
suspicious_reference  = st.sidebar.checkbox("Suspicious Reference",   value=False)
active_call           = st.sidebar.checkbox("Active Call",            value=False)
gambling_activity     = st.sidebar.checkbox("Gambling Activity",      value=False)

# Concerns — clearly labelled as analyst annotations
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Fraud Concerns")
st.sidebar.caption("Analyst annotations — **no score impact**")
login_irregularity = st.sidebar.checkbox("PPF — Login Irregularity", value=False)
sim_swap           = st.sidebar.checkbox("PPO — SIM Swap",           value=False)
identity_theft     = st.sidebar.checkbox("IDTO — Identity Theft",    value=False)

st.sidebar.subheader("⚠️ Scam Concerns")
st.sidebar.caption("Analyst annotations — **no score impact**")
phishing                  = st.sidebar.checkbox("PHI — Phishing",                value=False)
investment_scam           = st.sidebar.checkbox("Investment Scam",               value=False)
romance_scam              = st.sidebar.checkbox("Romance Scam",                  value=False)
business_email_compromise = st.sidebar.checkbox("Business Email Compromise",     value=False)
goods_services_scam       = st.sidebar.checkbox("Goods & Services Scam",         value=False)
remote_access_scam        = st.sidebar.checkbox("Remote Access Scam",            value=False)
hi_mum_scam               = st.sidebar.checkbox("Hi Mum Scam",                  value=False)
job_scam                  = st.sidebar.checkbox("Job Scam",                      value=False)

# Transaction details
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Transaction Details")
transaction_amount = st.sidebar.number_input(
    "Transaction Amount ($)", min_value=1.0, max_value=100000.0, value=250.0, step=50.0
)

# Date picker for account age — supports 60+ year old accounts
st.sidebar.markdown("**Account Opened Date**")
min_date = date(1920, 1, 1)   # supports 100+ year old accounts
max_date = date.today()
default_open = date(date.today().year - 1, date.today().month, date.today().day)
account_opened = st.sidebar.date_input(
    "Account Opened Date",
    value=default_open,
    min_value=min_date,
    max_value=max_date,
    help="Supports accounts opened up to 100+ years ago",
)
account_age_days = days_from_date(account_opened)
st.sidebar.caption(f"Account age: **{format_age(account_age_days)}**")

tx_count_24h = st.sidebar.number_input("Transactions in 24h", min_value=0, max_value=50, value=2)
hour_of_day  = st.sidebar.slider("Hour of Day", 0, 23, 12)


# -------------------------------------------------------
# Build input dict
# -------------------------------------------------------
input_dict = {
    "gps_at_address": int(gps_at_address), "registered_device": int(registered_device),
    "registered_ip": int(registered_ip),   "branch_device": int(branch_device),
    "paid_before": int(paid_before),        "n2n_payment": int(n2n_payment),
    "close_to_beneficiary": int(close_to_beneficiary), "name_match": int(name_match),
    "demographic_match": int(demographic_match),       "no_prior_fraud": int(no_prior_fraud),
    "new_ip": int(new_ip),               "new_device": int(new_device),
    "ekyc_account": int(ekyc_account),   "recently_created_account": int(recently_created_account),
    "multiple_device_logins": int(multiple_device_logins), "fast_traveller": int(fast_traveller),
    "password_reset": int(password_reset), "number_change": int(number_change),
    "account_takeover_alert": int(account_takeover_alert), "language_change": int(language_change),
    "weak_name_match": int(weak_name_match), "remote_access_session": int(remote_access_session),
    "crypto_payment": int(crypto_payment),  "remitter_payment": int(remitter_payment),
    "vulnerable_customer": int(vulnerable_customer), "international_payment": int(international_payment),
    "high_risk_bsb": int(high_risk_bsb),   "prior_victim": int(prior_victim),
    "suspicious_reference": int(suspicious_reference), "active_call": int(active_call),
    "gambling_activity": int(gambling_activity),
    # Concerns
    "login_irregularity": int(login_irregularity), "sim_swap": int(sim_swap),
    "identity_theft": int(identity_theft), "phishing": int(phishing),
    "investment_scam": int(investment_scam), "romance_scam": int(romance_scam),
    "business_email_compromise": int(business_email_compromise),
    "goods_services_scam": int(goods_services_scam), "remote_access_scam": int(remote_access_scam),
    "hi_mum_scam": int(hi_mum_scam), "job_scam": int(job_scam),
    # Numerical
    "transaction_amount": float(transaction_amount),
    "account_age_days": int(account_age_days),
    "tx_count_24h": int(tx_count_24h),
    "hour_of_day": int(hour_of_day),
}

input_dict = engineer_single_transaction(input_dict)

# -------------------------------------------------------
# Run prediction
# -------------------------------------------------------
feature_vector = [input_dict.get(feat, 0) for feat in feature_names]
feature_df     = pd.DataFrame([feature_vector], columns=feature_names)

ml_score   = float(model.predict_proba(feature_df)[0][1])
rule_result = run_rules_engine(input_dict)

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    importances = np.abs(model.coef_[0])

top_factors = get_top_risk_factors(feature_names, feature_vector, importances, top_n=5)
response    = build_full_prediction_response(ml_score=ml_score, rule_result=rule_result, top_risk_factors=top_factors)

fraud_score    = response["fraud_score"]
risk_level     = response["risk_level"]
recommendation = response["recommendation"]
rule_score     = response["rule_score"]

analyst_notes        = rule_result.get("analyst_notes", [])
analyst_explanations = rule_result.get("analyst_explanations", [])

# -------------------------------------------------------
# Score Cards
# -------------------------------------------------------
st.markdown("## 📊 Risk Assessment")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🤖 ML Score",       f"{ml_score:.2%}")
col2.metric("📋 Rule Score",     f"{rule_score:.2%}")
col3.metric("⚖️ Combined Score", f"{fraud_score:.2%}")
col4.metric("🎯 Risk Level",     risk_level)

# Recommendation banner
rec_fn = {"APPROVE": st.success, "REVIEW": st.info, "ESCALATE": st.warning, "BLOCK": st.error}
rec_icon = {"APPROVE": "✅", "REVIEW": "🔍", "ESCALATE": "⚠️", "BLOCK": "🚫"}
rec_fn[recommendation](
    f"{rec_icon[recommendation]} **{recommendation}** — combined score: {fraud_score:.1%} | {risk_level}"
)

# Gauge
st.markdown("### 🌡️ Fraud Risk Gauge")
st.progress(min(fraud_score, 1.0))
st.caption(f"Score: **{fraud_score:.1%}** — {risk_level} | Account age: {format_age(account_age_days)}")

st.markdown("---")

# -------------------------------------------------------
# Two-col: scores + top factors
# -------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("### 📊 Score Breakdown")
    bd = pd.DataFrame({
        "Source":  ["ML Model (60%)", "Rules Engine (40%)", "Combined"],
        "Score":   [ml_score, rule_score, fraud_score],
        "Weight":  ["60%", "40%", "Final"],
    })
    st.dataframe(bd.style.format({"Score": "{:.2%}"}), use_container_width=True)

    # Active signals
    green_keys   = ["gps_at_address","registered_device","registered_ip","branch_device",
                     "paid_before","n2n_payment","close_to_beneficiary","name_match",
                     "demographic_match","no_prior_fraud"]
    red_keys     = ["new_ip","new_device","ekyc_account","recently_created_account",
                     "multiple_device_logins","fast_traveller","password_reset","number_change",
                     "account_takeover_alert","language_change","weak_name_match",
                     "remote_access_session","crypto_payment","remitter_payment","vulnerable_customer",
                     "international_payment","high_risk_bsb","prior_victim","suspicious_reference",
                     "active_call","gambling_activity"]
    concern_keys = ["login_irregularity","sim_swap","identity_theft","phishing","investment_scam",
                     "romance_scam","business_email_compromise","goods_services_scam",
                     "remote_access_scam","hi_mum_scam","job_scam"]

    label_map = {
        "close_to_beneficiary": "Close to BSB", "n2n_payment": "N2N Pmt",
        "login_irregularity": "PPF", "sim_swap": "PPO",
        "identity_theft": "IDTO", "phishing": "PHI",
    }
    to_label = lambda k: label_map.get(k, format_feature_name(k))

    active_red     = [k for k in red_keys     if input_dict.get(k, 0)]
    active_green   = [k for k in green_keys   if input_dict.get(k, 0)]
    active_concern = [k for k in concern_keys if input_dict.get(k, 0)]

    if active_red:
        st.error(f"🔴 **Red flags:** {', '.join(to_label(k) for k in active_red)}")
    if active_green:
        st.success(f"🟢 **Green flags:** {', '.join(to_label(k) for k in active_green)}")
    if active_concern:
        st.info(f"📝 **Analyst annotations (no score impact):** {', '.join(to_label(k) for k in active_concern)}")
    if not active_red and not active_green and not active_concern:
        st.info("No signals active.")

with right:
    st.markdown("### 🏆 Top Risk Factors (ML Model)")
    for i, factor in enumerate(top_factors, 1):
        val  = input_dict.get(factor.replace(" ", "_").lower(), 0)
        icon = "🔴" if val else "⚪"
        st.markdown(f"**{i}.** {icon} {format_feature_name(factor)}")

st.markdown("---")

# -------------------------------------------------------
# Rules triggered (scoring rules only)
# -------------------------------------------------------
st.markdown("### 📋 Rules Engine — Triggered Scoring Rules")
triggered      = response["triggered_rules"]
explanations   = response["rule_explanations"]

if not triggered:
    st.success("✅ No scoring rules triggered.")
else:
    for name, explanation in zip(triggered, explanations):
        is_trust = "Lower Risk" in name or "Trust" in name or "N2N" in name
        icon = "🟢" if is_trust else "🔴"
        with st.expander(f"{icon} {name}", expanded=(not is_trust)):
            st.write(explanation)

# -------------------------------------------------------
# Analyst Notes (concerns — score = 0)
# -------------------------------------------------------
if analyst_notes:
    st.markdown("### 📝 Analyst Notes *(no score impact)*")
    st.caption("The following concerns are analyst annotations. They do not affect the ML score or rule score.")
    for name, explanation in zip(analyst_notes, analyst_explanations):
        with st.expander(f"📋 {name}", expanded=False):
            st.markdown(f'<div class="concern-box">{explanation}</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Feature Importance chart
# -------------------------------------------------------
st.markdown("### 📈 Model Feature Importance")
if Path("outputs/feature_importance.csv").exists():
    imp_df = pd.read_csv("outputs/feature_importance.csv").head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#D4001A")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Features — Fraud Detection Model")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Run `python model.py` to generate the feature importance chart.")

# -------------------------------------------------------
# Class distribution
# -------------------------------------------------------
st.markdown("### 📊 Dataset Class Distribution")
if Path("data/fraud_dataset.csv").exists():
    @st.cache_data
    def load_dist():
        return pd.read_csv("data/fraud_dataset.csv")["is_fraud"].value_counts()

    try:
        counts = load_dist()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(["Legitimate", "Fraud"], [counts.get(0, 0), counts.get(1, 0)],
                color=["#000000", "#D4001A"])
        ax2.set_ylabel("Count")
        ax2.set_title("Class Distribution in Training Data")
        for i, (_, cnt) in enumerate(zip(["Legitimate", "Fraud"], [counts.get(0,0), counts.get(1,0)])):
            ax2.text(i, cnt + 100, f"{cnt / counts.sum():.1%}", ha="center", fontsize=10, color="white" if i else "black")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    except Exception:
        st.info("Could not load dataset for chart.")
