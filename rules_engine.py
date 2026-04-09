"""
rules_engine.py
---------------
Rule-based fraud and scam detection layer.

KEY CHANGES:
- Fraud Concerns (PPF / PPO / IDTO) and Scam Concerns (PHI etc.) now have
  score=0 — they are ANALYST ANNOTATIONS and do NOT change the risk score.
  They still appear in the output so analysts can see them, separated under
  a dedicated "analyst_notes" key.
- n2n_payment (Name-to-Name payment) added as a green trust rule (-0.12 score).
- close_to_beneficiary is also known as "Close to BSB" in the UI.
- Rule names updated: PPO, PPF, IDTO, PHI.

NOTE: These rules are operationally inspired but not based on any
real bank's internal systems. They are designed for demo and interview purposes.
"""

from typing import Dict, Any, List


# ─────────────────────────────────────────────────────────────
# FRAUD RED FLAG RULES  — these affect the rule_score
# ─────────────────────────────────────────────────────────────
FRAUD_RULES = [
    {
        "name": "ATO Pattern: New Device + New IP + Password Reset",
        "check": lambda t: t.get("new_device") and t.get("new_ip") and t.get("password_reset"),
        "score": 0.35,
        "explanation": (
            "A new device, new IP address, and a recent password reset have all been detected together. "
            "This combination is a strong indicator of Account Takeover (ATO), where a fraudster "
            "gains access to an account by resetting the password and logging in from an unrecognised device."
        ),
    },
    {
        "name": "Account Takeover Alert Triggered",
        "check": lambda t: t.get("account_takeover_alert"),
        "score": 0.30,
        "explanation": (
            "An internal account takeover alert has been raised against this session. "
            "This flag is typically set by behavioural monitoring systems when the account "
            "shows access patterns inconsistent with the customer's normal behaviour."
        ),
    },
    {
        "name": "New Device + New IP + ATO Alert",
        "check": lambda t: t.get("new_device") and t.get("new_ip") and t.get("account_takeover_alert"),
        "score": 0.35,
        "explanation": (
            "A new device and new IP combined with an active ATO alert is a high-confidence "
            "account takeover signal. The system has already flagged unusual access and is now "
            "seeing further unrecognised access patterns."
        ),
    },
    {
        "name": "eKYC Account + Recently Created",
        "check": lambda t: t.get("ekyc_account") and t.get("recently_created_account"),
        "score": 0.20,
        "explanation": (
            "This is a digitally-created account (via eKYC) that was also recently opened. "
            "Fraudsters frequently open new accounts using stolen identity documents to use "
            "as money mule accounts."
        ),
    },
    {
        "name": "Fast Traveller + Multiple Device Logins",
        "check": lambda t: t.get("fast_traveller") and t.get("multiple_device_logins"),
        "score": 0.20,
        "explanation": (
            "The account shows signs of impossible travel combined with logins from multiple devices. "
            "This may indicate that credentials have been compromised and are being used "
            "simultaneously by the real customer and a fraudster."
        ),
    },
    {
        "name": "Language Change + Password Reset",
        "check": lambda t: t.get("language_change") and t.get("password_reset"),
        "score": 0.18,
        "explanation": (
            "The account's language setting was recently changed alongside a password reset. "
            "Fraudsters sometimes change the language to prevent the real customer from "
            "understanding security alert messages."
        ),
    },
    {
        "name": "Number Change + ATO Alert",
        "check": lambda t: t.get("number_change") and t.get("account_takeover_alert"),
        "score": 0.25,
        "explanation": (
            "A phone number change combined with an active ATO alert suggests the attacker may "
            "be attempting to update contact details to intercept authentication messages and "
            "lock the real customer out."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# SCAM RED FLAG RULES  — these affect the rule_score
# ─────────────────────────────────────────────────────────────
SCAM_RULES = [
    {
        "name": "Remote Access Session + Active Call + International Transfer",
        "check": lambda t: t.get("remote_access_session") and t.get("active_call") and t.get("international_payment"),
        "score": 0.40,
        "explanation": (
            "A remote access session is active, the customer appears to be on a phone call, "
            "and the payment is going overseas. This is the hallmark pattern of a remote access scam, "
            "where a fraudster (posing as tech support or a bank) remotely controls the customer's "
            "device and directs them to transfer money internationally while keeping them on the line."
        ),
    },
    {
        "name": "Crypto Payment + Suspicious Reference + Prior Victim",
        "check": lambda t: t.get("crypto_payment") and t.get("suspicious_reference") and t.get("prior_victim"),
        "score": 0.40,
        "explanation": (
            "A cryptocurrency payment to an address with a suspicious payment reference, "
            "made by a customer who has been a scam victim before. "
            "Cryptocurrency is increasingly used in investment scams and romance scams "
            "because payments are irreversible. Repeat victimisation is a serious concern."
        ),
    },
    {
        "name": "High-Risk BSB + Remitter Payment + Weak Name Match",
        "check": lambda t: t.get("high_risk_bsb") and t.get("remitter_payment") and t.get("weak_name_match"),
        "score": 0.30,
        "explanation": (
            "The destination BSB is flagged as high-risk, the payment type is remitter (third party), "
            "and the payee name only partially matches the account name. "
            "This combination suggests the customer may be sending money to a mule account or "
            "a fraudulent business."
        ),
    },
    {
        "name": "Vulnerable Customer + Active Call + Remote Access Session",
        "check": lambda t: t.get("vulnerable_customer") and t.get("active_call") and t.get("remote_access_session"),
        "score": 0.45,
        "explanation": (
            "A customer flagged as vulnerable is on an active phone call while a remote access session "
            "is running on their device. This is an extremely high-risk scenario. Vulnerable customers "
            "are disproportionately targeted by remote access scammers posing as bank staff or "
            "government agencies."
        ),
    },
    {
        "name": "Prior Victim — Elevated Re-targeting Risk",
        "check": lambda t: t.get("prior_victim"),
        "score": 0.18,
        "explanation": (
            "The customer has previously been a victim of a scam. Prior victims are statistically "
            "more likely to be re-targeted by scammers, and may have had their details shared "
            "on lists used by scam networks."
        ),
    },
    {
        "name": "Crypto Payment + International Transfer",
        "check": lambda t: t.get("crypto_payment") and t.get("international_payment"),
        "score": 0.22,
        "explanation": (
            "A cryptocurrency payment going overseas combines two significant scam red flags. "
            "This pattern is common in investment scams where funds are directed offshore and "
            "are virtually impossible to recover."
        ),
    },
    {
        "name": "Suspicious Reference + Remitter Payment",
        "check": lambda t: t.get("suspicious_reference") and t.get("remitter_payment"),
        "score": 0.18,
        "explanation": (
            "A suspicious payment reference combined with a remitter (third-party) payment raises "
            "concerns about the legitimacy of the beneficiary and the true purpose of the transfer."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# GREEN FLAG TRUST RULES  — these reduce the rule_score
# ─────────────────────────────────────────────────────────────
TRUST_RULES = [
    {
        "name": "Registered Device + Registered IP + Prior Payment — Lower Risk",
        "check": lambda t: t.get("registered_device") and t.get("registered_ip") and t.get("paid_before"),
        "score": -0.20,
        "explanation": (
            "The transaction was made from a registered device, a known IP address, and the customer "
            "has successfully paid this beneficiary before. This combination of trust signals "
            "suggests a lower probability of fraud."
        ),
    },
    {
        "name": "Branch Device or GPS at Address — Positive Trust Indicator",
        "check": lambda t: t.get("branch_device") or t.get("gps_at_address"),
        "score": -0.15,
        "explanation": (
            "The transaction was made from a bank branch device or the customer's GPS location "
            "matches their registered address. Physical presence at a known location significantly "
            "reduces remote fraud risk."
        ),
    },
    {
        "name": "N2N Payment — Name-to-Name Trust Signal",
        "check": lambda t: t.get("n2n_payment"),
        "score": -0.12,
        "explanation": (
            "This is a Name-to-Name (N2N) payment, meaning the sending and receiving account names "
            "match or are closely associated. N2N payments carry a lower scam risk as the customer "
            "is transacting with a known and named individual, reducing the likelihood of an "
            "impersonation or mule account scenario."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# CONCERN RULES  — score = 0 — ANALYST ANNOTATIONS ONLY
# These do NOT change the risk score. They are displayed
# separately so analysts can note the concern type.
#
# Renamed in UI: PPF (login_irregularity), PPO (sim_swap),
#                IDTO (identity_theft), PHI (phishing)
# ─────────────────────────────────────────────────────────────
CONCERN_RULES = [
    {
        "name": "⚠️ Analyst Note: PPO — SIM Swap / Portable Number Order",
        "check": lambda t: t.get("sim_swap"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A PPO (Portable Number Order / SIM swap) concern has been noted. "
            "This may indicate the customer's phone number has been ported to a new SIM without "
            "their knowledge, often a precursor to account takeover. The analyst should verify "
            "account contact details and consider outbound contact with the customer."
        ),
    },
    {
        "name": "⚠️ Analyst Note: PPF — Login Irregularity",
        "check": lambda t: t.get("login_irregularity"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A PPF (Prohibited Pattern Flag — login irregularity) concern has been noted. "
            "The analyst should review recent login history for unusual times, locations, or "
            "failure patterns that may inform a broader fraud assessment."
        ),
    },
    {
        "name": "⚠️ Analyst Note: IDTO — Identity Takeover",
        "check": lambda t: t.get("identity_theft"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "An IDTO (Identity Takeover) concern has been flagged. The analyst should consider "
            "whether the customer's personal details may have been compromised and whether "
            "additional verification is warranted before processing."
        ),
    },
    {
        "name": "⚠️ Analyst Note: PHI — Phishing Concern",
        "check": lambda t: t.get("phishing"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A PHI (Phishing) concern has been identified. The customer may have been directed "
            "to a fraudulent site and entered credentials or authorised a payment under false pretences."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Investment Scam Concern",
        "check": lambda t: t.get("investment_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "An investment scam context is suspected. The customer may have been contacted by a "
            "fake broker or trading platform promising high returns. Consider a welfare call before processing."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Romance Scam Concern",
        "check": lambda t: t.get("romance_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A romance scam context is suspected. The customer may have developed an online "
            "relationship with a fraudster who is now requesting money. Sensitive customer "
            "engagement is recommended."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Business Email Compromise (BEC) Concern",
        "check": lambda t: t.get("business_email_compromise"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "BEC is suspected. The customer may have received a fraudulent email impersonating a "
            "supplier or executive requesting a payment redirect. Verify directly with the intended "
            "payee using known contact details before proceeding."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Goods & Services Scam Concern",
        "check": lambda t: t.get("goods_services_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A goods or services scam is suspected. The customer may be paying for goods or "
            "services that will not be delivered, often via an online marketplace or fake retailer."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Remote Access Scam Concern",
        "check": lambda t: t.get("remote_access_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A remote access scam context has been noted. The customer may have allowed a scammer "
            "to access their device under the guise of tech support or a bank security team."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Hi Mum / Hi Dad Scam Concern",
        "check": lambda t: t.get("hi_mum_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A family impersonation ('Hi Mum / Hi Dad') scam is suspected. The customer may have "
            "received a message from someone claiming to be a family member in distress requesting "
            "an urgent transfer from a new number."
        ),
    },
    {
        "name": "⚠️ Analyst Note: Job Scam Concern",
        "check": lambda t: t.get("job_scam"),
        "score": 0,
        "explanation": (
            "ANALYST ANNOTATION — No score change. "
            "A job scam context has been noted. The customer may have responded to a fake job offer "
            "involving receiving and forwarding funds, making them an unwitting money mule."
        ),
    },
]


def run_rules_engine(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs all rules against a single transaction.

    Returns:
        triggered_rules:    names of scoring rules that fired (danger + trust)
        rule_explanations:  explanations for scoring rules
        analyst_notes:      names of concern rules that fired (score = 0)
        analyst_explanations: explanations for concern rules
        rule_score:         float 0-1 from scoring rules only
        rule_recommendation: APPROVE / REVIEW / ESCALATE / BLOCK
    """

    scoring_triggered = []
    scoring_explanations = []
    concern_triggered = []
    concern_explanations = []
    total_score = 0.0

    # Run scoring rules (fraud + scam red flags + trust rules)
    for rule in FRAUD_RULES + SCAM_RULES + TRUST_RULES:
        try:
            fired = rule["check"](transaction)
        except Exception:
            fired = False
        if fired:
            scoring_triggered.append(rule["name"])
            scoring_explanations.append(rule["explanation"])
            total_score += rule["score"]

    # Run concern rules — collect but do NOT add to score
    for rule in CONCERN_RULES:
        try:
            fired = rule["check"](transaction)
        except Exception:
            fired = False
        if fired:
            concern_triggered.append(rule["name"])
            concern_explanations.append(rule["explanation"])

    rule_score = max(0.0, min(1.0, total_score))

    return {
        "triggered_rules":      scoring_triggered,
        "rule_explanations":    scoring_explanations,
        "analyst_notes":        concern_triggered,
        "analyst_explanations": concern_explanations,
        "rule_score":           round(rule_score, 4),
        "rule_recommendation":  map_recommendation(rule_score),
    }


def map_recommendation(score: float) -> str:
    if score >= 0.60:
        return "BLOCK"
    elif score >= 0.35:
        return "ESCALATE"
    elif score >= 0.15:
        return "REVIEW"
    else:
        return "APPROVE"


if __name__ == "__main__":
    # Test: high-risk ATO — no concern flags should affect score
    ato = {
        "new_device": 1, "new_ip": 1, "password_reset": 1,
        "account_takeover_alert": 1, "number_change": 1,
        # Concern flags — should NOT change score
        "sim_swap": 1, "login_irregularity": 1, "identity_theft": 1,
    }
    r = run_rules_engine(ato)
    print("=== ATO Test ===")
    print(f"Rule Score:       {r['rule_score']}  (concerns excluded)")
    print(f"Recommendation:   {r['rule_recommendation']}")
    print(f"Scoring rules:    {r['triggered_rules']}")
    print(f"Analyst notes:    {r['analyst_notes']}")

    # Test: only concern flags — score must be 0
    concerns_only = {"sim_swap": 1, "login_irregularity": 1, "phishing": 1, "hi_mum_scam": 1}
    r2 = run_rules_engine(concerns_only)
    print("\n=== Concerns-Only Test (score must be 0) ===")
    print(f"Rule Score:       {r2['rule_score']}  {'✅' if r2['rule_score'] == 0 else '❌ FAIL'}")
    print(f"Analyst notes:    {r2['analyst_notes']}")

    # Test: N2N payment + trusted signals
    low_risk = {
        "registered_device": 1, "registered_ip": 1, "paid_before": 1,
        "gps_at_address": 1, "n2n_payment": 1,
    }
    r3 = run_rules_engine(low_risk)
    print("\n=== Low-Risk + N2N Test ===")
    print(f"Rule Score:       {r3['rule_score']}")
    print(f"Recommendation:   {r3['rule_recommendation']}")
    print(f"Triggered:        {r3['triggered_rules']}")
