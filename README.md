# NAB-Style Fraud Risk Scoring Engine

## Overview
This project is a full-stack fraud detection system designed to simulate real-world banking fraud prevention workflows.

It combines a machine learning model with a rule-based engine to generate real-time fraud risk scores for financial transactions. The system outputs a probability score (0–1), risk classification, key contributing factors, and actionable recommendations (approve, review, escalate, block).

## Key Features
- Machine learning fraud probability scoring (Random Forest / Logistic Regression)
- Rule-based fraud detection engine (ATO patterns, scam signals, trust indicators)
- Feature engineering aligned with real banking fraud signals
- Risk classification (LOW, MEDIUM, HIGH, CRITICAL)
- Actionable recommendations based on combined risk score
- FastAPI backend for real-time scoring
- Streamlit dashboard for analyst interaction and decision support

## Tech Stack
- Python
- pandas, numpy, scikit-learn
- FastAPI (API layer)
- Streamlit (dashboard)
- matplotlib (visualisation)

## How It Works
1. Synthetic transaction data is generated
2. Features are engineered (behavioural risk, trust score, scam patterns)
3. A machine learning model predicts fraud probability
4. A rules engine applies banking-style logic and explanations
5. Scores are combined into a final fraud risk score
6. The system outputs a decision recommendation

## Running the Project

Install dependencies:
pip install -r requirements.txt

Train the model:
python model.py

Evaluate performance:
python evaluate.py

Run API:
uvicorn app:app --reload --port 8000

Run dashboard:
streamlit run dashboard.py
