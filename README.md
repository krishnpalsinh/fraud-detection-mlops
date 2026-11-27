# Real-Time Fraud & Risk Detection Engine (LightGBM + SHAP + FastAPI)

An end-to-end machine learning microservice that scores financial transactions in real time (<25ms latency), routes payments into risk tiers (`APPROVED`, `MEDIUM_RISK_OTP`, `HIGH_RISK_BLOCK`), and returns per-transaction **SHAP reason codes**.

## Architecture
`Incoming Transaction JSON` -> `FastAPI (/predict)` -> `LightGBM Model (Optuna PR-AUC Tuned)` -> `Business-Cost Threshold (0.9037)` -> `Real-Time SHAP Reason Codes`

## Key Highlights
- **Imbalanced Learning:** Trained on 284,807 transactions (0.17% fraud rate) using `RobustScaler` and `LightGBM` (`is_unbalance=True`).
- **Hyperparameter & Threshold Tuning:** Optimized **PR-AUC** via **Optuna** and calibrated an F1-optimal decision threshold (`0.9037`).
- **Explainable AI (SHAP):** Extracts top 3 risk-driving features per transaction for regulatory auditability.
- **Production Packaging:** Served via **FastAPI** with **Docker** containerization.

## Project Structure
```text
├── app/main.py                        # FastAPI microservice & SHAP inference
├── models/fraud_detection_bundle.joblib # Trained model, threshold & features
├── Dockerfile                         # Container configuration
├── requirements.txt                   # Python dependencies
└── shap_summary.png                   # Global feature importance plot
```
