import time
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time Fraud & Risk Scoring API",
    description="Low-latency LightGBM fraud detection engine with real-time SHAP explainability.",
    version="1.0.0"
)

artifact = joblib.load("models/fraud_detection_bundle.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["features"]
explainer = shap.TreeExplainer(model)

class TransactionRequest(BaseModel):
    features: dict

@app.get("/health")
def health_check():
    return {"status": "healthy", "optimal_threshold": round(threshold, 4)}

@app.post("/predict")
def score_transaction(tx: TransactionRequest):
    start_time = time.perf_counter()

    row = pd.DataFrame([[tx.features.get(col, 0.0) for col in feature_names]], columns=feature_names)

    # Convert numpy array to Python list first, then pop class 1 (Fraud) probability
    fraud_prob = float(model.predict_proba(row).flatten().tolist().pop())
    is_fraud = fraud_prob >= threshold

    if fraud_prob >= threshold:
        risk_tier = "HIGH_RISK_BLOCK"
    elif fraud_prob >= threshold * 0.5:
        risk_tier = "MEDIUM_RISK_OTP"
    else:
        risk_tier = "APPROVED"

    # Compute local SHAP values for real-time reason codes
    shap_vals = explainer.shap_values(row)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals.pop()
    shap_1d = np.array(shap_vals).flatten().tolist()

    impacts = sorted(zip(feature_names, shap_1d), key=lambda item: abs(item[1]), reverse=True)[:3]

    reason_codes = [
        {
            "feature": feat,
            "impact_score": round(float(val), 4),
            "direction": "Increases Risk" if val > 0 else "Decreases Risk"
        }
        for feat, val in impacts
    ]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "decision": risk_tier,
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_prob, 4),
        "latency_ms": latency_ms,
        "top_risk_drivers": reason_codes
    }
