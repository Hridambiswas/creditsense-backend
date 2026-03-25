"""
main.py — FastAPI Credit Scoring Backend
Deploy on Render (free tier)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(title="Credit Score API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("model.pkl not found. Run train_model.py first.")


class CreditInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    monthly_income: float = Field(..., ge=0, description="Monthly income in USD")
    debt_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-income ratio (0–1)")


class CreditOutput(BaseModel):
    credit_score: int
    risk_level: str
    risk_color: str
    summary: str
    factors: list[dict]


def get_risk_level(score: int):
    if score >= 75:
        return "Low Risk", "#22c55e"
    elif score >= 50:
        return "Moderate Risk", "#f59e0b"
    elif score >= 30:
        return "High Risk", "#f97316"
    else:
        return "Very High Risk", "#ef4444"


def get_factors(age: int, monthly_income: float, debt_ratio: float):
    factors = []

    income_norm = min(monthly_income / 15000, 1.0)
    factors.append({
        "name": "Monthly Income",
        "value": f"${monthly_income:,.0f}",
        "impact": round(income_norm * 100),
        "positive": income_norm > 0.4
    })

    debt_impact = round((1 - debt_ratio) * 100)
    factors.append({
        "name": "Debt Ratio",
        "value": f"{debt_ratio * 100:.0f}%",
        "impact": debt_impact,
        "positive": debt_ratio < 0.4
    })

    age_norm = min((age - 18) / 62, 1.0)
    factors.append({
        "name": "Age",
        "value": str(age),
        "impact": round(age_norm * 100),
        "positive": age > 30
    })

    return sorted(factors, key=lambda x: x["impact"], reverse=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "Credit Score API is running"}


@app.post("/predict", response_model=CreditOutput)
def predict(data: CreditInput):
    X = np.array([[data.age, data.monthly_income, data.debt_ratio]])

    try:
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Convert probability to 0–100 credit score
    credit_score = int(round(prob * 100))
    risk_level, risk_color = get_risk_level(credit_score)
    factors = get_factors(data.age, data.monthly_income, data.debt_ratio)

    summary_map = {
        "Low Risk": "Strong credit profile. This applicant is likely to repay obligations on time.",
        "Moderate Risk": "Average credit profile. Some caution advised — review debt and income.",
        "High Risk": "Below average profile. Higher likelihood of default based on inputs.",
        "Very High Risk": "Poor credit profile. Significant risk of default indicated."
    }

    return CreditOutput(
        credit_score=credit_score,
        risk_level=risk_level,
        risk_color=risk_color,
        summary=summary_map[risk_level],
        factors=factors
    )
