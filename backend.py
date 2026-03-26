from fastapi import FastAPI
from pydantic import BaseModel
import random
import time

app = FastAPI(
    title="Fake CreditSense Backend",
    description="This is a replica backend API built for presentation purposes."
)

class ApplicantData(BaseModel):
    age: int
    annual_income: float
    total_debt: float
    past_due_count: int

@app.get("/")
def read_root():
    return {"status": "Fake Backend API is running."}

@app.post("/api/predict_risk")
def predict_credit_risk(applicant: ApplicantData):
    """
    Fake prediction endpoint. Returns a mock credit score 
    and risk category based on simple heuristic logic.
    """
    # Simulate processing delay
    time.sleep(0.5)
    
    debt_ratio = applicant.total_debt / max(applicant.annual_income, 1)
    
    # Generate Fake Score
    base_score = 750
    score = base_score - (applicant.past_due_count * 50) - int(debt_ratio * 100)
    
    # Clamp score between 300 and 850
    score = max(300, min(850, score))
    
    risk_category = "Low"
    if score < 600:
        risk_category = "High"
    elif score < 700:
        risk_category = "Medium"
        
    return {
        "success": True,
        "input_data": applicant.dict(),
        "prediction": {
            "credit_score": score,
            "risk_category": risk_category,
            "model_used": "RandomForest_v2_PPT_Edition",
            "confidence": round(random.uniform(0.85, 0.99), 2)
        }
    }

# To run this script for a demo:
# uvicorn fake_backend:app --reload
