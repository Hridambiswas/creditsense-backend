# CreditSense — ML Credit Scoring API

> **Real-time credit risk prediction using an ensemble of ML models, served via FastAPI**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What It Does

CreditSense takes three applicant inputs — **age, monthly income, and debt-to-income ratio** — and returns a credit score (0–100), a risk tier, and a plain-language explanation of the key risk drivers.

```
POST /score
{
  "age": 32,
  "monthly_income": 4500,
  "debt_ratio": 0.38
}

→ { "credit_score": 71, "risk_level": "Low Risk", "summary": "..." }
```

---

## Architecture

```
credit_risk_dataset.csv
        │
        ▼
┌──────────────────────────────────┐
│  train_model.py                  │
│                                  │
│  Logistic Regression             │
│  + Random Forest                 │
│  + Gradient Boosting             │
│  → VotingClassifier ensemble     │
│  → Serialised → model.pkl        │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  main.py — FastAPI               │
│                                  │
│  POST /score  → risk score       │
│  GET  /health → service status   │
└──────────────────────────────────┘
        │
        ▼
   Render (free tier)
```

---

## Setup

```bash
git clone https://github.com/Hridambiswas/creditsense-backend.git
cd creditsense-backend
pip install -r requirements.txt

# Train the model first
python train_model.py

# Start the API
uvicorn main:app --reload
```

API docs auto-generated at `http://localhost:8000/docs`

---

## Project Structure

```
creditsense-backend/
│
├── main.py                  # FastAPI app — POST /score endpoint
├── train_model.py           # Trains VotingClassifier, saves model.pkl
├── ml_decision.py           # Data preprocessing + model training utilities
├── frontend_app.py          # Streamlit frontend (optional UI)
├── credit_risk_dataset.csv  # Training data (23,000+ loan records)
├── requirements.txt
├── runtime.txt              # Python version pin for Render
└── render.yaml              # One-click Render deployment config
```

---

## Model Details

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline, interpretable coefficients |
| Random Forest | Non-linear patterns, handles missing data |
| Gradient Boosting | Sequential error correction, highest accuracy |
| **VotingClassifier** | **Soft-vote ensemble — final predictor** |

Training data: 23,000+ real loan applications with features including age, income, employment length, loan purpose, and historical default behavior.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/score` | Returns credit score + risk breakdown |

---

## Deployment

Pre-configured for [Render](https://render.com/) free tier via `render.yaml`. Push to GitHub and connect the repo — zero-config deploy.

---

## Author

**Hridam Biswas** — IEEE Researcher, KIIT University  
[GitHub](https://github.com/Hridambiswas) · [Portfolio](https://hridambiswas.github.io)

