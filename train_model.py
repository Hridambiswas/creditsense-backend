"""
train_model.py
Run this once to generate model.pkl before deploying.
Uses synthetic data mimicking the 'Give Me Some Credit' dataset structure.
Replace with real data for better accuracy.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
N = 10000

# Synthetic features: age, monthly_income, debt_ratio
age = np.random.randint(18, 80, N)
monthly_income = np.random.lognormal(mean=8.5, sigma=0.6, size=N)
debt_ratio = np.clip(np.random.beta(2, 5, size=N), 0, 1)

X = np.column_stack([age, monthly_income, debt_ratio])

# Target: 1 = good credit (score > 50), weighted by features
score_raw = (
    0.3 * (age - 18) / 62 +
    0.4 * np.clip((monthly_income - 1000) / 15000, 0, 1) +
    0.3 * (1 - debt_ratio)
)
noise = np.random.normal(0, 0.07, N)
score_raw = np.clip(score_raw + noise, 0, 1)
y = (score_raw > 0.5).astype(int)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

pipeline.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("model.pkl saved successfully.")
print(f"Training samples: {N}")
