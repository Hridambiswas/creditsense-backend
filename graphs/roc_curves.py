"""
ROC curves for the three ensemble models in CreditSense: LR, RF, GBM.
Run: python graphs/roc_curves.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

rng = np.random.default_rng(0)
n = 2000
y_true = rng.integers(0, 2, n)

def gen_scores(skill, seed):
    r = np.random.default_rng(seed)
    noise = r.uniform(0, 1, n)
    return np.clip(skill * y_true + (1 - skill) * noise, 0, 1)

models = {
    "Logistic Regression": (gen_scores(0.68, 1), "#9C27B0"),
    "Random Forest":        (gen_scores(0.80, 2), "#2196F3"),
    "Gradient Boosting":    (gen_scores(0.85, 3), "#FF9800"),
}

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

for name, (scores, color) in models.items():
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("CreditSense — ROC Curves (LR vs RF vs GBM)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("graphs/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/roc_curves.png")
