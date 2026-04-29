"""
Feature importance bar chart for the CreditSense GBM model.
Run: python graphs/feature_importance.py
"""
import matplotlib.pyplot as plt
import numpy as np

features = [
    "Payment History (12m)",
    "Debt-to-Income Ratio",
    "Credit Utilisation",
    "Loan Age (months)",
    "Number of Open Accounts",
    "Recent Hard Enquiries",
    "Employment Duration",
    "Monthly Income",
    "Revolving Balance",
    "Derogatory Marks",
]
importances = [0.231, 0.198, 0.157, 0.112, 0.088, 0.072, 0.055, 0.043, 0.028, 0.016]
colors = ["#F44336" if imp > 0.10 else "#2196F3" for imp in importances]

fig, ax = plt.subplots(figsize=(9, 6))
y = np.arange(len(features))
bars = ax.barh(y, importances, color=colors, alpha=0.87, edgecolor="white", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel("Feature Importance (normalised)", fontsize=11)
ax.set_title("CreditSense — GBM Feature Importance", fontsize=13, fontweight="bold")
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

for bar, val in zip(bars, importances):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#F44336", alpha=0.87, label="Top features (>10%)"),
                   Patch(color="#2196F3", alpha=0.87, label="Secondary features")],
          fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("graphs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/feature_importance.png")
