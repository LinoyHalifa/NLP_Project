import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\Results\tactics_per_tactic_model_comparison.csv")

# ======================
# PIVOT: tactic × model
# ======================
pivot_df = df.pivot(index="tactic", columns="model", values="f1")

tactics = pivot_df.index.tolist()
models = pivot_df.columns.tolist()

# ======================
# BAR POSITIONS
# ======================
x = np.arange(len(tactics))
width = 0.18   # adjust if you have many models

# ======================
# PLOT
# ======================
plt.figure(figsize=(18, 8))

for i, model in enumerate(models):
    plt.bar(
        x + i * width,
        pivot_df[model],
        width,
        label=model
    )

plt.xticks(
    x + width * (len(models) - 1) / 2,
    tactics,
    rotation=45,
    ha="right"
)

plt.ylabel("F1-score")
plt.title("Per-Tactic F1 Comparison Across Models")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("tactics_f1_grouped_bar_all_models.png")
plt.show()

print("Saved: tactics_f1_grouped_bar_all_models.png")
