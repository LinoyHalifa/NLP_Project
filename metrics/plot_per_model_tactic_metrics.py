import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\Results\tactics_per_tactic_model_comparison.csv")

models = df["model"].unique()

# ======================
# PLOT PER MODEL
# ======================
for model in models:
    model_df = df[df["model"] == model]

    tactics = model_df["tactic"]
    precision = model_df["precision"]
    recall = model_df["recall"]
    f1 = model_df["f1"]

    x = np.arange(len(tactics))
    width = 0.25

    plt.figure(figsize=(14, 6))

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1")

    plt.xticks(x, tactics, rotation=40, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title(f"Tactic-Level Metrics — {model}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    filename = f"tactics_metrics_{model.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename)
    plt.show()

    print(f"Saved: {filename}")
