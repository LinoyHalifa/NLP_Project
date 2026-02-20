import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import f1_score

FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_predictions_test.csv"

df = pd.read_csv(FILE)

# Convert string representations back to lists
df["y_true"] = df["y_true"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)

df["y_pred"] = df["y_pred"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)


y_true = np.array(df["y_true"].tolist())
y_pred = np.array(df["y_pred"].tolist())

# Count how many tactics exist in each headline
num_tactics = np.sum(y_true, axis=1)

print("\n===== F1 by Number of Tactics =====")

for k in [1, 2, 3]:
    indices = np.where(num_tactics == k)[0]
    if len(indices) > 0:
        f1 = f1_score(y_true[indices], y_pred[indices], average="micro")
        print(f"{k} tactic(s): {round(f1, 4)}  |  Samples: {len(indices)}")


# ======================================================
# PLOT: F1 vs Number of Tactics
# ======================================================

import matplotlib.pyplot as plt

tactic_counts = []
f1_scores = []
sample_counts = []

for k in [1, 2, 3]:
    indices = np.where(num_tactics == k)[0]
    if len(indices) > 0:
        f1 = f1_score(y_true[indices], y_pred[indices], average="micro")
        tactic_counts.append(k)
        f1_scores.append(f1)
        sample_counts.append(len(indices))

# Create plot
plt.figure(figsize=(6, 4))
plt.plot(tactic_counts, f1_scores, marker="o")

plt.xlabel("Number of Tactics in Headline")
plt.ylabel("Micro-F1 Score")
plt.title("Effect of Rhetorical Density on Multi-Label Performance")
plt.xticks([1, 2, 3])
plt.grid(True)

# Optional: annotate points with F1 values
for i, txt in enumerate(f1_scores):
    plt.annotate(f"{txt:.3f}", (tactic_counts[i], f1_scores[i]),
                 textcoords="offset points", xytext=(0,5), ha='center')

# Save figure
PLOT_PATH = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\f1_by_rhetorical_density.png"
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print(f"\nPlot saved to: {PLOT_PATH}")

