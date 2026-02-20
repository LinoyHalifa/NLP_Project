import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

ZERO_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\zero\gemini_two_stage_preds_zero_shot.csv"
FEW_FILE  = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_preds_few_shot.csv"


def normalize_vector(x, target_len=10):
    if isinstance(x, str):
        x = ast.literal_eval(x)

    if x == []:
        return np.zeros(target_len, dtype=int)

    x = list(map(int, x))

    if len(x) < target_len:
        x = x + [0] * (target_len - len(x))
    elif len(x) > target_len:
        x = x[:target_len]

    return np.array(x)


def compute_f1_by_density(file_path):

    df = pd.read_csv(file_path)

    df["true_vec"] = df["true_tactics_vector"].apply(normalize_vector)
    df["pred_vec"] = df["pred_tactics"].apply(normalize_vector)

    y_true = np.stack(df["true_vec"].values)
    y_pred = np.stack(df["pred_vec"].values)

    # Evaluate only GT clickbait
    mask = df["true_clickbait"] == 1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    num_tactics = np.sum(y_true, axis=1)

    f1_results = []

    for k in [1, 2, 3]:
        indices = np.where(num_tactics == k)[0]
        if len(indices) > 0:
            f1 = f1_score(
                y_true[indices],
                y_pred[indices],
                average="micro",
                zero_division=0
            )
            f1_results.append(f1)
        else:
            f1_results.append(0)

    return f1_results


# Compute results
zero_f1 = compute_f1_by_density(ZERO_FILE)
few_f1  = compute_f1_by_density(FEW_FILE)

print("Zero-Shot F1:", zero_f1)
print("Few-Shot F1 :", few_f1)

# ===== Plot =====

x = np.array([1, 2, 3])
offset = 0.07

plt.figure(figsize=(7,5))

x_zero = x - offset
x_few  = x + offset

plt.plot(x_zero, zero_f1, marker="o", linewidth=2, label="Gemini Zero-Shot")
plt.plot(x_few,  few_f1,  marker="s", linewidth=2, label="Gemini Few-Shot")

plt.xlabel("Number of Tactics in Headline")
plt.ylabel("Micro-F1 Score")
plt.title("Effect of Rhetorical Density on Multi-Label Performance (Gemini)")
plt.xticks([1,2,3])
plt.ylim(0.3, 0.45)
plt.grid(alpha=0.3)
plt.legend()

for i in range(len(x)):
    plt.text(
        x_zero[i],
        zero_f1[i] + 0.006,
        f"{zero_f1[i]:.3f}",
        ha="center",
        fontsize=9
    )

    plt.text(
        x_few[i],
        few_f1[i] + 0.008,
        f"{few_f1[i]:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()
plt.show()
