import pandas as pd
import numpy as np
import ast
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ===============================
# LOAD FILES
# ===============================
few_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\few\gpt_two_stage_preds_few_shot.csv"
zero_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\zero\gpt_two_stage_preds_zero_shot.csv"

df_few = pd.read_csv(few_path)
df_zero = pd.read_csv(zero_path)

# ===============================
# Convert string vectors → list
# ===============================
def parse_vec(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df_few["true_vec"] = df_few["true_tactics_vector"].apply(parse_vec)
df_few["pred_vec"] = df_few["pred_tactics"].apply(parse_vec)

df_zero["true_vec"] = df_zero["true_tactics_vector"].apply(parse_vec)
df_zero["pred_vec"] = df_zero["pred_tactics"].apply(parse_vec)

# ===============================
# Count tactics per headline
# ===============================
df_few["num_tactics"] = df_few["true_vec"].apply(sum)
df_zero["num_tactics"] = df_zero["true_vec"].apply(sum)

# ===============================
# Compute Micro-F1 per density
# ===============================
def compute_micro_f1(df):
    scores = []
    for k in [1,2,3]:
        subset = df[df["num_tactics"] == k]
        if len(subset) == 0:
            scores.append(0)
            continue

        y_true = np.array(subset["true_vec"].tolist())
        y_pred = np.array(subset["pred_vec"].tolist())

        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        scores.append(score)

    return scores

few_f1 = compute_micro_f1(df_few)
zero_f1 = compute_micro_f1(df_zero)

# ===============================
# Plot
# ===============================
plt.figure(figsize=(7,5))

plt.plot([1,2,3], zero_f1, marker='o', label="GPT Zero-Shot")
plt.plot([1,2,3], few_f1, marker='s', label="GPT Few-Shot")

# annotate values
for i in range(3):
    plt.text(i+1, zero_f1[i]+0.005, f"{zero_f1[i]:.3f}")
    plt.text(i+1, few_f1[i]-0.001, f"{few_f1[i]:.3f}")

plt.title("Effect of Rhetorical Density on Multi-Label Performance (GPT)")
plt.xlabel("Number of Tactics in Headline")
plt.ylabel("Micro-F1 Score")
plt.xticks([1,2,3])
plt.ylim(0.24, 0.45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
