import pandas as pd
import numpy as np
import ast
import re

# ======================================================
# TACTIC NAMES
# ======================================================
TACTIC_NAMES = [
    "Curiosity Gap",
    "Exaggeration",
    "Emotional Triggers",
    "Sensationalism",
    "Lists/Superlatives",
    "Ambiguous References",
    "Direct Appeals",
    "Unfinished Narratives",
    "Unexpected Associations",
    "Provocative Questions"
]

# ======================================================
# PATHS
# ======================================================
bert_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_predictions_test.csv"
gpt_few_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\few\gpt_two_stage_preds_few_shot.csv"
gpt_zero_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\zero\gpt_two_stage_preds_zero_shot.csv"
gemini_few_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_preds_few_shot.csv"
gemini_zero_path = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\zero\gemini_two_stage_preds_zero_shot.csv"

# ======================================================
# LOAD
# ======================================================
bert = pd.read_csv(bert_path)
gpt_few = pd.read_csv(gpt_few_path)
gpt_zero = pd.read_csv(gpt_zero_path)
gemini_few = pd.read_csv(gemini_few_path)
gemini_zero = pd.read_csv(gemini_zero_path)

min_len = min(len(bert), len(gpt_few), len(gpt_zero), len(gemini_few), len(gemini_zero))

bert = bert.iloc[:min_len].reset_index(drop=True)
gpt_few = gpt_few.iloc[:min_len].reset_index(drop=True)
gpt_zero = gpt_zero.iloc[:min_len].reset_index(drop=True)
gemini_few = gemini_few.iloc[:min_len].reset_index(drop=True)
gemini_zero = gemini_zero.iloc[:min_len].reset_index(drop=True)

# ======================================================
# PARSER
# ======================================================
def parse_vec(x):
    if isinstance(x, list):
        return list(map(int, x))
    if isinstance(x, str):
        try:
            return list(map(int, ast.literal_eval(x)))
        except:
            x = x.strip("[]").replace(",", " ")
            parts = [p for p in x.split() if p != ""]
            return [int(float(p)) for p in parts]
    return x

def normalize(vec):
    if vec is None or len(vec) == 0:
        return [0]*10
    return vec

def vec_to_tactics(vec):
    if sum(vec) == 0:
        return "Non-clickbait"
    return " + ".join([TACTIC_NAMES[i] for i, v in enumerate(vec) if v == 1])

# ======================================================
# APPLY PARSING
# ======================================================
bert_pred = bert["y_pred"].apply(parse_vec).apply(normalize)
gpt_true = gpt_few["true_tactics_vector"].apply(parse_vec).apply(normalize)
gpt_few_pred = gpt_few["pred_tactics"].apply(parse_vec).apply(normalize)
gpt_zero_pred = gpt_zero["pred_tactics"].apply(parse_vec).apply(normalize)
gemini_few_pred = gemini_few["pred_tactics"].apply(parse_vec).apply(normalize)
gemini_zero_pred = gemini_zero["pred_tactics"].apply(parse_vec).apply(normalize)

# ======================================================
# SELECT REPRESENTATIVE EXAMPLES
# ======================================================
selected = []

for i in range(min_len):

    true_vec = np.array(gpt_true[i])
    b = np.array(bert_pred[i])
    llms = [
        np.array(gpt_few_pred[i]),
        np.array(gpt_zero_pred[i]),
        np.array(gemini_few_pred[i]),
        np.array(gemini_zero_pred[i])
    ]

    # 1. כולם צודקים
    if len(selected) == 0:
        if np.array_equal(b, true_vec) and all(np.array_equal(x, true_vec) for x in llms):
            selected.append(i)
            continue

    # 2. LLMs טובים יותר מ-BERT
    if len(selected) == 1:
        if not np.array_equal(b, true_vec) and sum(np.array_equal(x, true_vec) for x in llms) >= 3:
            selected.append(i)
            continue

    # 3. BERT טוב יותר
    if len(selected) == 2:
        if np.array_equal(b, true_vec) and all(not np.array_equal(x, true_vec) for x in llms):
            selected.append(i)
            continue

    # 4. Over prediction
    if len(selected) == 3:
        if sum(gpt_few_pred[i]) > sum(true_vec):
            selected.append(i)
            continue

    # 5. Non-clickbait confusion
    if len(selected) == 4:
        if sum(true_vec) == 0 and (sum(b) > 0 or any(sum(x) > 0 for x in llms)):
            selected.append(i)
            continue

    # 6. Multi-tactic case
    if len(selected) == 5:
        if sum(true_vec) >= 2:
            selected.append(i)
            break

# ======================================================
# BUILD TABLE
# ======================================================
rows = []

for idx in selected:
    rows.append({
        "Headline": gpt_few.loc[idx, "text"],
        "Ground Truth": vec_to_tactics(gpt_true[idx]),
        "BERT": vec_to_tactics(bert_pred[idx]),
        "GPT-Few": vec_to_tactics(gpt_few_pred[idx]),
        "GPT-Zero": vec_to_tactics(gpt_zero_pred[idx]),
        "Gemini-Few": vec_to_tactics(gemini_few_pred[idx]),
        "Gemini-Zero": vec_to_tactics(gemini_zero_pred[idx]),
    })

qual_table = pd.DataFrame(rows)

print("\n===== QUALITATIVE TABLE =====\n")
print(qual_table)

qual_table.to_csv("qualitative_examples_table.csv", index=False)

print("\nSaved: qualitative_examples_table.csv")