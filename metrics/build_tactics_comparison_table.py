import pandas as pd

rows = []

def load_tactic_metrics(path, model_name):
    df = pd.read_csv(path)
    rows.append({
        "model": model_name,
        "tactics_precision_macro": df["tactics_precision_macro"].iloc[0],
        "tactics_recall_macro": df["tactics_recall_macro"].iloc[0],
        "tactics_f1_macro": df["tactics_f1_macro"].iloc[0],   # <-- F1 גדול
        "tactics_f1_micro": df["tactics_f1_micro"].iloc[0],   # <-- F1 גדול
    })



# ======================
# BERT
# ======================
load_tactic_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_overall_tactic_metrics_test.csv",
    "BERT"
)

# ======================
# GEMINI
# ======================
load_tactic_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_tactic_metrics_few_shot.csv",
    "Gemini (Few-shot)"
)

load_tactic_metrics(
     r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\zero\gemini_two_stage_tactic_metrics_zero_shot.csv",
     "Gemini (Zero-shot)"
)

# ======================
# GPT
# ======================
load_tactic_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\few\gpt_two_stage_tactic_metrics_few_shot.csv",
    "GPT (Few-shot)"
)

load_tactic_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\zero\gpt_two_stage_tactic_metrics_zero_shot.csv",
    "GPT (Zero-shot)"
)

tactics_comparison_df = pd.DataFrame(rows)
tactics_comparison_df.to_csv(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\Results\tactics_macro_micro_comparison.csv", index=False)

print(tactics_comparison_df)
