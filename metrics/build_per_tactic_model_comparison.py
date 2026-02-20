import pandas as pd

rows = []

def load_per_tactic_csv(path, model_name):
    df = pd.read_csv(path)

    for _, row in df.iterrows():
        rows.append({
            "tactic": row["tactic"],
            "model": model_name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"]
        })

# ======================
# BERT
# ======================
load_per_tactic_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_per_tactic_metrics_test.csv",
    "BERT"
)

# ======================
# GEMINI
# ======================
load_per_tactic_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_per_tactic_metrics_few_shot.csv",
    "Gemini (Few-shot)"
)

load_per_tactic_csv(
     r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\zero\gemini_two_stage_per_tactic_metrics_zero_shot.csv",
    "Gemini (zero-shot)"
)


# ======================
# GPT
# ======================
load_per_tactic_csv(
     r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\few\gpt_two_stage_per_tactic_metrics_few_shot.csv",
    "GPT (Few-shot)"
 )

load_per_tactic_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\zero\gpt_two_stage_per_tactic_metrics_zero_shot.csv",
    "GPT (Zero-shot)"
)

# ======================
# SAVE
# ======================
comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\Results\tactics_per_tactic_model_comparison.csv", index=False)

print("Saved: tactics_per_tactic_model_comparison.csv")
print(comparison_df.head())
