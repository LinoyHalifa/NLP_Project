import pandas as pd

rows = []

def load_binary_metrics(path, model_name):
    df = pd.read_csv(path)
    row = {
        "model": model_name,
        "accuracy": df["accuracy"].iloc[0],
        "precision": df["precision"].iloc[0],
        "recall": df["recall"].iloc[0],
        "f1": df["f1"].iloc[0]
    }
    rows.append(row)

# ======================
# BERT
# ======================
load_binary_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\bert_binary_clickbait_test_metrics.csv",
    "BERT"
)

# ======================
# GEMINI
# ======================
load_binary_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_clickbait_metrics_few_shot.csv",
    "Gemini (Few-shot)"
)

load_binary_metrics(
     r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\zero\gemini_two_stage_clickbait_metrics_zero_shot.csv",
     "Gemini (Zero-shot)"
)

# ======================
# GPT
# ======================
load_binary_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\few\gpt_two_stage_clickbait_metrics_few_shot.csv",
    "GPT (Few-shot)"
)

load_binary_metrics(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\GPT\zero\gpt_two_stage_clickbait_metrics_zero_shot.csv",
    "GPT (Zero-shot)"
)

# ======================
# FINAL TABLE
# ======================
binary_results_df = pd.DataFrame(rows)
binary_results_df.to_csv(r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\Results\binary_clickbait_comparison.csv", index=False)

print(binary_results_df)
