import pandas as pd

df = pd.read_csv("neutralization_evaluation.csv")

# Select only LOW rows
df_low = df[df["LLM_Judge_Level"] == "low"]

# Keep only the columns needed for a fresh run
columns_for_retry = [
    "original",
    "clickbait",
    "methods_vector"
]

df_low_retry = df_low[columns_for_retry].copy()

print(f"LOW samples for retry: {len(df_low_retry)}")

df_low_retry.to_csv("low_only_retry_input.csv", index=False)

print("Saved low_only_retry_input.csv")
