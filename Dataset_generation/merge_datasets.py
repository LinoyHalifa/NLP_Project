import pandas as pd

print("Loading files...")

# Load original dataset (from the Clickbait project)
original_df = pd.read_csv("news_data.csv")

# Load the cleaned extended dataset you created
extended_df = pd.read_csv("True_Titles.csv")

print(f"Original dataset size: {len(original_df)}")
print(f"Extended dataset size: {len(extended_df)}")

# Make sure column name matches format
# original_df must contain a column "title" or "headline"
# extended_df contains a column "title"

# Rename original if needed
if "title" not in original_df.columns:
    if "headline" in original_df.columns:
        original_df = original_df.rename(columns={"headline": "title"})
    else:
        raise ValueError("Cannot find 'title' or 'headline' column in news_data.csv")

# Add source identifier for clarity
original_df["source_tag"] = "original"
extended_df["source_tag"] = "extended"

# Concatenate both
combined = pd.concat([original_df[["title", "source_tag"]],
                      extended_df[["title", "source_tag"]]],
                     ignore_index=True)

print(f"After merging: {len(combined)}")

# Remove duplicates
combined = combined.drop_duplicates(subset=["title"])
print(f"After removing duplicates: {len(combined)}")

# Shuffle for training stability
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final merged file
combined.to_csv("combined_news_dataset.csv", index=False)

print("\nDone! Saved combined dataset:")
print("combined_news_dataset.csv")
