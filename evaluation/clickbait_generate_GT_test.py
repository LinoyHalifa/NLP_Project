import pandas as pd

# ===============================
# CONFIG
# ===============================
INPUT_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_test.csv"
OUTPUT_GT_FILE = "clickbait_test_GT_shuffled.csv"
RANDOM_SEED = 42

# ===============================
# 1. Load original dataset
# ===============================
df = pd.read_csv(INPUT_FILE)

print(f"Loaded dataset with {len(df)} rows")

# ===============================
# 2. Build GT rows
# ===============================



# כותרות אמיתיות → label = 0, tactics = []
df_real = pd.DataFrame({
    "text": df["original"],
    "label": 0,
    "tactics": [[] for _ in range(len(df))]
})

# כותרות קליקבייט → label = 1, tactics = הווקטור הקיים
df_clickbait = pd.DataFrame({
    "text": df["clickbait"],
    "label": 1,
    "tactics": df["methods_vector"]  # או איך שהעמודה נקראת אצלך במקור
})

# ===============================
# 3. Concatenate & shuffle
# ===============================
df_gt = pd.concat([df_real, df_clickbait], ignore_index=True)

df_gt = df_gt.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# ===============================
# 4. Save GT
# ===============================
df_gt.to_csv(OUTPUT_GT_FILE, index=False)

print("\n====================================")
print(f"GT file created: {OUTPUT_GT_FILE}")
print(f"Total samples: {len(df_gt)}")
print("Shuffled successfully")
print("====================================")
