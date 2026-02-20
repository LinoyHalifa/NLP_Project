import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# CONFIG
# ======================================================
TEST_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\evaluation\clickbait_test_GT_shuffled.csv"   # ← עדכני
MODEL_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_clickbait_finetuned"
OUTPUT_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\evaluation\test_clickbait_predictions.csv"

TEXT_COLUMN = "text"   # ← שם עמודת הכותרת
MAX_LEN = 8
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# 1. Load test data
# ======================================================
df = pd.read_csv(TEST_FILE)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

texts = df[TEXT_COLUMN].tolist()

print("Test samples:", len(texts))

# ======================================================
# 2. Load model & tokenizer
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# ======================================================
# 3. Dataset
# ======================================================
class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

dataset = TestDataset(texts)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# 4. Inference
# ======================================================
all_probs = []
all_preds = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Running BERT clickbait inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]   # probability of clickbait
        preds = (probs > 0.5).long()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ======================================================
# 5. Save results
# ======================================================
df["clickbait_prob"] = all_probs
df["clickbait_pred"] = all_preds

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved predictions to: {OUTPUT_FILE}")

# ======================================================
# 6. Evaluation metrics (TEST)
# ======================================================
y_true = df["label"].values
y_pred = np.array(all_preds)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)


# ======================================================
# 7. Save metrics to CSV
# ======================================================

METRICS_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\bert_binary_clickbait_test_metrics.csv"

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv(METRICS_OUT, index=False)

print(f"Saved metrics to: {METRICS_OUT}")


print("\n=== TEST METRICS (Clickbait Detection) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ======================================================
# 8. Confusion Matrix (Plot + Save)
# ======================================================

CONFUSION_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\bert_binary_clickbait_confusion_matrix.png"

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Clickbait", "Clickbait"],
    yticklabels=["Non-Clickbait", "Clickbait"]
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – BERT Clickbait Detection")
plt.tight_layout()

plt.savefig(CONFUSION_OUT, dpi=300)
plt.show()

print(f"Saved confusion matrix to: {CONFUSION_OUT}")


