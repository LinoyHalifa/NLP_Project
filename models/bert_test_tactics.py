import pandas as pd
import numpy as np
import torch

from ast import literal_eval
from sklearn.metrics import f1_score, classification_report

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

TACTIC_NAMES = [
    "Curiosity Gap",
    "Exaggeration",
    "Emotional Trigger",
    "Sensationalism",
    "Lists/Superlatives",
    "Ambiguous References",
    "Direct Appeals",
    "Unfinished Narratives",
    "Unexpected Associations",
    "Provocative Questions"
]




# ======================================================
# CONFIG
# ======================================================
TEST_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_test.csv"
MODEL_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_tactics_pairwise"
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# 1. Load test dataset
# ======================================================
df = pd.read_csv(TEST_FILE)

df["original"] = df["original"].astype(str)
df["clickbait"] = df["clickbait"].astype(str)
df["methods_vector"] = df["methods_vector"].apply(literal_eval)

X_orig = df["original"]
X_click = df["clickbait"]
y_true = np.array(df["methods_vector"].tolist())

num_labels = y_true.shape[1]

print("Test size:", len(df))
print("Number of tactics:", num_labels)

# ======================================================
# 2. Load tokenizer & model
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

model = BertForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

model.to(DEVICE)
model.eval()

# ======================================================
# 3. Tokenization (PAIRWISE)
# ======================================================
def tokenize_pairwise(orig, click):
    return tokenizer(
        orig.tolist(),
        click.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

encodings = tokenize_pairwise(X_orig, X_click)

# ======================================================
# 4. Dataset class (TEST)
# ======================================================
class PairwiseTestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

test_dataset = PairwiseTestDataset(encodings, y_true)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================================================
# 5. Inference
# ======================================================
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()

        all_logits.append(logits)
        all_labels.append(labels)

all_logits = np.vstack(all_logits)
all_labels = np.vstack(all_labels)

# ======================================================
# 6. Metrics (SAME AS TRAINING)
# ======================================================
probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
preds = (probs > 0.5).astype(int)

micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)

print("Micro F1:", round(micro_f1, 4))
print("Macro F1:", round(macro_f1, 4))


# ======================================================
# SAVE PREDICTIONS WITH TRUE LABELS
# ======================================================

pred_df = pd.DataFrame({
    "y_true": list(all_labels),
    "y_pred": list(preds)
})

PRED_SAVE_PATH = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_predictions_test.csv"

pred_df.to_csv(PRED_SAVE_PATH, index=False)

print(f"Saved predictions to: {PRED_SAVE_PATH}")



# ======================================================
# 7. SAVE OVERALL F1 METRICS (MICRO / MACRO)
# ======================================================

overall_f1_metrics = {
    "micro_f1": micro_f1,
    "macro_f1": macro_f1
}

overall_f1_df = pd.DataFrame([overall_f1_metrics])

overall_f1_df.to_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_overall_f1_metrics_test.csv",
    index=False
)

print("Saved overall F1 metrics (micro / macro) to CSV")



# ======================================================
# 8. PER-TACTIC METRICS (LIKE GEMINI)
# ======================================================

print("\n===== PER-TACTIC METRICS (BERT Pairwise) =====")

per_tactic_rows = []

for i, tactic_name in enumerate(TACTIC_NAMES):
    y_true_i = all_labels[:, i]
    y_pred_i = preds[:, i]

    precision_i = precision_score(y_true_i, y_pred_i, zero_division=0)
    recall_i = recall_score(y_true_i, y_pred_i, zero_division=0)
    f1_i = f1_score(y_true_i, y_pred_i, zero_division=0)

    per_tactic_rows.append({
        "tactic": tactic_name,
        "precision": precision_i,
        "recall": recall_i,
        "f1": f1_i,
        "support": int(y_true_i.sum())
    })

per_tactic_df = pd.DataFrame(per_tactic_rows)
print(per_tactic_df)

per_tactic_df.to_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_per_tactic_metrics_test.csv",
    index=False
)

# ======================================================
# 9. TACTIC METRICS (MACRO) — LIKE GEMINI
# ======================================================

tactic_metrics = {
    "tactics_precision_macro": precision_score(
        all_labels, preds, average="macro", zero_division=0
    ),
    "tactics_recall_macro": recall_score(
        all_labels, preds, average="macro", zero_division=0
    ),
    "tactics_f1_macro": f1_score(
        all_labels, preds, average="macro", zero_division=0
    )
}

print("\n===== TACTIC METRICS (MACRO) =====")
print(tactic_metrics)


pd.DataFrame([tactic_metrics]).to_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_overall_tactic_metrics_test.csv",
    index=False
)

