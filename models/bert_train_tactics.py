import pandas as pd
import numpy as np
import torch

from ast import literal_eval
from sklearn.model_selection import train_test_split


from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from torch.utils.data import Dataset
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
DATA_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_train_val.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
RANDOM_SEED = 42

SAVE_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_tactics_pairwise"

# ======================================================
# 1. Load dataset
# ======================================================
df = pd.read_csv(DATA_FILE)

df["original"] = df["original"].astype(str)
df["clickbait"] = df["clickbait"].astype(str)
df["methods_vector"] = df["methods_vector"].apply(literal_eval)

X_orig = df["original"]
X_click = df["clickbait"]
y = np.array(df["methods_vector"].tolist())

num_labels = y.shape[1]

print("Dataset size:", len(df))
print("Number of tactics:", num_labels)
print("Tactic distribution:", y.sum(axis=0))

# ======================================================
# 2. Train / Validation split
# ======================================================
Xo_train, Xo_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
    X_orig,
    X_click,
    y,
    test_size=0.15,
    random_state=RANDOM_SEED,
    shuffle=True
)

print("Train size:", len(Xo_train))
print("Validation size:", len(Xo_val))

# ======================================================
# 3. Tokenizer (PAIRWISE)
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_pairwise(orig, click):
    return tokenizer(
        orig.tolist(),
        click.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

train_enc = tokenize_pairwise(Xo_train, Xc_train)
val_enc   = tokenize_pairwise(Xo_val, Xc_val)

# ======================================================
# 4. Dataset class
# ======================================================
class PairwiseTacticsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = PairwiseTacticsDataset(train_enc, y_train)
val_dataset   = PairwiseTacticsDataset(val_enc, y_val)

# ======================================================
# 5. Model (MULTI-LABEL)
# ======================================================
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# ======================================================
# 6. Metrics
# ======================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        # === SAME AS GEMINI (TACTICS) ===
        "tactics_precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "tactics_recall_macro": recall_score(
            labels, preds, average="macro", zero_division=0
        ),
        "tactics_f1_macro": f1_score(
            labels, preds, average="macro", zero_division=0
        ),

        # === EXTRA (useful, not mandatory) ===
        "micro_f1": f1_score(
            labels, preds, average="micro", zero_division=0
        )
    }


# ======================================================
# 7. Training arguments
# ======================================================
training_args = TrainingArguments(
    output_dir="./bert_tactics_pairwise_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=1,
    report_to="none"
)

# ======================================================
# 8. Trainer
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ======================================================
# 9. Train
# ======================================================
trainer.train()


# ======================================================
# PLOT TRAIN / VALIDATION LOSS PER EPOCH
# ======================================================

import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_epochs = []
train_losses = []
eval_epochs = []
eval_losses = []

for log in log_history:
    if "loss" in log and "epoch" in log and "eval_loss" not in log:
        train_epochs.append(log["epoch"])
        train_losses.append(log["loss"])

    if "eval_loss" in log and "epoch" in log:
        eval_epochs.append(log["epoch"])
        eval_losses.append(log["eval_loss"])

plt.figure(figsize=(6, 4))
plt.plot(train_epochs, train_losses, label="Training Loss")
plt.plot(eval_epochs, eval_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Pairwise Tactics)")
plt.legend()
plt.grid(True)

LOSS_PLOT_PATH = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_tactics_loss_curve.png"

plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.show()

print(f"Loss curve saved to: {LOSS_PLOT_PATH}")

# ======================================================
# 10. SAVE OVERALL TACTIC METRICS (MACRO / MICRO) TO CSV
# ======================================================

overall_metrics = trainer.evaluate(eval_dataset=val_dataset)

# Keep only tactic-related metrics
overall_metrics_clean = {
    "tactics_precision_macro": overall_metrics.get("eval_tactics_precision_macro"),
    "tactics_recall_macro": overall_metrics.get("eval_tactics_recall_macro"),
    "tactics_f1_macro": overall_metrics.get("eval_tactics_f1_macro"),
    "micro_f1": overall_metrics.get("eval_micro_f1")
}

overall_metrics_df = pd.DataFrame([overall_metrics_clean])

OVERALL_METRICS_PATH = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_overall_metrics.csv"

overall_metrics_df.to_csv(OVERALL_METRICS_PATH, index=False)

print(f"Overall tactic metrics saved to: {OVERALL_METRICS_PATH}")




# ======================================================
# 11. Save model
# ======================================================
trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Pairwise tactics model saved to: {SAVE_DIR}")

# ======================================================
# 11. Inference on validation set (for per-tactic metrics)
# ======================================================

pred_output = trainer.predict(val_dataset)

logits = pred_output.predictions
labels = pred_output.label_ids

probs = torch.sigmoid(torch.tensor(logits)).numpy()
preds = (probs > 0.5).astype(int)

# ======================================================
# 12. PER-TACTIC METRICS (LIKE GEMINI)
# ======================================================

print("\n===== PER-TACTIC METRICS (BERT Pairwise) =====")

per_tactic_rows = []

for i, tactic_name in enumerate(TACTIC_NAMES):
    y_true_i = labels[:, i]
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
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\tactics\bert_pairwise_per_tactic_metrics.csv",
    index=False
)



