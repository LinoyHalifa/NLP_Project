import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

from sklearn.metrics import precision_score, recall_score, f1_score
print("RUNNING FILE:", __file__)




# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "gemini-2.5-flash"
MAX_SAMPLES = 5000

PRED_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_preds_few_shot.csv"
CLICK_METRICS_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_clickbait_metrics_few_shot.csv"
TACTIC_METRICS_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_tactic_metrics_few_shot.csv"
CONFUSION_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_confusion_few_shot.png"


# ======================================================
# TACTIC NAMES (ORDER MUST MATCH DATASET VECTOR)
# ======================================================
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
# LOAD DATASET (TEST ONLY)
# ======================================================
df = pd.read_csv(
    r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\evaluation\clickbait_test_GT_shuffled.csv"
)



X = df["text"].tolist()
y = df["label"].tolist()
tactics_gt = df["tactics"].tolist()

print("X ORIGIN CHECK:")
print(X[:5])


assert len(X) == len(y) == len(tactics_gt)

if len(X) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    X = df["text"].tolist()
    y = df["label"].tolist()
    tactics_gt = df["tactics"].tolist()

print("Running on samples:", len(X))


# ======================================================
# FEW-SHOT EXAMPLES
# ======================================================

FEW_SHOT = """
Example 1:
Headline: "va chief presses congress to make it easier to fire workers for misconduct"
Output:
{"is_clickbait": 0}

Example 2:
Headline: "southeast asian ministers urge north korea to rein in weapons programs"
Output:
{"is_clickbait": 0}

Example 3:
Headline: "as obamacare repeal falters, insurers start to press on subsidies"
Output:
{"is_clickbait": 0}

Example 4:
Headline: "trump expected to endorse ryan for re-election later on friday: fox news"
Output:
{"is_clickbait": 0}

Example 5:
Headline: "the looming deadline that could change northern ireland's future"
Output:
{"is_clickbait": 1}

Example 6:
Headline: "can britain really resolve the northern ireland impasse"
Output:
{"is_clickbait": 1}

Example 7:
Headline: "what iceland's election results could mean for you"
Output:
{"is_clickbait": 1}

Example 8:
Headline: "the surprising move by fired u.s. attorney bharara that everyone is talking about"
Output:
{"is_clickbait": 1}
"""


# ======================================================
# TACTIC FEW-SHOT EXAMPLES
# ======================================================

TACTIC_FEW_SHOT = """
Example 1:
Headline: "the looming deadline that could change northern ireland's future"
Output:
{
  "methods_vector": [0,0,1,0,0,0,0,0,0,0] 
}

Example 2:
Headline: "can britain really resolve the northern ireland impasse"
Output:
{
  "methods_vector": [0,0,0,0,0,0,0,0,0,1] 
}

Example 3:
Headline: "what iceland's election results could mean for you"
Output:
{
  "methods_vector": [0,0,1,0,0,0,1,0,0,0] 
}

Example 4:
Headline: "the surprising move by fired u.s. attorney bharara that everyone is talking about"
Output:
{
  "methods_vector": [0,1,1,1,0,0,0,0,0,0]  
}

Example 5:
Headline: "what will happen if trump refuses to release his tax returns"
Output:
{
  "methods_vector": [0,0,0,0,0,0,0,0,0,1] 
}

Example 6:
Headline: "is the looming election the nail in the coffin for japan's fiscal reform"
Output:
{
  "methods_vector": [0,0,0,0,1,0,0,0,1,1]  
}

Example 7:
Headline: "the impact of u.s. strikes on syria's operational aircraft is far-reaching"
Output:
{
  "methods_vector": [0,0,0,1,0,0,0,0,1,0]  
}

Example 8:
Headline: "the shocking details behind the senate's advance of ross and chao"
Output:
{
  "methods_vector": [0,0,0,1,1,1,0,0,0,0] 
}
"""





# ======================================================
# MODEL
# ======================================================
model = genai.GenerativeModel("gemini-2.5-flash")


# ======================================================
# JSON SAFE PARSER
# ======================================================
def extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))


# ======================================================
# STAGE A — CLICKBAIT BINARY
# ======================================================
def gemini_stage_a_clickbait(text):
    prompt = f"""
You are an expert analyst for binary clickbait detection.

Your task is to determine whether a headline is clickbait.

--------------------------------------------------
FEW-SHOT EXAMPLES:

{FEW_SHOT}

--------------------------------------------------
Definition:

A clickbait headline is one that intentionally uses
attention-manipulating techniques to induce clicks
beyond straightforward news reporting.

A non-clickbait headline presents information
in a clear, factual, and journalistic manner.

--------------------------------------------------
Analysis instructions (MANDATORY):

Analyze the headline carefully and objectively.

Consider whether it contains any of the following:
- Withholding of key information that creates curiosity
- Framing designed to provoke anticipation or intrigue
- Emphasis or wording that encourages the reader to click
  rather than simply informing
- Narrative hooks or suggestive phrasing

Also consider whether the headline:
- Clearly states the event or outcome
- Matches professional newswire style
- Does not rely on curiosity-driven framing

--------------------------------------------------
Decision rule:

- Classify as clickbait (1) if the headline appears
  intentionally designed to attract clicks
  through curiosity or engagement tactics.
- Classify as non-clickbait (0) if it primarily
  serves an informational purpose.



--------------------------------------------------
Return STRICT JSON only:
{{
  "is_clickbait": 0 or 1
}}

Headline:
"{text}"

"""
    try:
        response = model.generate_content(prompt)
        raw = response.candidates[0].content.parts[0].text
        data = extract_json(raw)
        return int(data["is_clickbait"])
    except Exception as e:
        print("Stage A error:", e)
        return 1   # fallback: recall-oriented


# ======================================================
# STAGE B — TACTIC ATTRIBUTION
# ======================================================
def gemini_stage_b_tactics(text):
    prompt = f"""
You are an expert analyst for clickbait tactic attribution.

--------------------------------------------------
CLICKBAIT TACTICS (FIXED SET AND ORDER):

You MUST choose from the following 10 tactics ONLY.
Each tactic corresponds to a fixed index in the output vector.

Index 0: Curiosity Gap:
Create explicit information withholding by signaling that a specific but unnamed piece of knowledge is missing.
Required cues:
- Phrases such as “what you don’t see”, “behind the scenes”, “this detail”, “what remains hidden”.

Index 1: Exaggeration:
Amplify importance, scale, or impact using intensity modifiers without adding new facts.
Required cues:
- Intensity modifiers such as “major”, “dramatic”, “significant”, “unprecedented”.


Index 2: Emotional Triggers:
Evoke a specific emotion (fear, concern, outrage, hope) through explicit emotional wording.
Required cues:
- Clear emotional terms such as “fear”, “outrage”, “concern”, “anger”, “hope”.


Index 3: Sensationalism:
Create dramatic impact through heightened framing or spectacle without emotional wording.
Required cues:
- Strong dramatic framing suggesting shock or spectacle.

Index 4: Lists / Superlatives:
Present information using list-based structures or extreme ranking language.
Required cues:
- Explicit list or ranking terms such as “top”, “first”, “largest”, “most”.

Index 5: Ambiguous References:
Use deliberately vague or non-specific references to create uncertainty.
Required cues:
- Indefinite references such as “this”, “they”, “something”, “a certain move”.

Index 6: Direct Appeals:
Address the reader or a specific audience directly to prompt engagement.
Required cues:
- Direct address such as “you”, “voters”, “investors”, “parents”.


Index 7: Unfinished Narratives:
Present the event as ongoing or unresolved, emphasizing that the outcome is not yet known.
Required cues:
- Explicit continuation signals such as “what happens next”, “the outcome remains unclear”.

Index 8: Unexpected Associations:
Explicitly link two concepts or domains that are not commonly connected.
Required cues:
- Clear mention of both elements and their surprising linkage.

Index 9: Provocative Questions:
Frame the headline as an implicit or explicit challenge to an assumption.
Required cues:
- Question-like syntactic structure, even without a question mark.



--------------------------------------------------
FEW-SHOT TACTICS EXAMPLES:
{TACTIC_FEW_SHOT}


Example 1:
Headline: "the looming deadline that could change northern ireland's future"
Output:
{{
  "methods_vector": [0,0,1,0,0,0,0,0,0,0] 
}}

Example 2:
Headline: "can britain really resolve the northern ireland impasse"
Output:
{{
  "methods_vector": [0,0,0,0,0,0,0,0,0,1] 
}}

Example 3:
Headline: "what iceland's election results could mean for you"
Output:
{{
  "methods_vector": [0,0,1,0,0,0,1,0,0,0] 
}}

Example 4:
Headline: "the surprising move by fired u.s. attorney bharara that everyone is talking about"
Output:
{{
  "methods_vector": [0,1,1,1,0,0,0,0,0,0]  
}}

Example 5:
Headline: "what will happen if trump refuses to release his tax returns"
Output:
{{
  "methods_vector": [0,0,0,0,0,0,0,0,0,1] 
}}

Example 6:
Headline: "is the looming election the nail in the coffin for japan's fiscal reform"
Output:
{{
  "methods_vector": [0,0,0,0,1,0,0,0,1,1]  
}}

Example 7:
Headline: "the impact of u.s. strikes on syria's operational aircraft is far-reaching"
Output:
{{
  "methods_vector": [0,0,0,1,0,0,0,0,1,0]  
}}

Example 8:
Headline: "the shocking details behind the senate's advance of ross and chao"
Output:
{{
  "methods_vector": [0,0,0,1,1,1,0,0,0,0] 
}}




--------------------------------------------------
TASK DEFINITION:

Your task is to identify which of the above clickbait tactics
are used in the headline below.

IMPORTANT:
- The headline HAS ALREADY been classified as clickbait.
- This task must NOT be executed for non-clickbait headlines.

--------------------------------------------------
CRITICAL LOGICAL CONSTRAINT (MANDATORY):

Because a headline is classified as clickbait ONLY if it uses
attention-manipulating tactics, you MUST select AT LEAST ONE tactic.

It is NOT allowed to return zero tactics.

--------------------------------------------------
TACTIC SELECTION RULES:

- Select BETWEEN 1 AND 3 tactics.
- Select a tactic (1) ONLY if there is CLEAR textual evidence.
- If more than 3 tactics appear applicable,
  select ONLY the 3 MOST DOMINANT ones.
- Ignore weak, indirect, or speculative signals.
- Every selected tactic must be defensible based on the text.

--------------------------------------------------
OUTPUT FORMAT (STRICT):

Return a VALID JSON object with EXACTLY this structure:

{{
  "methods_vector": [0 or 1, 0 or 1, 0 or 1, 0 or 1, 0 or 1,
                     0 or 1, 0 or 1, 0 or 1, 0 or 1, 0 or 1]
}}


--------------------------------------------------
OUTPUT CONSTRAINTS:

- The vector MUST contain BETWEEN 1 AND 3 ones.
- The vector MUST follow the EXACT index order above.
- No explanations, comments, or text outside JSON.

--------------------------------------------------
Headline:
"{text}"


"""
    try:
        response = model.generate_content(prompt)
        raw = response.candidates[0].content.parts[0].text
        data = extract_json(raw)

        return data["methods_vector"]



    except Exception as e:
        raise RuntimeError(f"Stage B failed: {e}")


# ======================================================
# SANITY CHECK (RUN ON 5 SAMPLES ONLY)
# ======================================================
print("\n===== SANITY CHECK (FIRST 5 SAMPLES) =====\n")

for x in X[:5]:
    print("Headline:", x)
    lbl = gemini_stage_a_clickbait(x)
    print("Clickbait:", lbl)

    if lbl == 1:
        print("Tactics:", gemini_stage_b_tactics(x))

    print("-" * 80)

print("\n===== SANITY CHECK DONE =====\n")




# ======================================================
# RUN TWO-STAGE PIPELINE
# ======================================================
pred_labels = []
pred_tactics = []

for headline in tqdm(X, desc="Gemini Two-Stage"):
    label = gemini_stage_a_clickbait(headline)
    pred_labels.append(label)

    if label == 1:
        tactics = gemini_stage_b_tactics(headline)
    else:
        tactics = [0] * len(TACTIC_NAMES)

    pred_tactics.append(tactics)


# ======================================================
# SAVE PREDICTIONS
# ======================================================
pd.DataFrame({
    "text": X,
    "true_clickbait": y,
    "pred_clickbait": pred_labels,
    "true_tactics_vector": tactics_gt,
    "pred_tactics": pred_tactics
}).to_csv(PRED_OUT, index=False)

print("Saved:", PRED_OUT)


# ======================================================
# CLICKBAIT METRICS
# ======================================================
clickbait_metrics = {
    "accuracy": accuracy_score(y, pred_labels),
    "precision": precision_score(y, pred_labels),
    "recall": recall_score(y, pred_labels),
    "f1": f1_score(y, pred_labels)
}

print("\n===== CLICKBAIT METRICS =====")
print(clickbait_metrics)

pd.DataFrame([clickbait_metrics]).to_csv(CLICK_METRICS_OUT, index=False)
print("Saved:", CLICK_METRICS_OUT)


# ======================================================
# CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y, pred_labels)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Gemini Two-Stage")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONFUSION_OUT)
plt.show()

print("Saved:", CONFUSION_OUT)



def normalize_tactics_vector(vec, target_len):
    vec = list(map(int, vec))
    if len(vec) < target_len:
        vec = vec + [0] * (target_len - len(vec))
    elif len(vec) > target_len:
        vec = vec[:target_len]
    return vec



# ======================================================
# TACTIC UTILITIES
# ======================================================

import ast
import numpy as np

def tactics_to_binary_matrix(tactics_list, tactic_names):
    binary = []

    for t in tactics_list:
        if isinstance(t, str):
            vec = ast.literal_eval(t)
            vec = normalize_tactics_vector(vec, len(tactic_names))
            binary.append(vec)
            continue

        if isinstance(t, (list, tuple)):
            vec = normalize_tactics_vector(t, len(tactic_names))
            binary.append(vec)
            continue

        raise ValueError(f"Unexpected tactics format: {t}")

    return np.array(binary)

# ======================================================
# TACTIC METRICS (CLICKBAIT ONLY)
# ======================================================


# evaluate tactics only on GT clickbait samples
mask = np.array(y) == 1

y_true_tactics = tactics_to_binary_matrix(
    [tactics_gt[i] for i in range(len(tactics_gt)) if mask[i]],
    TACTIC_NAMES
)

y_pred_tactics = tactics_to_binary_matrix(
    [pred_tactics[i] for i in range(len(pred_tactics)) if mask[i]],
    TACTIC_NAMES
)


# ======================================================
# PER-TACTIC METRICS
# ======================================================

per_tactic_rows = []

for i, tactic_name in enumerate(TACTIC_NAMES):
    y_true_i = y_true_tactics[:, i]
    y_pred_i = y_pred_tactics[:, i]

    precision_i = precision_score(
        y_true_i, y_pred_i, zero_division=0
    )
    recall_i = recall_score(
        y_true_i, y_pred_i, zero_division=0
    )
    f1_i = f1_score(
        y_true_i, y_pred_i, zero_division=0
    )

    per_tactic_rows.append({
        "tactic": tactic_name,
        "precision": precision_i,
        "recall": recall_i,
        "f1": f1_i,
        "support": int(y_true_i.sum())
    })

per_tactic_df = pd.DataFrame(per_tactic_rows)

print("\n===== PER-TACTIC METRICS =====")
print(per_tactic_df)

PER_TACTIC_OUT = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\gemini\few\gemini_two_stage_per_tactic_metrics_few_shot.csv"
per_tactic_df.to_csv(PER_TACTIC_OUT, index=False)
print("Saved:", PER_TACTIC_OUT)



tactic_metrics = {
    "tactics_precision_macro": precision_score(
        y_true_tactics, y_pred_tactics,
        average="macro", zero_division=0
    ),
    "tactics_recall_macro": recall_score(
        y_true_tactics, y_pred_tactics,
        average="macro", zero_division=0
    ),
    "tactics_f1_macro": f1_score(
        y_true_tactics, y_pred_tactics,
        average="macro", zero_division=0
    )
}


tactic_metrics["tactics_f1_micro"] = f1_score(
    y_true_tactics, y_pred_tactics,
    average="micro", zero_division=0
)


print("\n===== TACTIC METRICS =====")
print(tactic_metrics)

pd.DataFrame([tactic_metrics]).to_csv(
    TACTIC_METRICS_OUT,
    index=False
)

print("Saved:", TACTIC_METRICS_OUT)
