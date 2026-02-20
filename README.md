NLP Project Name: Beyond Binary Clickbait Detection: Tactic-Level Attribution for Explainable Headline Analysis

📌 Project Overview

This project presents a two-stage Natural Language Processing (NLP) framework for:

Binary Clickbait Detection – Classifying whether a headline is clickbait or non-clickbait.

Tactic Attribution – Identifying which rhetorical clickbait tactics are used in a headline (multi-label classification, up to 3 tactics per headline).

The project combines:

Real news headlines

Synthetic clickbait generation using GPT

Supervised fine-tuning of BERT

Zero-shot and Few-shot evaluation using GPT and Gemini models

The goal is to compare supervised learning (BERT) against large language models (LLMs) under zero-shot and few-shot settings.

📂 Data Sources
1️⃣ Kaggle Dataset (Real Headlines Only)

Dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

From this dataset:

Only the real news headlines were used.

Fake headlines were discarded.

2️⃣ Additional News Dataset

news_data.csv

Merged with Kaggle real headlines.

Duplicate headlines were removed.

🧹 Data Preprocessing

Merged datasets into a unified dataset.

Removed duplicate headlines.

Split the dataset:

80% Training + Validation

20% Test

Prevented data leakage by splitting before synthetic clickbait generation.

🤖 Synthetic Clickbait Generation

Clickbait headlines were generated using GPT-4o with controlled tactic prompting.

Generation Constraints:

Each synthetic headline could include up to 3 clickbait tactics

Minimum: 1 tactic

Maximum: 3 tactics

This ensured:

Multi-label structure

Realistic rhetorical combinations

Controlled tactic attribution

The generated file:

ClickBait_generated.csv
🧠 Clickbait Tactics (10 Labels)

The project includes 10 rhetorical clickbait tactics:

Curiosity Gap

Exaggeration

Emotional Triggers

Sensationalism

Lists / Superlatives

Ambiguous References

Direct Appeals

Unfinished Narratives

Unexpected Associations

Provocative Questions

Each clickbait headline may contain 1–3 tactics.

🏗 Model Training
Stage 1: BERT – Binary Classification

Task:
Classify headline as:

Clickbait

Non-clickbait

Data Split (within training set):

85% Training

15% Validation

Evaluation performed on the held-out 20% test set.

Stage 2: BERT – Multi-Label Tactic Attribution

Task:
Predict which of the 10 tactics appear in the headline.

Multi-label classification

Sigmoid output layer

Same 85% / 15% training-validation split

Tested on the same 20% held-out test set

🔍 LLM Evaluation (Zero-shot & Few-shot)

The same held-out test set was evaluated using:

GPT (Few-shot & Zero-shot)

Tasks:

Binary clickbait detection

Multi-label tactic attribution

Gemini 2.5 (Few-shot & Zero-shot)

Tasks:

Binary clickbait detection

Multi-label tactic attribution

This allows direct comparison between:

Fine-tuned supervised model (BERT)

Large language models without training (zero-shot)

LLMs with minimal guidance (few-shot)

📊 Evaluation Strategy

Evaluation was performed on the same test file for all models.

Binary Classification Metrics:

Accuracy

Precision

Recall

F1-score

Multi-label Tactic Metrics:

Macro F1

Micro F1

Per-class F1

Exact match accuracy

🧩 Full Pipeline Summary

Collect real headlines (Kaggle + additional dataset)

Remove duplicates

Split 80/20 (train+val / test)

Generate synthetic clickbait with GPT (max 3 tactics per headline)

Train BERT (binary)

Train BERT (multi-label tactics)

Evaluate BERT on test set

Run GPT (zero-shot & few-shot) on same test set

Run Gemini (zero-shot & few-shot) on same test set

Compare performance across all systems

🎯 Research Contribution

This project provides:

A controlled synthetic clickbait generation framework

A 10-tactic multi-label taxonomy

Direct comparison between:

Supervised fine-tuned models

Zero-shot LLMs

Few-shot LLMs

Analysis of tactic-level attribution behavior across models

It demonstrates the strengths and limitations of:

Fine-tuned transformer classifiers

Prompt-based reasoning in LLMs

Multi-label rhetorical modeling

🛠 Technologies Used

Python

PyTorch

HuggingFace Transformers

BERT

GPT-4o

Gemini 2.5

Pandas / NumPy / Sklearn

🚀 Future Work

Improve tactic-level calibration

Expand tactic taxonomy

Explore contrastive learning

Investigate explainability alignment between BERT and LLM reasoning

If you'd like, I can also:

Write an academic-style version (for thesis submission)

Write a short GitHub-friendly version

Add architecture diagrams section

Add reproducibility instructions (requirements.txt, environment setup)

Add results table template

Just tell me which format you need.