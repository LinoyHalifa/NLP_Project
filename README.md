---

# 📝 Beyond Binary Clickbait Detection

### Tactic-Level Attribution for Explainable Headline Analysis

---

## 🎯 Project Overview

This project introduces a **two-stage NLP framework** designed to move beyond simple clickbait classification. While most systems stop at "Is this clickbait?", this framework identifies the **specific rhetorical tactics** used to manipulate readers, providing a layer of explainability to headline analysis.

### Key Features

* **Binary Classification:** High-accuracy detection of clickbait vs. non-clickbait.
* **Tactic Attribution:** Multi-label classification identifying up to 3 rhetorical tactics per headline.
* **Hybrid Dataset:** Combines real-world news with high-quality synthetic clickbait generated via GPT-4o.
* **Benchmarking:** A direct head-to-head comparison between **Fine-tuned BERT** and **State-of-the-Art LLMs** (Zero-shot & Few-shot).

---

## 🧠 The 10-Tactic Taxonomy

Each clickbait headline is analyzed through 10 distinct rhetorical lenses:

| Tactic | Description |
| --- | --- |
| **Curiosity Gap** | Withholding crucial information to force a click. |
| **Exaggeration** | Overpromising the significance of the content. |
| **Emotional Triggers** | Leveraging fear, joy, or anger. |
| **Sensationalism** | Using shocking language for mundane facts. |
| **Lists / Superlatives** | "10 things you won't believe..." |
| **Ambiguous Refs** | Using "This" or "He/She" without context. |
| **Direct Appeals** | Explicitly telling the user to "Click here" or "Watch". |
| **Unfinished Narratives** | Starting a story and cutting it off. |
| **Unexpected Assoc.** | Linking two unrelated topics for shock value. |
| **Provocative Qs** | Questions where the answer is usually "No". |

---

## 🏗 System Architecture

The pipeline is divided into four distinct phases:

### 1. Data Engineering

* **Sources:** Kaggle (Real News) + Merged news_data.csv.
* **Synthetic Generation:** GPT-4o controlled prompting to create multi-label clickbait (1-3 tactics per sample).
* **Integrity:** Strict 80/20 split **before** generation to prevent data leakage.

### 2. Supervised Learning (BERT)

* **Stage 1:** Binary Classifier (Clickbait vs. Real).
* **Stage 2:** Multi-label Tactic Predictor (Sigmoid output layer).
* **Training:** Fine-tuned on 85% of the training set with 15% validation.

### 3. LLM Benchmarking

* **Models:** GPT-4o, Gemini 2.5.
* **Settings:** Zero-shot (no examples) vs. Few-shot (context-aware).
* **Evaluation:** Tested on the exact same 20% held-out test set as BERT.

---

## 📊 Evaluation Metrics

To ensure a fair comparison, we use a robust set of metrics:

* **Binary:** Accuracy, Precision, Recall, F1-Score.
* **Multi-label:** * **Macro/Micro F1:** Overall tactic performance.
* **Exact Match Ratio:** How often the model got *all* tactics right.
* **Per-class F1:** Identifying which tactics are "hardest" to detect.



---

## 🛠 Tech Stack

* **Core:** Python, Pandas, NumPy, Scikit-learn.
* **Deep Learning:** PyTorch, HuggingFace Transformers.
* **Models:** BERT-base-uncased.
* **APIs:** OpenAI (GPT-4o), Google AI (Gemini).

---

## 🚀 Future Roadmap

* [ ] **Calibration:** Improve probability thresholds for tactic attribution.
* [ ] **Extended Taxonomy:** Add visual-based clickbait detection (Image+Text).
* [ ] **Contrastive Learning:** Explore if SimCLR/CLIP can improve feature representation.
* [ ] **Explainability:** Map BERT's attention heads to the specific tactics identified.

---

## 📄 License

This project is for academic research purposes. Data sources belong to their respective owners.


**Developed as part of the M.Sc in Intelligent Systems & AI @ Afeka College.**
