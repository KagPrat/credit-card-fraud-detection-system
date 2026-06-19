# Credit Card Fraud Detection System

A machine learning pipeline that detects fraudulent credit card transactions using classification models on a severely imbalanced real-world dataset. Built as a portfolio project targeting data science and risk analyst roles.

---

## Problem Statement

Credit card fraud accounts for billions in losses annually. Detecting fraud is a classic **imbalanced classification problem** — fraudulent transactions make up only **0.17%** of all records in this dataset. A naive model that predicts "not fraud" every time achieves 99.8% accuracy while being completely useless.

This project addresses that challenge end-to-end: from raw data ingestion through model deployment in a live Streamlit dashboard.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property                | Value                               |
| ----------------------- | ----------------------------------- |
| Total transactions      | 284,807                             |
| Fraudulent transactions | 492 (0.17%)                         |
| Features                | 30 (V1–V28 via PCA, Time, Amount)   |
| Target column           | `Class` (0 = legitimate, 1 = fraud) |

> Note: V1–V28 are PCA-transformed by the dataset authors to protect cardholder privacy. `Time` and `Amount` are the only raw features.

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

---

## Pipeline Overview

```
Raw CSV → Preprocessing → Class Imbalance Handling → Feature Engineering
       → Model Training → Evaluation → Streamlit Dashboard
```

### Stage 1 — Preprocessing (`src/preprocess.py`)

- Load and inspect the raw CSV
- Verify nulls and duplicates
- **Stratified train/test split (80/20)** before any scaling to prevent data leakage
- `StandardScaler` fit on training data only, applied to both sets
- Saves processed arrays and the fitted scaler to `data/processed/`

### Stage 2 — Class Imbalance Handling (`src/imbalance.py`)

- Applies **SMOTE** (Synthetic Minority Oversampling Technique) to the training set only
- Generates synthetic fraud samples so the model sees a balanced class distribution
- Test set is never resampled — it reflects real-world conditions

### Stage 3 — Feature Engineering (`src/features.py`)

- Log-transform `Amount` to reduce skew
- Extract transaction hour from `Time`
- Derive rolling average amount features

### Stage 4 — Model Training (`src/train.py`)

- **Logistic Regression** — interpretable baseline
- **XGBoost** — high-performance gradient boosting

### Stage 5 — Evaluation (`src/evaluate.py`)

- Precision, Recall, F1-score, AUC-ROC
- Confusion matrix heatmap
- SHAP feature importance plots (XGBoost)

### Stage 6 — Streamlit Dashboard (`app.py`)

- Upload a CSV of transactions
- Get per-row fraud probability scores
- View confusion matrix and SHAP plots interactively

---

## Key Concepts Demonstrated

**Data leakage prevention** — the scaler is fit exclusively on training data, then applied to the test set. SMOTE is applied after splitting, never before.

**Imbalanced classification** — accuracy is a misleading metric here. The project evaluates models on precision-recall tradeoff and AUC-ROC, which are meaningful for skewed targets.

**Model interpretability** — SHAP values explain which features drove each prediction, making results auditable — a key requirement in financial risk contexts.

---

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                  # place creditcard.csv here (gitignored)
│   └── processed/            # output of preprocess.py (gitignored)
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis
├── src/
│   ├── preprocess.py
│   ├── imbalance.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at `data/raw/creditcard.csv`.

Then run the pipeline in order:

```bash
python src/preprocess.py
python src/imbalance.py
python src/features.py
python src/train.py
python src/evaluate.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
streamlit
joblib
```

---

## Results

> To be updated once model training is complete.

| Model               | Precision | Recall | F1  | AUC-ROC |
| ------------------- | --------- | ------ | --- | ------- |
| Logistic Regression | —         | —      | —   | —       |
| XGBoost             | —         | —      | —   | —       |

---

## Why This Matters for Risk & Analytics Roles

Fraud detection sits at the intersection of data science and financial risk. This project demonstrates:

- Handling real-world class imbalance without overfitting
- Choosing evaluation metrics appropriate to the business problem
- Building explainable models that a risk team can audit
- Deploying predictions through a usable interface

---

## Author

**Pratik**
Portfolio project — open to data analyst, financial analyst, data science, and risk analyst roles.
[LinkedIn](#) · [GitHub](#)

```

```
