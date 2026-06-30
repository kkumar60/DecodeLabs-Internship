# Task 2 - Supervised Learning (Fraud Detection Pipeline)

## Objective

The objective of this task was to build and tune a classification pipeline to identify fraudulent transactions in an imbalanced e-commerce dataset. The focus was on correct handling of class imbalance using SMOTE, building leak-free pipelines, and evaluating models with the right metrics — completely discarding Accuracy in favor of Precision, Recall, and ROC-AUC.

---

## Dataset

| Property | Detail |
|---|---|
| Source File | `cleaned_dataset.xls` |
| Total Records | 1,200 orders |
| Original Features | 17 columns |
| ML-Ready Features | 11 numeric features + 1 target |
| Target Column | `IsFraud` (0 = Legitimate, 1 = Fraud) |
| Fraud Rate | 41.4% (497 fraud / 703 legitimate) |

### How the Fraud Label was Created

Since no explicit fraud column existed, a proxy label was engineered from `OrderStatus`:

```python
df["IsFraud"] = df["OrderStatus"].isin(["Cancelled", "Returned"]).astype(int)
```

- `Cancelled` + `Returned` → **1 (Fraud)**
- `Shipped` + `Delivered` + `Pending` → **0 (Legitimate)**

---

## Pipeline Steps

### Step 1 — Dataset Preparation
- Created `IsFraud` target column from `OrderStatus`
- Extracted `Month` and `DayOfWeek` from `Date`
- Label-encoded `PaymentMethod`, `Product`, `ReferralSource`
- Dropped ID columns, zero-variance columns, and leakage columns
- Final dataset: **1200 rows × 11 features**

### Step 2 — Stratified Train/Test Split (80/20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# X_train: (960, 11) | X_test: (240, 11)
```

### Step 3 — Leak-Free imblearn Pipelines

```python
# Pipeline A — Logistic Regression
lr_pipeline = ImbPipeline([
    ('scaler',     StandardScaler()),
    ('smote',      SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Pipeline B — Random Forest
rf_pipeline = ImbPipeline([
    ('smote',      SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

> SMOTE lives **inside** the pipeline — applied only on training folds, never on test data. Zero data leakage.

### Step 4 — Hyperparameter Tuning (GridSearchCV)

- 5-Fold Stratified Cross-Validation
- Scoring metric: `roc_auc` (not accuracy)
- Best LR: `C=0.1, k_neighbors=5`
- Best RF: `max_depth=20, n_estimators=200, k_neighbors=5`

### Step 5 — Model Evaluation

> Accuracy was intentionally excluded from evaluation.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.367 | 0.378 |
| Recall | 0.404 | 0.343 |
| F1-Score | 0.385 | 0.360 |
| **ROC-AUC** | 0.463 | **0.469** |

**Winner: Random Forest** (higher ROC-AUC)

### Step 6 — Visualizations

- ROC Curve comparison (LR vs RF)
- Random Forest Feature Importance chart
- Confusion Matrices for both models

---

## Key Findings

- **Random Forest** outperformed Logistic Regression on ROC-AUC (0.469 vs 0.463)
- **TotalPrice**, **UnitPrice**, and **RevenuePerItem** were the top 3 most important features
- **Quantity** contributed the least predictive signal
- ROC-AUC scores near 0.50 indicate that the available features carry limited fraud signal — this is a data finding, not a pipeline error. The dataset is a general e-commerce log where Cancelled/Returned orders reflect normal customer behavior as much as actual fraud

---

## Why Accuracy Was Discarded

In this dataset, predicting "Legitimate" for every single transaction would give ~58% accuracy — while catching **zero fraud**. Accuracy is completely misleading in imbalanced classification. Instead:

- **Precision** → When we flag fraud, how often are we right?
- **Recall** → Of all real fraud cases, how many did we catch?
- **ROC-AUC** → Overall ability to separate fraud from legitimate

---

## Project Structure

```
Task-2-Fraud-Detection/
├── README.md                     → This file
├── cleaned_dataset.xls           → Pre-cleaned source dataset
├── ml_ready_dataset.csv          → Final ML-ready dataset (11 features + target)
├── dcode_task2.ipynb             → Jupyter Notebook with all 6 steps
└── Fraud_Detection_Report.docx   → Full project report with results and charts
```

---

## Libraries Used

```python
pandas, numpy, matplotlib
scikit-learn (train_test_split, GridSearchCV, LogisticRegression, RandomForestClassifier)
imbalanced-learn (SMOTE, imblearn.pipeline.Pipeline)
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

---

## Key Concepts Applied

| Concept | Implementation |
|---|---|
| Class Imbalance Handling | SMOTE inside imblearn Pipeline |
| Data Leakage Prevention | SMOTE applied only inside CV training folds |
| Hyperparameter Tuning | GridSearchCV with 5-Fold Stratified CV |
| Correct Evaluation | Precision, Recall, F1, ROC-AUC only |
| Stratified Splitting | `stratify=y` preserves fraud % in both sets |

---

*DecodeLabs Industrial Training Kit — Batch 2026 | Project 2*
