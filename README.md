# KD34403_Group3
# Bank Marketing — Term Deposit Subscription Prediction

> **KD34403 Machine Learning for Data Science — Group 3 Project**

| No. | Name | Matric No. |
|-----|------|------------|
| 1 | Goo Xin Yi | BI23110093 |
| 2 | Tam Yi Qing | BI23110092 |
| 3 | Ong Chia Yin | BI23110241 |
| 4 | Chok Weng Hin | BI23110182 |
| 5 | Voo Xin Rou | BI23110233 |

=====================

## Project Overview

This project builds a machine learning pipeline to predict whether a bank client will **subscribe to a term deposit** following a telephone marketing campaign.

Using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`, 41,188 rows, 20 features), we address a classic **binary classification** problem on highly imbalanced data (~11% positive class).

=====================

## Repository Structure

```
├── bank-additional-full.csv      # Dataset (auto-downloaded if missing)
├── notebook.ipynb                # Main Jupyter notebook (full pipeline)
├── saved_models/
│   ├── best_lr.pkl               # Best Logistic Regression model
│   ├── best_rf.pkl               # Best Random Forest model
│   └── best_xgb.pkl              # Best XGBoost model
└── README.md
```

=====================

## MACHINE LEARNING Pipeline Summary

### 1. Problem Definition
- **Type:** Binary Classification
- **Target:** `y` — subscribed (1) / not subscribed (0)
- **Key challenge:** Class imbalance (~8.9:1), sentinel values, and post-call leakage feature (`duration`)

### 2. Exploratory Data Analysis (EDA)
- Class distribution and subscription rates by job type
- Numeric feature distributions split by target
- Correlation heatmap (multicollinearity identified in economic indicators)
- Categorical subscription rates across all features
- `unknown` value analysis and `pdays` sentinel investigation
- Outlier detection via IQR method

### 3. Data Preprocessing
| Decision | Justification |
|----------|---------------|
| Drop `duration` | Post-call leakage — unavailable before a call is made |
| `pdays = 999` → binary flag | Sentinel value; converted to `was_previously_contacted` |
| `'unknown'` → `NaN` | Enables proper imputation |
| Ordinal encode `education` | Natural hierarchy (illiterate → university) |
| One-hot encode other categoricals | No inherent order |
| Median impute numerics | Robust to skew and outliers |
| Most-frequent impute categoricals | Maintains realistic distributions |
| Train/val/test split **before** scaling | Prevents data leakage |
| `StandardScaler` on continuous features | Required for Logistic Regression; improves stability |

**Split:** 70% train / 15% validation / 15% test (stratified)

### 4. Feature Engineering
- **PCA** on correlated economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed` → `econ_index`; `cons.price.idx`, `cons.conf.idx` → `cons_index`) to reduce multicollinearity
- **Interaction terms:** `poutcome × was_previously_contacted`, `campaign × (previous + 1)`
- **Age binning:** young (<30), middle (30–45), senior (45–60), retired (60+)

### 5. Imbalance Handling
Three techniques were evaluated for each model:
- **Tomek Links** —> undersampling; removes borderline majority samples
- **SMOTE** —> oversampling; synthesises new minority samples using k-NN
- **SMOTETomek** —> hybrid; combines both approaches

### 6. Models Trained
| Model | Best Imbalance Technique | Tuning Methods Tried |
|-------|--------------------------|----------------------|
| Logistic Regression | Selected automatically | Grid Search, Random Search, Bayesian Opt, Manual |
| Random Forest | Selected automatically | Grid Search, Random Search, Bayesian Opt, Manual |
| XGBoost | Selected automatically | Grid Search, Random Search, Bayesian Opt, Manual |
| **Stacking Ensemble** | — | Meta-learner: Logistic Regression (5-fold OOF) |

All models also underwent **threshold tuning** on the validation set to maximise Balanced Accuracy before final evaluation on the test set.

### 7. Final Results (Test Set)

| Model | Accuracy | Balanced Accuracy | F1-Score | ROC-AUC |
|-------|----------|-------------------|----------|---------|
| LR Baseline (demographics only) | 0.6250 | 0.5993 | 0.2538 | 0.6395 |
| Logistic Regression (enhanced) | 0.8283 | 0.7502 | 0.4601 | 0.8026 |
| Random Forest | 0.8686 | 0.7554 | 0.5108 | 0.8045 |
| **XGBoost** ✅✅ | 0.8480 | 0.7613 | 0.4905 | 0.8084 |
| Stacking Ensemble | 0.8844 | 0.7448 | 0.5240 | 0.8089 |

> **XGBoost was selected as the final model.** The stacking ensemble performed marginally lower, likely due to the linear meta-learner limiting combination of non-linear base outputs.

=====================

## Requirements

```
python >= 3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
scikit-optimize
joblib
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost scikit-optimize joblib
```

=====================

## **Using Google Colab**

### Method 1 — Open directly from GitHub into Colab

In Colab:
* File → Open notebook → GitHub tab
* paste your repo URL
* select notebook.ipynb 

### Step 2 — Install Required Libraries

```python
!pip install pandas numpy matplotlib seaborn scikit-learn \
imbalanced-learn xgboost scikit-optimize joblib ucimlrepo
```

### Step 3 — Upload Dataset (Optional)

If bank-additional-full.csv is not already included:

```python
from google.colab import files
uploaded = files.upload()
```
Then select bank-additional-full.csv.

If the notebook already auto-downloads the dataset from the UCI repository, this step can be skipped.

### Step 4 — Create Model Save Directory

Since Colab starts with a clean environment each session, create the model directory manually:

```python
import os
os.makedirs("saved_models", exist_ok=True)
```

### Step 5 — Run All Cells

Run the notebook sequentially from top to bottom.

Training may take 20–60 minutes depending on Colab hardware availability.

### Step 6 — Download Saved Models

```python
from google.colab import files

files.download('saved_models/best_xgb.pkl')
files.download('saved_models/best_rf.pkl')
files.download('saved_models/best_lr.pkl')
```

=====================

## **If using other notebooks**

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually using the command in the Requirements section above.

### Step 3 — Launch the notebook

```bash
jupyter notebook notebook.ipynb
```

### Step 4 — Run all cells

The notebook is fully self-contained and sequential. Run all cells from top to bottom.

- If `bank-additional-full.csv` is not present, the notebook will **automatically download** it from the UCI repository.
- Trained models are saved to `saved_models/` as `.pkl` files after each model section completes.

> **Note:** Bayesian Optimisation and Random Forest grid search steps are computationally intensive. Expect the full run to take **20–60 minutes** depending on your hardware.

=====================

## How to Use Saved Models

After running the notebook, you can load and use any saved model independently:

```python
import joblib
import pandas as pd

# Load the best XGBoost model
payload = joblib.load('saved_models/best_xgb.pkl')

model          = payload['model']
feature_names  = payload['feature_names']
threshold      = payload['best_threshold']
scaler         = payload['scaler']

# Prepare input data (must match the engineered feature set)
# X_new = pd.DataFrame(...)  # the new data

X_new_aligned = X_new[feature_names]
proba = model.predict_proba(X_new_aligned)[:, 1]
pred  = (proba >= threshold).astype(int)

print("Subscription probability:", proba)
print("Predicted class (1=subscribe, 0=not):", pred)
```

Each `.pkl` payload contains:

| Key | Description |
|-----|-------------|
| `model` | Fitted sklearn/XGBoost estimator |
| `feature_names` | Ordered list of features the model was trained on |
| `best_threshold` | Tuned decision threshold |
| `best_technique` | Best imbalance handling method used |
| `best_search` | Best hyperparameter search method used |
| `best_params` | Final model hyperparameters |
| `test_metrics` | Metrics at default threshold (0.5) |
| `tuned_metrics` | Metrics at tuned threshold |
| `scaler` | Fitted `StandardScaler` for numeric features |

=====================

## Key Findings

- **Economic indicators** (`nr.employed`, `euribor3m`, `emp.var.rate`) are the strongest predictors but are highly correlated (r up to 0.97). PCA reduces this multicollinearity effectively.
- **`poutcome = success`** (previous campaign outcome) is the strongest categorical predictor.
- **Call timing** (month, contact method) significantly influences subscription likelihood.
- **`duration`** was dropped to prevent data leakage — it cannot be known before a call is made.
- The **top 30%** of clients ranked by XGBoost score capture ~74% of all subscribers, enabling targeted and cost-efficient marketing.

=====================

## Dataset
Dataset: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)