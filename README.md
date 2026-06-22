# 🏠 House Prices — Advanced Regression Techniques

Kaggle competition: predicting residential home sale prices in Ames, Iowa.

🔗 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Overview
The dataset contains 79 features describing almost every aspect of residential
homes — from basement ceiling height to proximity to railroads. The goal is to
predict the final sale price of each home.

Submissions are evaluated on **RMSE between log-transformed predictions and
log-transformed observed prices**.

## Approach
- Dropped high-missingness columns (Alley, PoolQC, Fence, MiscFeature, MasVnrType)
- Built a sklearn `Pipeline` with a `ColumnTransformer`:
  - Numeric features → `SimpleImputer(strategy='median')`
  - Categorical features → `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder`
- Model: `RandomForestRegressor` tuned with `GridSearchCV` (max_depth, 5-fold CV)
- Evaluation metric: `neg_root_mean_squared_log_error`

## Stack
- Python, pandas, scikit-learn

## Usage
```bash
python3 my_prediction.py
```
Outputs `submission.csv` ready for Kaggle upload.
