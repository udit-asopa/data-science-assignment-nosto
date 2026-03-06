# Customer Return Time Prediction — Documentation

## Overview

This project predicts how many hours a customer will take to return to a webshop after their current visit, using session-level behavioural data. The goal is to support marketing and retention decisions by providing a data-driven estimate of visitor return cadence.

---

## Documentation Index

| File | Contents |
|---|---|
| [01_problem_and_data.md](01_problem_and_data.md) | Problem definition, dataset description, EDA findings |
| [02_feature_engineering.md](02_feature_engineering.md) | Feature construction, leak prevention, feature catalogue |
| [03_modelling.md](03_modelling.md) | Train/val split, baseline, model architectures, hyperparameter tuning |
| [04_results_and_findings.md](04_results_and_findings.md) | Model comparison, drift analysis, feature importance, future directions |

---

## Quick Summary

- **Task**: Regression — predict `return_hours` (hours until next visit) from current visit features
- **Data**: ~40k visit rows, ~7 weeks (Jul 20 – Sep 9 2024), single webshop
- **Split**: Chronological 80/20 on `end_dt` (no data leakage)
- **Best model**: LightGBM DART — **MAE 29.98h** (beats global median baseline by 7.2%)
- **Key finding**: 48.7% target drift between train and val periods is the dominant performance constraint

---

## Notebook

The end-to-end implementation lives in [`temp2.ipynb`](../temp2.ipynb), structured as:

```
Step 0 — Loading & Data Prep
Step 1 — Target Variable Definition
Step 2 — Exploratory Data Analysis  (5 sub-analyses)
Step 3 — Feature Engineering
Step 4 — Modelling
  4.1  Train/Val Split + Drift Check
  4.2  Distribution Check (train vs val)
  4.3  Baseline Models
  4.4  Model Training & Evaluation (HGB → RF → XGBoost → LightGBM DART)
  4.5  Feature Importance (permutation-based)
Summary & Findings
```

---

## Tech Stack

| Library | Role |
|---|---|
| `pandas` / `numpy` | Data wrangling and numerical ops |
| `matplotlib` | All visualisations |
| `scikit-learn` | Pipeline, imputers, RF, HGB, evaluation, permutation importance |
| `xgboost` | XGBoost regressor |
| `lightgbm` | LightGBM DART regressor |
