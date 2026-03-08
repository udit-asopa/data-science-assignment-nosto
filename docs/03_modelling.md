# Modelling

## Train / Validation Split

A **chronological 80/20 split** on `end_dt` is used throughout.

```
Split point : end_dt quantile(0.80)
Train       : ~24,700 rows  (Jul 20 – ~Aug 28 2024)
Val         :  ~6,200 rows  (~Aug 28 – Sep 9 2024)
```

**Why chronological, not random?**  
Random splitting would allow future visit information to leak into training (a visit on Sep 1 in train, one on Aug 15 in val for the same customer). Chronological splitting simulates real deployment: the model is trained on historical data and evaluated on unseen future data.

### Drift Check

At split time a target drift check is computed:

```
Train median : 30.9h  (1.3 days)
Val   median : 15.8h  (0.7 days)
Drift        : 48.7%  ⚠ SIGNIFICANT
```

The validation period customers return roughly **twice as fast** as training period customers. This is the single largest performance constraint - models trained on the training distribution are systematically over-calibrated for longer return times.

---

## Target Transform

All models are trained on `log1p(return_hours)` and predictions are inverted with `expm1` before evaluation:

```python
y_train_log = np.log1p(y_train)
# training...
preds = np.expm1(model.predict(X_val)).clip(min=0)
```

**Why log1p?**
- The raw target is heavily right-skewed (range: 0 – 3,000h+)
- Log transform compresses the long tail and gives the model a near-normal signal to optimise
- MAE on the log scale corresponds approximately to proportional errors - appropriate for a target that spans orders of magnitude
- `expm1` (not `exp`) correctly inverts `log1p`, preserving the zero case

---

## Baseline Models

Established before adding ML complexity to set a meaningful performance floor.

### Global Median Baseline

Predicts the training set median (30.9h) for every row.

```
MAE : 32.32h
```

### Segmented Median Baseline

Groups the training set by `(ever_bought × start_dayofweek)` and predicts the segment median. Falls back to the global median for unseen segments.

```
MAE : 32.71h  (worse than global - segments don't generalise to val due to drift)
```

The segmented baseline performing worse than global is an early indicator that train/val segment compositions differ.

---

## Models

All tree-based models are evaluated with MAE on the validation set in original hours. A consistent pattern is used:

- MAE objective (not MSE) - more robust to the heavy right tail
- `log1p` target transform
- Scikit-learn `Pipeline` wrapping imputer + model

---

### 4.4.1 HistGradientBoosting (HGB)

Scikit-learn's native gradient boosting - **handles NaN natively** (no imputation step needed).

```python
HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=400,
    max_depth=6,
    min_samples_leaf=30,
    random_state=42,
)
```

| Val MAE | Val RMSE |
|---|---|
| 36.11h | - |

---

### 4.4.2 Random Forest

Ensemble of decision trees grown independently. Requires `SimpleImputer` for NaN features.

**Hyperparameter tuning** via `GridSearchCV` + `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)`:

```python
# Best params (from RandomizedSearchCV then GridSearchCV)
RandomForestRegressor(
    n_estimators=500,
    max_depth=10,        # shallower than default → better generalisation
    min_samples_leaf=10,
    max_features=0.5,    # feature subsampling per split
    n_jobs=-1,
    random_state=42,
)
```

Key tuning insight: `max_depth=10` outperformed deeper trees, confirming the model was overfitting with unrestricted depth.

| Val MAE | Val RMSE |
|---|---|
| 32.00h | - |

---

### 4.4.3 XGBoost

Gradient boosted trees with a native MAE objective (`reg:absoluteerror`).

**Hyperparameter tuning** via `RandomizedSearchCV` (30 iterations, `TimeSeriesSplit(n_splits=5)`) followed by fine-grained `GridSearchCV` around the best region:

```python
xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,      # slower, more conservative than default 0.05
    max_depth=8,
    min_child_weight=50,     # analogous to min_samples_leaf
    subsample=0.8,           # row subsampling per tree
    objective="reg:absoluteerror",
    random_state=42,
)
```

| Val MAE | Val RMSE |
|---|---|
| 36.06h (tuned) | - |

---

### 4.4.4 LightGBM DART

LightGBM with **DART boosting** (Dropouts meet Multiple Additive Regression Trees). DART applies dropout during the boosting process - each iteration, a random subset of trees is dropped, preventing any single tree from dominating.

```python
lgb.LGBMRegressor(
    boosting_type="dart",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_samples=30,
    objective="mae",
    random_state=42,
)
```

**Why DART outperforms here**: under significant target drift, standard gradient boosting aggressively over-commits to the training distribution. DART's dropout regularisation forces the model to be less deterministic - it effectively learns a more generalised representation that happens to be more robust to the distribution shift in the validation period.

The `.set_output(transform="pandas")` flag on the imputer preserves column names, which LightGBM uses to match features at predict time.

| Val MAE | Notes |
|---|---|
| **29.98h** | **Best model - beats baseline by 7.2%** |

---

## Hyperparameter Tuning Strategy

### Cross-validation approach

`TimeSeriesSplit(n_splits=5)` is used throughout - this creates 5 expanding-window train/val folds that respect temporal order, preventing future data from appearing in any CV training fold.

```
Fold 1: train [0..20%]     val [20..40%]
Fold 2: train [0..40%]     val [40..60%]
Fold 3: train [0..60%]     val [60..80%]
Fold 4: train [0..80%]     val [80..100%]  ← closest to true val
```

### Two-stage tuning (RF and XGBoost)

1. **Stage 1 - `RandomizedSearchCV`**: wide parameter ranges, 30 random samples → identifies the best region cheaply
2. **Stage 2 - `GridSearchCV`**: tight grid centred on Stage 1 winner → exhaustive search of nearby values

### Caveat

CV MAE scores are in `log1p` scale (≈1.4) and are not directly comparable to validation MAE in hours (≈30–38h). CV scores also use older data as validation - they cannot account for the drift to the true val period, so CV performance is an imperfect proxy for true generalisation.

---

## Pipeline Design

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # handles NaN from lag/history features
    ("model",   <estimator>),
])
```

Benefits:
- Imputation is fitted on training data only - no val statistics leak into imputed values
- `GridSearchCV`/`RandomizedSearchCV` wraps the whole pipeline, so imputation is re-fitted on each CV fold
- Predict-time calls go through the same imputation automatically
