# Results & Findings

## Model Comparison

| Model | Val MAE (h) | vs Global Baseline | Notes |
|---|---|---|---|
| Global Median Baseline | 32.32 | — | Predicts 30.9h for every row |
| Segmented Median (ever_bought × dow) | 32.71 | −1.2% ↓ | Worse — drift breaks segment medians |
| HistGradientBoosting | 36.11 | −11.8% ↓ | |
| Random Forest (tuned) | 32.00 | +1.0% ↑ | Marginal — near-baseline level |
| XGBoost (tuned) | 36.06 | −11.6% ↓ | |
| **LightGBM DART** | **29.98** | **+7.2% ↑** | **Best — only model to beat baseline** |

All models trained on `log1p(return_hours)`, evaluated on original hours with `expm1(prediction)`.

---

## Why Most Models Underperform the Baseline

The **48.7% target drift** is the dominant factor:

- **Train median**: 30.9h — the global median baseline predicts this value
- **Val median**: 15.8h — the true distribution has shifted significantly to shorter return times
- Models trained on train data learn to predict a value near 30.9h; the global median baseline already does exactly this
- Any additional complexity (features, gradient boosting) is essentially learning noise around the same wrong centre

This is not a modelling failure. It is the correct diagnostic: **the data distribution changed between train and val**, making generalisation structurally difficult regardless of model quality.

### Why LightGBM DART is the exception

DART's dropout mechanism prevents the model from over-memorising the training distribution. By randomly dropping entire trees during boosting, DART is forced to maintain a more generalised, averaged representation of the target. Under distribution shift, this acts as an implicit regulariser that makes the model less biased toward the (stale) training distribution.

---

## Feature Importance (Permutation-Based)

Permutation importance is used across all 4 models: a feature is masked by shuffling its values and the resulting increase in MAE (on `log1p(y_val)`) measures its contribution. Positive = useful; negative = harmful.

### Consistent top features across models

| Rank | Feature | Why it matters |
|---|---|---|
| 1 | `start_hour` | Time of day is the strongest signal for return timing — a late-night visit has a different return cadence than a morning visit |
| 2 | `time_spent_in_minutes` | Engagement depth: longer visits indicate higher purchase intent, correlating with sooner return |
| 3 | `visit_counter_index` | Customer lifecycle position — new vs. regular visitors have fundamentally different return patterns |
| 4 | `total_viewed_products` | Browse depth as an engagement proxy |

### The `prev_gap_hours` artefact

`prev_gap_hours` appears **near zero or negative** in permutation importance despite being theoretically the most informative feature (a customer's historical return interval directly predicts future intervals).

**Why**: 63% of rows are first-visit customers for whom `prev_gap_hours = NaN`. Shuffling NaN→NaN changes nothing, so permutation registers no importance. For the 37% of repeat visitors it is likely strongly predictive, but the signal is diluted by the dominant NaN group.

This is a **measurement artefact**, not a model finding. Permutation importance should be interpreted cautiously on datasets with high NaN rates in key features.

---

## Covariate Drift Analysis

The train vs. val distribution plots (Section 4.2) reveal:

| Feature group | Drift | Interpretation |
|---|---|---|
| `return_hours` (target) | **Strong left shift** | Val customers return faster — distribution shifted toward shorter times |
| Session features (`total_viewed`, `time_spent`, etc.) | Stable | No meaningful covariate shift |
| Purchase flags (`visit_bought_flag`, `ever_bought`) | Stable | Purchase behaviour unchanged |
| Gap features (`prev_gap_hours`, `rolling_avg_gap`, `cumulative_avg_gap`) | Slight left shift | Consistent with target drift — recent gaps were already shorter in val period |

**Conclusion**: this is **target drift without covariate drift**. The input features look the same in both periods, but the outcome distribution changed — possibly driven by a seasonal event (back-to-school, sale period) that compressed return times in late August/September.

---

## What I Would Try Next

### 1. Survival Modelling (highest priority)

The current approach **drops last-visit rows** (censored observations — ~24% of data). A survival model treats these correctly:

- **Cox Proportional Hazards**: estimates hazard ratios for each feature — interpretable and handles censoring natively
- **Weibull AFT (Accelerated Failure Time)**: directly models return time distribution — more appropriate for prediction

This would fix a structural problem in the current target definition rather than a modelling choice.

### 2. Drift-Aware Training

Re-weight training samples by recency so that recent observations (closer to the val period) have higher influence during training:

```python
sample_weight = np.linspace(0.5, 1.0, len(X_train))  # linear ramp by time
model.fit(X_train, y_train_log, sample_weight=sample_weight)
```

This directly addresses the drift without requiring more data.

### 3. Reframe as a Classification Task

"Will this customer return within 7 days?" is:
- More learnable under drift (a binary boundary is more stable than a continuous distribution)
- More actionable for marketing (binary trigger for re-engagement campaign)
- Less sensitive to the heavy tail of non-returning customers

### 4. Longer Historical Window

7 weeks is insufficient to:
- Capture weekly seasonality reliably (need 8–12 weeks minimum)
- Distinguish genuine churn from a long gap
- Build stable recency features for occasional shoppers

A 12-month window would substantially improve the lag/recency feature signal for the 37% repeat visitor segment.

### 5. Customer-Level Models

For the top 5–10% of customers (high visit frequency), a **per-customer time-series model** (ARIMA on gap sequence, or a simple exponential smoother of past gaps) would likely outperform a global model — their return patterns are stable and idiosyncratic.

---

## Reproducibility

The notebook runs end-to-end from raw `data/visits.tsv` with no external seeds beyond `random_state=42`. All random states are fixed. The chronological split is deterministic (based on quantile of `end_dt`). Runtime: approximately 10–15 minutes on a standard laptop (dominated by grid search and permutation importance).
