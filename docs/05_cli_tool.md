# CLI Tool — Implementation Guide

This document explains the design, workflows, commands, and implementation of the
script-based CLI tool for the customer return-time prediction pipeline.

---

## Overview

The CLI tool (`main.py`) exposes the full machine-learning pipeline as three composable
commands. It is built with [Typer](https://typer.tiangolo.com/) and acts as a thin
orchestration layer over the modules in `scripts/`.

```
project root
├── main.py                     ← CLI entry point (Typer app)
└── scripts/
    ├── data_preprocessing.py   ← data loading & target construction
    ├── feature_engineering.py  ← feature computation + FEATURE_COLS registry
    ├── data_modelling.py       ← ModelSuite class + chronological_split()
    ├── evaluation.py           ← metrics, leaderboard, permutation importance
    └── helper_functions.py     ← shared utilities (JSON I/O, section printing)
```

---

## Quick Start

```bash
# 1. Train all models (saves to scripts/models/)
python main.py train

# 2. Generate predictions for the same dataset
python main.py predict --output output/predictions.tsv

# 3. Evaluate the saved models and print the leaderboard
python main.py evaluate
```

Run `python main.py --help` or `python main.py <command> --help` for full option
details.

---

## Commands

### `train`

Runs the complete training pipeline end-to-end and persists the fitted `ModelSuite`
to disk.

```
python main.py train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` / `-d` | `data/visits.tsv` | Input visits TSV file |
| `--model-dir` / `-m` | `scripts/models` | Directory to save the trained suite |
| `--val-fraction` | `0.20` | Fraction of data (by time) held out for validation |

**What it does — step by step:**

1. Load and preprocess the TSV (`load_and_prepare`)
2. Run all feature engineering (`build_features`)
3. Drop last-visit rows (target is undefined for them)
4. Perform a chronological 80/20 split by `end_dt`
5. Log a target-drift warning if the val median diverges > 10 % from the train median
6. Train all six models in order (see [Models](#models))
7. Print the evaluation leaderboard on the val set
8. Save the entire `ModelSuite` to `<model-dir>/model_suite.joblib`

**Example — custom paths:**

```bash
python main.py train --data data/visits.tsv --model-dir scripts/models --val-fraction 0.15
```

---

### `predict`

Loads a saved `ModelSuite` and generates return-time predictions for every row in
the input file, including customers whose last visit has no known target.

```
python main.py predict [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` / `-d` | `data/visits.tsv` | Input visits TSV file |
| `--model-dir` / `-m` | `scripts/models` | Directory containing the saved suite |
| `--output` / `-o` | `output/predictions.tsv` | Output file path (TSV) |
| `--model` | `lgb` | Model to use: `lgb`, `hgb`, `rf`, `xgb`, `global_baseline`, `seg_baseline` |

**Output columns:**

| Column | Description |
|--------|-------------|
| `customer_id` | Customer identifier |
| `start_dt` | Session start datetime |
| `end_dt` | Session end datetime |
| `predicted_return_hours` | Predicted hours until next visit |
| `predicted_return_days` | Same value in days (rounded to 2 d.p.) |
| `return_hours` | Actual target value (only present for non-last-visit rows) |

**Example — use the faster HGB model:**

```bash
python main.py predict --model hgb --output output/hgb_predictions.tsv
```

---

### `evaluate`

Re-runs evaluation on the validation split using an already-trained `ModelSuite`,
reproducing the exact train/val split from training.

```
python main.py evaluate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` / `-d` | `data/visits.tsv` | Input visits TSV file |
| `--model-dir` / `-m` | `scripts/models` | Directory containing the saved suite |
| `--val-fraction` | `0.20` | Validation fraction (should match the training run) |

**Example:**

```bash
python main.py evaluate
```

---

## Workflows

### End-to-End Training and Prediction

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────────┐
│ data/        │────▶│  python main.py train │────▶│ scripts/    │
│ visits.tsv   │     │                      │     │ models/     │
└──────────────┘     └──────────────────────┘     │ model_suite │
                                                   │ .joblib     │
┌──────────────┐     ┌────────────────────────┐   └──────┬──────┘
│ data/        │────▶│ python main.py predict │◀──────────┘
│ visits.tsv   │     │                        │
└──────────────┘     └───────────┬────────────┘
                                 ▼
                     output/predictions.tsv
```

### Re-evaluating Without Retraining

If the data has not changed, you can call `evaluate` independently after `train`
without repeating the expensive training step:

```bash
python main.py train   --data data/visits.tsv   # run once
python main.py evaluate                          # run any number of times
```

### Experimenting with Individual Models

Use `--model` in the `predict` command to swap the inference model without
retraining:

```bash
python main.py predict --model hgb
python main.py predict --model xgb
python main.py predict --model global_baseline  # sanity check
```

---

## Implementation

### `main.py` — Entry Point

```
app = typer.Typer(name="nosto-return-time")
 │
 ├── @app.command() train(...)
 ├── @app.command() predict(...)
 └── @app.command() evaluate(...)
```

`main.py` imports only the public API of each script module and wires them together.
It contains no data-processing or statistical logic of its own — all of that lives
in `scripts/`.

---

### `scripts/data_preprocessing.py`

Responsible for loading the raw TSV and constructing the regression target.

| Function | Role |
|----------|------|
| `load_visits_data(data_path)` | Read TSV, validate required columns |
| `process_product_columns(df)` | Parse list strings → `list[int]`, compute `total_*` counts |
| `add_datetime_features(df)` | UNIX-ms → `end_dt`, derive `start_dt`, `*_hour`, `*_dayofweek` |
| `build_return_time_target(df)` | Sort by `(customer_id, start_dt)`, compute `return_hours` via time-to-next-visit |
| `load_and_prepare(data_path)` | Orchestrator: calls all four functions above in sequence |
| `audit_dataset(df)` | Optional: compute quality statistics (row counts, null rates, etc.) |

**Target construction:**  
Visits are sorted per-customer by `start_dt`. For each visit, `next_start_dt` is
the start of the same customer's next visit. The target is:

```
return_hours = (next_start_dt − end_dt).total_seconds() / 3600   [clipped ≥ 0]
```

The last visit of each customer has no successor → `return_hours = NaN`.

---

### `scripts/feature_engineering.py`

Defines **23 features** across five groups, all computed per-visit with no
information leakage from future visits.

```
FEATURE_COLS (single source of truth, consumed by data_modelling and evaluation)
 │
 ├── Within-session (10): total_viewed, total_cart, total_bought,
 │       search_count, time_spent_in_minutes, buy_ratio, cart_ratio,
 │       start_hour, start_dayofweek, visit_bought_flag
 │
 ├── Sequence (1): visit_counter_index
 │
 ├── Purchase history (3): ever_bought, cumulative_bought_visits, cumulative_buy_rate
 │       (shift(1) applied — values reflect history *before* the current visit)
 │
 ├── Lag / recency (5): prev_gap_hours, prev_time_spent, prev_search_count,
 │       prev_viewed_count, prev_bought
 │
 └── Rolling + expanding (4): rolling_avg_gap, rolling_avg_time_spent,
         rolling_avg_viewed, cumulative_avg_gap
```

| Function | Role |
|----------|------|
| `add_session_features(df)` | Convert timedelta → float, add `buy_ratio`, `cart_ratio` |
| `add_purchase_history_features(df)` | Cumulative purchase flags (shift(1)-safe) |
| `add_lag_features(df)` | Per-customer shift of previous visit metrics |
| `add_rolling_features(df)` | 3-visit window averages + expanding mean |
| `build_features(df)` | Orchestrator: calls all four functions in sequence |

---

### `scripts/data_modelling.py`

Contains `chronological_split()` and the `ModelSuite` class.

#### `chronological_split(df, time_col, val_fraction)`

Splits the DataFrame at the `(1 − val_fraction)` quantile of `time_col` (defaults
to `end_dt`). This mirrors a real deployment scenario where the model is trained on
historical data and evaluated on future data — no random shuffling.

#### `ModelSuite`

Wraps all six models under one object with a consistent `predict(model_name, X)`
interface.

| Name | Type | Notes |
|------|------|-------|
| `global_baseline` | Training-set median | Simplest possible predictor |
| `seg_baseline` | Median per `(ever_bought, start_dayofweek)` | Segment-aware baseline |
| `hgb` | `HistGradientBoostingRegressor` | Handles NaN natively, MAE loss |
| `rf` | `Pipeline(SimpleImputer + RandomForest)` | 500 trees, `max_features=0.5` |
| `xgb` | `Pipeline(SimpleImputer + XGBRegressor)` | `objective="reg:absoluteerror"` |
| `lgb` | `Pipeline(SimpleImputer + LGBMRegressor)` | DART boosting, `objective="mae"` |

All ML models are trained on **log1p(return_hours)** and predictions are back-transformed
with `expm1` before returning, keeping residuals symmetric on the heavy right-skewed
distribution.

**Key methods:**

| Method | Description |
|--------|-------------|
| `train_all(train_df, verbose)` | Train all 6 models and log progress |
| `predict(model_name, X)` | Predict in hours scale for any of the six models |
| `save(model_dir)` | `joblib.dump(self, model_dir / model_suite.joblib)` |

The `ModelSuite` is loaded via a standalone module-level function (not a classmethod):

| Function | Description |
|----------|-------------|
| `load_model_suite(model_dir)` | `joblib.load(...)`, raises `FileNotFoundError` with a helpful message |

---

### `scripts/evaluation.py`

| Function | Description |
|----------|-------------|
| `regression_metrics(y_true, y_pred)` | Returns `{"mae": float, "rmse": float}` |
| `check_target_drift(y_train, y_val, threshold_pct)` | Flags if val median diverges > 10 % from train median |
| `print_leaderboard(results, baseline_key)` | Formatted table: MAE, RMSE, % improvement vs baseline, marks best model |
| `compute_permutation_importance(model, X_val, y_val, feature_cols)` | sklearn permutation importance scored against log1p(y_val), returns sorted DataFrame |

---

### `scripts/helper_functions.py`

| Function | Description |
|----------|-------------|
| `ensure_parent_dir(file_path)` | Create parent directories as needed |
| `save_json(data, output_path)` | Serialise a dict to JSON |
| `load_json(input_path)` | Load JSON into a dict |
| `print_section(title, width)` | Print a labelled divider line to stdout |

---

## Design Decisions

**Functional modules, one class.**  
Each `scripts/` module is a collection of pure functions. The single exception is
`ModelSuite`, which is a class because it must hold mutable training state (fitted
models, baselines) and provide a clean `save`/`load` interface.

**`FEATURE_COLS` as the single source of truth.**  
The 23-element list in `feature_engineering.py` is imported by both `data_modelling`
and `evaluation`. This guarantees that training, prediction, and importance
computations always operate on exactly the same feature set.

**No leakage in cumulative/lag features.**  
All history-based features (`ever_bought`, `cumulative_buy_rate`, lag and rolling
features) apply `shift(1)` within each customer group, so every value reflects
only information available *before* the current visit.

**Chronological split, not random.**  
Using `end_dt` quantile for the split ensures validation measures true
out-of-sample generalisation to future data, rather than interpolation within
the observed time range.

**log1p → train → expm1.**  
`return_hours` has a heavy right tail (most returns within hours; a long tail of
weeks). Training on `log1p(y)` and recovering `expm1(ŷ)` keeps MAE-based
gradients well-conditioned without distorting the evaluation metric.
