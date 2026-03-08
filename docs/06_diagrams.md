# Pipeline Diagrams

Mermaid diagrams covering the CLI commands, data flow, and decision logic.

---

## 1. System Architecture

```mermaid
flowchart LR
    TSV[("visits.tsv")]
    SUITE[("model_suite.joblib")]
    OUT[("predictions.tsv")]

    subgraph CLI ["main.py"]
        TRAIN["train"]
        PREDICT["predict"]
        EVALUATE["evaluate"]
    end

    subgraph Scripts ["scripts/"]
        PRE["data_preprocessing"]
        FE["feature_engineering"]
        MOD["data_modelling"]
        EVAL["evaluation"]
    end

    TSV -->|input| TRAIN & PREDICT & EVALUATE
    TRAIN --> PRE --> FE --> MOD --> EVAL
    TRAIN -->|saves| SUITE
    SUITE -->|loads| PREDICT & EVALUATE
    PREDICT -->|writes| OUT
```

---

## 2. `train` Command

```mermaid
flowchart TD
    START(["train"])
    LOAD["load_and_prepare"]
    FEAT["build_features"]
    FILTER["drop NaN target rows"]
    SPLIT["chronological_split"]
    DRIFT{target drift > 10%?}
    WARN["log warning"]
    SUITE["ModelSuite()"]
    EVAL["evaluate on val set"]
    BOARD["print_leaderboard"]
    SAVE["suite.save"]
    DONE(["Done"])

    START --> LOAD --> FEAT --> FILTER --> SPLIT --> DRIFT
    DRIFT -->|Yes| WARN --> SUITE
    DRIFT -->|No| SUITE
    SUITE --> TRAIN["train_all: all 6 models"] --> EVAL
    EVAL --> BOARD --> SAVE --> DONE
```

---

## 3. `predict` Command

```mermaid
flowchart TD
    START(["predict"])
    CHK{"model file exists?"}
    ERR["FileNotFoundError"]
    LOAD["load_model_suite"]
    PREP["load_and_prepare + build_features"]
    PRED["suite.predict(model_name, X)"]
    ROUTER{model_name?}
    GLOB["return global_median constant"]
    SEG["lookup seg_medians, fallback to global_median"]
    ML["expm1(model.predict(X)).clip(0)"]
    VALERR["ValueError: invalid model name"]
    WRITE["write predictions TSV"]
    DONE(["Done"])

    START --> CHK
    CHK -->|No| ERR
    CHK -->|Yes| LOAD --> PREP --> PRED --> ROUTER
    ROUTER -->|global_baseline| GLOB --> WRITE
    ROUTER -->|seg_baseline| SEG --> WRITE
    ROUTER -->|hgb, rf, xgb, lgb| ML --> WRITE
    ROUTER -->|other| VALERR
    WRITE --> DONE
```

---

## 4. `evaluate` Command

```mermaid
flowchart TD
    START(["evaluate"])
    CHK{"model file exists?"}
    ERR["FileNotFoundError"]
    LOAD["load_model_suite"]
    PREP["load_and_prepare + build_features"]
    SPLIT["chronological_split"]
    DRIFT{target drift > 10%?}
    WARN["log warning"]
    EVAL["evaluate all available models"]
    BOARD["print_leaderboard"]
    DONE(["Done"])

    START --> CHK
    CHK -->|No| ERR
    CHK -->|Yes| LOAD --> PREP --> SPLIT --> DRIFT
    DRIFT -->|Yes| WARN --> EVAL
    DRIFT -->|No| EVAL
    EVAL --> BOARD --> DONE([" Done "])
```

---

## 5. Preprocessing Pipeline

```mermaid
flowchart TD
    RAW[("visits.tsv")]
    VAL["load_visits_data: validate columns"]
    COLQ{columns missing?}
    ERR["ValueError"]
    PARSE["process_product_columns: parse lists, deduplicate, add total_* counts"]
    DT["add_datetime_features: UNIX-ms to end_dt, derive start_dt, extract hour + dayofweek"]
    TARGET["build_return_time_target: sort by customer + time, compute return_hours via next visit gap"]
    NEGQ{negative gaps?}
    CLIP["clip to 0, log warning"]
    OUT[("prepared DataFrame")]

    RAW --> VAL --> COLQ
    COLQ -->|Yes| ERR
    COLQ -->|No| PARSE --> DT --> TARGET --> NEGQ
    NEGQ -->|Yes| CLIP --> OUT
    NEGQ -->|No| OUT
```

---

## 6. Feature Engineering Pipeline

All features use `shift(1)` within customer groups - no future leakage.

```mermaid
flowchart TD
    IN[("preprocessed DataFrame")]
    SESS["add_session_features: time_spent float, buy_ratio, cart_ratio"]
    HIST["add_purchase_history_features: ever_bought, cumulative_bought_visits, cumulative_buy_rate"]
    LAG["add_lag_features: prev_gap_hours, prev_time_spent, prev_search_count, prev_viewed, prev_bought"]
    ROLL["add_rolling_features: 3-visit rolling averages, expanding cumulative_avg_gap"]
    OUT[("23 features in FEATURE_COLS")]

    IN --> SESS --> HIST --> LAG --> ROLL --> OUT
```

---

## 7. `ModelSuite.predict()` Routing

```mermaid
flowchart TD
    CALL["predict(model_name, X)"]
    R{model_name?}

    GBQ{global_median fitted?}
    GBERR["RuntimeError"]
    GBRET["return constant global_median"]

    SBQ{seg_medians fitted?}
    SBERR["RuntimeError"]
    SBRET["lookup seg_medians, fallback to global_median"]

    ML["look up in model map"]
    MLNULL{model is None?}
    MLERR["ValueError"]
    MLRET["expm1(model.predict(X)).clip(0)"]

    UNK["ValueError: invalid name"]

    CALL --> R
    R -->|global_baseline| GBQ
    GBQ -->|No| GBERR
    GBQ -->|Yes| GBRET

    R -->|seg_baseline| SBQ
    SBQ -->|No| SBERR
    SBQ -->|Yes| SBRET

    R -->|hgb, rf, xgb, lgb| ML --> MLNULL
    MLNULL -->|Yes| MLERR
    MLNULL -->|No| MLRET

    R -->|other| UNK
```

---

## 8. Chronological Split

No shuffling - the val set is always in the future relative to train.

```mermaid
flowchart TD
    IN["chronological_split(df, val_fraction)"]
    CHK{val_fraction in range?}
    ERR["ValueError"]
    Q["split_time = quantile(1 - val_fraction) of end_dt"]
    TRAIN["train_df: end_dt <= split_time"]
    VAL["val_df:   end_dt >  split_time"]
    OUT[("train_df, val_df - no overlap")]

    IN --> CHK
    CHK -->|No| ERR
    CHK -->|Yes| Q --> TRAIN & VAL --> OUT
```

---

## 9. Target Construction (`return_hours`)

```mermaid
flowchart TD
    SORT["sort by customer_id, start_dt"]
    SHIFT["next_start_dt = shift(-1) per customer"]
    NULLQ{next_start_dt is NaN?}
    LAST["return_hours = NaN, visit_is_this_last = True"]
    GAP["gap = (next_start_dt - end_dt) in hours"]
    NEGQ{gap < 0?}
    CLIP["return_hours = 0"]
    KEEP["return_hours = gap"]

    SORT --> SHIFT --> NULLQ
    NULLQ -->|Yes| LAST
    NULLQ -->|No| GAP --> NEGQ
    NEGQ -->|Yes| CLIP
    NEGQ -->|No| KEEP
```
