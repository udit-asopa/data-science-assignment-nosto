# Pipeline Diagrams

Decision logic and data-flow visualisations for the CLI tool and all supporting
modules. Every diagram is self-contained and can be rendered in any Mermaid-aware
viewer (VS Code with the Markdown Preview Mermaid Support extension, GitHub, etc.).

---

## 1. System Architecture

High-level view of how the three CLI commands relate to the script modules,
input data, and persisted artefacts.

```mermaid
flowchart LR
    TSV[("data/visits.tsv")]
    SUITE[("scripts/models/\nmodel_suite.joblib")]
    OUT[("output/predictions.tsv")]

    subgraph CLI ["main.py — Typer CLI"]
        TRAIN["train"]
        PREDICT["predict"]
        EVALUATE["evaluate"]
    end

    subgraph Scripts ["scripts/"]
        PRE["data_preprocessing.py"]
        FE["feature_engineering.py"]
        MOD["data_modelling.py\n(ModelSuite)"]
        EVAL["evaluation.py"]
    end

    TSV -->|input| TRAIN
    TSV -->|input| PREDICT
    TSV -->|input| EVALUATE

    TRAIN --> PRE --> FE --> MOD
    MOD --> EVAL
    TRAIN -->|saves| SUITE

    SUITE -->|loads| PREDICT
    PREDICT --> PRE
    PREDICT -->|writes| OUT

    SUITE -->|loads| EVALUATE
    EVALUATE --> PRE
```

---

## 2. `train` Command — Decision Flow

```mermaid
flowchart TD
    START(["python main.py train"])
    LOAD["load_and_prepare(data_path)\n─────────────────────\nRead TSV → parse product lists\nadd datetime features\nbuild return_hours target"]
    FEAT["build_features(df)\n─────────────────────\nSession ratios · purchase history\nlag features · rolling averages"]
    FILTER["Filter rows where return_hours is not NaN\n(last visit per customer has no known target)"]
    SPLIT["chronological_split(df, val_fraction)\n─────────────────────────────────────\nSplit at end_dt quantile(1 − val_fraction)\nTrain = older rows · Val = newer rows"]
    DRIFT["check_target_drift(y_train, y_val)"]
    DRIFTQ{drift > 10 %?}
    WARN["⚠  Print warning:\ndistributions may differ"]
    OK["✓  Distribution stable"]
    SUITE["suite = ModelSuite()"]
    LGBQ{--skip-lgb?}
    TRAIN5["train_all(train_df, skip_lgb=True)\nTrain 5 models:\nGlobal Median · Segmented Median\nHGB · RandomForest · XGBoost"]
    TRAIN6["train_all(train_df)\nTrain 6 models:\nGlobal Median · Segmented Median\nHGB · RandomForest · XGBoost\nLightGBM DART"]
    EVALALL["Predict all trained models on val set\nregression_metrics(y_val, y_pred)\nfor each model"]
    BOARD["print_leaderboard(results)\nMAE · RMSE · % vs Global Median baseline"]
    SAVE["suite.save(model_dir)\nWrite model_suite.joblib"]
    DONE(["Done"])

    START --> LOAD --> FEAT --> FILTER --> SPLIT --> DRIFT --> DRIFTQ
    DRIFTQ -->|Yes| WARN --> SUITE
    DRIFTQ -->|No| OK --> SUITE
    SUITE --> LGBQ
    LGBQ -->|Yes| TRAIN5 --> EVALALL
    LGBQ -->|No| TRAIN6 --> EVALALL
    EVALALL --> BOARD --> SAVE --> DONE
```

---

## 3. `predict` Command — Decision Flow

```mermaid
flowchart TD
    START(["python main.py predict"])
    CHKFILE{"model_suite.joblib\nexists?"}
    ERR["FileNotFoundError\n'Run python main.py train first'"]
    LOADM["ModelSuite.load(model_dir)\njoblib.load(model_suite.joblib)"]
    LOAD["load_and_prepare(data_path)\nAll rows kept — including\nlast-visit rows (no known target)"]
    FEAT["build_features(df)\nCompute all 23 FEATURE_COLS"]
    PRED["suite.predict(model_name, X)"]
    ROUTER{model_name}

    GLOB["return np.full(len(X), global_median)\nSame scalar for every row"]
    SEG["Join seg_medians on\n(ever_bought, start_dayofweek)\nfillna(global_median) for unseen segments"]
    ML["raw = model.predict(X)   ← log1p scale\npreds = expm1(raw).clip(min=0)   ← hours"]
    VALERR["ValueError\nPrint valid model name options"]

    BUILD["Build output DataFrame:\ncustomer_id · start_dt · end_dt\npredicted_return_hours · predicted_return_days"]
    ACTQ{"return_hours present\nin input data?"}
    WITHACT["Include actual return_hours\ncolumn in output"]
    NOACT["Omit column\n(unseen / future data)"]
    WRITE["Write TSV to output path\n(creates parent dirs if needed)"]
    DONE(["Done"])

    START --> CHKFILE
    CHKFILE -->|No| ERR
    CHKFILE -->|Yes| LOADM --> LOAD --> FEAT --> PRED --> ROUTER

    ROUTER -->|global_baseline| GLOB --> BUILD
    ROUTER -->|seg_baseline| SEG --> BUILD
    ROUTER -->|hgb · rf · xgb · lgb| ML --> BUILD
    ROUTER -->|other| VALERR

    BUILD --> ACTQ
    ACTQ -->|Yes| WITHACT --> WRITE
    ACTQ -->|No| NOACT --> WRITE
    WRITE --> DONE
```

---

## 4. `evaluate` Command — Decision Flow

```mermaid
flowchart TD
    START(["python main.py evaluate"])
    CHKFILE{"model_suite.joblib\nexists?"}
    ERR["FileNotFoundError"]
    LOADM["ModelSuite.load(model_dir)"]
    PREP["load_and_prepare + build_features\nReproduce same feature columns\nas used during training"]
    FILTER["Filter: return_hours is not NaN"]
    SPLIT["chronological_split(df, val_fraction)\nMust use same val_fraction as training\nto reproduce the identical split"]
    DRIFT["check_target_drift(y_train, y_val)"]
    DRIFTQ{drift > 10 %?}
    WARN["⚠  Print drift warning"]
    OK["✓  Stable"]
    EVAL5["Evaluate 5 always-present models:\nGlobal Median · Segmented Median\nHGB · RandomForest · XGBoost"]
    LGBQ{suite.lgb\nis not None?}
    EVAL6["Also evaluate LightGBM DART"]
    SKIPLGB["Skip LGB\n(was trained with --skip-lgb)"]
    BOARD["print_leaderboard(results)"]
    IMPQ{--importance\nflag set?}
    LGBAVAIL{suite.lgb\navailable?}
    NOWARN["⚠  LightGBM not available\n(skip-lgb was used at training time)"]
    PERM["compute_permutation_importance\n(suite.lgb, X_val, log1p(y_val))\nn_repeats=10, n_jobs=-1"]
    TOP15["Print top 15 features\nby importance_mean"]
    DONE(["Done"])

    START --> CHKFILE
    CHKFILE -->|No| ERR
    CHKFILE -->|Yes| LOADM --> PREP --> FILTER --> SPLIT --> DRIFT --> DRIFTQ
    DRIFTQ -->|Yes| WARN --> EVAL5
    DRIFTQ -->|No| OK --> EVAL5
    EVAL5 --> LGBQ
    LGBQ -->|Yes| EVAL6 --> BOARD
    LGBQ -->|No| SKIPLGB --> BOARD
    BOARD --> IMPQ
    IMPQ -->|No| DONE
    IMPQ -->|Yes| LGBAVAIL
    LGBAVAIL -->|No| NOWARN --> DONE
    LGBAVAIL -->|Yes| PERM --> TOP15 --> DONE
```

---

## 5. Preprocessing Pipeline (`data_preprocessing.py`)

```mermaid
flowchart TD
    RAW[("data/visits.tsv\nRaw tab-separated input")]
    VALIDATE["load_visits_data()\n──────────────────\npd.read_csv(sep=tab)\nCheck all REQUIRED_COLUMNS present"]
    COLCHECK{"Required columns\nmissing?"}
    COLERR["ValueError:\nlist missing column names"]
    PARSE["process_product_columns()\n──────────────────────────\nFor each of: viewed_products,\nbought_products, put_in_cart_products:\n  parse string → list[int]\n  deduplicate (order-preserving)\n  add total_* integer count column"]
    DATETIME["add_datetime_features()\n────────────────────────\nend (UNIX-ms) → end_dt (floor 10ms)\nend_hour, end_dayofweek\ntime_spent_in_minutes → timedelta\nstart_dt = end_dt − time_spent\nstart_hour, start_dayofweek"]
    TARGET["build_return_time_target()\n──────────────────────────\nSort by (customer_id, start_dt)\nnext_start_dt = shift(-1) per customer\nreturn_hours = (next_start_dt − end_dt)\n               in hours, clipped ≥ 0\nvisit_counter_index = cumcount()\nvisit_bought_flag = bought_products > 0\nvisit_is_this_last = next_start_dt is NaN"]
    NEGQ{"Negative gaps\ndetected?"}
    NEGWARN["Print: 'N overlapping sessions\n(negative gap) clipped to 0'"]
    OUT[("Prepared DataFrame\nSorted by (customer_id, start_dt)\nLast-visit rows kept\n(return_hours = NaN for them)")]

    RAW --> VALIDATE --> COLCHECK
    COLCHECK -->|Yes| COLERR
    COLCHECK -->|No| PARSE --> DATETIME --> TARGET --> NEGQ
    NEGQ -->|Yes| NEGWARN --> OUT
    NEGQ -->|No| OUT
```

---

## 6. Feature Engineering Pipeline (`feature_engineering.py`)

All steps operate within each customer group to prevent cross-customer leakage.
All lag/history features apply `shift(1)` so a visit cannot see its own outcome.

```mermaid
flowchart TD
    IN[("Preprocessed DataFrame\nfrom load_and_prepare()")]

    SESS["add_session_features()\n─────────────────────\nConvert time_spent timedelta → float minutes\nbuy_ratio  = total_bought / total_viewed\n             (0 if no products viewed)\ncart_ratio = total_carted / total_viewed\n             (0 if no products viewed)"]

    HIST["add_purchase_history_features()\n────────────────────────────────\npast_bought = shift(1) of visit_bought_flag\n              (looks only backwards)\never_bought           = cummax(past_bought)\ncumulative_bought_visits = cumsum(past_bought)\ncumulative_buy_rate = cumulative_bought / visit_counter\n                      (NaN for first visit)"]

    LAG["add_lag_features()\n───────────────────\nFor each customer, shift(1):\n  prev_end_dt\n  prev_time_spent\n  prev_search_count\n  prev_viewed_count\n  prev_bought\nprev_gap_hours = (start_dt − prev_end_dt)\n                 in hours, clipped ≥ 0"]

    ROLL["add_rolling_features()\n──────────────────────\nrolling(window=3, min_periods=1):\n  rolling_avg_gap\n  rolling_avg_time_spent\n  rolling_avg_viewed\nshift(1) + expanding().mean():\n  cumulative_avg_gap\n  (extra shift keeps current gap out)"]

    OUT[("Feature-complete DataFrame\n23 features in FEATURE_COLS\nReady for modelling")]

    IN --> SESS --> HIST --> LAG --> ROLL --> OUT
```

---

## 7. `ModelSuite.predict()` Routing Logic

```mermaid
flowchart TD
    CALL["predict(model_name, X)"]
    ROUTER{model_name}

    GB{"global_median\nfitted?"}
    GBERR["RuntimeError:\nModel not trained —\ncall train_all() first"]
    GBRET["return np.full(len(X), global_median)\nConstant prediction for all rows"]

    SB{"seg_medians\nfitted?"}
    SBERR["RuntimeError:\nModel not trained"]
    SBRET["Join seg_medians on\n(ever_bought × start_dayofweek)\nRows with unseen segment → fillna(global_median)\nreturn as numpy array"]

    ML["Look up model in\n{hgb, rf, xgb, lgb} map"]
    MLNULL{model is None?}
    MLERR["ValueError:\nModel unknown or not yet trained"]
    MLRET["raw_log = model.predict(X)\npreds    = expm1(raw_log).clip(min=0)\nreturn preds as numpy array"]

    UNK["ValueError:\nUnknown model_name\nPrint list of valid options"]

    CALL --> ROUTER
    ROUTER -->|"'global_baseline'"| GB
    GB -->|No| GBERR
    GB -->|Yes| GBRET

    ROUTER -->|"'seg_baseline'"| SB
    SB -->|No| SBERR
    SB -->|Yes| SBRET

    ROUTER -->|"'hgb' / 'rf' / 'xgb' / 'lgb'"| ML --> MLNULL
    MLNULL -->|Yes| MLERR
    MLNULL -->|No| MLRET

    ROUTER -->|anything else| UNK
```

---

## 8. Chronological Split Logic

Random shuffling would allow future data to leak into the training set.
The quantile-based split below guarantees the val set is strictly in the future
relative to the training set.

```mermaid
flowchart TD
    IN["chronological_split(df, time_col, val_fraction)"]
    CHECK{0 < val_fraction < 1?}
    ERR["ValueError: val_fraction out of range"]
    QUANTILE["split_time = df[time_col].quantile(1 − val_fraction)\ne.g. val_fraction=0.20 → quantile(0.80)"]
    TRAINDF["train_df = df[ end_dt <= split_time ]\nAll older / contemporary rows"]
    VALDF["val_df = df[ end_dt > split_time ]\nAll strictly newer rows"]
    NOOVERLAP{"Any row\noverlap?"}
    SAFE["No overlap by construction:\ntrain rows are never in val,\nval rows are never in train"]
    OUT[("Return (train_df, val_df)")]

    IN --> CHECK
    CHECK -->|No| ERR
    CHECK -->|Yes| QUANTILE
    QUANTILE --> TRAINDF & VALDF
    TRAINDF --> NOOVERLAP
    VALDF --> NOOVERLAP
    NOOVERLAP -->|Never| SAFE --> OUT
```

---

## 9. Target Construction Detail

How `return_hours` is built for each row within `build_return_time_target()`.

```mermaid
flowchart TD
    SORT["Sort all rows by (customer_id, start_dt)"]
    SHIFT["next_start_dt = shift(-1)\nwithin each customer group"]
    LAST{"Is next_start_dt\nNaN?"}
    LASTROW["This is the customer's\nlast recorded visit\nreturn_hours = NaN\nvisit_is_this_last = True"]
    GAP["raw_gap_hours =\n(next_start_dt − end_dt).total_seconds() / 3600"]
    CLIPQ{"raw_gap_hours < 0?\n(overlapping / multi-device sessions)"}
    CLIP["return_hours = 0\n(clipped — preserves row,\navoids negative targets)"]
    USE["return_hours = raw_gap_hours"]
    FLAG["visit_is_this_last = False"]

    SORT --> SHIFT --> LAST
    LAST -->|Yes| LASTROW
    LAST -->|No| GAP --> CLIPQ
    CLIPQ -->|Yes| CLIP --> FLAG
    CLIPQ -->|No| USE --> FLAG
```
