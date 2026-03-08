# Feature Engineering

## Design Principles

1. **No leakage**: every feature uses only information available at the moment the current visit ends
2. **`shift(1)` discipline**: all per-customer lag/cumulative features are shifted by one visit before aggregating, so the current visit never contributes to its own features
3. **NaN tolerance**: features for first-visit rows are left as `NaN` and imputed downstream (median imputation inside the sklearn `Pipeline`)

---

## Feature Groups

### 1 - Session Features (current visit)

What happened during this visit. Computed directly from raw columns - no look-back required.

| Feature | Source | Description |
|---|---|---|
| `total_viewed_products` | `len(viewed_products)` | Distinct product types viewed |
| `total_bought_products` | `len(bought_products)` | Distinct product types purchased |
| `total_put_in_cart_products` | `len(put_in_cart_products)` | Distinct product types carted |
| `num_of_times_search_was_used` | raw column | Search interactions |
| `time_spent_in_minutes` | raw column (converted to float) | Session duration |
| `buy_ratio` | `bought / viewed` | Purchase conversion within visit (0 if viewed=0) |
| `cart_ratio` | `carted / viewed` | Cart conversion within visit |
| `start_hour` | `start_dt.hour` | Hour of day the visit began (UTC) |
| `start_dayofweek` | `start_dt.dayofweek + 1` | Day of week (1=Mon, 7=Sun) |
| `visit_bought_flag` | `total_bought > 0` | Boolean - did the customer buy anything? |

### 2 - Sequence Position

| Feature | Description |
|---|---|
| `visit_counter_index` | 0-based position in this customer's visit history (0 = first ever visit) |

Captures where the customer is in their lifecycle - new visitors behave differently from regulars.

### 3 - Purchase History (leak-safe cumulative)

Computed from **past visits only** using `shift(1)` before cumulative aggregations.

| Feature | Formula | Description |
|---|---|---|
| `ever_bought` | `cummax(shift(visit_bought_flag))` | Has this customer ever bought before this visit? |
| `cumulative_bought_visits` | `cumsum(shift(visit_bought_flag))` | Number of past visits where a purchase was made |
| `cumulative_buy_rate` | `cumulative_bought_visits / visit_counter_index` | Fraction of past visits with a purchase (NaN on first visit) |

**Why `shift(1)`?** Without the shift, a visit where the customer buys would see `ever_bought=1` for itself - using the outcome to predict itself. The shift ensures the value reflects history *before* the current visit.

```python
# Example implementation (vectorised, no loops)
feature_engineered_df["ever_bought"] = (
    grouped_by_bought_flag
    .shift(1)
    .fillna(False)
    .astype(bool)
    .groupby(feature_engineered_df["customer_id"])
    .cummax()
    .astype(int)
)
```

### 4 - Lag / Recency Features

Information about the immediately preceding visit.

| Feature | Description |
|---|---|
| `prev_gap_hours` | Hours between end of previous visit and start of current visit (clipped to 0) |
| `prev_time_spent` | Duration of previous visit (minutes) |
| `prev_search_count` | Search count from previous visit |
| `prev_viewed_count` | Products viewed in previous visit |
| `prev_bought` | Whether the customer bought in the previous visit (float; NaN for first visit) |

All lag features are `NaN` for a customer's first visit.

### 5 - Rolling History (3-visit window)

Smoothed signal over the last 3 visits, computed on the shifted lag columns so the current visit is excluded.

| Feature | Description |
|---|---|
| `rolling_avg_gap` | 3-visit rolling mean of `prev_gap_hours` |
| `rolling_avg_time_spent` | 3-visit rolling mean of `prev_time_spent` |
| `rolling_avg_viewed` | 3-visit rolling mean of `prev_viewed_count` |

`min_periods=1` is used so single-visit customers still get a value from their one available lag.

### 6 - Customer-level Recency

| Feature | Description |
|---|---|
| `cumulative_avg_gap` | Expanding mean of all past gaps (excluding current), captures the customer's long-run return rhythm |

`cumulative_avg_gap` is `shift(1)` applied to `prev_gap_hours` before `expanding().mean()` - ensures neither the current gap nor the immediately previous gap pollute the long-run average.

---

## Final Feature Set

23 features total, passed to all models:

```
Session       : total_viewed_products, total_put_in_cart_products, total_bought_products,
                num_of_times_search_was_used, time_spent_in_minutes, buy_ratio, cart_ratio,
                start_hour, start_dayofweek, visit_bought_flag

Sequence      : visit_counter_index

Purchase hist : ever_bought, cumulative_bought_visits, cumulative_buy_rate

Lag / recency : prev_gap_hours, prev_time_spent, prev_search_count, prev_viewed_count, prev_bought

Rolling       : rolling_avg_gap, rolling_avg_time_spent, rolling_avg_viewed

Customer-level: cumulative_avg_gap
```

---

## NaN Handling

| Source | Affected features | Treatment |
|---|---|---|
| First-visit rows | All lag/recency/rolling/history features | Median imputation inside sklearn `Pipeline` |
| `cumulative_buy_rate` | Rows with `past_visits_count = 0` | Left as `NaN` (denominator guard) → imputed |
| `buy_ratio`, `cart_ratio` | Visits with 0 viewed products | Explicitly set to `0.0` (no division by zero) |

HGB handles NaN natively; RF and XGBoost use a `SimpleImputer(strategy="median")` as the first pipeline step.
