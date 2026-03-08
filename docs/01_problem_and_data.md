# Problem Definition & Data

## Problem Statement

Predict **how many hours** a customer will take to return to the webshop after their current visit.

This is a **regression task** on a right-skewed, censored target derived from raw event logs. The prediction is made using only information available at the moment the current visit ends - no future data is used.

**Business value**: knowing the expected return window allows marketers to time re-engagement campaigns more precisely and identify customers at risk of churn.

---

## Raw Dataset

**File**: `data/visits.tsv` - 40,306 rows, 7 columns  
**Period**: ~7 weeks (20 Jul 2024 – 09 Sep 2024), single webshop

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique visitor identifier |
| `viewed_products` | list[int] | Product type IDs viewed this visit |
| `bought_products` | list[int] | Product type IDs purchased |
| `put_in_cart_products` | list[int] | Product type IDs added to cart |
| `num_of_times_search_was_used` | int | Search interactions this visit |
| `end` | int (ms) | UNIX timestamp of last action |
| `time_spent_in_minutes` | float | Duration from first to last action |

### Preprocessing

1. **List columns** (`viewed_products`, `bought_products`, `put_in_cart_products`) are stored as string-encoded lists - parsed with `ast.literal_eval` and deduplicated (order-preserving) to remove product ID repeats within the same visit
2. **Derived counts** (`total_viewed_products`, `total_bought_products`, `total_put_in_cart_products`) extracted as integer features
3. **Timestamps**: `end` converted from UNIX ms → `datetime`; `start_dt` back-computed as `end_dt − time_spent`
4. **Temporal features**: `start_hour`, `start_dayofweek` (1=Mon) extracted for both visit endpoints

---

## Target Variable Definition

**Target**: `return_hours` - hours from `end_dt` of the current visit to `start_dt` of the customer's next visit.

### Construction

```
visits sorted by (customer_id, start_dt)
return_hours[i] = next_visit.start_dt[i] − current_visit.end_dt[i]   (in hours)
```

- **Last visits** (no subsequent visit in the window): `return_hours = NaN` - **excluded** from modelling (these are censored observations; the true return could happen after the data window ends)
- **Negative gaps** (11 rows): caused by overlapping sessions across devices - clipped to 0

### Distribution

```
Count   : 30,916 rows (after dropping last visits)
Median  : 25.0h  (1.0 days)
P75     : 97.4h  (4.1 days)
P90     : 339.7h (14.2 days)
P99     : ~1,200h
```

The target is heavily **right-skewed** with a log-normal shape. A `log1p` transform is applied before modelling to stabilise variance and put the target in a range where gradient boosters optimise effectively.

---

## Exploratory Data Analysis

### EDA 2.1 - Visit frequency per customer

- **71.6% of customers appear only once** in the 7-week window
- Median visits per customer: 1; mean: ~1.7
- Long tail: a small number of customers visit 10–20+ times

**Implication**: the dataset is dominated by one-time visitors for whom no inter-visit history exists. Features relying on past visits (lag gaps, rolling averages) will be `NaN` for these customers and must be imputed.

### EDA 2.2 - Return time distribution

- Raw distribution is strongly right-skewed - unsuitable for regression without transformation
- `log1p(return_hours)` is approximately normal, motivating its use as the training target
- ECDF shows: ~50% of customers return within 25h, ~75% within 97h

### EDA 2.3 - Buyers vs browsers

| Segment | Median return |
|---|---|
| Bought this visit | 21.7h |
| Browse only | 25.8h |

**Buyers return ~4h sooner** - a subtle but consistent signal. Purchasing indicates higher engagement and drives a modestly faster return cadence.

### EDA 2.4 - Temporal patterns

- **Visit volume**: peaks mid-week (Tue–Thu), drops on weekends; peaks in the late afternoon/evening (UTC)
- **Median return time**: relatively flat across days; slightly longer from weekend visits, consistent with customers not returning until working hours

### EDA 2.5 - Visit sequence position

Customers on their **1st visit** have the longest median return (~30h); return time shortens as visit sequence progresses, stabilising around the 4th–6th visit (~20–22h). Regulars have more predictable, shorter return cadences.

---

## Key Structural Challenges

| Challenge | Impact |
|---|---|
| 63% one-time visitors | Lag/recency features are NaN for most rows |
| 7-week window only | Cannot capture weekly seasonality reliably |
| Last-visit censoring | True return time unknown for ~24% of rows - simply dropped, not modelled |
| 48.7% target drift | Val period customers return twice as fast as train period customers |
