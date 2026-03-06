from __future__ import annotations

import pandas as pd

# ── Feature column registry ─────────────────────────────────────────────────
# All 23 model features, grouped by type for clarity.
# This list is the single source of truth consumed by modelling and evaluation.

FEATURE_COLS: list[str] = [
	# ── Within-session engagement ────────────────────────────────────────────
	"total_viewed_products",
	"total_put_in_cart_products",
	"total_bought_products",
	"num_of_times_search_was_used",
	"time_spent_in_minutes",   # float minutes (converted from timedelta)
	"buy_ratio",               # bought / viewed  (0 when viewed == 0)
	"cart_ratio",              # carted / viewed  (0 when viewed == 0)
	"start_hour",              # 0–23
	"start_dayofweek",         # 1=Mon … 7=Sun
	"visit_bought_flag",       # 1 if any purchase on this visit
	# ── Visit sequence position ──────────────────────────────────────────────
	"visit_counter_index",     # 0 = first ever visit for this customer
	# ── Cumulative purchase history (past visits only — leak-safe) ───────────
	"ever_bought",             # 1 if customer has ever bought before this visit
	"cumulative_bought_visits",# count of all prior buying visits
	"cumulative_buy_rate",     # cumulative_bought_visits / past_visits_count  (NaN for 1st visit)
	# ── Lag / recency features ───────────────────────────────────────────────
	"prev_gap_hours",          # hours between previous visit end and this visit start
	"prev_time_spent",         # session duration of the previous visit (minutes)
	"prev_search_count",       # searches used in the previous visit
	"prev_viewed_count",       # products viewed in the previous visit
	"prev_bought",             # 1.0 if customer purchased on the previous visit
	# ── Rolling averages (3-visit window over past visits) ───────────────────
	"rolling_avg_gap",
	"rolling_avg_time_spent",
	"rolling_avg_viewed",
	# ── Customer-level expanding average ─────────────────────────────────────
	"cumulative_avg_gap",      # expanding mean of prev_gap_hours (excluding current visit)
]


# ── Step 3a: Session features ───────────────────────────────────────────────

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute within-session engagement ratios.

	- time_spent_in_minutes: converted from timedelta → float minutes so it
	  can be used directly by ML models.
	- buy_ratio : total_bought / total_viewed  (0 when no products viewed).
	- cart_ratio: total_put_in_cart / total_viewed  (0 when no products viewed).
	"""
	out = df.copy()

	# Convert timedelta → float minutes (was stored as timedelta in preprocessing
	# so that start_dt could be derived; now we materialise it as a plain number).
	if pd.api.types.is_timedelta64_dtype(out["time_spent_in_minutes"]):
		out["time_spent_in_minutes"] = out["time_spent_in_minutes"].dt.total_seconds() / 60

	viewed = out["total_viewed_products"]
	out["buy_ratio"] = (out["total_bought_products"] / viewed).where(viewed > 0, 0.0)
	out["cart_ratio"] = (out["total_put_in_cart_products"] / viewed).where(viewed > 0, 0.0)

	return out


# ── Step 3b: Cumulative purchase history (leak-safe) ────────────────────────

def add_purchase_history_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Build cumulative purchase history using only information from *past* visits.

	The critical detail: shift(1) is applied before cummax / cumsum so the
	current visit's own outcome never leaks into its feature value.

	- ever_bought            : 1 if the customer purchased on any prior visit.
	- cumulative_bought_visits: how many prior visits included a purchase.
	- cumulative_buy_rate    : cumulative_bought_visits / visits seen so far
	                           (NaN for the very first visit, no prior history).
	"""
	out = df.copy()

	# shift(1) → use only information from visits *before* the current one
	past_bought = (
		out.groupby("customer_id")["visit_bought_flag"]
		.shift(1)
		.fillna(False)
		.astype(bool)
	)

	out["ever_bought"] = (
		past_bought
		.groupby(out["customer_id"])
		.cummax()
		.astype(int)
	)
	out["cumulative_bought_visits"] = (
		past_bought
		.groupby(out["customer_id"])
		.cumsum()
		.astype(int)
	)
	out["past_visits_count"] = out["visit_counter_index"]
	out["cumulative_buy_rate"] = (
		out["cumulative_bought_visits"] / out["past_visits_count"]
	).where(out["past_visits_count"] > 0)  # NaN for first visit (no history)

	return out


# ── Step 3c: Lag / recency features ─────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Carry forward key metrics from the previous visit per customer.

	- prev_gap_hours: hours between the *end* of the previous visit and the
	  *start* of the current visit (NaN for each customer's first visit;
	  negative gaps clipped to 0 for multi-device overlapping sessions).
	- prev_time_spent / prev_search_count / prev_viewed_count / prev_bought:
	  raw metrics from the immediately preceding visit.
	"""
	out = df.copy()
	grp = out.groupby("customer_id")

	out["prev_end_dt"] = grp["end_dt"].shift(1)
	out["prev_time_spent"] = grp["time_spent_in_minutes"].shift(1)
	out["prev_search_count"] = grp["num_of_times_search_was_used"].shift(1)
	out["prev_viewed_count"] = grp["total_viewed_products"].shift(1)
	out["prev_bought"] = grp["visit_bought_flag"].shift(1).astype(float)

	gap_td = out["start_dt"] - out["prev_end_dt"]
	out["prev_gap_hours"] = (gap_td.dt.total_seconds() / 3600).clip(lower=0)

	return out


# ── Step 3d: Rolling & expanding averages ───────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute rolling (3-visit) and expanding statistics over past inter-visit gaps.

	- rolling_avg_gap        : 3-visit rolling mean of prev_gap_hours.
	- rolling_avg_time_spent : 3-visit rolling mean of prev_time_spent.
	- rolling_avg_viewed     : 3-visit rolling mean of prev_viewed_count.
	- cumulative_avg_gap     : expanding mean of prev_gap_hours, but shifted
	  by one *additional* step so the current visit's gap does not contribute
	  to its own cumulative average.
	"""
	out = df.copy()

	def rolling3(s: pd.Series) -> pd.Series:
		return s.rolling(3, min_periods=1).mean()

	def expanding_mean(s: pd.Series) -> pd.Series:
		return s.expanding().mean()

	grp = out.groupby("customer_id")

	out["rolling_avg_gap"] = grp["prev_gap_hours"].transform(rolling3)
	out["rolling_avg_time_spent"] = grp["prev_time_spent"].transform(rolling3)
	out["rolling_avg_viewed"] = grp["prev_viewed_count"].transform(rolling3)

	# One extra shift: exclude the current visit's own lag value from the
	# expanding average so the feature is strictly historical.
	out["cumulative_avg_gap"] = (
		grp["prev_gap_hours"]
		.shift(1)
		.groupby(out["customer_id"])
		.transform(expanding_mean)
	)

	return out


# ── Orchestrator ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply all feature engineering steps in sequence.

	Input : preprocessed DataFrame from data_preprocessing.load_and_prepare().
	Output: same DataFrame with all columns in FEATURE_COLS present in addition
	        to the original columns.
	"""
	df = add_session_features(df)
	df = add_purchase_history_features(df)
	df = add_lag_features(df)
	df = add_rolling_features(df)
	return df
