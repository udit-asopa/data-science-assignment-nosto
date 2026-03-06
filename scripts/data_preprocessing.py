from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.helper_functions import parse_product_list

# ── Column contracts ────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
	"customer_id",
	"viewed_products",
	"bought_products",
	"put_in_cart_products",
	"num_of_times_search_was_used",
	"end",
	"time_spent_in_minutes",
]

PRODUCT_LIST_COLUMNS: list[str] = [
	"viewed_products",
	"bought_products",
	"put_in_cart_products",
]


# ── Step 0: Loading & Cleaning ──────────────────────────────────────────────

def load_visits_data(data_path: str | Path) -> pd.DataFrame:
	"""Load the raw visits TSV and validate that all required columns are present."""
	path = Path(data_path)
	df = pd.read_csv(path, sep="\t")
	missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
	if missing:
		raise ValueError(f"Input file is missing required columns: {missing}")
	return df


def process_product_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Parse the three product-list columns from raw strings into real Python lists,
	deduplicate each list while preserving insertion order, and add a 'total_*'
	integer count column for each.

	e.g.  "viewed_products" → list[int]  +  "total_viewed_products" → int
	"""
	out = df.copy()
	for col in PRODUCT_LIST_COLUMNS:
		out[col] = out[col].apply(parse_product_list)
		# order-preserving deduplication via dict.fromkeys
		out[col] = out[col].apply(lambda xs: list(dict.fromkeys(xs)))
		out[f"total_{col}"] = out[col].apply(len)
	return out


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Derive datetime columns and extract temporal features.

	- end_dt         : UNIX-ms timestamp → datetime, floored to 10 ms
	- end_hour       : hour of day (0–23)
	- end_dayofweek  : day of week (1=Mon … 7=Sun)
	- time_spent_in_minutes : converted to timedelta (used to derive start_dt,
	                          later converted back to float in feature engineering)
	- start_dt       : end_dt – time_spent_in_minutes, floored to 10 ms
	- start_hour     : hour of day the session started
	- start_dayofweek: day of week the session started
	"""
	out = df.copy()

	out["end_dt"] = pd.to_datetime(out["end"], unit="ms", utc=False).dt.floor("10ms")
	out["end_hour"] = out["end_dt"].dt.hour
	out["end_dayofweek"] = out["end_dt"].dt.dayofweek + 1  # 1=Mon

	# Store as timedelta so we can compute start_dt correctly.
	# Feature engineering will convert it back to float (minutes).
	out["time_spent_in_minutes"] = pd.to_timedelta(out["time_spent_in_minutes"], unit="m")
	out["start_dt"] = (out["end_dt"] - out["time_spent_in_minutes"]).dt.floor("10ms")
	out["start_hour"] = out["start_dt"].dt.hour
	out["start_dayofweek"] = out["start_dt"].dt.dayofweek + 1  # 1=Mon

	return out


# ── Step 1: Target variable ─────────────────────────────────────────────────

def build_return_time_target(df: pd.DataFrame) -> pd.DataFrame:
	"""Attach the regression target and visit-level metadata to each row.

	For each customer, visits are sorted chronologically by start_dt.
	The target is:
	    return_hours = next_visit.start_dt − current_visit.end_dt   (in hours)

	Negative gaps are clipped to 0 (caused by overlapping/multi-device sessions).
	Last visits per customer have no known next visit → return_hours is NaN.

	Added columns:
	  - return_hours       : target (hours until next visit; NaN for last visits)
	  - visit_is_this_last : True for each customer's final recorded visit
	  - visit_counter_index: 0-based chronological visit sequence number per customer
	  - visit_bought_flag  : True if ≥1 product was purchased on this visit
	"""
	out = df.sort_values(["customer_id", "start_dt"]).reset_index(drop=True)

	# For each visit, look up the start time of the customer's NEXT visit.
	out["next_start_dt"] = out.groupby("customer_id")["start_dt"].shift(-1)

	raw_gap = (out["next_start_dt"] - out["end_dt"]).dt.total_seconds() / 3600
	n_negative = int((raw_gap < 0).sum())
	if n_negative:
		print(f"  ⚠  {n_negative} overlapping sessions (negative gap) clipped to 0.")

	out["return_hours"] = raw_gap.clip(lower=0)
	out["visit_is_this_last"] = out["next_start_dt"].isna()
	out["visit_counter_index"] = out.groupby("customer_id").cumcount()
	out["visit_bought_flag"] = out["total_bought_products"] > 0

	return out


# ── Orchestrator ────────────────────────────────────────────────────────────

def load_and_prepare(data_path: str | Path) -> pd.DataFrame:
	"""Run the full preprocessing pipeline in one call.

	Steps: load TSV → parse product lists → datetime features → return-time target.

	The returned DataFrame is sorted by (customer_id, start_dt).
	Last-visit rows (return_hours = NaN) are *kept* so callers can decide
	whether to drop them (modelling needs to drop them; prediction keeps them).
	"""
	df = load_visits_data(data_path)
	df = process_product_columns(df)
	df = add_datetime_features(df)
	df = build_return_time_target(df)
	return df


# ── Optional: data quality audit ────────────────────────────────────────────

def audit_visits_data(df: pd.DataFrame) -> dict[str, Any]:
	"""Return a summary of basic data quality statistics for the raw DataFrame."""
	visits_per_customer = df.groupby("customer_id").size()
	return {
		"row_count": int(len(df)),
		"unique_customers": int(df["customer_id"].nunique()),
		"missing_values": {k: int(v) for k, v in df.isna().sum().items()},
		"duplicate_rows": int(df.duplicated().sum()),
		"negative_time_spent_count": int((df["time_spent_in_minutes"] < 0).sum()),
		"visits_per_customer_min": int(visits_per_customer.min()),
		"visits_per_customer_median": float(visits_per_customer.median()),
		"visits_per_customer_max": int(visits_per_customer.max()),
	}
