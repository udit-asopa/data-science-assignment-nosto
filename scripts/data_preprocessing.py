from __future__ import annotations

from pathlib import Path
from typing import Any

import ast
import pandas as pd

# ── Column ───────────────────────────────────────────────────────────────────
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
    """Load visits TSV and validate required columns."""
    path = Path(data_path)
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    return df


def process_product_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse product-list columns, dedupe in-order, and add total_* counts."""
    out = df.copy()
    for col in PRODUCT_LIST_COLUMNS:
        out[col] = out[col].apply(ast.literal_eval)
        # order-preserving deduplication via dict.fromkeys
        out[col] = out[col].apply(lambda xs: list(dict.fromkeys(xs)))
        out[f"total_{col}"] = out[col].apply(len)
    return out


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build end/start datetime features and day/hour columns."""
    out = df.copy()

    out["end_dt"] = pd.to_datetime(
        out["end"], unit="ms", utc=False).dt.floor("10ms")
    out["end_hour"] = out["end_dt"].dt.hour
    out["end_dayofweek"] = out["end_dt"].dt.dayofweek + 1  # 1=Mon

    # Keep as timedelta here; feature engineering converts it back to float minutes.
    out["time_spent_in_minutes"] = pd.to_timedelta(
        out["time_spent_in_minutes"], unit="m")

    out["start_dt"] = (
        out["end_dt"] - out["time_spent_in_minutes"]
        ).dt.floor("10ms")
    out["start_hour"] = out["start_dt"].dt.hour
    out["start_dayofweek"] = out["start_dt"].dt.dayofweek + 1  # 1=Mon

    return out


# ── Step 1: Target variable ─────────────────────────────────────────────────
def build_return_time_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add return_hours target and visit metadata per customer timeline."""
    out = df.sort_values(["customer_id", "start_dt"]).reset_index(drop=True)
    grp = out.groupby("customer_id")

    out["next_start_dt"] = grp["start_dt"].shift(-1)

    raw_gap = (out["next_start_dt"] - out["end_dt"]).dt.total_seconds() / 3600
    n_negative = int((raw_gap < 0).sum())
    if n_negative:
        print(
            f"  ⚠  {n_negative} overlapping sessions (negative gap) clipped to 0."
        )

    out["return_hours"] = raw_gap.clip(lower=0)
    out["visit_is_this_last"] = out["next_start_dt"].isna()
    out["visit_counter_index"] = grp.cumcount()
    out["visit_bought_flag"] = out["total_bought_products"] > 0

    return out


# ── Orchestrator ────────────────────────────────────────────────────────────
def load_and_prepare(data_path: str | Path) -> pd.DataFrame:
    """Run full preprocessing: load, products, datetime features, target."""
    df = load_visits_data(data_path)
    df = process_product_columns(df)
    df = add_datetime_features(df)
    df = build_return_time_target(df)
    return df


# ── Optional: data quality check ────────────────────────────────────────────
def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a summary of basic data quality statistics for the DataFrame."""
    visits_per_customer = df.groupby("customer_id").size()
    return {
        "row_count": int(len(df)),
        "unique_customers": int(df["customer_id"].nunique()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_time_spent_count": int(
            (df["time_spent_in_minutes"] < pd.Timedelta(0)).sum()
        ),
        "visits_per_customer_min": int(visits_per_customer.min()),
        "visits_per_customer_median": float(visits_per_customer.median()),
        "visits_per_customer_max": int(visits_per_customer.max()),
    }