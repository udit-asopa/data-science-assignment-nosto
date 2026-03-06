from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from scripts.helper_functions import parse_product_list

REQUIRED_COLUMNS = [
	"customer_id",
	"viewed_products",
	"bought_products",
	"put_in_cart_products",
	"num_of_times_search_was_used",
	"end",
	"time_spent_in_minutes",
]

PRODUCT_LIST_COLUMNS = [
	"viewed_products",
	"bought_products",
	"put_in_cart_products",
]


def load_visits_data(data_path: str | Path) -> pd.DataFrame:
	path = Path(data_path)
	dataframe = pd.read_csv(path, sep="\t")

	missing_columns = [col for col in REQUIRED_COLUMNS if col not in dataframe.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")

	return dataframe


def normalize_product_lists(dataframe: pd.DataFrame) -> pd.DataFrame:
	processed = dataframe.copy()
	for column in PRODUCT_LIST_COLUMNS:
		processed[column] = processed[column].apply(parse_product_list)
	return processed


def audit_visits_data(dataframe: pd.DataFrame) -> dict[str, Any]:
	parse_errors: dict[str, int] = {}
	empty_list_counts: dict[str, int] = {}

	for column in PRODUCT_LIST_COLUMNS:
		raw_series = dataframe[column]
		parsed_series = raw_series.apply(parse_product_list)
		parse_errors[column] = int(
			((raw_series.astype(str).str.strip() != "") \
	          & \
			 (parsed_series.apply(len) == 0)
			).sum()
		)
		empty_list_counts[column] = int((parsed_series.apply(len) == 0).sum())

	visits_per_customer = dataframe.groupby("customer_id").size()

	return {
		"row_count": int(len(dataframe)),
		"column_count": int(len(dataframe.columns)),
		"unique_customers": int(dataframe["customer_id"].nunique()),
		"missing_values": {k: int(v) for k, v in dataframe.isna().sum().to_dict().items()},
		"duplicate_rows": int(dataframe.duplicated().sum()),
		"list_parse_errors": parse_errors,
		"empty_list_counts": empty_list_counts,
		"negative_time_spent_count": int((dataframe["time_spent_in_minutes"] < 0).sum()),
		"negative_search_count": int((dataframe["num_of_times_search_was_used"] < 0).sum()),
		"end_timestamp_min": int(dataframe["end"].min()),
		"end_timestamp_max": int(dataframe["end"].max()),
		"time_spent_min": float(dataframe["time_spent_in_minutes"].min()),
		"time_spent_max": float(dataframe["time_spent_in_minutes"].max()),
		"visits_per_customer_min": int(visits_per_customer.min()),
		"visits_per_customer_median": float(median(visits_per_customer.tolist())),
		"visits_per_customer_max": int(visits_per_customer.max()),
	}


def load_and_audit_visits(data_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
	dataframe = load_visits_data(data_path)
	audit_summary = audit_visits_data(dataframe)
	normalized = normalize_product_lists(dataframe)
	return normalized, audit_summary


def build_return_time_target(
	dataframe: pd.DataFrame,
	customer_column: str = "customer_id",
	end_column: str = "end",
	target_column: str = "return_time_hours",
	drop_censored: bool = True,
) -> pd.DataFrame:
	target_df = dataframe.copy()
	target_df = target_df.sort_values([customer_column, end_column]).reset_index(drop=True)
	target_df["next_end"] = target_df.groupby(customer_column)[end_column].shift(-1)
	target_df[target_column] = (target_df["next_end"] - target_df[end_column]) / (1000 * 60 * 60)

	if drop_censored:
		target_df = target_df[target_df[target_column].notna()].reset_index(drop=True)

	return target_df
