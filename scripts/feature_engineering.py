from __future__ import annotations

import pandas as pd


def add_basic_behavior_features(dataframe: pd.DataFrame) -> pd.DataFrame:
	features = dataframe.copy()
	features["n_viewed_products"] = features["viewed_products"].apply(len)
	features["n_bought_products"] = features["bought_products"].apply(len)
	features["n_cart_products"] = features["put_in_cart_products"].apply(len)
	features["buy_to_view_ratio"] = (
		features["n_bought_products"] / features["n_viewed_products"].clip(lower=1)
	)
	features["cart_to_view_ratio"] = (
		features["n_cart_products"] / features["n_viewed_products"].clip(lower=1)
	)
	return features


def add_temporal_features(dataframe: pd.DataFrame, timestamp_column: str = "end") -> pd.DataFrame:
	features = dataframe.copy()
	dt = pd.to_datetime(features[timestamp_column], unit="ms", utc=True)
	features["visit_hour"] = dt.dt.hour
	features["visit_day_of_week"] = dt.dt.dayofweek
	features["visit_is_weekend"] = features["visit_day_of_week"].isin([5, 6]).astype(int)
	return features
