from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MedianBaselineModel:
	median_target: float


def time_based_split(
	dataframe: Any,
	time_column: str = "end",
	validation_fraction: float = 0.2,
) -> tuple[Any, Any]:
	if not 0 < validation_fraction < 1:
		raise ValueError("validation_fraction must be between 0 and 1")

	sorted_df = dataframe.sort_values(time_column).reset_index(drop=True)
	split_index = int(len(sorted_df) * (1 - validation_fraction))
	train_df = sorted_df.iloc[:split_index].copy()
	validation_df = sorted_df.iloc[split_index:].copy()
	return train_df, validation_df


def train_median_baseline(target_series: Any) -> MedianBaselineModel:
	if target_series.empty:
		raise ValueError("Target series is empty")
	return MedianBaselineModel(median_target=float(target_series.median()))


def predict_median_baseline(model: MedianBaselineModel, size: int) -> list[float]:
	return [model.median_target] * size
