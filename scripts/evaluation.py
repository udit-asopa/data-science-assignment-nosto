from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
	return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
	return float(np.sqrt(np.mean((y_true.to_numpy() - y_pred.to_numpy()) ** 2)))


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
	return {
		"mae": mean_absolute_error(y_true, y_pred),
		"rmse": root_mean_squared_error(y_true, y_pred),
	}
