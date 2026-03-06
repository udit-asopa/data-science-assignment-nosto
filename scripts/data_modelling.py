from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import FEATURE_COLS

TARGET_COL: str = "return_hours"
MODEL_FILENAME: str = "model_suite.joblib"


# ── Utility: chronological train / val split ─────────────────────────────────

def chronological_split(
	df: pd.DataFrame,
	time_col: str = "end_dt",
	val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Split *df* into train / validation by time, not by random shuffle.

	The split point is the (1 - val_fraction) quantile of *time_col*,
	mirroring a real deployment scenario where the model trains on past data
	and is validated against future data.

	Returns (train_df, val_df).
	"""
	if not 0 < val_fraction < 1:
		raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
	split_time = df[time_col].quantile(1 - val_fraction)
	train_df = df[df[time_col] <= split_time].copy()
	val_df = df[df[time_col] > split_time].copy()
	return train_df, val_df


# ── ModelSuite class ──────────────────────────────────────────────────────────

class ModelSuite:
	"""Container for all return-time regression models.

	Trains and stores six models that predict *return_hours* (how many hours
	until a customer's next visit):

	  global_baseline : always predict the training-set median (simplest baseline)
	  seg_baseline    : segment-aware median keyed on (ever_bought, start_dayofweek)
	  hgb             : HistGradientBoostingRegressor (MAE loss)
	  rf              : RandomForest inside an sklearn Pipeline (median imputation)
	  xgb             : XGBoost inside an sklearn Pipeline (median imputation)
	  lgb             : LightGBM DART inside an sklearn Pipeline (median imputation)

	All ML models train on log1p(return_hours) and output predictions in the
	original hours scale (via expm1), which handles the heavy right-skew of
	the target without distorting residuals.
	"""

	def __init__(self) -> None:
		self.feature_cols: list[str] = FEATURE_COLS
		self.target_col: str = TARGET_COL

		# Baseline state
		self.global_median: float | None = None
		self.seg_medians: pd.Series | None = None   # MultiIndex: (ever_bought, start_dayofweek)

		# ML model state (populated by train_all)
		self.hgb: HistGradientBoostingRegressor | None = None
		self.rf: Pipeline | None = None
		self.xgb: Pipeline | None = None
		self.lgb: Pipeline | None = None

	# ── Private: individual model trainers ───────────────────────────────────

	def _train_global_median(self, train_df: pd.DataFrame) -> None:
		self.global_median = float(train_df[self.target_col].median())

	def _train_seg_median(self, train_df: pd.DataFrame) -> None:
		# Compute median return_hours per (ever_bought × start_dayofweek) segment.
		self.seg_medians = (
			train_df.groupby(["ever_bought", "start_dayofweek"])[self.target_col]
			.median()
		)

	def _train_hgb(self, X_train: pd.DataFrame, y_log: pd.Series) -> None:
		# HistGradientBoosting handles NaN natively — no imputer needed.
		self.hgb = HistGradientBoostingRegressor(
			loss="absolute_error",
			learning_rate=0.05,
			max_iter=400,
			max_depth=6,
			min_samples_leaf=30,
			random_state=42,
		)
		self.hgb.fit(X_train, y_log)

	def _train_rf(self, X_train: pd.DataFrame, y_log: pd.Series) -> None:
		self.rf = Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("model", RandomForestRegressor(
				n_estimators=500,
				max_depth=10,
				min_samples_leaf=10,
				max_features=0.5,
				n_jobs=-1,
				random_state=42,
			)),
		])
		self.rf.fit(X_train, y_log)

	def _train_xgb(self, X_train: pd.DataFrame, y_log: pd.Series) -> None:
		self.xgb = Pipeline([
			("imputer", SimpleImputer(strategy="median")),
			("model", xgb.XGBRegressor(
				n_estimators=500,
				learning_rate=0.01,      # slow learner: more robust to drift
				max_depth=8,
				min_child_weight=50,     # strong regularisation
				subsample=0.8,
				colsample_bytree=1.0,
				objective="reg:absoluteerror",
				n_jobs=-1,
				random_state=42,
				verbosity=0,
			)),
		])
		self.xgb.fit(X_train, y_log)

	def _train_lgb(self, X_train: pd.DataFrame, y_log: pd.Series) -> None:
		# set_output(transform="pandas") preserves column names so LightGBM
		# can use them for its internal feature-name tracking.
		self.lgb = Pipeline([
			("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
			("model", lgb.LGBMRegressor(
				boosting_type="dart",   # dropout prevents over-fitting the training distribution
				n_estimators=400,
				learning_rate=0.05,
				max_depth=6,
				min_child_samples=30,
				objective="mae",
				n_jobs=-1,
				random_state=42,
				verbose=-1,
			)),
		])
		self.lgb.fit(X_train, y_log)

	# ── Public: orchestration ────────────────────────────────────────────────

	def train_all(
		self,
		train_df: pd.DataFrame,
		verbose: bool = True,
		skip_lgb: bool = False,
	) -> None:
		"""Train every model in the suite on *train_df*.

		Args:
		    train_df : DataFrame after feature engineering, filtered to rows
		               with a known target (return_hours is not NaN).
		    verbose  : print progress lines while training.
		    skip_lgb : skip LightGBM DART (slow) for quick experimentation.
		"""
		X_train = train_df[self.feature_cols]
		y_train = train_df[self.target_col]
		y_log = np.log1p(y_train)

		# (label, callable, needs_X_y)
		# Baselines only need train_df; ML models need X_train + y_log.
		steps: list[tuple[str, bool]] = [
			("Global median baseline", False),
			("Segmented median baseline", False),
			("HistGradientBoosting", True),
			("RandomForest", True),
			("XGBoost", True),
		]
		if not skip_lgb:
			steps.append(("LightGBM DART", True))

		trainers = {
			"Global median baseline": lambda: self._train_global_median(train_df),
			"Segmented median baseline": lambda: self._train_seg_median(train_df),
			"HistGradientBoosting": lambda: self._train_hgb(X_train, y_log),
			"RandomForest": lambda: self._train_rf(X_train, y_log),
			"XGBoost": lambda: self._train_xgb(X_train, y_log),
			"LightGBM DART": lambda: self._train_lgb(X_train, y_log),
		}

		for label, _ in steps:
			if verbose:
				print(f"  Training {label}...", end=" ", flush=True)
			trainers[label]()
			if verbose:
				print("done.")

	# ── Public: prediction ───────────────────────────────────────────────────

	def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
		"""Return predictions in the original *hours* scale for *model_name*.

		For "seg_baseline", *X* must contain the columns "ever_bought" and
		"start_dayofweek" (both present in FEATURE_COLS, so X is sufficient).

		model_name choices:
		    "global_baseline", "seg_baseline", "hgb", "rf", "xgb", "lgb"
		"""
		if model_name == "global_baseline":
			if self.global_median is None:
				raise RuntimeError("Model not trained — call train_all() first.")
			return np.full(len(X), self.global_median)

		if model_name == "seg_baseline":
			if self.seg_medians is None or self.global_median is None:
				raise RuntimeError("Model not trained — call train_all() first.")
			# Join the per-segment median onto X using (ever_bought, start_dayofweek).
			# Rows without a matching segment fall back to the global median.
			seg = X[["ever_bought", "start_dayofweek"]].join(
				self.seg_medians.rename("seg_median"),
				on=["ever_bought", "start_dayofweek"],
			)
			return seg["seg_median"].fillna(self.global_median).to_numpy()

		model_map: dict[str, object] = {
			"hgb": self.hgb,
			"rf": self.rf,
			"xgb": self.xgb,
			"lgb": self.lgb,
		}
		model = model_map.get(model_name)
		if model is None:
			valid = list(model_map.keys()) + ["global_baseline", "seg_baseline"]
			raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")

		return np.expm1(model.predict(X)).clip(min=0)

	def predict_best(self, X: pd.DataFrame) -> np.ndarray:
		"""Predict using LightGBM DART — the best-performing model."""
		return self.predict("lgb", X)

	# ── Persistence ──────────────────────────────────────────────────────────

	def save(self, model_dir: str | Path) -> Path:
		"""Serialise the entire ModelSuite to *model_dir* using joblib."""
		path = Path(model_dir) / MODEL_FILENAME
		path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self, path)
		return path

	@classmethod
	def load(cls, model_dir: str | Path) -> "ModelSuite":
		"""Load a previously saved ModelSuite from *model_dir*."""
		path = Path(model_dir) / MODEL_FILENAME
		if not path.exists():
			raise FileNotFoundError(
				f"No saved model at '{path}'. Run  python main.py train  first."
			)
		return joblib.load(path)
