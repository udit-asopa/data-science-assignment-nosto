from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import FEATURE_COLS

TARGET_COL: str = "return_hours"
MODEL_FILENAME: str = "model_suite.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# ── Utility: chronological train / val split ─────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────
# ── ModelSuite class ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────


class ModelSuite:
    """Train, store, and serve all return-time regression models."""

    def __init__(self) -> None:
        self.feature_cols: list[str] = FEATURE_COLS
        self.target_col: str = TARGET_COL

        # Baseline state
        self.global_median: float | None = None
        self.seg_medians: pd.Series | None = None

        # ML model state (populated by train_all)
        self.hgb: HistGradientBoostingRegressor | None = None
        self.rf: Pipeline | None = None
        self.xgb: Pipeline | None = None
        self.lgb: Pipeline | None = None

    def _fit_pipeline(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_log: pd.Series,
        keep_feature_names: bool = False,
    ) -> Pipeline:
        imputer = SimpleImputer(strategy="median")
        if keep_feature_names:
            imputer = imputer.set_output(transform="pandas")
        model = Pipeline([("imputer", imputer), ("model", estimator)])
        model.fit(X_train, y_log)
        return model

    def _log_step(self, label: str, verbose: bool) -> None:
        if verbose:
            print(f"  Training {label}...", end=" ", flush=True)

    def _done_step(self, verbose: bool) -> None:
        if verbose:
            print("done.")

    def train_all(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """Train every model in the suite on *train_df*."""
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        y_log = np.log1p(y_train)

        self._log_step("Global median baseline", verbose)
        self.global_median = float(y_train.median())
        self._done_step(verbose)

        self._log_step("Segmented median baseline", verbose)
        self.seg_medians = train_df.groupby(["ever_bought", "start_dayofweek"])[
            self.target_col
        ].median()
        self._done_step(verbose)

        self._log_step("HistGradientBoosting", verbose)
        self.hgb = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=400,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42,
        )
        self.hgb.fit(X_train, y_log)
        self._done_step(verbose)

        self._log_step("RandomForest", verbose)
        self.rf = self._fit_pipeline(
            RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=10,
                max_features=0.5,
                n_jobs=-1,
                random_state=42,
            ),
            X_train,
            y_log,
        )
        self._done_step(verbose)

        self._log_step("XGBoost", verbose)
        self.xgb = self._fit_pipeline(
            xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=8,
                min_child_weight=50,
                subsample=0.8,
                objective="reg:absoluteerror",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            ),
            X_train,
            y_log,
        )
        self._done_step(verbose)

        self._log_step("LightGBM DART", verbose)
        self.lgb = self._fit_pipeline(
            lgb.LGBMRegressor(
                boosting_type="dart",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                min_child_samples=30,
                objective="mae",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
            X_train,
            y_log,
            keep_feature_names=True,
        )
        self._done_step(verbose)

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Return predictions in hours for the requested model."""
        if model_name == "global_baseline":
            if self.global_median is None:
                raise RuntimeError(
                    "Model not trained - call train_all() first."
                )
            return np.full(len(X), self.global_median)

        if model_name == "seg_baseline":
            if self.seg_medians is None or self.global_median is None:
                raise RuntimeError(
                    "Model not trained - call train_all() first."
                )
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
        if model_name not in model_map:
            valid = list(model_map.keys()) + ["global_baseline", "seg_baseline"]
            raise ValueError(
                f"Unknown model '{model_name}'. Valid options: {valid}"
            )
        model = model_map[model_name]
        if model is None:
            raise RuntimeError("Model not trained — call train_all() first.")

        return np.expm1(model.predict(X)).clip(min=0)

    def save(self, model_dir: str | Path) -> Path:
        """Save the trained suite to disk."""
        path = Path(model_dir) / MODEL_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path


def load_model_suite(model_dir: str | Path) -> ModelSuite:
    """Load a trained ModelSuite from disk."""
    path = Path(model_dir) / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at '{path}'. Run python main.py train first."
        )
    return joblib.load(path)
