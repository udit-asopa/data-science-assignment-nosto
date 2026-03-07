from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import (
    permutation_importance as sklearn_permutation_importance,
)


def mean_absolute_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.to_numpy() - np.asarray(y_pred))))


def root_mean_squared_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((y_true.to_numpy() - np.asarray(y_pred)) ** 2))
    )


def regression_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Return both MAE and RMSE as a dict."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"mae": mae, "rmse": rmse}


def check_target_drift(
    y_train: pd.Series,
    y_val: pd.Series,
    threshold_pct: float = 10.0,
) -> dict[str, float | bool]:
    """
    Compare the median return_hours between train and validation sets.

    A drift > threshold_pct % is flagged as significant; this typically means
    the validation distribution is different enough from training that models
    trained naively will be systematically biased.

    Returns a dict with keys: train_median_h, val_median_h, drift_pct, significant.
    """
    train_median = float(y_train.median())
    val_median = float(y_val.median())
    if train_median == 0:
        drift_pct = 0.0 if val_median == 0 else float("inf")
    else:
        drift_pct = abs(val_median - train_median) / train_median * 100

    return {
        "train_median_h": train_median,
        "val_median_h": val_median,
        "drift_pct": drift_pct,
        "significant": drift_pct > threshold_pct,
    }


def print_leaderboard(
    results: dict[str, dict[str, float]],
    baseline_key: str = "Global Median",
) -> None:
    """Print a formatted model leaderboard to stdout.

    Args:
        results     : {model_name: {"mae": float, "rmse": float}}
        baseline_key: which entry acts as the reference for % improvement.
    """
    baseline_mae = results[baseline_key]["mae"]
    best_mae = min(m["mae"] for m in results.values())
    col_w = max(len(k) for k in results) + 2

    header = f"  {'Model':<{col_w}} {'MAE (h)':>9} {'RMSE (h)':>10} {'vs Baseline':>13}"
    print(header)
    print("  " + "─" * (col_w + 35))

    for name, metrics in results.items():
        mae_v = metrics["mae"]
        rmse_v = metrics["rmse"]
        pct = (baseline_mae - mae_v) / baseline_mae * 100
        marker = " ← BEST" if mae_v == best_mae else ""
        print(
            f"  {name:<{col_w}} {mae_v:>9.2f} {rmse_v:>10.2f}"
            f"  {pct:>+10.1f}%{marker}"
        )


def compute_permutation_importance(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Run sklearn permutation importance against log1p(y_val).

    Scoring against log1p(y_val) matches the training objective and avoids
    extreme outliers dominating the importance scores.

    Returns a DataFrame with columns [feature, importance_mean, importance_std],
    sorted by importance_mean descending.
    """
    result = sklearn_permutation_importance(
        model,
        X_val,
        np.log1p(y_val),
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance_df.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
