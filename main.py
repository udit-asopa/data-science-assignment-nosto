"""
Nosto - Customer Return Time Prediction CLI
===========================================
Predicts how many hours until a customer's next visit to the webshop.

Commands
--------
  train    : Preprocess data, train all models, evaluate on val set, save to disk.
  predict  : Load a trained ModelSuite and generate predictions for a dataset.
  evaluate : Load a trained ModelSuite and re-run the evaluation leaderboard.

Usage examples
--------------
  python main.py train   --data data/visits.tsv
  python main.py predict --data data/visits.tsv --output output/predictions.tsv
  python main.py evaluate --data data/visits.tsv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.data_modelling import (
    MODEL_FILENAME,
    TARGET_COL,
    ModelSuite,
    chronological_split,
    load_model_suite,
)
from scripts.data_preprocessing import audit_dataset, load_and_prepare
from scripts.evaluation import (
    check_target_drift,
    print_leaderboard,
    regression_metrics,
)
from scripts.feature_engineering import FEATURE_COLS, build_features

app = typer.Typer(
    name="nosto-return-time",
    help="Customer return-time prediction pipeline (Nosto data-science assignment).",
    add_completion=False,
)

# Shared defaults
_DEFAULT_DATA = Path("data/visits.tsv")
_DEFAULT_MODEL_DIR = Path("scripts/models")
_DEFAULT_OUTPUT = Path("output/predictions.tsv")

_EVAL_MODEL_ORDER: list[tuple[str, str]] = [
    ("Global Median", "global_baseline"),
    ("Segmented Median", "seg_baseline"),
    ("HGB", "hgb"),
    ("RandomForest", "rf"),
    ("XGBoost", "xgb"),
    ("LightGBM DART", "lgb"),
]


def _section(title: str) -> None:
    typer.echo(f"\n{title}")


def _build_leaderboard_results(
    suite: ModelSuite,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute metrics for all configured models."""
    results: dict[str, dict[str, float]] = {}
    for label, model_key in _EVAL_MODEL_ORDER:
        results[label] = regression_metrics(
            y_val, suite.predict(model_key, X_val)
        )
    return results


@app.command()
def train(
    data: Path = typer.Option(
        _DEFAULT_DATA, "--data", "-d",
        help="Path to the visits TSV file."
    ),
    model_dir: Path = typer.Option(
        _DEFAULT_MODEL_DIR, "--model-dir", "-m",
        help="Directory where the trained ModelSuite will be saved.",
    ),
    val_fraction: float = typer.Option(
        0.20, "--val-fraction", "-v",
        help="Fraction of data (by time) reserved for validation. Default: 0.20.",
    ),
) -> None:
    """Train all models, evaluate on validation, and save the suite."""
    typer.echo(f"Data      : {data}")
    typer.echo(f"Model dir : {model_dir}")
    typer.echo(
        f"Val split : {val_fraction:.0%} of data reserved for validation"
    )

    _section("Preprocessing")
    df = load_and_prepare(data)
    n_customers = df["customer_id"].nunique()
    typer.echo(
        f"  Loaded {len(df):,} rows from {n_customers:,} unique customers."
    )

    audit = audit_dataset(df)
    typer.echo(
        f"  Audit: duplicates={audit['duplicate_rows']:,}, "
        f"neg_time={audit['negative_time_spent_count']:,}"
    )

    _section("Feature Engineering")
    df = build_features(df)
    model_df = df[df[TARGET_COL].notna()].copy()
    n_dropped = len(df) - len(model_df)
    typer.echo(
        f"  Model-ready rows : {len(model_df):,}  ({n_dropped:,} last-visit rows excluded)."
    )

    _section("Chronological Split")
    train_df, val_df = chronological_split(model_df, val_fraction=val_fraction)
    typer.echo(
        f"  Train : {len(train_df):,} rows  ({train_df['end_dt'].min().date()} → {train_df['end_dt'].max().date()})"
    )
    typer.echo(
        f"  Val   : {len(val_df):,} rows  ({val_df['end_dt'].min().date()} → {val_df['end_dt'].max().date()})"
    )

    drift = check_target_drift(train_df[TARGET_COL], val_df[TARGET_COL])
    flag = "⚠" if drift["significant"] else "✓"
    typer.echo(
        f"  {flag}  Target drift: train median={drift['train_median_h']:.1f}h, "
        f"val median={drift['val_median_h']:.1f}h  ({drift['drift_pct']:.1f}%)"
    )

    _section("Training")
    suite = ModelSuite()
    suite.train_all(train_df, verbose=True)

    _section("Evaluation - Val Set")
    X_val: pd.DataFrame = val_df[FEATURE_COLS]
    y_val: pd.Series = val_df[TARGET_COL]
    results = _build_leaderboard_results(suite, X_val, y_val)
    print_leaderboard(results)

    _section("Saving")
    saved_path = suite.save(model_dir)
    typer.echo(f"  Saved → {saved_path}")
    typer.echo(
        "\nDone.  Run 'python main.py predict --help' to generate predictions."
    )


@app.command()
def predict(
    data: Path = typer.Option(
        _DEFAULT_DATA, "--data", "-d",
        help="Path to the visits TSV file (can be new, unseen data).",
    ),
    model_dir: Path = typer.Option(
        _DEFAULT_MODEL_DIR, "--model-dir", "-mp",
        help="Directory containing the saved ModelSuite.",
    ),
    output: Path = typer.Option(
        _DEFAULT_OUTPUT, "--output", "-o",
        help="Output TSV file path for predictions.",
    ),
    model_name: str = typer.Option(
        "lgb", "--model", "-mo",
        help="Which model to use: lgb (default), hgb, rf, xgb, global_baseline, seg_baseline.",
    )) -> None:
    """Load a trained suite and generate predictions for all rows."""

    typer.echo(f"Data   : {data}")
    typer.echo(f"Model  : {model_name}  (from {model_dir / MODEL_FILENAME})")
    typer.echo(f"Output : {output}")

    _section("Loading Model")
    suite = load_model_suite(model_dir)
    typer.echo(f"  ModelSuite loaded from {model_dir / MODEL_FILENAME}")

    _section("Preprocessing & Feature Engineering")
    df = load_and_prepare(data)
    df = build_features(df)
    typer.echo(f"  {len(df):,} rows ready for prediction.")

    _section("Prediction")
    X = df[FEATURE_COLS]
    preds = suite.predict(model_name, X)
    typer.echo(
        f"  Predictions: mean={preds.mean():.1f}h, median={np.median(preds):.1f}h"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[["customer_id", "start_dt", "end_dt"]].copy()
    out_df["predicted_return_hours"] = preds
    out_df["predicted_return_days"] = (preds / 24).round(2)
    if TARGET_COL in df.columns:
        out_df[TARGET_COL] = df[TARGET_COL]
    out_df.to_csv(output, sep="\t", index=False)
    typer.echo(f"  Saved {len(out_df):,} rows → {output}")


@app.command()
def evaluate(
    data: Path = typer.Option(
        _DEFAULT_DATA, "--data", "-d",
        help="Path to the visits TSV file."
    ),
    model_dir: Path = typer.Option(
        _DEFAULT_MODEL_DIR, "--model-dir", "-m",
        help="Directory containing the saved ModelSuite.",
    ),
    val_fraction: float = typer.Option(
        0.20, "--val-fraction", "-v",
        help="Fraction of data reserved for validation (must match training split).",
    )) -> None:
    """Evaluate a saved suite on the chronological validation split."""

    typer.echo(f"Data        : {data}")
    typer.echo(f"Model dir   : {model_dir}")
    typer.echo(f"Val fraction: {val_fraction:.0%}")

    _section("Loading Model")
    suite = load_model_suite(model_dir)
    typer.echo(f"  ModelSuite loaded from {model_dir / MODEL_FILENAME}")

    _section("Preprocessing & Feature Engineering")
    df = load_and_prepare(data)
    df = build_features(df)
    model_df = df[df[TARGET_COL].notna()].copy()

    train_df, val_df = chronological_split(model_df, val_fraction=val_fraction)
    X_val: pd.DataFrame = val_df[FEATURE_COLS]
    y_val: pd.Series = val_df[TARGET_COL]
    typer.echo(f"  Validation rows: {len(val_df):,}")

    drift = check_target_drift(train_df[TARGET_COL], y_val)
    flag = "⚠" if drift["significant"] else "✓"
    typer.echo(
        f"  {flag}  Target drift: train median={drift['train_median_h']:.1f}h, "
        f"val median={drift['val_median_h']:.1f}h  ({drift['drift_pct']:.1f}%)"
    )

    _section("Leaderboard - Val Set")
    results = _build_leaderboard_results(suite, X_val, y_val)
    print_leaderboard(results)
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
