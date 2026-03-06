"""
Nosto — Customer Return Time Prediction CLI
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
  python main.py evaluate --data data/visits.tsv --importance
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
)
from scripts.data_preprocessing import load_and_prepare
from scripts.evaluation import (
	check_target_drift,
	compute_permutation_importance,
	print_leaderboard,
	regression_metrics,
)
from scripts.feature_engineering import FEATURE_COLS, build_features

# ── CLI app ──────────────────────────────────────────────────────────────────

app = typer.Typer(
	name="nosto-return-time",
	help="Customer return-time prediction pipeline (Nosto data-science assignment).",
	add_completion=False,
)

# Shared defaults
_DEFAULT_DATA = Path("data/visits.tsv")
_DEFAULT_MODEL_DIR = Path("scripts/models")
_DEFAULT_OUTPUT = Path("output/predictions.tsv")


def _section(title: str) -> None:
	pad = max(0, 60 - len(title))
	typer.echo(f"\n── {title} {'─' * pad}")


# ── train ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
	data: Path = typer.Option(_DEFAULT_DATA, "--data", "-d",
		help="Path to the visits TSV file."),
	model_dir: Path = typer.Option(_DEFAULT_MODEL_DIR, "--model-dir", "-m",
		help="Directory where the trained ModelSuite will be saved."),
	val_fraction: float = typer.Option(0.20, "--val-fraction",
		help="Fraction of data (by time) reserved for validation. Default: 0.20."),
	skip_lgb: bool = typer.Option(False, "--skip-lgb",
		help="Skip LightGBM DART training (slow). Useful for quick runs."),
) -> None:
	"""Train all models and save the ModelSuite to disk.

	Full pipeline: load → preprocess → feature engineering →
	chronological split → train models → evaluate on val set → save.
	"""
	typer.echo(f"Data      : {data}")
	typer.echo(f"Model dir : {model_dir}")
	typer.echo(f"Val split : {val_fraction:.0%} of data reserved for validation")
	if skip_lgb:
		typer.echo("  ⚠  --skip-lgb: LightGBM DART will NOT be trained.")

	# ── Preprocessing ─────────────────────────────────────────────────────────
	_section("Preprocessing")
	df = load_and_prepare(data)
	n_customers = df["customer_id"].nunique()
	typer.echo(f"  Loaded {len(df):,} rows from {n_customers:,} unique customers.")

	# ── Feature engineering ───────────────────────────────────────────────────
	_section("Feature Engineering")
	df = build_features(df)
	model_df = df[df[TARGET_COL].notna()].copy()
	n_dropped = len(df) - len(model_df)
	typer.echo(f"  Model-ready rows : {len(model_df):,}  ({n_dropped:,} last-visit rows excluded).")

	# ── Train / val split ─────────────────────────────────────────────────────
	_section("Chronological Split")
	train_df, val_df = chronological_split(model_df, val_fraction=val_fraction)
	typer.echo(f"  Train : {len(train_df):,} rows  ({train_df['end_dt'].min().date()} → {train_df['end_dt'].max().date()})")
	typer.echo(f"  Val   : {len(val_df):,} rows  ({val_df['end_dt'].min().date()} → {val_df['end_dt'].max().date()})")

	drift = check_target_drift(train_df[TARGET_COL], val_df[TARGET_COL])
	flag = "⚠" if drift["significant"] else "✓"
	typer.echo(
		f"  {flag}  Target drift: train median={drift['train_median_h']:.1f}h, "
		f"val median={drift['val_median_h']:.1f}h  ({drift['drift_pct']:.1f}%)"
	)

	# ── Training ──────────────────────────────────────────────────────────────
	_section("Training")
	suite = ModelSuite()
	suite.train_all(train_df, verbose=True, skip_lgb=skip_lgb)

	# ── Evaluation ────────────────────────────────────────────────────────────
	_section("Evaluation — Val Set")
	X_val = val_df[FEATURE_COLS]
	y_val = val_df[TARGET_COL]

	results: dict[str, dict[str, float]] = {
		"Global Median":    regression_metrics(y_val, suite.predict("global_baseline", X_val)),
		"Segmented Median": regression_metrics(y_val, suite.predict("seg_baseline", X_val)),
		"HGB":              regression_metrics(y_val, suite.predict("hgb", X_val)),
		"RandomForest":     regression_metrics(y_val, suite.predict("rf", X_val)),
		"XGBoost":          regression_metrics(y_val, suite.predict("xgb", X_val)),
	}
	if not skip_lgb:
		results["LightGBM DART"] = regression_metrics(y_val, suite.predict("lgb", X_val))

	print_leaderboard(results)

	# ── Save ──────────────────────────────────────────────────────────────────
	_section("Saving")
	saved_path = suite.save(model_dir)
	typer.echo(f"  Saved → {saved_path}")
	typer.echo("\nDone.  Run 'python main.py predict --help' to generate predictions.")


# ── predict ───────────────────────────────────────────────────────────────────

@app.command()
def predict(
	data: Path = typer.Option(_DEFAULT_DATA, "--data", "-d",
		help="Path to the visits TSV file (can be new, unseen data)."),
	model_dir: Path = typer.Option(_DEFAULT_MODEL_DIR, "--model-dir", "-m",
		help="Directory containing the saved ModelSuite."),
	output: Path = typer.Option(_DEFAULT_OUTPUT, "--output", "-o",
		help="Output TSV file path for predictions."),
	model_name: str = typer.Option("lgb", "--model",
		help="Which model to use: lgb (default), hgb, rf, xgb, global_baseline, seg_baseline."),
) -> None:
	"""Generate return-time predictions for a dataset.

	Loads a trained ModelSuite, runs preprocessing and feature engineering,
	then writes a TSV with predicted_return_hours for every row.
	"""
	typer.echo(f"Data   : {data}")
	typer.echo(f"Model  : {model_name}  (from {model_dir / MODEL_FILENAME})")
	typer.echo(f"Output : {output}")

	# ── Load model ────────────────────────────────────────────────────────────
	_section("Loading Model")
	suite = ModelSuite.load(model_dir)
	typer.echo(f"  ModelSuite loaded from {model_dir / MODEL_FILENAME}")

	# ── Preprocess & feature engineering ─────────────────────────────────────
	_section("Preprocessing & Feature Engineering")
	df = load_and_prepare(data)
	df = build_features(df)
	typer.echo(f"  {len(df):,} rows ready for prediction.")

	# ── Predict ───────────────────────────────────────────────────────────────
	_section("Prediction")
	X = df[FEATURE_COLS]
	preds = suite.predict(model_name, X)
	typer.echo(f"  Predictions: mean={preds.mean():.1f}h, median={np.median(preds):.1f}h")

	# ── Save output ───────────────────────────────────────────────────────────
	output.parent.mkdir(parents=True, exist_ok=True)
	out_df = df[["customer_id", "start_dt", "end_dt"]].copy()
	out_df["predicted_return_hours"] = preds
	out_df["predicted_return_days"] = (preds / 24).round(2)
	# Include the actual target if this is a labelled dataset (e.g. validation run).
	if TARGET_COL in df.columns:
		out_df[TARGET_COL] = df[TARGET_COL]
	out_df.to_csv(output, sep="\t", index=False)
	typer.echo(f"  Saved {len(out_df):,} rows → {output}")


# ── evaluate ──────────────────────────────────────────────────────────────────

@app.command()
def evaluate(
	data: Path = typer.Option(_DEFAULT_DATA, "--data", "-d",
		help="Path to the visits TSV file."),
	model_dir: Path = typer.Option(_DEFAULT_MODEL_DIR, "--model-dir", "-m",
		help="Directory containing the saved ModelSuite."),
	val_fraction: float = typer.Option(0.20, "--val-fraction",
		help="Fraction of data reserved for validation (must match training split)."),
	importance: bool = typer.Option(False, "--importance",
		help="Compute permutation feature importance for LightGBM (slow, ~minutes)."),
) -> None:
	"""Re-run evaluation on the validation split using a saved ModelSuite.

	Reproduces the exact train/val split used during training and prints
	the full model leaderboard (MAE, RMSE, % improvement over baseline).
	"""
	typer.echo(f"Data        : {data}")
	typer.echo(f"Model dir   : {model_dir}")
	typer.echo(f"Val fraction: {val_fraction:.0%}")

	# ── Load model ────────────────────────────────────────────────────────────
	suite = ModelSuite.load(model_dir)
	typer.echo(f"  ModelSuite loaded from {model_dir / MODEL_FILENAME}")

	# ── Preprocess & features ─────────────────────────────────────────────────
	_section("Preprocessing & Feature Engineering")
	df = load_and_prepare(data)
	df = build_features(df)
	model_df = df[df[TARGET_COL].notna()].copy()

	# ── Split ─────────────────────────────────────────────────────────────────
	train_df, val_df = chronological_split(model_df, val_fraction=val_fraction)
	X_val = val_df[FEATURE_COLS]
	y_val = val_df[TARGET_COL]
	typer.echo(f"  Validation rows: {len(val_df):,}")

	drift = check_target_drift(train_df[TARGET_COL], y_val)
	flag = "⚠" if drift["significant"] else "✓"
	typer.echo(
		f"  {flag}  Target drift: train median={drift['train_median_h']:.1f}h, "
		f"val median={drift['val_median_h']:.1f}h  ({drift['drift_pct']:.1f}%)"
	)

	# ── Evaluate all models ───────────────────────────────────────────────────
	_section("Leaderboard — Val Set")
	results: dict[str, dict[str, float]] = {
		"Global Median":    regression_metrics(y_val, suite.predict("global_baseline", X_val)),
		"Segmented Median": regression_metrics(y_val, suite.predict("seg_baseline", X_val)),
		"HGB":              regression_metrics(y_val, suite.predict("hgb", X_val)),
		"RandomForest":     regression_metrics(y_val, suite.predict("rf", X_val)),
		"XGBoost":          regression_metrics(y_val, suite.predict("xgb", X_val)),
	}
	if suite.lgb is not None:
		results["LightGBM DART"] = regression_metrics(y_val, suite.predict("lgb", X_val))

	print_leaderboard(results)

	# ── Optional: permutation feature importance ──────────────────────────────
	if importance:
		if suite.lgb is None:
			typer.echo("\n  ⚠  LightGBM model not available — was training run with --skip-lgb?")
		else:
			_section("Permutation Feature Importance — LightGBM DART")
			typer.echo("  Computing (this may take a few minutes)...")
			imp_df = compute_permutation_importance(suite.lgb, X_val, y_val, FEATURE_COLS)
			typer.echo("\n  Top 15 features:")
			typer.echo(imp_df.head(15).to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
	app()
