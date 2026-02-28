"""
train_electricity_model.py
==========================
Trains a GradientBoosting and RandomForest model on the Tokyo electricity load
data (half-hourly, 365 days) and saves the best one as:
    data/electricity_load_model.pkl

The saved artifact is a dict:
    {
        "pipeline":       sklearn Pipeline (ColumnTransformer + model),
        "feature_names":  list[str]  — exact column names fed to the pipeline,
        "target":         "Real_Used_Volume",
        "metrics":        {mae, rmse, r2, mape},
        "feature_importances": {col: importance},
        "trained_at":     ISO timestamp,
        "model_type":     "GradientBoosting" | "RandomForest",
        "version":        "1.0",
    }

Once saved, the Flask app can load this pkl and predict without the CSV.

Usage:
    .\\flasking_py311\\Scripts\\python.exe train_electricity_model.py
            [--csv  data/Generated_electricity_load_japan_past365days.csv]
            [--out  data/electricity_load_model.pkl]
            [--compare]     # compare both models
            [--no-plot]     # skip feature importance plot
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CSV = "data/Generated_electricity_load_japan_past365days.csv"
DEFAULT_PKL = "data/electricity_load_model.pkl"
TARGET      = "Real_Used_Volume"

# Columns that are constants or fully NaN — always dropped
DROP_COLS = [
    "Plant_Area",          # always 'Tokyo'
    "Plant_Type",          # always 'Hydropower'
    "Plant_Volume",        # always 839520 (constant)
    "Non_Renewable_Supply_URL",  # 100% NaN
    "Holiday_Name",        # 100% NaN
    # Actual* readings not available at prediction time
    "Actual_Temperature(°C)",
    "Actual_Humidity(%)",
    "Actual_Solar(kWh/m²/day)",
    "Actual_WindSpeed(m/s)",
    "Actual_Rainfall(mm)",
    "Actual_CloudCover(0-10)",
]

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


# ─── Feature engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from Datetime and drop unusable columns.
    Returns a clean feature DataFrame (target column still present).
    """
    df = df.copy()

    # Parse datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["hour"]         = df["Datetime"].dt.hour
    df["minute"]       = df["Datetime"].dt.minute
    df["month"]        = df["Datetime"].dt.month
    df["day_of_month"] = df["Datetime"].dt.day
    # Cyclic hour encoding keeps distance meaningful (23:00 ~ 00:00)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)

    # Day of week numeric (0=Mon … 6=Sun) from the string column
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    df["day_of_week_num"] = df["Day_of_Week"].map(dow_map)

    # Net available capacity
    df["net_capacity"] = df["Gen_Capacity(MW)"] - df["Planned_Outage(MW)"]

    # Drop original Datetime + raw string columns that are now encoded
    df = df.drop(columns=["Datetime", "Day_of_Week"] + DROP_COLS, errors="ignore")

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list, str]:
    """
    Returns (numeric_features, categorical_features, target).
    Categorical features here are only 'Season' (ordinal).
    """
    exclude = {TARGET}
    cat_cols = ["Season"]
    num_cols = [
        c for c in df.columns
        if c not in exclude and c not in cat_cols
    ]
    return num_cols, cat_cols, TARGET


# ─── Build sklearn pipeline ───────────────────────────────────────────────────

def build_pipeline(model_type: str = "GradientBoosting") -> Pipeline:
    """
    ColumnTransformer:
        numeric  → StandardScaler
        Season   → OrdinalEncoder (Spring < Summer < Autumn < Winter)
    Then the chosen regressor.
    """
    numeric_transformer = StandardScaler()
    season_transformer  = OrdinalEncoder(
        categories=[SEASON_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # Placeholder; actual column lists are set in train()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, []),   # filled in train()
            ("cat", season_transformer,  ["Season"]),
        ],
        remainder="drop",
    )

    if model_type == "GradientBoosting":
        regressor = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=5,
            min_samples_split=4,
            subsample=0.85,
            random_state=42,
            verbose=0,
        )
    elif model_type == "RandomForest":
        regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE: avoid division by zero
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    return {
        "MAE":        round(float(mae),  3),
        "RMSE":       round(float(rmse), 3),
        "R2":         round(float(r2),   5),
        "MAPE_%":     round(float(mape), 3),
    }


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    csv_path: str,
    out_path:  str,
    model_type: str = "GradientBoosting",
    test_ratio: float = 0.2,
) -> dict:
    """
    Full pipeline: load → engineer → split → train → evaluate → save.
    Returns the metrics dict.
    """
    log.info("── Loading CSV: %s", csv_path)
    raw = pd.read_csv(csv_path)
    log.info("   Loaded %d rows × %d cols", *raw.shape)

    log.info("── Engineering features …")
    df = build_features(raw)
    log.info("   Feature matrix: %d rows × %d cols", *df.shape)

    num_cols, cat_cols, target = get_feature_columns(df)
    all_features = num_cols + cat_cols

    X = df[all_features]
    y = df[target].values

    # Chronological split (keep temporal order to avoid data leakage)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx],       y[split_idx:]

    log.info(
        "── Chronological split: train=%d  test=%d  (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        (1 - test_ratio) * 100,
        test_ratio * 100,
    )

    # Wire the numeric feature list into the pipeline
    pipe = build_pipeline(model_type)
    pipe.named_steps["preprocessor"].transformers[0] = (
        "num", StandardScaler(), num_cols
    )

    log.info("── Training %s …", model_type)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info("   Training done in %.1fs", elapsed)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics  = compute_metrics(y_test,  y_pred_test)

    log.info("── Train metrics: %s", json.dumps(train_metrics))
    log.info("── Test  metrics: %s", json.dumps(test_metrics))

    # Feature importances (from regressor)
    regressor = pipe.named_steps["regressor"]
    # After ColumnTransformer: num_cols come first, then season ordinal
    transformed_names = num_cols + ["Season"]
    importances = {
        name: round(float(imp), 6)
        for name, imp in zip(
            transformed_names,
            regressor.feature_importances_,
        )
    }
    top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    log.info("── Top-5 features: %s", top5)

    # Save artifact
    artifact = {
        "pipeline":             pipe,
        "feature_names":        all_features,
        "num_feature_names":    num_cols,
        "cat_feature_names":    cat_cols,
        "target":               TARGET,
        "train_metrics":        train_metrics,
        "test_metrics":         test_metrics,
        "feature_importances":  importances,
        "trained_at":           datetime.now().isoformat(),
        "model_type":           model_type,
        "train_size":           len(X_train),
        "test_size":            len(X_test),
        "version":              "1.0",
        "csv_source":           str(csv_path),
        "season_order":         SEASON_ORDER,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    size_kb = out.stat().st_size / 1024
    log.info("── Saved → %s  (%.1f KB)", out, size_kb)

    return artifact


# ─── Comparison mode ─────────────────────────────────────────────────────────

def compare_models(csv_path: str) -> dict:
    """Train both models and print comparison; return metrics dict."""
    log.info("═══ Comparing GradientBoosting vs RandomForest ═══")
    results = {}

    for mtype in ["GradientBoosting", "RandomForest"]:
        log.info("")
        log.info(">>> %s", mtype)
        artifact = train(csv_path, f"/tmp/{mtype}.tmp.pkl", model_type=mtype)
        results[mtype] = artifact["test_metrics"]

    log.info("")
    log.info("═══ Comparison summary ═══")
    for mtype, m in results.items():
        log.info("  %-20s  MAE=%.2f  RMSE=%.2f  R²=%.4f  MAPE=%.2f%%",
                 mtype, m["MAE"], m["RMSE"], m["R2"], m["MAPE_%"])

    # Return name of winner by R²
    winner = max(results, key=lambda k: results[k]["R2"])
    log.info("  Winner by R²: %s", winner)
    return results, winner


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train electricity load forecasting model")
    p.add_argument("--csv",     default=DEFAULT_CSV, help="Path to source CSV")
    p.add_argument("--out",     default=DEFAULT_PKL, help="Where to save .pkl")
    p.add_argument("--model",   default="GradientBoosting",
                   choices=["GradientBoosting", "RandomForest"],
                   help="Model type to train")
    p.add_argument("--compare", action="store_true",
                   help="Train both models and save the winner")
    p.add_argument("--test-ratio", type=float, default=0.2)
    args = p.parse_args()

    if args.compare:
        results, winner = compare_models(args.csv)
        log.info("")
        log.info("Training winner (%s) and saving to %s …", winner, args.out)
        train(args.csv, args.out, model_type=winner, test_ratio=args.test_ratio)
    else:
        train(args.csv, args.out, model_type=args.model, test_ratio=args.test_ratio)

    log.info("")
    log.info("✅  Model ready at: %s", args.out)
    log.info("    Run:  python test_electricity_model.py  to verify predictions")


if __name__ == "__main__":
    main()
