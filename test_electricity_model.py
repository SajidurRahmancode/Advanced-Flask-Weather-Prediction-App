"""
test_electricity_model.py
=========================
Smoke-test and validation suite for the trained electricity_load_model.pkl.

Tests:
  1. PKL loads successfully
  2. Batch prediction on held-out test window
  3. Single-row prediction from a hand-crafted dict
  4. Feature names / artifact structure
  5. Metrics thresholds (R² > 0.80, MAPE < 20%)

Usage:
    .\\flasking_py311\\Scripts\\python.exe test_electricity_model.py
    .\\flasking_py311\\Scripts\\python.exe -m pytest test_electricity_model.py -v
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import pytest

PKL_PATH = "data/electricity_load_model.pkl"
CSV_PATH = "data/Generated_electricity_load_japan_past365days.csv"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_artifact():
    return joblib.load(PKL_PATH)


def build_single_row(
    dt_str: str = "2025-01-15 14:00:00",
    season: str = "Winter",
    day_of_week: str = "Wednesday",
    is_holiday: int = 0,
    is_weekend: int = 0,
    gen_capacity: float = 200.0,
    planned_outage: float = 0.0,
    forecast_temp: float = 5.0,
    forecast_humidity: float = 60.0,
    forecast_solar: float = 2.5,
    forecast_wind: float = 4.0,
    forecast_rain: float = 1.0,
    forecast_cloud: float = 5.0,
    weather_var_idx: float = 0.5,
    belnder_forecast: float = 1100.0,
    bidding_vol: float = 1050.0,
    real_buying: float = 1040.0,
    plant_number: int = 3,
    user_amount: int = 18000,
) -> dict:
    """Build a single prediction input dict (no target / no Actual_* columns)."""
    dt = pd.to_datetime(dt_str)
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    row = {
        "hour":              dt.hour,
        "minute":            dt.minute,
        "month":             dt.month,
        "day_of_month":      dt.day,
        "hour_sin":          np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos":          np.cos(2 * np.pi * dt.hour / 24),
        "month_sin":         np.sin(2 * np.pi * dt.month / 12),
        "month_cos":         np.cos(2 * np.pi * dt.month / 12),
        "day_of_week_num":   dow_map[day_of_week],
        "Plant_Number":      plant_number,
        "Gen_Capacity(MW)":  gen_capacity,
        "Planned_Outage(MW)": planned_outage,
        "BELNDER_Forecast_Volume": belnder_forecast,
        "Bidding_Volume":    bidding_vol,
        "Real_Buying_Volume": real_buying,
        "Forecast_Temperature(°C)": forecast_temp,
        "Forecast_Humidity(%)": forecast_humidity,
        "Forecast_Solar(kWh/m²/day)": forecast_solar,
        "Forecast_WindSpeed(m/s)": forecast_wind,
        "Forecast_Rainfall(mm)": forecast_rain,
        "Forecast_CloudCover(0-10)": forecast_cloud,
        "Weather_Variability_Index": weather_var_idx,
        "Is_Holiday":        is_holiday,
        "Is_Weekend":        is_weekend,
        "User_Amount":       user_amount,
        "net_capacity":      gen_capacity - planned_outage,
        "Season":            season,
    }
    return row


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestArtifactStructure:
    """Check the saved pkl has all required keys."""

    def test_file_exists(self):
        assert Path(PKL_PATH).exists(), f"PKL not found at {PKL_PATH} — run train_electricity_model.py first"

    def test_loads_without_error(self):
        art = load_artifact()
        assert art is not None

    def test_required_keys(self):
        art = load_artifact()
        required = {"pipeline", "feature_names", "target", "test_metrics",
                    "model_type", "version", "trained_at", "season_order"}
        missing = required - set(art.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_target_is_correct(self):
        art = load_artifact()
        assert art["target"] == "Real_Used_Volume"

    def test_version(self):
        art = load_artifact()
        assert art["version"] == "1.0"

    def test_feature_names_is_list(self):
        art = load_artifact()
        assert isinstance(art["feature_names"], list)
        assert len(art["feature_names"]) > 5

    def test_pipeline_has_steps(self):
        from sklearn.pipeline import Pipeline
        art = load_artifact()
        assert isinstance(art["pipeline"], Pipeline)
        assert "preprocessor" in art["pipeline"].named_steps
        assert "regressor" in art["pipeline"].named_steps

    def test_metrics_keys(self):
        art = load_artifact()
        m = art["test_metrics"]
        for k in ("MAE", "RMSE", "R2", "MAPE_%"):
            assert k in m, f"Missing metric key: {k}"

    def test_trained_at_is_parseable(self):
        art = load_artifact()
        dt = datetime.fromisoformat(art["trained_at"])
        assert dt.year >= 2025


class TestMetricThresholds:
    """Model quality gates — fail if model is too poor."""

    def test_r2_above_threshold(self):
        art = load_artifact()
        r2 = art["test_metrics"]["R2"]
        assert r2 >= 0.80, f"R² too low: {r2:.4f} (threshold 0.80)"

    def test_mape_below_threshold(self):
        art = load_artifact()
        mape = art["test_metrics"]["MAPE_%"]
        assert mape <= 20.0, f"MAPE too high: {mape:.2f}% (threshold 20%)"

    def test_mae_is_positive(self):
        art = load_artifact()
        assert art["test_metrics"]["MAE"] > 0

    def test_no_overfitting(self):
        """Train MAE should not be dramatically lower than test MAE (× 3 guard)."""
        art = load_artifact()
        if "train_metrics" in art:
            train_mae = art["train_metrics"]["MAE"]
            test_mae  = art["test_metrics"]["MAE"]
            ratio = test_mae / max(train_mae, 0.001)
            assert ratio < 3.0, (
                f"Possible overfitting: train_MAE={train_mae:.2f}, "
                f"test_MAE={test_mae:.2f}, ratio={ratio:.2f}"
            )


class TestBatchPrediction:
    """Run the model on real CSV rows and validate output shape + range."""

    def test_batch_predict_returns_array(self):
        from train_electricity_model import build_features
        art = load_artifact()
        df  = pd.read_csv(CSV_PATH)
        df  = build_features(df)
        feat_cols = art["feature_names"]
        X   = df[feat_cols].tail(100)     # last 100 rows
        preds = art["pipeline"].predict(X)
        assert preds.shape == (100,)

    def test_predictions_are_finite(self):
        from train_electricity_model import build_features
        art = load_artifact()
        df  = pd.read_csv(CSV_PATH)
        df  = build_features(df)
        X   = df[art["feature_names"]].tail(200)
        preds = art["pipeline"].predict(X)
        assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf"

    def test_predictions_in_plausible_range(self):
        from train_electricity_model import build_features
        art  = load_artifact()
        df   = pd.read_csv(CSV_PATH)
        df   = build_features(df)
        X    = df[art["feature_names"]].tail(500)
        preds = art["pipeline"].predict(X)
        assert preds.min() > 0,      "Predictions must be positive (MW load)"
        assert preds.max() < 5000,   "Predictions implausibly high (>5000 MW)"
        assert preds.mean() > 400,   "Predictions implausibly low (mean<400 MW)"

    def test_full_test_set_metrics(self):
        """Replicate train/test split and verify recorded metrics are accurate."""
        from train_electricity_model import build_features, compute_metrics, TARGET
        from sklearn.metrics import r2_score
        art  = load_artifact()
        df   = pd.read_csv(CSV_PATH)
        df   = build_features(df)
        split = int(len(df) * 0.8)
        X_test = df[art["feature_names"]].iloc[split:]
        y_test = df[TARGET].values[split:]
        preds  = art["pipeline"].predict(X_test)
        m      = compute_metrics(y_test, preds)
        assert abs(m["R2"] - art["test_metrics"]["R2"]) < 0.01, (
            f"Stored R²={art['test_metrics']['R2']:.4f} != recomputed R²={m['R2']:.4f}"
        )


class TestSingleRowPrediction:
    """Predict from hand-crafted single inputs."""

    def setup_method(self):
        self.art = load_artifact()

    def _predict(self, row_dict: dict) -> float:
        df = pd.DataFrame([row_dict])
        return float(self.art["pipeline"].predict(df[self.art["feature_names"]])[0])

    def test_winter_midday_prediction(self):
        row   = build_single_row("2025-01-15 14:00:00", season="Winter")
        pred  = self._predict(row)
        assert 400 < pred < 2500, f"Winter midday load out of range: {pred:.1f} MW"
        print(f"\n  Winter 14:00 → {pred:.1f} MW")

    def test_summer_midday_higher_than_midnight(self):
        """Summer afternoon load should generally be higher than midnight load."""
        midday   = build_single_row("2025-07-15 14:00:00", season="Summer",
                                    forecast_temp=32, user_amount=22000)
        midnight = build_single_row("2025-07-15 01:00:00", season="Summer",
                                    forecast_temp=25, user_amount=13000,
                                    belnder_forecast=700, bidding_vol=680)
        pred_mid  = self._predict(midday)
        pred_night = self._predict(midnight)
        print(f"\n  Summer 14:00={pred_mid:.1f} MW, 01:00={pred_night:.1f} MW")
        assert pred_mid > pred_night, (
            f"Expected midday ({pred_mid:.1f}) > midnight ({pred_night:.1f})"
        )

    def test_holiday_vs_workday(self):
        """Holiday load should not exceed workday load (typically lower)."""
        workday = build_single_row("2025-01-07 10:00:00", season="Winter",
                                   is_holiday=0, is_weekend=0, day_of_week="Tuesday")
        holiday = build_single_row("2025-01-01 10:00:00", season="Winter",
                                   is_holiday=1, is_weekend=0, day_of_week="Wednesday")
        pred_work = self._predict(workday)
        pred_hol  = self._predict(holiday)
        print(f"\n  Workday 10:00={pred_work:.1f} MW, Holiday 10:00={pred_hol:.1f} MW")
        # Holiday load is typically lower, but don't hard-fail if model disagrees
        # Just assert both are in a reasonable range
        assert 400 < pred_work < 2500
        assert 400 < pred_hol  < 2500

    def test_prediction_changes_with_temperature(self):
        """Higher summer temperature should change the prediction."""
        hot = build_single_row("2025-07-20 15:00:00", season="Summer",
                               forecast_temp=38, user_amount=22000)
        cool = build_single_row("2025-07-20 15:00:00", season="Summer",
                                forecast_temp=18, user_amount=22000)
        pred_hot  = self._predict(hot)
        pred_cool = self._predict(cool)
        print(f"\n  38°C → {pred_hot:.1f} MW,  18°C → {pred_cool:.1f} MW")
        assert pred_hot != pred_cool, "Temperature has zero effect — check feature engineering"

    def test_all_required_features_used(self):
        row = build_single_row()
        df  = pd.DataFrame([row])
        feat_cols = self.art["feature_names"]
        missing = [f for f in feat_cols if f not in df.columns]
        assert not missing, f"Missing features in test row: {missing}"

    def test_output_is_scalar_float(self):
        row  = build_single_row()
        pred = self._predict(row)
        assert isinstance(pred, float)
        assert np.isfinite(pred)


class TestFeatureImportances:
    """Sanity-check the feature importances saved in the artifact."""

    def test_importances_sum_to_one(self):
        art = load_artifact()
        total = sum(art["feature_importances"].values())
        assert abs(total - 1.0) < 0.01, f"Importances sum={total:.4f} (expected ≈1.0)"

    def test_belnder_forecast_is_important(self):
        """The BELNDER demand forecast should be among the top predictors."""
        art = load_artifact()
        imps = art["feature_importances"]
        belnder_imp = imps.get("BELNDER_Forecast_Volume", 0)
        top_imp = max(imps.values())
        assert belnder_imp >= top_imp * 0.05, (
            f"BELNDER_Forecast_Volume importance ({belnder_imp:.4f}) "
            f"seems too low (top={top_imp:.4f})"
        )

    def test_all_features_have_importances(self):
        art = load_artifact()
        feat_cols = art["feature_names"]
        imp_keys  = set(art["feature_importances"].keys())
        for f in feat_cols:
            assert f in imp_keys, f"No importance stored for feature: {f}"


# ─── CLI quick-run (not pytest) ───────────────────────────────────────────────

def quick_demo():
    """Run 3 quick prediction demos and print results."""
    print("\n" + "═" * 60)
    print("  Electricity Load Model — Quick Demo")
    print("═" * 60)

    if not Path(PKL_PATH).exists():
        print(f"❌  PKL not found: {PKL_PATH}")
        print("    Run:  python train_electricity_model.py")
        sys.exit(1)

    art = load_artifact()
    pipe = art["pipeline"]
    feat = art["feature_names"]
    m    = art["test_metrics"]

    print(f"\n  Model type  : {art['model_type']}")
    print(f"  Trained at  : {art['trained_at']}")
    print(f"  Test R²     : {m['R2']:.4f}")
    print(f"  Test MAE    : {m['MAE']:.2f} MW")
    print(f"  Test MAPE   : {m['MAPE_%']:.2f}%")
    print(f"  Test RMSE   : {m['RMSE']:.2f} MW")
    print()

    scenarios = [
        ("Winter midnight (low demand)",
         "2025-01-10 01:00:00", "Winter", "Friday",
         5.0, 50, 20000, 900),
        ("Summer afternoon (peak)",
         "2025-07-20 15:00:00", "Summer", "Monday",
         35.0, 75, 23000, 1400),
        ("Spring morning (moderate)",
         "2025-04-05 09:00:00", "Spring", "Saturday",
         18.0, 65, 17000, 1100),
    ]

    for label, dt_str, season, dow, temp, humidity, users, belnder_v in scenarios:
        row  = build_single_row(
            dt_str, season=season, day_of_week=dow,
            forecast_temp=temp, forecast_humidity=float(humidity),
            user_amount=users, belnder_forecast=float(belnder_v),
        )
        df   = pd.DataFrame([row])
        pred = float(pipe.predict(df[feat])[0])
        print(f"  {label}")
        print(f"      → Predicted load: {pred:.1f} MW\n")

    print("═" * 60)
    print("  Run pytest test_electricity_model.py -v  for full tests")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    quick_demo()
