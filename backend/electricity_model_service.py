"""
backend/electricity_model_service.py
=====================================
Production service wrapping the trained GradientBoosting pkl model.
The Flask app uses this class instead of reading the CSV at runtime.

Usage inside a route:
    from backend.electricity_model_service import electricity_model

    result = electricity_model.predict(
        datetime_str="2025-07-20 15:00:00",
        season="Summer",
        day_of_week="Sunday",
        forecast_temp=34.0,
        forecast_humidity=70.0,
        forecast_solar=5.5,
        forecast_wind=3.2,
        forecast_rain=0.0,
        forecast_cloud=2.0,
        belnder_forecast=1300.0,     # optional — uses model mean if unknown
    )
    # → {"predicted_load_mw": 1247.3, "confidence_band": (1180.0, 1315.0), ...}

The model artifact lives at:
    data/electricity_load_model.pkl

Re-train by running:
    python train_electricity_model.py --compare
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── Defaults (from observed data statistics) ─────────────────────────────────

_DEFAULTS = {
    "gen_capacity_mw":    200.0,
    "planned_outage_mw":  0.0,
    "plant_number":       5,
    "belnder_forecast":   1076.5,   # dataset mean
    "bidding_volume":     1056.1,
    "real_buying_volume": 1067.5,
    "user_amount":        18481,
    "weather_var_idx":    0.0,
    "forecast_temp":      16.6,
    "forecast_humidity":  64.5,
    "forecast_solar":     4.5,
    "forecast_wind":      6.0,
    "forecast_rain":      4.8,
    "forecast_cloud":     3.3,
}

_DOW_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

_SEASON_FOR_MONTH = {
    1: "Winter", 2: "Winter", 3: "Spring",
    4: "Spring", 5: "Spring", 6: "Summer",
    7: "Summer", 8: "Summer", 9: "Autumn",
    10: "Autumn", 11: "Autumn", 12: "Winter",
}


# ─── Service class ────────────────────────────────────────────────────────────

class ElectricityModelService:
    """
    Lazy-loading singleton wrapper around the trained pkl model.

    Thread-safe: model is loaded once on first call.
    """

    PKL_PATH = "data/electricity_load_model.pkl"

    def __init__(self):
        self._artifact   = None
        self._lock       = threading.Lock()
        self._load_error: Optional[str] = None

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self) -> bool:
        """Load artifact from disk (called once, thread-safe)."""
        if self._artifact is not None:
            return True
        with self._lock:
            if self._artifact is not None:          # double-check
                return True
            try:
                import joblib
                path = Path(self.PKL_PATH)
                if not path.exists():
                    self._load_error = (
                        f"Model file not found: {self.PKL_PATH}. "
                        "Run: python train_electricity_model.py"
                    )
                    log.warning("⚡ %s", self._load_error)
                    return False
                self._artifact = joblib.load(path)
                m = self._artifact["test_metrics"]
                log.info(
                    "⚡ ElectricityModel loaded  type=%s  R²=%.4f  MAE=%.1f MW  "
                    "MAPE=%.2f%%  trained=%s",
                    self._artifact["model_type"],
                    m["R2"], m["MAE"], m["MAPE_%"],
                    self._artifact["trained_at"][:19],
                )
                return True
            except Exception as exc:
                self._load_error = str(exc)
                log.error("⚡ Failed to load electricity model: %s", exc)
                return False

    def _build_row(self, **kwargs) -> pd.DataFrame:
        """
        Construct a single-row DataFrame with all required model features.
        Falls back to sensible defaults for any unspecified parameter.
        """
        dt_str   = kwargs.get("datetime_str") or datetime.now().isoformat()
        dt       = pd.to_datetime(dt_str)

        dow_str  = str(kwargs.get("day_of_week", "Monday")).strip().lower()
        dow_num  = _DOW_MAP.get(dow_str, dt.weekday())

        month    = dt.month
        season   = kwargs.get("season") or _SEASON_FOR_MONTH.get(month, "Spring")

        is_weekend = 1 if dow_num >= 5 else int(kwargs.get("is_weekend", 0))
        is_holiday = int(kwargs.get("is_holiday", 0))

        gen_cap   = float(kwargs.get("gen_capacity_mw",    _DEFAULTS["gen_capacity_mw"]))
        outage    = float(kwargs.get("planned_outage_mw",  _DEFAULTS["planned_outage_mw"]))

        row = {
            "hour":              dt.hour,
            "minute":            dt.minute,
            "month":             month,
            "day_of_month":      dt.day,
            "hour_sin":          np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos":          np.cos(2 * np.pi * dt.hour / 24),
            "month_sin":         np.sin(2 * np.pi * month / 12),
            "month_cos":         np.cos(2 * np.pi * month / 12),
            "day_of_week_num":   dow_num,
            "Plant_Number":      int(kwargs.get("plant_number",    _DEFAULTS["plant_number"])),
            "Gen_Capacity(MW)":  gen_cap,
            "Planned_Outage(MW)": outage,
            "net_capacity":      gen_cap - outage,
            "BELNDER_Forecast_Volume": float(kwargs.get("belnder_forecast",  _DEFAULTS["belnder_forecast"])),
            "Bidding_Volume":    float(kwargs.get("bidding_volume",   _DEFAULTS["bidding_volume"])),
            "Real_Buying_Volume": float(kwargs.get("real_buying_volume", _DEFAULTS["real_buying_volume"])),
            "Forecast_Temperature(°C)":    float(kwargs.get("forecast_temp",     _DEFAULTS["forecast_temp"])),
            "Forecast_Humidity(%)":        float(kwargs.get("forecast_humidity", _DEFAULTS["forecast_humidity"])),
            "Forecast_Solar(kWh/m²/day)":  float(kwargs.get("forecast_solar",   _DEFAULTS["forecast_solar"])),
            "Forecast_WindSpeed(m/s)":     float(kwargs.get("forecast_wind",    _DEFAULTS["forecast_wind"])),
            "Forecast_Rainfall(mm)":       float(kwargs.get("forecast_rain",    _DEFAULTS["forecast_rain"])),
            "Forecast_CloudCover(0-10)":   float(kwargs.get("forecast_cloud",   _DEFAULTS["forecast_cloud"])),
            "Weather_Variability_Index":   float(kwargs.get("weather_var_idx",  _DEFAULTS["weather_var_idx"])),
            "Is_Holiday":        is_holiday,
            "Is_Weekend":        is_weekend,
            "User_Amount":       int(kwargs.get("user_amount", _DEFAULTS["user_amount"])),
            "Season":            season,
        }
        return pd.DataFrame([row])

    # ── Public API ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the model pkl is loaded and ready."""
        return self._load()

    def predict(self, **kwargs) -> dict:
        """
        Predict electricity load for one half-hour slot.

        Keyword args (all optional — fall back to dataset means):
            datetime_str        str   ISO format. Default: now
            season              str   Spring|Summer|Autumn|Winter
            day_of_week         str   Monday … Sunday
            is_holiday          int   0 or 1
            is_weekend          int   0 or 1
            gen_capacity_mw     float  MW
            planned_outage_mw   float  MW
            plant_number        int    1-10
            belnder_forecast    float  forecast demand MW
            bidding_volume      float  MW
            real_buying_volume  float  MW
            forecast_temp       float  °C
            forecast_humidity   float  %
            forecast_solar      float  kWh/m²/day
            forecast_wind       float  m/s
            forecast_rain       float  mm
            forecast_cloud      float  0-10
            weather_var_idx     float
            user_amount         int

        Returns dict:
            {
              "predicted_load_mw":  float,
              "confidence_band":    [lower, upper],   # ±8% empirical band
              "confidence_level":   "High"|"Medium"|"Low",
              "model_type":         str,
              "model_r2":           float,
              "model_mape_pct":     float,
              "success":            bool,
              "error":              str | None,
            }
        """
        if not self._load():
            return {
                "success": False,
                "error":   self._load_error or "Model not loaded",
                "predicted_load_mw": None,
            }

        try:
            art  = self._artifact
            df   = self._build_row(**kwargs)
            pred = float(art["pipeline"].predict(df[art["feature_names"]])[0])

            # Empirical confidence band (±8% based on MAPE ≈ 2.7%)
            mape = art["test_metrics"]["MAPE_%"] / 100
            band_pct = max(mape * 3, 0.08)     # 3× MAPE or 8%, whichever larger
            lower = round(pred * (1 - band_pct), 1)
            upper = round(pred * (1 + band_pct), 1)

            r2 = art["test_metrics"]["R2"]
            if r2 >= 0.90:
                conf_level = "High"
            elif r2 >= 0.80:
                conf_level = "Medium"
            else:
                conf_level = "Low"

            return {
                "success":            True,
                "predicted_load_mw":  round(pred, 2),
                "confidence_band":    [lower, upper],
                "confidence_level":   conf_level,
                "model_type":         art["model_type"],
                "model_r2":           art["test_metrics"]["R2"],
                "model_mape_pct":     art["test_metrics"]["MAPE_%"],
                "model_mae_mw":       art["test_metrics"]["MAE"],
                "error":              None,
            }

        except Exception as exc:
            log.error("⚡ Prediction error: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc), "predicted_load_mw": None}

    def predict_for_weather(
        self,
        location:     str,
        days:         int = 7,
        forecast_temp: float = 20.0,
        forecast_humidity: float = 65.0,
        season:       str = "Autumn",
        **extra,
    ) -> dict:
        """
        High-level helper called from weather prediction routes.
        Accepts the same parameters as the weather routes.
        Returns a weather-style response with electricity load embedded.
        """
        now = datetime.now()
        results = []
        for day_offset in range(days):
            for hour in [0, 6, 12, 18]:      # 4 samples per day
                dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                pred = self.predict(
                    datetime_str=dt.isoformat(),
                    season=season,
                    day_of_week=dt.strftime("%A"),
                    forecast_temp=forecast_temp + (2 if hour in (12, 18) else -2),
                    forecast_humidity=forecast_humidity,
                    is_weekend=1 if dt.weekday() >= 5 else 0,
                    **extra,
                )
                if pred["success"]:
                    results.append(pred["predicted_load_mw"])

        if not results:
            return {"success": False, "error": "No predictions generated"}

        avg = round(float(np.mean(results)), 2)
        peak = round(float(np.max(results)), 2)
        low  = round(float(np.min(results)), 2)

        art = self._artifact
        conf_level = "High" if art["test_metrics"]["R2"] >= 0.90 else "Medium"

        return {
            "success":            True,
            "method":             "electricity_model_pkl",
            "location":           location,
            "prediction_days":    days,
            "predicted_avg_mw":   avg,
            "predicted_peak_mw":  peak,
            "predicted_low_mw":   low,
            "confidence_level":   conf_level,
            "prediction": (
                f"Based on the trained GradientBoosting model (R²={art['test_metrics']['R2']:.3f}, "
                f"MAPE={art['test_metrics']['MAPE_%']:.2f}%), the predicted electricity load for "
                f"{location} over the next {days} days averages {avg:.1f} MW "
                f"(peak {peak:.1f} MW, low {low:.1f} MW). "
                f"The model was trained on 365 days of Tokyo Hydropower data and captures "
                f"seasonal, weather, and demand patterns without accessing the original CSV."
            ),
            "model_info": {
                "model_type":   art["model_type"],
                "r2_score":     art["test_metrics"]["R2"],
                "mape_pct":     art["test_metrics"]["MAPE_%"],
                "mae_mw":       art["test_metrics"]["MAE"],
                "trained_at":   art["trained_at"][:19],
                "version":      art["version"],
            },
            "error": None,
        }

    def get_model_info(self) -> dict:
        """Return model metadata for status/monitoring endpoints."""
        if not self._load():
            return {
                "available":  False,
                "error":      self._load_error,
                "pkl_path":   self.PKL_PATH,
            }
        art = self._artifact
        return {
            "available":     True,
            "pkl_path":      self.PKL_PATH,
            "model_type":    art["model_type"],
            "version":       art["version"],
            "trained_at":    art["trained_at"][:19],
            "target":        art["target"],
            "num_features":  len(art["feature_names"]),
            "feature_names": art["feature_names"],
            "test_metrics":  art["test_metrics"],
            "train_metrics": art.get("train_metrics", {}),
            "top_features": dict(
                sorted(
                    art["feature_importances"].items(),
                    key=lambda x: -x[1],
                )[:10]
            ),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

electricity_model = ElectricityModelService()
