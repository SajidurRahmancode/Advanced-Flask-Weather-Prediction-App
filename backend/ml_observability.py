"""
ml_observability.py — MLOps prediction tracing and metrics.

Writes structured JSONL traces to logs/ml_predictions.jsonl.
Maintains a rolling 1 000-entry in-memory metrics window for the
monitoring dashboard.

Usage:
    from backend.ml_observability import observability

    trace = observability.start_trace(location, days, "ensemble")
    try:
        result = do_prediction(...)
        observability.end_trace(trace, success=True, result=result)
    except Exception as e:
        trace.errors.append(str(e))
        observability.end_trace(trace, success=False)
        raise

    # Decorator shortcut (sync only)
    from backend.ml_observability import trace_prediction

    @trace_prediction("rag_llm")
    def predict_with_rag(self, location, days):
        ...
"""

import os
import json
import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "ml_predictions.jsonl")


# ---------------------------------------------------------------------------
# Trace dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionTrace:
    """Full trace for one prediction request."""

    trace_id:   str
    location:   str
    days:       int
    method:     str

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time:   Optional[float] = None
    duration_ms: Optional[float] = None

    # Model / quality
    model_used:        str   = ""
    confidence_score:  float = 0.0
    quality_score:     float = 0.0

    # RAG
    rag_docs_retrieved: int = 0

    # Ensemble breakdown
    methods_succeeded: int = 0
    methods_failed:    int = 0

    # Cache
    cache_hit:        bool  = False
    cache_similarity: float = 0.0

    # Security
    security_risk_level:   str = "low"
    injection_attempts:    int = 0

    # Diagnostics
    errors:        List[str] = field(default_factory=list)
    warnings:      List[str] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)

    # Per-agent traces (LangGraph)
    agent_traces: List[Dict] = field(default_factory=list)

    def finish(self) -> None:
        self.end_time   = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)

    def add_agent(
        self,
        name:       str,
        duration_ms: float,
        success:    bool,
        model:      str = "",
        output_len: int = 0,
    ) -> None:
        self.agent_traces.append({
            "agent":       name,
            "duration_ms": round(duration_ms, 2),
            "success":     success,
            "model":       model,
            "output_chars": output_len,
            "ts":          datetime.now().isoformat(),
        })

    def to_summary(self) -> Dict:
        return {
            "trace_id":      self.trace_id,
            "location":      self.location,
            "days":          self.days,
            "method":        self.method,
            "duration_ms":   round(self.duration_ms or 0, 2),
            "model":         self.model_used,
            "confidence":    round(self.confidence_score, 3),
            "cache_hit":     self.cache_hit,
            "rag_docs":      self.rag_docs_retrieved,
            "agents_run":    len(self.agent_traces),
            "errors":        len(self.errors),
            "fallbacks":     len(self.fallbacks_used),
            "security_risk": self.security_risk_level,
            "success":       len(self.errors) == 0,
        }


# ---------------------------------------------------------------------------
# Observability service
# ---------------------------------------------------------------------------

class MLObservabilityService:
    """
    Thread-safe MLOps observability for the weather prediction system.
    """

    WINDOW_SIZE = 1000

    def __init__(self, log_file: str = LOG_FILE) -> None:
        self._log_file = log_file
        self._lock     = threading.Lock()

        # Active (in-flight) traces
        self._active: Dict[str, PredictionTrace] = {}

        # Rolling metrics window
        self._window: List[Dict] = []

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        logger.info("📊 MLObservabilityService ready → %s", log_file)

    # ------------------------------------------------------------------
    def start_trace(
        self,
        location: str,
        days:     int,
        method:   str,
    ) -> PredictionTrace:
        """Begin tracking a prediction request; returns the trace object."""

        trace_id = hashlib.md5(
            f"{location}{days}{method}{time.time()}".encode()
        ).hexdigest()[:12]

        trace = PredictionTrace(
            trace_id=trace_id,
            location=location,
            days=days,
            method=method,
        )

        with self._lock:
            self._active[trace_id] = trace

        logger.debug("🔍 Trace [%s] START %s %dd via %s", trace_id, location, days, method)
        return trace

    # ------------------------------------------------------------------
    def end_trace(
        self,
        trace:   PredictionTrace,
        success: bool = True,
        result:  Optional[Dict] = None,
    ) -> PredictionTrace:
        """Complete, log, and remove a trace."""

        trace.finish()

        # Pull confidence / quality from result if not already set
        if result:
            conf = result.get("confidence_level") or result.get("confidence_score")
            if isinstance(conf, str):
                trace.confidence_score = {
                    "very high": 0.95, "high": 0.80,
                    "medium":    0.60, "low":  0.40,
                    "very low":  0.20,
                }.get(conf.lower(), 0.5)
            elif isinstance(conf, (int, float)):
                trace.confidence_score = float(conf)

            if not trace.model_used:
                trace.model_used = result.get("model_used", "")

        # --- Write JSONL ---
        log_entry = {**trace.to_summary(), "_full": asdict(trace)}
        self._write_log(log_entry)

        # --- Update rolling window ---
        with self._lock:
            self._window.append({
                "duration_ms": trace.duration_ms,
                "confidence":  trace.confidence_score,
                "cache_hit":   trace.cache_hit,
                "success":     success,
                "method":      trace.method,
                "ts":          datetime.now().isoformat(),
            })
            if len(self._window) > self.WINDOW_SIZE:
                self._window = self._window[-self.WINDOW_SIZE:]

            self._active.pop(trace.trace_id, None)

        status = "✅" if success else "❌"
        logger.info(
            "%s Trace [%s] %.0fms | conf=%.2f | cache=%s | errors=%d",
            status, trace.trace_id,
            trace.duration_ms or 0,
            trace.confidence_score,
            "HIT" if trace.cache_hit else "MISS",
            len(trace.errors),
        )
        return trace

    # ------------------------------------------------------------------
    def get_dashboard_metrics(self) -> Dict:
        """Aggregate metrics for the monitoring API endpoint."""
        with self._lock:
            window = list(self._window)
            active_count = len(self._active)

        if not window:
            return {
                "summary":  {"total_predictions": 0, "active_traces": active_count},
                "latency":  {},
                "quality":  {},
                "methods":  {},
                "window_size": 0,
                "generated_at": datetime.now().isoformat(),
            }

        total     = len(window)
        successes = [m for m in window if m["success"]]
        hits      = [m for m in window if m["cache_hit"]]

        durations    = [m["duration_ms"] for m in window if m["duration_ms"]]
        confidences  = [m["confidence"]  for m in window if m["confidence"] > 0]

        def pct(n: int) -> float:
            return round(n / total * 100, 1)

        def percentile(data: List[float], p: int) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return round(sorted(data)[min(idx, len(data) - 1)], 1)

        method_counts: Dict[str, int] = {}
        for m in window:
            key = m.get("method", "unknown")
            method_counts[key] = method_counts.get(key, 0) + 1

        return {
            "summary": {
                "total_predictions": total,
                "success_rate_pct":  pct(len(successes)),
                "cache_hit_rate_pct": pct(len(hits)),
                "active_traces":     active_count,
            },
            "latency": {
                "avg_ms": round(sum(durations) / len(durations), 1) if durations else 0,
                "p50_ms": percentile(durations, 50),
                "p95_ms": percentile(durations, 95),
                "p99_ms": percentile(durations, 99),
            },
            "quality": {
                "avg_confidence": round(
                    sum(confidences) / len(confidences), 3
                ) if confidences else 0,
                "high_conf_rate_pct": round(
                    len([c for c in confidences if c >= 0.8]) /
                    len(confidences) * 100, 1
                ) if confidences else 0,
            },
            "methods":     method_counts,
            "window_size": total,
            "generated_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def _write_log(self, entry: Dict) -> None:
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to write trace log: %s", exc)


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------

def trace_prediction(method_name: str):
    """
    Sync decorator that automatically starts/ends an observability trace.

    Usage:
        @trace_prediction("ensemble")
        def predict_weather_ensemble(self, location, days, ...):
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # Best-effort extraction of location / days
            location = kwargs.get("location") or (args[0] if args else "unknown")
            days     = kwargs.get("prediction_days") or kwargs.get("days") or (
                args[1] if len(args) > 1 else 3
            )

            trace = observability.start_trace(str(location), int(days), method_name)
            try:
                result = fn(self, *args, **kwargs)
                observability.end_trace(trace, success=True, result=result)
                if isinstance(result, dict):
                    result["trace_id"] = trace.trace_id
                return result
            except Exception as exc:
                trace.errors.append(str(exc))
                observability.end_trace(trace, success=False)
                raise
        return wrapper
    return decorator


# Global singleton
observability = MLObservabilityService()
