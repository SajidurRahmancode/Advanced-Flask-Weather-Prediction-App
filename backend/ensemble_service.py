"""
Week 4: Ensemble Prediction Service

Combines predictions from multiple methods using confidence-weighted aggregation
and Qwen3 chain-of-thought meta-synthesis for maximum accuracy.

Methods combined:
  1. LangGraph Multi-Agent (Week 3)   - highest reasoning depth
  2. Multi-query RAG + LLM  (Week 2)  - widest retrieval coverage
  3. Standard RAG + LLM               - fast reliable baseline
  4. Statistical Analysis              - always-available fallback

Pipeline:
  Run all available methods -> Score each by confidence/quality ->
  Build weighted summary -> Qwen3 CoT meta-synthesis -> Final prediction
"""

import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Observability (lazy import to avoid circular dependency at load time)
def _get_observability():
    try:
        from backend.ml_observability import observability
        return observability
    except Exception:
        return None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default weights (can be overridden by confidence-based dynamic weights)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "langgraph":      0.40,
    "multiquery_rag": 0.30,
    "rag":            0.20,
    "statistical":    0.10,
}

# Quality floor: methods scoring below this are excluded from the synthesis
MIN_QUALITY_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Helper: extract numeric confidence from a confidence label or float
# ---------------------------------------------------------------------------
def _score_from_label(label: str | float | None) -> float:
    """Convert a confidence label ('High', 'Medium', …) or numeric value to 0-1 float."""
    if isinstance(label, (int, float)):
        return float(min(1.0, max(0.0, label)))
    if isinstance(label, str):
        mapping = {"very high": 0.95, "high": 0.80, "medium": 0.60,
                   "low": 0.40, "very low": 0.20}
        return mapping.get(label.lower(), 0.50)
    return 0.50


class EnsemblePredictionService:
    """
    Confidence-weighted ensemble that meta-synthesises multiple prediction
    methods via Qwen3 chain-of-thought reasoning.
    """

    def __init__(
        self,
        lm_studio_service=None,
        rag_service=None,
        langgraph_service=None,
        langchain_service=None,
        weather_service=None,
    ):
        self.lm_studio   = lm_studio_service
        self.rag         = rag_service
        self.langgraph   = langgraph_service
        self.langchain   = langchain_service
        self.weather_svc = weather_service

        self.available = lm_studio_service is not None
        logger.info("Ensemble service initialised (lm_studio=%s)", self.available)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def predict_ensemble(
        self,
        location: str = "Tokyo",
        prediction_days: int = 3,
        season: str = "auto",
        enable_multiquery: bool = True,
        timeout_seconds: int = 90,
    ) -> Dict[str, Any]:
        """
        Generate an ensemble weather prediction.

        Steps:
          1. Collect predictions from every available method.
          2. Score each by confidence + quality.
          3. Build a structured summary of the best results.
          4. Run Qwen3 CoT meta-synthesis.
          5. Return ensemble result with full breakdown.
        """
        start = datetime.now()
        logger.info("Ensemble prediction: location=%s days=%d", location, prediction_days)

        # Start observability trace
        obs = _get_observability()
        trace = obs.start_trace(location, prediction_days, "ensemble") if obs else None

        # ------------------------------------------------------------------ #
        # 1. Gather predictions from all methods                              #
        # ------------------------------------------------------------------ #
        raw_results = self._gather_all_predictions(
            location, prediction_days, season, enable_multiquery, timeout_seconds
        )

        # ------------------------------------------------------------------ #
        # 2. Score and weight each result                                     #
        # ------------------------------------------------------------------ #
        scored = self._score_predictions(raw_results)
        if not scored:
            if trace and obs:
                trace.errors.append("No valid predictions after scoring")
                obs.end_trace(trace, success=False)
            return self._fallback_result(location, prediction_days, "No valid predictions")

        # ------------------------------------------------------------------ #
        # 3. Meta-synthesis with Qwen3 CoT (only when ≥2 methods available)  #
        # ------------------------------------------------------------------ #
        if len(scored) >= 2 and self.lm_studio and self.lm_studio.available:
            final_prediction, meta_reasoning = self._meta_synthesise(
                location, prediction_days, scored
            )
        else:
            # Single method or no LLM: use best individual result directly
            best_single = max(scored, key=lambda x: x["weighted_score"])
            final_prediction = best_single["prediction"]
            meta_reasoning = (
                f"Meta-synthesis skipped – only {len(scored)} method(s) contributed. "
                f"Using {best_single['method']} result directly."
            )
            logger.info("Meta-synthesis skipped (%d methods), using %s", len(scored), best_single['method'])

        # ------------------------------------------------------------------ #
        # 4. Build response                                                   #
        # ------------------------------------------------------------------ #
        duration = (datetime.now() - start).total_seconds()

        methods_used = [s["method"] for s in scored]
        best = max(scored, key=lambda x: x["weighted_score"])

        # Update trace with final metrics
        if trace:
            trace.methods_succeeded = len([v for v in raw_results.values() if v.get("success")])
            trace.methods_failed    = len([v for v in raw_results.values() if not v.get("success")])
            trace.model_used = "Qwen3-14B Ensemble"

        result = {
            "success": True,
            "prediction": final_prediction,
            "location": location,
            "prediction_days": prediction_days,
            "timeframe": prediction_days,
            "generated_at": datetime.now().isoformat(),
            "method": "ensemble_cot",
            "model_used": "Qwen3-14B Ensemble (Week 4)",
            "duration_seconds": round(duration, 2),
            "confidence_level": self._ensemble_confidence_label(scored),
            "quality_score": round(best["quality_score"], 3),
            "ensemble": {
                "methods_attempted": list(raw_results.keys()),
                "methods_used": methods_used,
                "method_count": len(scored),
                "best_method": best["method"],
                "best_score": round(best["weighted_score"], 3),
                "weights_applied": {s["method"]: round(s["weight"], 3) for s in scored},
                "confidence_scores": {s["method"]: round(s["confidence"], 3) for s in scored},
                "quality_scores":    {s["method"]: round(s["quality_score"], 3) for s in scored},
                "meta_reasoning_length": len(meta_reasoning),
                "meta_reasoning_preview": meta_reasoning[:300] + "..." if len(meta_reasoning) > 300 else meta_reasoning,
            },
            "features": [
                "Multi-Method Ensemble",
                "Confidence-Weighted Aggregation",
                "Qwen3 CoT Meta-Synthesis",
                "Dynamic Method Selection",
                "Quality Filtering",
            ],
            "enhancement": "Week 4 ensemble combining LangGraph, Multi-query RAG, standard RAG and statistical analysis",
        }

        if trace and obs:
            obs.end_trace(trace, success=True, result=result)
            result["trace_id"] = trace.trace_id

        return result

    # ------------------------------------------------------------------
    # Step 1: gather predictions
    # ------------------------------------------------------------------

    def _gather_all_predictions(
        self, location, days, season, enable_multiquery, timeout
    ) -> Dict[str, Dict[str, Any]]:
        """Run all prediction methods and collect results (with timeouts)."""

        results: Dict[str, Dict[str, Any]] = {}

        def run(name, fn):
            try:
                result = fn()
                if result:
                    results[name] = result
                    logger.info("  [%s] OK  (success=%s)", name, result.get("success"))
                else:
                    logger.warning("  [%s] returned None", name)
            except Exception as exc:
                logger.warning("  [%s] failed: %s", name, exc)
                results[name] = {"success": False, "error": str(exc), "method": name}

        tasks = []

        # LangGraph multi-agent
        if self.langgraph and self.langgraph.available:
            tasks.append(("langgraph", lambda: self.langgraph.predict_weather_with_langgraph(
                location=location, prediction_days=days
            )))

        # Multi-query RAG
        if enable_multiquery and self.rag and self.lm_studio and self.lm_studio.available:
            tasks.append(("multiquery_rag", lambda: self._run_multiquery_rag(location, days, season)))

        # Standard RAG + LLM
        if self.rag and self.lm_studio and self.lm_studio.available:
            tasks.append(("rag", lambda: self._run_standard_rag(location, days)))

        # Statistical (always available when weather_svc exists)
        if self.weather_svc:
            tasks.append(("statistical", lambda: self._run_statistical(location, days)))

        if not tasks:
            logger.warning("No prediction tasks available - all services offline")
            return results

        # Run in thread pool with a per-method timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(run, name, fn): name for name, fn in tasks}
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                pass  # results populated inside run()

        return results

    # ------------------------------------------------------------------
    # Individual wrapped method runners
    # ------------------------------------------------------------------

    def _run_multiquery_rag(self, location, days, season) -> Dict[str, Any]:
        """Run multi-query RAG prediction (Week 2)."""
        if season == "auto":
            month = datetime.now().month
            season = (
                "Winter" if month in [12, 1, 2] else
                "Spring" if month in [3, 4, 5] else
                "Summer" if month in [6, 7, 8] else "Autumn"
            )

        docs = self.rag.multi_query_retrieval(
            location=location,
            days=days,
            season=season,
            k=6,
            lm_studio_service=self.lm_studio,
        )
        if not docs:
            return {"success": False, "method": "multiquery_rag"}

        context = "\n---\n".join(d.page_content for d in docs[:5])
        prompt = (
            f"Based on historical weather patterns for {location}, "
            f"generate a detailed {days}-day forecast. "
            f"Use this context:\n\n{context}"
        )
        messages = [{"role": "user", "content": prompt}]
        prediction = self.lm_studio.generate_chat(messages, max_tokens=1500, temperature=0.3)

        return {
            "success": bool(prediction and len(prediction) > 100),
            "prediction": prediction or "",
            "method": "multiquery_rag",
            "model_used": "Qwen3 + Multi-Query RAG",
            "confidence_level": "High" if len(docs) >= 5 else "Medium",
            "quality_score": min(0.90, 0.5 + len(docs) * 0.05),
            "rag_docs_found": len(docs),
        }

    def _run_standard_rag(self, location, days) -> Dict[str, Any]:
        """Run standard single-query RAG prediction."""
        query = f"weather forecast {location} {days} days temperature humidity"
        try:
            docs = self.rag.retrieve_similar_weather(query, k=5)
        except Exception:
            docs = []

        if not docs:
            return {"success": False, "method": "rag"}

        context = "\n---\n".join(d.page_content for d in docs[:4])
        prompt = (
            f"Using historical weather data for {location}, "
            f"generate a {days}-day forecast.\n\nContext:\n{context}"
        )
        messages = [{"role": "user", "content": prompt}]
        prediction = self.lm_studio.generate_chat(messages, max_tokens=1200, temperature=0.3)

        return {
            "success": bool(prediction and len(prediction) > 80),
            "prediction": prediction or "",
            "method": "rag",
            "model_used": "Qwen3 + Standard RAG",
            "confidence_level": "Medium",
            "quality_score": 0.60,
            "rag_docs_found": len(docs),
        }

    def _run_statistical(self, location, days) -> Dict[str, Any]:
        """Run statistical baseline using weather service data."""
        try:
            data = self.weather_svc.data
            if data is None or data.empty:
                return {"success": False, "method": "statistical"}

            recent = data.tail(7)
            temp_avg = recent.get("Actual_Temperature(°C)", recent.iloc[:, 0]).mean()
            humidity_avg = recent.get("Actual_Humidity(%)", recent.iloc[:, 1]).mean() if len(recent.columns) > 1 else 60.0

            prediction = (
                f"Statistical Weather Forecast for {location} ({days} days)\n\n"
                f"Based on recent 7-day average conditions:\n"
                f"- Temperature: ~{temp_avg:.1f}°C\n"
                f"- Humidity: ~{humidity_avg:.0f}%\n\n"
                f"Outlook: Conditions expected to remain similar to recent averages "
                f"with typical day-to-day variability. "
                f"Monitor for seasonal transitions over the {days}-day period."
            )
        except Exception as exc:
            logger.warning("Statistical fallback error: %s", exc)
            prediction = (
                f"Statistical Weather Forecast for {location} ({days} days):\n"
                f"Forecast based on historical seasonal averages. "
                f"Conditions typical for current season with standard variability expected."
            )

        return {
            "success": True,
            "prediction": prediction,
            "method": "statistical",
            "model_used": "Statistical Analysis",
            "confidence_level": "Low",
            "quality_score": 0.40,
        }

    # ------------------------------------------------------------------
    # Step 2: scoring
    # ------------------------------------------------------------------

    def _score_predictions(self, raw: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach confidence/quality scores and compute weighted scores."""
        scored = []

        for method, result in raw.items():
            if not result.get("success"):
                continue
            prediction_text = result.get("prediction", "")
            if not prediction_text or len(prediction_text) < 50:
                continue

            # Raw confidence
            conf = _score_from_label(
                result.get("confidence_level") or result.get("confidence_score", 0.5)
            )
            # If LangGraph gave us a numeric score
            if method == "langgraph":
                lga = result.get("langgraph_analysis", {})
                if lga.get("rag_confidence"):
                    conf = max(conf, float(lga["rag_confidence"]))

            # Quality score
            quality = float(result.get("quality_score", 0.5))
            # Penalise very short predictions
            if len(prediction_text) < 200:
                quality *= 0.7
            if len(prediction_text) > 800:
                quality = min(1.0, quality * 1.1)

            if quality < MIN_QUALITY_THRESHOLD:
                logger.info("  [%s] excluded (quality=%.2f below threshold)", method, quality)
                continue

            base_weight = DEFAULT_WEIGHTS.get(method, 0.10)
            # Boost weight by confidence delta from 0.5 baseline
            adjusted_weight = base_weight * (0.7 + 0.6 * conf)

            weighted_score = adjusted_weight * conf * quality

            scored.append({
                "method":         method,
                "prediction":     prediction_text,
                "confidence":     round(conf, 3),
                "quality_score":  round(quality, 3),
                "weight":         round(adjusted_weight, 3),
                "weighted_score": round(weighted_score, 4),
                "raw_result":     result,
            })

        # Sort best first
        scored.sort(key=lambda x: x["weighted_score"], reverse=True)
        logger.info(
            "Scored %d/%d methods: %s",
            len(scored), len(raw),
            [(s["method"], round(s["weighted_score"], 3)) for s in scored],
        )
        return scored

    # ------------------------------------------------------------------
    # Step 3: Qwen3 meta-synthesis
    # ------------------------------------------------------------------

    def _meta_synthesise(
        self, location: str, days: int, scored: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Use Qwen3 CoT to synthesise scored predictions into a final forecast."""

        if not self.lm_studio or not self.lm_studio.available:
            # No LLM – return best individual prediction
            best = scored[0]["prediction"]
            return best, "Meta-synthesis skipped (LM Studio unavailable)"

        # Build per-method summaries for the prompt (cap each at 500 chars)
        method_summaries = []
        for rank, s in enumerate(scored[:4], start=1):
            snippet = s["prediction"][:500].replace("\n", " ")
            if len(s["prediction"]) > 500:
                snippet += "..."
            method_summaries.append(
                f"Method {rank} [{s['method'].upper()}] "
                f"(confidence={s['confidence']:.2f}, quality={s['quality_score']:.2f}, "
                f"weight={s['weight']:.2f}):\n  {snippet}"
            )

        summaries_text = "\n\n".join(method_summaries)
        method_names = ", ".join(s["method"] for s in scored[:4])

        cot_prompt = f"""You are a meteorologist. Synthesise these {len(scored)} forecasts for {location} ({days} days) into one concise authoritative forecast.

{summaries_text}

Write a {days}-day forecast covering temperature, precipitation and overall conditions.
Where methods agree, state that confidently. Where they differ, express uncertainty.
Be concise (3-5 sentences per day max). End with an overall confidence level (Low/Medium/High)."""

        try:
            messages = [{"role": "user", "content": cot_prompt}]
            synthesis = self.lm_studio.generate_chat(messages, max_tokens=800, temperature=0.25)
            if synthesis and len(synthesis) > 150:
                logger.info("Meta-synthesis: %d characters generated", len(synthesis))
                return synthesis, synthesis
        except Exception as exc:
            logger.warning("Meta-synthesis failed: %s", exc)

        # Fallback: return best prediction
        return scored[0]["prediction"], "Meta-synthesis failed – returning best individual prediction"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensemble_confidence_label(self, scored: List[Dict[str, Any]]) -> str:
        if not scored:
            return "Very Low"
        avg_conf = sum(s["confidence"] for s in scored) / len(scored)
        method_count = len(scored)
        # Boost for agreement among multiple methods
        agreement_boost = min(0.10, (method_count - 1) * 0.04)
        final = min(0.98, avg_conf + agreement_boost)
        if final >= 0.80:
            return "High"
        if final >= 0.60:
            return "Medium"
        if final >= 0.40:
            return "Low"
        return "Very Low"

    def _fallback_result(self, location, days, reason) -> Dict[str, Any]:
        """Return a graceful failure response."""
        prediction = (
            f"Weather Forecast for {location} ({days} days):\n"
            f"Forecast service temporarily unavailable. "
            f"Please check that LM Studio is running on port 1234 and try again."
        )
        return {
            "success": False,
            "prediction": prediction,
            "location": location,
            "prediction_days": days,
            "timeframe": days,
            "generated_at": datetime.now().isoformat(),
            "method": "ensemble_fallback",
            "model_used": "Fallback",
            "confidence_level": "Very Low",
            "quality_score": 0.0,
            "error": reason,
            "ensemble": {"methods_attempted": [], "methods_used": [], "method_count": 0},
        }

    def get_status(self) -> Dict[str, Any]:
        """Return capability status for the ensemble service."""
        return {
            "service": "Ensemble Prediction (Week 4)",
            "available": self.available,
            "lm_studio": bool(self.lm_studio and self.lm_studio.available),
            "langgraph": bool(self.langgraph and self.langgraph.available),
            "rag": bool(self.rag),
            "statistical": bool(self.weather_svc),
            "default_weights": DEFAULT_WEIGHTS,
            "min_quality_threshold": MIN_QUALITY_THRESHOLD,
        }
