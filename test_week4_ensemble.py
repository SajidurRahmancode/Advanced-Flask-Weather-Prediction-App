"""
Week 4 Test: Ensemble Prediction System

Tests the confidence-weighted ensemble that combines:
  1. LangGraph Multi-Agent  (Week 3)
  2. Multi-query RAG + LLM  (Week 2)
  3. Standard RAG + LLM
  4. Statistical Analysis

Run with:
  $env:PYTHONIOENCODING='utf-8'
  .\\flasking_py311\\Scripts\\python.exe test_week4_ensemble.py
"""

import sys, os, json
from datetime import datetime

# Windows UTF-8 fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
def section(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def subsection(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def ok(msg):   print(f"  [OK]  {msg}")
def warn(msg): print(f"  [!!]  {msg}")
def fail(msg): print(f"  [XX]  {msg}")


# ---------------------------------------------------------------------------
def run_tests():
    section("WEEK 4 TEST: Ensemble Prediction System")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")

    # -----------------------------------------------------------------------
    # 1. Imports
    # -----------------------------------------------------------------------
    subsection("1. Module imports")
    try:
        from backend.lmstudio_service  import LMStudioService
        from backend.rag_service       import WeatherRAGService
        from backend.langgraph_service import LangGraphWeatherService
        from backend.ensemble_service  import EnsemblePredictionService
        from backend.weather_service   import WeatherPredictionService
        ok("All modules imported")
    except ImportError as e:
        fail(f"Import error: {e}")
        return

    # -----------------------------------------------------------------------
    # 2. Service init
    # -----------------------------------------------------------------------
    subsection("2. Service initialisation")

    lm_studio = LMStudioService()
    print(f"  LM Studio : {'available' if lm_studio.available else 'UNAVAILABLE'}")
    if not lm_studio.available:
        warn("LM Studio offline - ensemble will run statistical-only tests")

    weather_svc = WeatherPredictionService()
    print(f"  Weather   : {len(weather_svc.data) if weather_svc.data is not None else 0} CSV rows loaded")

    try:
        rag_svc = WeatherRAGService()
        ok("RAG service ready")
    except Exception as e:
        warn(f"RAG service: {e}")
        rag_svc = None

    langgraph_svc = LangGraphWeatherService(
        weather_service=weather_svc,
        rag_service=rag_svc,
        lm_studio_service=lm_studio,
    )
    print(f"  LangGraph : {'available' if langgraph_svc.available else 'unavailable'}")

    ensemble = EnsemblePredictionService(
        lm_studio_service=lm_studio,
        rag_service=rag_svc,
        langgraph_service=langgraph_svc,
        weather_service=weather_svc,
    )
    ok("Ensemble service created")

    # -----------------------------------------------------------------------
    # 3. Unit test: EnsemblePredictionService directly
    # -----------------------------------------------------------------------
    subsection("3. Direct ensemble service tests")

    CASES = [
        {"location": "Tokyo",    "days": 3, "season": "auto"},
        {"location": "New York", "days": 5, "season": "auto"},
        {"location": "London",   "days": 7, "season": "auto"},
    ]

    results = []
    for tc in CASES:
        loc, days, season = tc["location"], tc["days"], tc["season"]
        print(f"\n  -> {loc}, {days} days, season={season}")
        t0 = datetime.now()

        try:
            result = ensemble.predict_ensemble(
                location=loc, prediction_days=days, season=season
            )
            elapsed = (datetime.now() - t0).total_seconds()

            success     = result.get("success", False)
            method      = result.get("method", "?")
            conf        = result.get("confidence_level", "?")
            qual        = result.get("quality_score", 0.0)
            pred_len    = len(result.get("prediction", ""))
            ens_info    = result.get("ensemble", {})
            methods_used = ens_info.get("methods_used", [])
            best        = ens_info.get("best_method", "?")

            status = "OK" if success else "FAIL"
            print(f"  [{status}] {loc}: method={method}, conf={conf}, quality={qual:.2f}")
            print(f"         methods_used={methods_used}, best={best}")
            print(f"         prediction_len={pred_len} chars, time={elapsed:.1f}s")

            if pred_len > 0:
                preview = result["prediction"][:200].replace("\n", " ")
                print(f"         preview: {preview}...")

            records = {
                "location": loc, "days": days,
                "success": success, "elapsed": elapsed,
                "method": method, "confidence": conf, "quality": qual,
                "prediction_length": pred_len,
                "methods_used": methods_used,
            }
            results.append(records)

            if success:
                ok(f"{loc} passed")
            else:
                fail(f"{loc} failed: {result.get('error', 'no error message')}")

        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            fail(f"{loc} exception: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"location": loc, "success": False, "error": str(exc)})

    # -----------------------------------------------------------------------
    # 4. Via WeatherPredictionService wrapper
    # -----------------------------------------------------------------------
    subsection("4. Via WeatherPredictionService.predict_weather_ensemble()")

    try:
        result = weather_svc.predict_weather_ensemble(
            location="Osaka", prediction_days=4
        )
        if result and result.get("success"):
            ok(f"WeatherPredictionService wrapper: OK, method={result.get('method')}")
        else:
            warn(f"WeatherPredictionService wrapper: non-success ({result.get('error', '?')})")
    except Exception as exc:
        fail(f"WeatherPredictionService wrapper: {exc}")

    # -----------------------------------------------------------------------
    # 5. Status endpoint
    # -----------------------------------------------------------------------
    subsection("5. Ensemble status check")
    status = ensemble.get_status()
    print(f"  available       : {status.get('available')}")
    print(f"  lm_studio       : {status.get('lm_studio')}")
    print(f"  langgraph       : {status.get('langgraph')}")
    print(f"  rag             : {status.get('rag')}")
    print(f"  statistical     : {status.get('statistical')}")
    print(f"  default_weights : {status.get('default_weights')}")

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    section("SUMMARY")
    total   = len(results)
    passed  = sum(1 for r in results if r.get("success"))
    failed  = total - passed
    rate    = passed / total * 100 if total else 0

    print(f"  Tests     : {total}")
    print(f"  Passed    : {passed}")
    print(f"  Failed    : {failed}")
    print(f"  Pass rate : {rate:.1f}%")

    if passed:
        avg_time  = sum(r.get("elapsed", 0) for r in results if r.get("success")) / passed
        avg_len   = sum(r.get("prediction_length", 0) for r in results if r.get("success")) / passed
        all_methods = [m for r in results if r.get("success") for m in r.get("methods_used", [])]
        unique_methods = sorted(set(all_methods))
        print(f"\n  Avg time        : {avg_time:.1f}s")
        print(f"  Avg pred length : {avg_len:.0f} chars")
        print(f"  Methods active  : {unique_methods}")

    # Save results
    output_file = "week4_ensemble_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "summary": {"total": total, "passed": passed, "failed": failed, "pass_rate": rate},
            "results": results,
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    verdict = "PASSED" if rate >= 80 else ("PARTIAL" if rate >= 50 else "FAILED")
    print(f"\n  *** WEEK 4 TEST {verdict} ({rate:.0f}%) ***\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_tests()
