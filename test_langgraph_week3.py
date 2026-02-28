"""
Test Script for Week 3: LangGraph Multi-Agent CoT Enhancement

Tests the 5 enhanced agents with Qwen3 chain-of-thought prompting:
1. Pattern Analysis Agent (AI-enhanced RAG pattern analysis)
2. Meteorological Expert Agent (AI-powered expert analysis)
3. Confidence Assessment Agent (AI-driven confidence evaluation)
4. Quality Control Agent (AI-based quality validation)
5. Prediction Generator Agent (Improved CoT prediction)
"""

import sys
import os
import json
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_langgraph_week3_enhancements():
    """Test LangGraph multi-agent system with Week 3 CoT enhancements"""
    
    print("=" * 80)
    print("WEEK 3 TEST: LangGraph Multi-Agent CoT Enhancement")
    print("=" * 80)
    print()
    
    # Import services
    try:
        from backend.lmstudio_service import LMStudioService
        from backend.rag_service import WeatherRAGService
        from backend.langgraph_service import LangGraphWeatherService
        from backend.weather_service import WeatherPredictionService
        print("✅ All services imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("-" * 80)
    print("INITIALIZING SERVICES")
    print("-" * 80)
    
    # Initialize services
    lm_studio = LMStudioService()
    print(f"LM Studio Service: {'✅ Available' if lm_studio.available else '❌ Unavailable'}")
    
    if not lm_studio.available:
        print("⚠️ LM Studio not available - Week 3 CoT enhancements cannot be tested")
        return
    
    weather_service = WeatherPredictionService()
    print(f"Weather Service: ✅ Initialized")
    
    try:
        rag_service = WeatherRAGService()
        print(f"RAG Service: {'✅ Available' if rag_service else '❌ Unavailable'}")
    except Exception as e:
        print(f"RAG Service: ⚠️ Error: {e}")
        rag_service = None
    
    langgraph_service = LangGraphWeatherService(
        weather_service=weather_service,
        rag_service=rag_service,
        langchain_service=None,  # Not testing LangChain
        lm_studio_service=lm_studio,
        websocket_service=None  # Not testing WebSocket
    )
    print(f"LangGraph Service: {'✅ Available' if langgraph_service.available else '❌ Unavailable'}")
    
    if not langgraph_service.available:
        print("❌ LangGraph service not available - cannot run tests")
        return
    
    print()
    print("-" * 80)
    print("TEST CASES")
    print("-" * 80)
    print()
    
    test_cases = [
        {
            "name": "Test 1: Tokyo 3-day forecast",
            "location": "Tokyo",
            "days": 3,
            "expected_agents": ["data_collector", "pattern_analyzer", "meteorologist", "confidence_assessor", "prediction_generator", "quality_controller"]
        },
        {
            "name": "Test 2: New York 5-day forecast",
            "location": "New York",
            "days": 5,
            "expected_agents": ["data_collector", "pattern_analyzer", "meteorologist", "confidence_assessor", "prediction_generator", "quality_controller"]
        },
        {
            "name": "Test 3: London 7-day forecast",
            "location": "London",
            "days": 7,
            "expected_agents": ["data_collector", "pattern_analyzer", "meteorologist", "confidence_assessor", "prediction_generator", "quality_controller"]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"{test_case['name']}")
        print(f"{'='*80}")
        print(f"Location: {test_case['location']}")
        print(f"Days: {test_case['days']}")
        print()
        
        try:
            # Run prediction
            start_time = datetime.now()
            result = langgraph_service.predict_weather_with_langgraph(
                location=test_case['location'],
                prediction_days=test_case['days']
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Analyze result
            success = result.get('success', False)
            prediction_text = result.get('prediction', '')
            method_used = result.get('method', 'Unknown')
            model_used = result.get('model_used', 'Unknown')
            confidence_level = result.get('confidence_level', 'Unknown')
            quality_score = result.get('quality_score', 0.0)
            
            # Check agent reports
            langgraph_analysis = result.get('langgraph_analysis', {})
            agent_reports = langgraph_analysis.get('agent_reports', {})
            analysis_results = langgraph_analysis.get('analysis_results', {})
            
            print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Method: {method_used}")
            print(f"Model: {model_used}")
            print(f"Confidence: {confidence_level}")
            print(f"Quality Score: {quality_score:.2f}")
            print()
            
            print("AGENT REPORTS:")
            for agent, report in agent_reports.items():
                print(f"  • {agent}: {report}")
            print()
            
            # Check for Week 3 CoT enhancements
            print("WEEK 3 COT ENHANCEMENTS:")
            
            # Pattern Analysis AI enhancement
            pattern_analysis = analysis_results.get('pattern_analysis', {})
            if pattern_analysis:
                patterns_found = pattern_analysis.get('patterns_found', 0)
                has_ai_reasoning = bool(pattern_analysis.get('ai_reasoning'))
                print(f"  • Pattern Analysis: {patterns_found} patterns, AI-enhanced: {'✅' if has_ai_reasoning else '❌'}")
                if has_ai_reasoning:
                    print(f"    AI Reasoning: {pattern_analysis['ai_reasoning'][:100]}...")
            
            # Meteorological AI enhancement
            meteorological = analysis_results.get('meteorological', {})
            if meteorological:
                method = meteorological.get('analysis_method', 'unknown')
                has_ai = 'qwen3' in method or 'ai_analysis' in meteorological
                print(f"  • Meteorological: Method = {method}, AI-enhanced: {'✅' if has_ai else '❌'}")
                if has_ai and meteorological.get('ai_analysis'):
                    print(f"    AI Analysis: {meteorological['ai_analysis'][:100]}...")
            
            # Confidence AI enhancement
            confidence_data = analysis_results.get('confidence', {})
            if confidence_data:
                method = confidence_data.get('method', 'unknown')
                has_ai_reasoning = bool(confidence_data.get('ai_reasoning'))
                print(f"  • Confidence: Method = {method}, AI-enhanced: {'✅' if has_ai_reasoning else '❌'}")
                if has_ai_reasoning:
                    print(f"    AI Reasoning: {confidence_data['ai_reasoning'][:100]}...")
            
            # Check prediction text for CoT indicators
            has_reasoning = "<think>" in prediction_text or "step-by-step" in prediction_text.lower() or "reasoning" in prediction_text.lower()
            print(f"  • Prediction: Contains reasoning indicators: {'✅' if has_reasoning else '❌'}")
            
            print()
            print("PREDICTION PREVIEW:")
            print("-" * 80)
            print(prediction_text[:500] + ("..." if len(prediction_text) > 500 else ""))
            print("-" * 80)
            
            # Record result
            test_result = {
                "test_name": test_case['name'],
                "success": success,
                "duration": duration,
                "method": method_used,
                "confidence": confidence_level,
                "quality_score": quality_score,
                "agents_executed": len(agent_reports),
                "cot_enhancements": {
                    "pattern_analysis_ai": bool(pattern_analysis.get('ai_reasoning')),
                    "meteorological_ai": meteorological.get('analysis_method') == 'qwen3_cot',
                    "confidence_ai": confidence_data.get('method') == 'qwen3_cot',
                    "prediction_reasoning": has_reasoning
                },
                "prediction_length": len(prediction_text)
            }
            results.append(test_result)
            
            print(f"\n✅ Test {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Test {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_name": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print()
    print("=" * 80)
    print("WEEK 3 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get('success', False))
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    # CoT Enhancement Analysis
    if successful_tests > 0:
        print("COT ENHANCEMENT ADOPTION:")
        cot_stats = {
            "pattern_analysis_ai": 0,
            "meteorological_ai": 0,
            "confidence_ai": 0,
            "prediction_reasoning": 0
        }
        
        for r in results:
            if r.get('success'):
                enhancements = r.get('cot_enhancements', {})
                for key in cot_stats:
                    if enhancements.get(key):
                        cot_stats[key] += 1
        
        print(f"  • Pattern Analysis AI: {cot_stats['pattern_analysis_ai']}/{successful_tests} ({cot_stats['pattern_analysis_ai']/successful_tests*100:.1f}%)")
        print(f"  • Meteorological AI: {cot_stats['meteorological_ai']}/{successful_tests} ({cot_stats['meteorological_ai']/successful_tests*100:.1f}%)")
        print(f"  • Confidence Assessment AI: {cot_stats['confidence_ai']}/{successful_tests} ({cot_stats['confidence_ai']/successful_tests*100:.1f}%)")
        print(f"  • Prediction Reasoning: {cot_stats['prediction_reasoning']}/{successful_tests} ({cot_stats['prediction_reasoning']/successful_tests*100:.1f}%)")
        print()
        
        # Average metrics
        avg_duration = sum(r.get('duration', 0) for r in results if r.get('success')) / successful_tests
        avg_quality = sum(r.get('quality_score', 0) for r in results if r.get('success')) / successful_tests
        avg_prediction_length = sum(r.get('prediction_length', 0) for r in results if r.get('success')) / successful_tests
        
        print("AVERAGE METRICS:")
        print(f"  • Duration: {avg_duration:.2f} seconds")
        print(f"  • Quality Score: {avg_quality:.2f}")
        print(f"  • Prediction Length: {avg_prediction_length:.0f} characters")
    
    # Save results to file
    output_file = 'week3_langgraph_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful': successful_tests,
                'failed': total_tests - successful_tests,
                'success_rate': success_rate
            },
            'results': results
        }, f, indent=2)
    
    print()
    print(f"📄 Detailed results saved to: {output_file}")
    print()
    print("=" * 80)
    print(f"WEEK 3 TEST {'✅ PASSED' if success_rate >= 80 else '⚠️ NEEDS ATTENTION' if success_rate >= 60 else '❌ FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    test_langgraph_week3_enhancements()
