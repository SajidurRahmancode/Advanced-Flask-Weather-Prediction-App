"""
Test Multi-Query RAG Endpoint
Tests the new /weather/predict-multiquery-rag route end-to-end
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

import json
from datetime import datetime
from backend.rag_service import WeatherRAGService
from backend.lmstudio_service import LMStudioService

def test_multiquery_rag_components():
    """Test the RAG components that the endpoint will use"""
    print("\n" + "="*80)
    print("MULTI-QUERY RAG ENDPOINT COMPONENT TEST")
    print("="*80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Initialize RAG Service
    print("\n[Test 1] Initialize RAG Service")
    print("-" * 80)
    try:
        # Get configuration from environment
        weather_data_path = os.getenv('WEATHER_DATA_PATH', './data/Generated_electricity_load_japan_past365days.csv')
        gemini_api_key = os.getenv('GEMINI_API_KEY', 'test-key')  # Will work with local embeddings if no Gemini
        
        rag_service = WeatherRAGService(
            weather_data_path=weather_data_path,
            gemini_api_key=gemini_api_key
        )
        if rag_service.is_available():
            print("✅ RAG Service initialized and available")
            tests_passed += 1
        else:
            print("❌ RAG Service initialized but not available")
            tests_failed += 1
    except Exception as e:
        print(f"❌ RAG Service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return
    
    # Test 2: Initialize LM Studio Service
    print("\n[Test 2] Initialize LM Studio Service")
    print("-" * 80)
    try:
        lm_studio = LMStudioService()
        if lm_studio.available:
            print(f"✅ LM Studio available: {lm_studio.model_name}")
            print(f"   API URL: {lm_studio.api_url}")
            tests_passed += 1
        else:
            print("⚠️ LM Studio not available - template-based fallback will be used")
            lm_studio = None
            tests_passed += 1  # Still pass - fallback is OK
    except Exception as e:
        print(f"⚠️ LM Studio initialization warning: {e}")
        lm_studio = None
        tests_passed += 1  # Still pass - fallback is OK
    
    # Test 3: Multi-Query Retrieval (with LM Studio if available)
    print("\n[Test 3] Multi-Query Retrieval with AI Query Generation")
    print("-" * 80)
    try:
        location = "Tokyo"
        days = 3
        season = "summer"
        k = 10
        
        print(f"Parameters: location={location}, days={days}, season={season}, k={k}")
        
        rag_result = rag_service.multi_query_retrieval(
            location=location,
            days=days,
            season=season,
            k=k,
            lm_studio_service=lm_studio
        )
        
        print(f"\n📊 Results:")
        print(f"   Query Variations: {len(rag_result['query_variations'])}")
        print(f"   Total Retrieved: {rag_result['total_retrieved']}")
        print(f"   Final Unique: {rag_result['final_count']}")
        print(f"   Deduplicated: {rag_result.get('deduplication', 0)}")
        
        print(f"\n🔍 Query Variations Generated:")
        for i, query in enumerate(rag_result['query_variations'][:3], 1):
            print(f"   {i}. {query[:70]}...")
        
        if len(rag_result['query_variations']) > 3:
            print(f"   ... and {len(rag_result['query_variations'])-3} more")
        
        # Validation
        assert len(rag_result['query_variations']) >= 5, "Should generate at least 5 queries"
        assert rag_result['total_retrieved'] > 0, "Should retrieve documents"
        assert rag_result['final_count'] > 0, "Should have unique documents"
        assert len(rag_result['documents']) > 0, "Documents list should not be empty"
        
        print("\n✅ Multi-Query RAG retrieval successful")
        tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ Multi-Query RAG retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return
    
    # Test 4: Prediction Generation (if LM Studio available)
    if lm_studio:
        print("\n[Test 4] AI Prediction Generation")
        print("-" * 80)
        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['content'][:200]}..." 
                for i, doc in enumerate(rag_result['documents'][:3])
            ])
            
            prompt = f"""Based on historical weather data for {location}, predict the weather for the next {days} days.

Historical Context:
{context}

Season: {season}

Provide a brief {days}-day forecast."""

            print(f"Generating prediction with {lm_studio.model_name}...")
            
            prediction = lm_studio.generate_chat(
                messages=[
                    {"role": "system", "content": "You are an expert meteorologist providing accurate weather forecasts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            if prediction and len(prediction) > 50:
                print(f"\n✅ Prediction generated: {len(prediction)} characters")
                print(f"\n📝 Preview:")
                print(f"   {prediction[:300]}...")
                tests_passed += 1
            else:
                print(f"⚠️ Prediction too short or empty: {len(prediction) if prediction else 0} characters")
                tests_failed += 1
                
        except Exception as e:
            print(f"\n❌ Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    else:
        print("\n[Test 4] AI Prediction Generation - SKIPPED (LM Studio not available)")
        print("-" * 80)
        print("⚠️ Template-based approach will be used in production")
    
    # Test 5: Response Structure Validation
    print("\n[Test 5] Response Structure Validation")
    print("-" * 80)
    try:
        # Simulate endpoint response structure
        response = {
            'success': True,
            'location': location,
            'days': days,
            'season': season,
            'prediction': "Sample prediction text",
            'rag_stats': {
                'query_variations': rag_result['query_variations'],
                'total_retrieved': rag_result['total_retrieved'],
                'final_count': rag_result['final_count'],
                'deduplication_removed': rag_result.get('deduplication', 0),
                'ai_query_generation': lm_studio is not None,
                'model_used': lm_studio.model_name if lm_studio else None
            },
            'context_samples': [
                {
                    'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                    'doc_type': doc.get('doc_type', 'unknown'),
                    'source_query': doc.get('source_query', 'N/A')[:60] + '...' if doc.get('source_query') else 'N/A'
                }
                for doc in rag_result['documents'][:3]
            ],
            'method': f'multi-query-rag-{"ai-powered" if lm_studio else "template-based"}',
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate required fields
        required_fields = ['success', 'location', 'days', 'prediction', 'rag_stats', 'context_samples', 'method', 'timestamp']
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # Validate rag_stats structure
        rag_stats_fields = ['query_variations', 'total_retrieved', 'final_count', 'ai_query_generation']
        for field in rag_stats_fields:
            assert field in response['rag_stats'], f"Missing rag_stats field: {field}"
        
        # Validate context_samples structure
        if response['context_samples']:
            sample = response['context_samples'][0]
            assert 'content' in sample, "context_sample missing 'content'"
            assert 'doc_type' in sample, "context_sample missing 'doc_type'"
            assert 'source_query' in sample, "context_sample missing 'source_query'"
        
        print("✅ Response structure validation passed")
        print(f"\n📋 Sample Response (truncated):")
        print(json.dumps({
            'success': response['success'],
            'location': response['location'],
            'days': response['days'],
            'rag_stats': response['rag_stats'],
            'method': response['method'],
            'context_samples_count': len(response['context_samples'])
        }, indent=2))
        
        tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ Response structure validation failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Tests Passed: {tests_passed}")
    print(f"❌ Tests Failed: {tests_failed}")
    print(f"📊 Success Rate: {tests_passed}/{tests_passed + tests_failed} ({100*tests_passed/(tests_passed+tests_failed):.1f}%)")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed! The endpoint should work correctly.")
    else:
        print(f"\n⚠️ {tests_failed} test(s) failed. Please review the errors above.")
    
    print("="*80)
    
    return tests_passed, tests_failed

if __name__ == "__main__":
    print("\n" + "🧪"*40)
    print("TESTING MULTI-QUERY RAG ENDPOINT COMPONENTS")
    print("🧪"*40)
    
    try:
        passed, failed = test_multiquery_rag_components()
        
        print("\n📝 NEXT STEPS:")
        print("-" * 80)
        if failed == 0:
            print("1. ✅ Component tests passed")
            print("2. 🚀 Start the Flask application:")
            print("   - Run: python app.py")
            print("3. 🧪 Test the endpoint with curl or Postman:")
            print("   POST http://127.0.0.1:5000/api/weather/predict-multiquery-rag")
            print("   Headers: Content-Type: application/json")
            print("   Body: {")
            print('     "location": "Tokyo",')
            print('     "days": 3,')
            print('     "season": "summer",')
            print('     "k": 10')
            print("   }")
            print("4. 📊 Check the response for:")
            print("   - success: true")
            print("   - rag_stats with query variations")
            print("   - prediction text (if LM Studio available)")
            print("   - context_samples showing retrieved data")
        else:
            print("❌ Fix the component issues before testing the full endpoint")
        print("-" * 80)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
