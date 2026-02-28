"""
Test Multi-Query RAG Enhancement
"""

import sys
sys.path.insert(0, 'o:\\portfolio\\Github upload\\Flask-Application')

from backend.rag_service import WeatherRAGService
from backend.lmstudio_service import get_lm_studio_service
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_multi_query():
    """Test multi-query retrieval"""
    print("🧪 Testing Multi-Query RAG Enhancement...\n")
    
    # Initialize services
    weather_data_path = "data\\Generated_electricity_load_japan_past365days.csv"
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    print("📦 Initializing RAG Service...")
    rag_service = WeatherRAGService(weather_data_path, gemini_api_key)
    
    if not rag_service.is_available():
        print("❌ RAG Service not available")
        return
    
    print("✅ RAG Service initialized\n")
    
    # Initialize LM Studio
    print("📦 Initializing LM Studio...")
    lm_studio = get_lm_studio_service()
    
    if lm_studio and lm_studio.available:
        print(f"✅ LM Studio available: {lm_studio.model_name}\n")
    else:
        print("⚠️ LM Studio not available - using template-based queries\n")
    
    # Test multi-query retrieval
    print("🔍 Testing Multi-Query Retrieval...")
    print("=" * 60)
    
    result = rag_service.multi_query_retrieval(
        location="Tokyo",
        days=3,
        season="winter",
        k=5,
        lm_studio_service=lm_studio if lm_studio and lm_studio.available else None
    )
    
    print(f"\n📊 Results:")
    print(f"  Query Variations: {len(result['query_variations'])}")
    print(f"  Total Retrieved: {result['total_retrieved']}")
    print(f"  Final Count: {result['final_count']}")
    print(f"  Deduplicated: {result.get('deduplication', 0)}")
    
    print(f"\n📝 Query Variations:")
    for i, query in enumerate(result['query_variations'], 1):
        print(f"  {i}. {query}")
    
    print(f"\n📄 Sample Documents ({min(3, len(result['documents']))}):")
    for i, doc in enumerate(result['documents'][:3], 1):
        content_preview = doc['content'][:150].replace('\n', ' ')
        print(f"\n  {i}. {content_preview}...")
        print(f"     Type: {doc.get('doc_type', 'unknown')}")
        print(f"     Source Query: {doc.get('source_query', 'N/A')[:60]}...")
    
    print("\n" + "=" * 60)
    print("✅ Multi-Query RAG Test Complete!")
    
    # Compare with single query
    print("\n🔍 Comparing with Single Query Retrieval...")
    single_result = rag_service.retrieve_similar_weather("weather prediction Tokyo 3 days", k=5)
    
    print(f"  Single Query: {len(single_result)} documents")
    print(f"  Multi-Query: {result['final_count']} documents")
    print(f"  Improvement: {result['total_retrieved'] - len(single_result)} more documents found")


if __name__ == "__main__":
    test_multi_query()
