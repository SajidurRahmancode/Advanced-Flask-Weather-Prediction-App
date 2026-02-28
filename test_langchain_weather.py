#!/usr/bin/env python3
"""
LangChain Weather Prediction Test Script

This script demonstrates how to use LangChain to call the internal weather prediction 
functions of the Flask project. It can be used for testing, debugging, or as an 
example of how to integrate LangChain with the weather prediction system.

Usage:
    python test_langchain_weather.py
    
Requirements:
    - Flask app dependencies installed
    - LM Studio running (optional)
    - XAMPP MySQL running
    - Environment configured
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import Flask app components
from backend.weather_service import WeatherPredictionService
from backend.langchain_rag_service import LangChainRAGService, get_langchain_rag_service
from backend.lmstudio_service import LMStudioService
from backend.rag_service import WeatherRAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('langchain_weather_test.log')
    ]
)

logger = logging.getLogger(__name__)

class LangChainWeatherTester:
    """Test class for LangChain weather prediction functionality"""
    
    def __init__(self):
        """Initialize the tester with all required services"""
        self.weather_service = None
        self.langchain_service = None
        self.lm_studio_service = None
        self.rag_service = None
        
        logger.info("🧪 Initializing LangChain Weather Tester...")
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services for testing"""
        try:
            # Initialize Weather Service
            logger.info("🌤️ Initializing Weather Service...")
            self.weather_service = WeatherPredictionService()
            
            # Initialize LM Studio Service
            logger.info("🏠 Initializing LM Studio Service...")
            self.lm_studio_service = LMStudioService()
            
            # Initialize RAG Service with proper parameters
            logger.info("📚 Initializing RAG Service...")
            weather_data_path = os.path.join(os.path.dirname(__file__), 'data', 'Generated_electricity_load_japan_past365days.csv')
            gemini_api_key = os.getenv('GEMINI_API_KEY', '')  # Empty string if not set
            self.rag_service = WeatherRAGService(weather_data_path, gemini_api_key)
            
            # Initialize LangChain Service
            logger.info("⛓️ Initializing LangChain Service...")
            self.langchain_service = get_langchain_rag_service(
                weather_service=self.weather_service,
                lm_studio_service=self.lm_studio_service,
                rag_service=self.rag_service
            )
            
            logger.info("✅ All services initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise
    
    def test_service_availability(self) -> Dict[str, bool]:
        """Test availability of all services"""
        logger.info("🔍 Testing service availability...")
        
        availability = {
            "weather_service": False,
            "lm_studio_service": False,
            "rag_service": False,
            "langchain_service": False
        }
        
        try:
            # Test Weather Service
            if self.weather_service and hasattr(self.weather_service, 'data'):
                availability["weather_service"] = len(self.weather_service.data) > 0
                logger.info(f"✅ Weather Service: {len(self.weather_service.data)} records loaded")
            
            # Test LM Studio Service
            if self.lm_studio_service:
                availability["lm_studio_service"] = self.lm_studio_service.available
                status = "✅ Available" if self.lm_studio_service.available else "❌ Not available"
                logger.info(f"{status} LM Studio Service")
            
            # Test RAG Service
            if self.rag_service:
                availability["rag_service"] = hasattr(self.rag_service, 'vectorstore')
                status = "✅ Available" if availability["rag_service"] else "❌ Not available"
                logger.info(f"{status} RAG Service")
            
            # Test LangChain Service
            if self.langchain_service:
                availability["langchain_service"] = self.langchain_service.available
                status = "✅ Available" if self.langchain_service.available else "❌ Not available"
                logger.info(f"{status} LangChain Service")
                
        except Exception as e:
            logger.error(f"❌ Error testing service availability: {e}")
        
        return availability
    
    def test_langchain_prediction(self, location: str = "Tokyo", days: int = 3) -> Dict[str, Any]:
        """Test LangChain weather prediction"""
        logger.info(f"🧠 Testing LangChain prediction for {location}, {days} days...")
        
        try:
            if not self.weather_service:
                raise Exception("Weather service not initialized")
            
            # Check if LangChain method is available
            if not hasattr(self.weather_service, 'predict_weather_langchain_rag'):
                raise Exception("LangChain prediction method not available")
            
            # Call the LangChain prediction method
            result = self.weather_service.predict_weather_langchain_rag(location, days)
            
            if result and result.get('success'):
                logger.info("✅ LangChain prediction successful!")
                logger.info(f"📊 Method: {result.get('method', 'Unknown')}")
                logger.info(f"🎯 Confidence: {result.get('confidence_level', 'Unknown')}")
                logger.info(f"📝 Prediction length: {len(result.get('prediction', ''))} characters")
            else:
                logger.warning("⚠️ LangChain prediction failed or returned no results")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ LangChain prediction test failed: {e}")
            return {"error": str(e), "success": False}
    
    def test_rag_search(self, query: str = "temperature 20 degrees humidity 60 percent") -> Dict[str, Any]:
        """Test RAG similarity search functionality"""
        logger.info(f"📚 Testing RAG search for: '{query}'...")
        
        try:
            if not self.rag_service:
                raise Exception("RAG service not initialized")
            
            # Perform similarity search
            similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
            
            result = {
                "query": query,
                "patterns_found": len(similar_patterns),
                "patterns": []
            }
            
            for i, pattern in enumerate(similar_patterns):
                pattern_info = {
                    "index": i + 1,
                    "content": pattern.page_content[:200] + "..." if len(pattern.page_content) > 200 else pattern.page_content,
                    "metadata": getattr(pattern, 'metadata', {})
                }
                result["patterns"].append(pattern_info)
            
            logger.info(f"✅ RAG search successful! Found {len(similar_patterns)} similar patterns")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG search test failed: {e}")
            return {"error": str(e), "patterns_found": 0}
    
    def test_lm_studio_connection(self) -> Dict[str, Any]:
        """Test LM Studio connection and model availability"""
        logger.info("🏠 Testing LM Studio connection...")
        
        try:
            if not self.lm_studio_service:
                raise Exception("LM Studio service not initialized")
            
            # Test connection
            status = self.lm_studio_service.test_connection()
            
            if status.get('available', False):
                logger.info("✅ LM Studio connection successful!")
                logger.info(f"🤖 Model: {status.get('model_name', 'Unknown')}")
            else:
                logger.warning("⚠️ LM Studio not available")
                logger.info("💡 Tip: Make sure LM Studio is running with a loaded model")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ LM Studio connection test failed: {e}")
            return {"error": str(e), "available": False}
    
    def test_langchain_memory(self) -> Dict[str, Any]:
        """Test LangChain conversation memory functionality"""
        logger.info("🧠 Testing LangChain conversation memory...")
        
        try:
            if not self.langchain_service:
                raise Exception("LangChain service not initialized")
            
            # Clear memory first
            self.langchain_service.clear_conversation_memory()
            
            # Test memory storage and retrieval
            test_input = "What's the weather like in Tokyo?"
            test_output = "The weather in Tokyo is currently 22°C and partly cloudy."
            
            # Save to memory
            self.langchain_service.memory.save_context(
                {"input": test_input},
                {"output": test_output}
            )
            
            # Retrieve memory
            memory_content = self.langchain_service.get_conversation_history()
            
            result = {
                "memory_test": "successful",
                "test_input": test_input,
                "test_output": test_output,
                "retrieved_memory": memory_content,
                "memory_length": len(memory_content)
            }
            
            logger.info("✅ LangChain memory test successful!")
            logger.info(f"📝 Memory content length: {len(memory_content)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LangChain memory test failed: {e}")
            return {"error": str(e), "memory_test": "failed"}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and provide comprehensive results"""
        logger.info("🚀 Starting comprehensive LangChain weather prediction test...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "service_availability": {},
            "langchain_prediction": {},
            "rag_search": {},
            "lm_studio_connection": {},
            "langchain_memory": {},
            "overall_status": "unknown"
        }
        
        try:
            # Test service availability
            test_results["service_availability"] = self.test_service_availability()
            
            # Test LM Studio connection
            test_results["lm_studio_connection"] = self.test_lm_studio_connection()
            
            # Test RAG search
            test_results["rag_search"] = self.test_rag_search()
            
            # Test LangChain memory
            test_results["langchain_memory"] = self.test_langchain_memory()
            
            # Test LangChain prediction
            test_results["langchain_prediction"] = self.test_langchain_prediction()
            
            # Determine overall status
            all_services_available = all(test_results["service_availability"].values())
            langchain_success = test_results["langchain_prediction"].get("success", False)
            
            if all_services_available and langchain_success:
                test_results["overall_status"] = "excellent"
            elif langchain_success:
                test_results["overall_status"] = "good"
            elif any(test_results["service_availability"].values()):
                test_results["overall_status"] = "partial"
            else:
                test_results["overall_status"] = "failed"
            
            logger.info(f"🎯 Comprehensive test completed! Overall status: {test_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            test_results["error"] = str(e)
            test_results["overall_status"] = "error"
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results"""
        print("\n" + "="*80)
        print("🧪 LANGCHAIN WEATHER PREDICTION TEST SUMMARY")
        print("="*80)
        
        print(f"\n📅 Test Date: {results.get('timestamp', 'Unknown')}")
        print(f"🎯 Overall Status: {results.get('overall_status', 'Unknown').upper()}")
        
        # Service Availability
        print("\n🔧 SERVICE AVAILABILITY:")
        availability = results.get('service_availability', {})
        for service, available in availability.items():
            status = "✅ Available" if available else "❌ Unavailable"
            print(f"  • {service.replace('_', ' ').title()}: {status}")
        
        # LM Studio Connection
        print("\n🏠 LM STUDIO CONNECTION:")
        lm_status = results.get('lm_studio_connection', {})
        if lm_status.get('available'):
            print(f"  • Status: ✅ Connected")
            print(f"  • Model: {lm_status.get('model_name', 'Unknown')}")
        else:
            print(f"  • Status: ❌ Not connected")
            if 'error' in lm_status:
                print(f"  • Error: {lm_status['error']}")
        
        # RAG Search
        print("\n📚 RAG SEARCH TEST:")
        rag_results = results.get('rag_search', {})
        if 'error' not in rag_results:
            print(f"  • Query: '{rag_results.get('query', 'Unknown')}'")
            print(f"  • Patterns Found: {rag_results.get('patterns_found', 0)}")
            print(f"  • Status: ✅ Successful")
        else:
            print(f"  • Status: ❌ Failed")
            print(f"  • Error: {rag_results['error']}")
        
        # LangChain Prediction
        print("\n⛓️ LANGCHAIN PREDICTION TEST:")
        prediction_results = results.get('langchain_prediction', {})
        if prediction_results.get('success'):
            print(f"  • Status: ✅ Successful")
            print(f"  • Method: {prediction_results.get('method', 'Unknown')}")
            print(f"  • Confidence: {prediction_results.get('confidence_level', 'Unknown')}")
            prediction_preview = prediction_results.get('prediction', '')[:100]
            print(f"  • Preview: {prediction_preview}...")
        else:
            print(f"  • Status: ❌ Failed")
            if 'error' in prediction_results:
                print(f"  • Error: {prediction_results['error']}")
        
        # Memory Test
        print("\n🧠 MEMORY TEST:")
        memory_results = results.get('langchain_memory', {})
        if memory_results.get('memory_test') == 'successful':
            print(f"  • Status: ✅ Successful")
            print(f"  • Memory Length: {memory_results.get('memory_length', 0)} characters")
        else:
            print(f"  • Status: ❌ Failed")
            if 'error' in memory_results:
                print(f"  • Error: {memory_results['error']}")
        
        print("\n" + "="*80)
        print("🎉 Test completed! Check langchain_weather_test.log for detailed logs.")
        print("="*80)


def main():
    """Main function to run the LangChain weather prediction tests"""
    try:
        # Initialize tester
        tester = LangChainWeatherTester()
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Print summary
        tester.print_test_summary(results)
        
        # Optionally save results to file
        import json
        with open('langchain_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("📁 Test results saved to langchain_test_results.json")
        
    except Exception as e:
        logger.error(f"❌ Main test execution failed: {e}")
        print(f"\n❌ Test failed: {e}")
        print("💡 Make sure all dependencies are installed and services are running.")


if __name__ == "__main__":
    main()