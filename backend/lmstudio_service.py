"""
LM Studio Local LLM Service with Qwen3-14B Optimization
Integrates with LM Studio's OpenAI-compatible API for local AI inference
Enhanced with chain-of-thought prompting and advanced response parsing
"""

import os
import requests
import json
import re
import logging
from typing import Dict, Any, Optional, List, Literal
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Circuit breaker — imported lazily to avoid circular imports at module load
def _get_circuit_breakers():
    try:
        from backend.circuit_breaker import circuit_breakers
        return circuit_breakers
    except Exception:
        return None

logger = logging.getLogger(__name__)

class LMStudioService:
    """Service for interacting with LM Studio local LLM API"""
    
    def __init__(self):
        """Initialize LM Studio service with configuration from environment"""
        self.api_url = os.getenv('LM_STUDIO_API_URL', 'http://127.0.0.1:1234')  # Updated default port
        self.timeout = 90  # Allow enough time for Qwen3-14b to respond; agents fall back to stat on timeout
        self.available = False
        self.model_info = {}
        self.model_name = 'unknown'
        self.is_qwen3 = False
        self.optimal_params = self._get_default_params()
        
        logger.info(f"🏠 Initializing LM Studio service at {self.api_url}")
        self._check_availability()
        self._detect_model_capabilities()
    
    def _check_availability(self) -> bool:
        """Check if LM Studio server is running and model is loaded"""
        try:
            # Check if server is running
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                
                if 'data' in models_data and len(models_data['data']) > 0:
                    self.model_info = models_data['data'][0]
                    self.model_name = self.model_info.get('id', 'unknown')
                    self.available = True
                    logger.info(f"✅ LM Studio available with model: {self.model_name}")
                    return True
                else:
                    logger.warning("⚠️ LM Studio running but no model loaded")
                    return False
            else:
                logger.warning(f"⚠️ LM Studio server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ LM Studio server not reachable - ensure LM Studio is running")
            return False
        except Exception as e:
            logger.warning(f"⚠️ LM Studio availability check failed: {str(e)}")
            return False
    
    def _detect_model_capabilities(self):
        """Detect if Qwen3 is loaded and optimize parameters accordingly"""
        if not self.available:
            return
        
        model_name_lower = self.model_name.lower()
        
        # Detect Qwen3 variants
        if 'qwen3' in model_name_lower or 'qwen-3' in model_name_lower:
            self.is_qwen3 = True
            logger.info("🎯 Qwen3 model detected - applying optimizations")
            self.optimal_params = {
                'temperature': 0.3,
                'top_p': 0.9,
                'max_tokens': 800,  # Enough for 7-14 day forecasts without excessive slowness
                'frequency_penalty': 0.2,
                'presence_penalty': 0.1
            }
        elif 'qwen2.5' in model_name_lower:
            self.is_qwen3 = False
            logger.info("🎯 Qwen2.5 model detected - applying optimizations")
            self.optimal_params = {
                'temperature': 0.3,
                'top_p': 0.9,
                'max_tokens': 800,  # Enough for 7-14 day forecasts without excessive slowness
                'frequency_penalty': 0.2,
                'presence_penalty': 0.1
            }
        elif 'mistral' in model_name_lower or 'mixtral' in model_name_lower:
            logger.info("🎯 Mistral/Mixtral model detected")
            self.optimal_params = {
                'temperature': 0.3,
                'top_p': 0.95,
                'max_tokens': 1800,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.1
            }
        else:
            logger.info(f"📝 Using default parameters for model: {self.model_name}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default generation parameters"""
        return {
            'temperature': 0.3,
            'top_p': 0.9,
            'max_tokens': 1500,
            'frequency_penalty': 0.1,
            'presence_penalty': 0.1
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if not self.available:
            return {"error": "LM Studio not available"}
        
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
        except Exception as e:
            return {"error": f"Model info request failed: {str(e)}"}
    
    def strip_thinking(self, text: str) -> str:
        """Strip Qwen3 <think>...</think> CoT blocks from output"""
        import re
        # Remove complete <think>...</think> blocks (including multiline)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any stray opening/closing tags
        text = re.sub(r'</?think>', '', text)
        # Collapse excess blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clear_prompt_cache(self) -> bool:
        """Clear LM Studio KV/prompt cache to free RAM before large generation requests.
        This prevents stalls caused by cache pressure (e.g. 2021/2172 MiB used),
        which cause Qwen3 generation to freeze mid-output on 7-14 day forecasts.
        LM Studio >= 0.3.5 supports POST /slots/{id}/action {"action":"erase"}.
        """
        try:
            cleared = 0
            for slot_id in range(4):
                resp = requests.post(
                    f"{self.api_url}/slots/{slot_id}/action",
                    json={"action": "erase"},
                    timeout=5
                )
                if resp.status_code in (200, 204):
                    cleared += 1
            logger.info(f"✅ LM Studio prompt cache cleared ({cleared} slots erased)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Cache clear skipped (non-fatal): {e}")
            return False

    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> Optional[str]:
        """Generate text using the local LLM with simple completion"""
        if not self.available:
            logger.error("❌ LM Studio not available for text generation")
            return None
        
        try:
            payload = {
                "model": self.model_info.get('id', 'local-model'),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "stop": ["\n\n", "User:", "Human:"]
            }
            
            logger.info("🤖 Calling LM Studio for text completion...")
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text'].strip()
                logger.info("✅ LM Studio text generation successful")
                return generated_text
            else:
                logger.error(f"❌ LM Studio completion error: {response.status_code} - {response.text}")
                cb = _get_circuit_breakers()
                if cb:
                    cb.lm_studio.record_failure(f"generate_text_http_{response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ LM Studio text generation failed: {str(e)}")
            cb = _get_circuit_breakers()
            if cb:
                cb.lm_studio.record_failure(f"generate_text_exception: {type(e).__name__}")
            return None
    
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.3, request_timeout: int = None, clear_cache: bool = False) -> Optional[str]:
        """Generate response using chat completion format.
        request_timeout overrides self.timeout for this call.
        clear_cache=True erases LM Studio's KV cache first — use for 7+ day forecasts
        to prevent cache-pressure stalls that cause generation to freeze mid-output.
        """
        if not self.available:
            logger.error("❌ LM Studio not available for chat generation")
            return None
        if clear_cache:
            self.clear_prompt_cache()
        _timeout = request_timeout if request_timeout is not None else self.timeout
        
        try:
            # Inject /no_think into system messages to suppress Qwen3 CoT reasoning output
            processed_messages = []
            for msg in messages:
                if msg.get('role') == 'system':
                    content = msg['content']
                    if '/no_think' not in content:
                        content = content + "\n/no_think"
                    processed_messages.append({**msg, 'content': content})
                else:
                    processed_messages.append(msg)
            # If no system message exists, prepend one with /no_think
            if not any(m.get('role') == 'system' for m in processed_messages):
                processed_messages.insert(0, {"role": "system", "content": "/no_think"})

            payload = {
                "model": self.model_info.get('id', 'local-model'),
                "messages": processed_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            
            logger.info("💬 Calling LM Studio for chat completion...")
            logger.info(f"🔧 Using timeout: {_timeout}s, max_tokens: {max_tokens}")
            
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['message']['content']
                    generated_text = self.strip_thinking(generated_text)
                    logger.info("✅ LM Studio chat generation successful")
                    logger.info(f"📝 Generated {len(generated_text)} characters")
                    return generated_text
                else:
                    logger.error("❌ LM Studio response missing choices")
                    return None
            else:
                logger.error(f"❌ LM Studio chat error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ LM Studio timeout after {_timeout}s - prediction may be too complex")
            cb = _get_circuit_breakers()
            if cb:
                cb.lm_studio.record_failure("generate_chat_timeout")
            return "⚠️ Prediction generation timed out. The model was processing but took longer than expected. Try with a shorter timeframe or simpler request."
        except requests.exceptions.ConnectionError:
            logger.error("❌ LM Studio connection failed - check if LM Studio is running")
            cb = _get_circuit_breakers()
            if cb:
                cb.lm_studio.record_failure("generate_chat_connection_error")
            return None
        except Exception as e:
            logger.error(f"❌ LM Studio chat generation failed: {str(e)}")
            cb = _get_circuit_breakers()
            if cb:
                cb.lm_studio.record_failure(f"generate_chat_exception: {type(e).__name__}")
            return None
    
    def generate_weather_prediction(
        self, 
        prompt: str,
        method: Literal['basic', 'advanced', 'chain_of_thought'] = 'advanced'
    ) -> Optional[str]:
        """Generate weather prediction using optimized parameters
        
        Args:
            prompt: Base prediction request
            method: 'basic' (simple), 'advanced' (structured), 'chain_of_thought' (step-by-step reasoning)
        
        Returns:
            Generated prediction text or None if failed
        """
        if method == 'chain_of_thought':
            return self._predict_with_cot(prompt)
        elif method == 'advanced':
            return self._predict_advanced(prompt)
        else:
            return self._predict_basic(prompt)
    
    def _predict_basic(self, prompt: str) -> Optional[str]:
        """Basic prediction with simple prompt"""
        messages = [
            {
                "role": "system", 
                "content": "You are an expert meteorologist and weather prediction specialist. Provide accurate, detailed weather forecasts based on the provided historical data and current conditions. Format your predictions clearly with daily breakdowns and confidence levels."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        return self.generate_chat(
            messages=messages,
            max_tokens=self.optimal_params['max_tokens'],
            temperature=0.2  # Lower temperature for consistent predictions
        )
    
    def _predict_advanced(self, prompt: str) -> Optional[str]:
        """Advanced prediction with structured output format"""
        structured_prompt = self._create_structured_prompt(prompt)
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": structured_prompt
            }
        ]
        
        return self.generate_chat(
            messages=messages,
            max_tokens=self.optimal_params['max_tokens'],
            temperature=self.optimal_params['temperature']
        )
    
    def _predict_with_cot(self, prompt: str) -> Optional[str]:
        """Chain-of-thought prediction with explicit reasoning steps"""
        cot_prompt = self._create_cot_prompt(prompt)
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": cot_prompt
            }
        ]
        
        logger.info("🧠 Using chain-of-thought prompting for enhanced reasoning")
        
        return self.generate_chat(
            messages=messages,
            max_tokens=self.optimal_params['max_tokens'],
            temperature=self.optimal_params['temperature']
        )
    
    def _get_system_prompt(self) -> str:
        """Get optimized system prompt for weather predictions"""
        return """You are an expert meteorological AI assistant specializing in weather prediction and analysis.

Your capabilities:
- Analyze historical weather patterns and trends
- Identify seasonal and cyclical patterns
- Generate accurate short-term weather forecasts
- Provide confidence assessments based on data quality
- Explain reasoning clearly and scientifically

Guidelines:
- Always use scientific reasoning based on atmospheric principles
- Consider geographical and seasonal factors
- Be honest about uncertainty and limitations
- Provide actionable insights and clear explanations
- Format responses in clear JSON when requested for structured data
- Use Celsius for temperature, percentage for humidity, km/h for wind speed"""
    
    def _create_structured_prompt(self, base_prompt: str) -> str:
        """Create structured prompt for consistent JSON output"""
        return f"""{base_prompt}

**Required Output Format (JSON):**
```json
{{
  "location": "location name",
  "forecast_days": number_of_days,
  "predictions": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "temperature_c": float,
      "humidity_percent": float,
      "wind_speed_kmh": float,
      "conditions": "description",
      "precipitation_chance": float,
      "confidence": "high|medium|low"
    }}
  ],
  "analysis": "Brief analysis of weather patterns and reasoning",
  "confidence_level": "high|medium|low",
  "factors": ["key factor 1", "key factor 2"]
}}
```

Provide the forecast in this exact JSON format."""
    
    def _create_cot_prompt(self, base_prompt: str) -> str:
        """Create chain-of-thought prompt for explicit reasoning"""
        return f"""{base_prompt}

**Think step-by-step and show your reasoning:**

1. **Analyze Historical Context**: Review the provided historical weather patterns and identify key trends
2. **Consider Seasonal Factors**: Determine what season and typical weather patterns apply
3. **Identify Patterns**: Look for cyclical patterns, trends, or anomalies in the data
4. **Apply Meteorological Principles**: Use atmospheric science to understand likely weather evolution
5. **Make Day-by-Day Predictions**: Predict weather for each requested day with reasoning
6. **Assess Confidence**: Evaluate prediction reliability based on data quality and pattern consistency

**Format your response as:**
```json
{{
  "reasoning": "Your step-by-step analysis and thought process here",
  "forecast": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "temperature_c": float,
      "humidity_percent": float,
      "conditions": "description",
      "confidence": "high|medium|low",
      "reasoning": "Why this prediction for this day"
    }}
  ],
  "overall_confidence": "high|medium|low",
  "key_factors": ["factor 1", "factor 2"],
  "uncertainties": ["uncertainty 1", "uncertainty 2"]
}}
```

Think carefully and show your work."""
    
    def parse_prediction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse prediction response and extract structured data
        
        Args:
            response_text: Raw text response from LLM
        
        Returns:
            Parsed structured data with fallback handling
        """
        if not response_text:
            return {
                'success': False,
                'error': 'Empty response from LLM',
                'raw_text': ''
            }
        
        # Try to extract JSON from response
        parsed_json = self._extract_json(response_text)
        
        if parsed_json:
            logger.info("✅ Successfully parsed structured JSON from response")
            return {
                'success': True,
                'structured': True,
                'data': parsed_json,
                'raw_text': response_text,
                'confidence': self._extract_confidence(parsed_json)
            }
        else:
            logger.warning("⚠️ Could not parse JSON, using text extraction")
            # Fallback to text parsing
            extracted_data = self._extract_from_text(response_text)
            return {
                'success': True,
                'structured': False,
                'data': extracted_data,
                'raw_text': response_text,
                'confidence': self._extract_confidence_from_text(response_text)
            }
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text using multiple strategies"""
        # Strategy 1: Look for JSON code block
        json_block_pattern = r'```json\s*({.*?})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for any JSON object
        json_pattern = r'({\s*"[^"]+"\s*:.*})'  
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Find the complete JSON object
                obj = self._extract_complete_json(text, match)
                if obj:
                    return json.loads(obj)
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Try parsing entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_complete_json(self, text: str, start_match: str) -> Optional[str]:
        """Extract complete JSON object handling nested braces"""
        try:
            start_idx = text.index(start_match)
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start_idx:i+1]
            
            return None
        except (ValueError, IndexError):
            return None
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract weather data from plain text using pattern matching"""
        extracted = {
            'predictions': [],
            'analysis': '',
            'source': 'text_extraction'
        }
        
        # Extract temperature values
        temp_pattern = r'(\d+\.?\d*)\s*(?:°C|celsius|degrees)'
        temps = re.findall(temp_pattern, text, re.IGNORECASE)
        
        # Extract humidity values  
        humidity_pattern = r'(\d+\.?\d*)\s*%?\s*humidity'
        humidities = re.findall(humidity_pattern, text, re.IGNORECASE)
        
        # Extract day predictions
        day_pattern = r'day\s*(\d+)'
        days = re.findall(day_pattern, text, re.IGNORECASE)
        
        # Build predictions from extracted data
        for i, day in enumerate(days):
            prediction = {'day': int(day)}
            if i < len(temps):
                prediction['temperature_c'] = float(temps[i])
            if i < len(humidities):
                prediction['humidity_percent'] = float(humidities[i])
            extracted['predictions'].append(prediction)
        
        # Extract first substantial paragraph as analysis
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
        if paragraphs:
            extracted['analysis'] = paragraphs[0][:500]
        
        return extracted
    
    def _extract_confidence(self, data: Dict[str, Any]) -> str:
        """Extract confidence level from structured data"""
        # Check various possible confidence keys
        for key in ['confidence_level', 'overall_confidence', 'confidence']:
            if key in data:
                value = str(data[key]).lower()
                if value in ['high', 'medium', 'low']:
                    return value
        
        return 'medium'  # Default
    
    def _extract_confidence_from_text(self, text: str) -> str:
        """Extract confidence from plain text"""
        text_lower = text.lower()
        
        if 'high confidence' in text_lower or 'very confident' in text_lower:
            return 'high'
        elif 'low confidence' in text_lower or 'uncertain' in text_lower:
            return 'low'
        else:
            return 'medium'
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return detailed status"""
        logger.info("🔍 Testing LM Studio connection...")
        
        # Refresh availability
        self._check_availability()
        
        status = {
            "available": self.available,
            "api_url": self.api_url,
            "timestamp": str(logger.info("Testing at this time")),
        }
        
        if self.available:
            status.update({
                "status": "✅ Connected",
                "model": self.model_info.get('id', 'unknown'),
                "model_info": self.model_info
            })
            
            # Test with a simple generation
            try:
                test_response = self.generate_text("Hello", max_tokens=10)
                status["test_generation"] = "✅ Success" if test_response else "❌ Failed"
                if test_response:
                    status["test_response_length"] = len(test_response)
            except Exception as e:
                status["test_generation"] = f"❌ Error: {str(e)}"
        else:
            status["status"] = "❌ Not available"
            status["help"] = [
                "1. Ensure LM Studio is running",
                "2. Load a model in LM Studio",
                "3. Start the local server in LM Studio",
                f"4. Verify server is running on {self.api_url}"
            ]
        
        return status
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics and health info"""
        stats = {
            "service": "LM Studio Local LLM (Qwen3-Optimized)",
            "api_url": self.api_url,
            "available": self.available,
            "timeout": self.timeout,
            "model_name": self.model_name,
            "is_qwen3": self.is_qwen3,
            "optimal_params": self.optimal_params,
            "features": {
                "chain_of_thought": True,
                "structured_output": True,
                "json_parsing": True,
                "model_detection": True
            }
        }
        
        if self.available and self.model_info:
            stats.update({
                "model_id": self.model_info.get('id'),
                "model_object": self.model_info.get('object'),
                "model_owned_by": self.model_info.get('owned_by', 'local'),
            })
        
        return stats

# Global instance for easy access
lm_studio_service = None

def get_lm_studio_service() -> Optional[LMStudioService]:
    """Get the global LM Studio service instance"""
    global lm_studio_service
    
    if lm_studio_service is None:
        try:
            lm_studio_service = LMStudioService()
            if lm_studio_service.available:
                logger.info("🏠 LM Studio service initialized successfully")
            else:
                logger.warning("⚠️ LM Studio service initialized but not available")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio service: {e}")
            lm_studio_service = None
    
    return lm_studio_service

# Test function for development
def test_lm_studio():
    """Test function to verify LM Studio integration"""
    service = get_lm_studio_service()
    
    if not service:
        print("❌ LM Studio service not available")
        return
    
    print(f"🔍 Testing LM Studio at {service.api_url}")
    status = service.test_connection()
    
    print("\n📊 Connection Status:")
    for key, value in status.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    {item}")
        else:
            print(f"  {key}: {value}")
    
    # Test new features if connected
    if service.available:
        print("\n🧪 Testing New Features:")
        print(f"  ✅ Model Detection: {service.model_name}")
        print(f"  ✅ Qwen3 Optimized: {service.is_qwen3}")
        print(f"  ✅ Optimal Params: {service.optimal_params}")
        
        # Test chain-of-thought prediction
        print("\n🧠 Testing Chain-of-Thought Prediction...")
        test_prompt = "Predict weather for Tokyo for 2 days based on current sunny conditions."
        
        try:
            result = service.generate_weather_prediction(test_prompt, method='chain_of_thought')
            if result:
                print("  ✅ Chain-of-thought generation successful")
                parsed = service.parse_prediction_response(result)
                print(f"  📊 Parsed successfully: {parsed.get('structured', False)}")
                print(f"  🎯 Confidence: {parsed.get('confidence', 'unknown')}")
            else:
                print("  ⚠️ No response generated")
        except Exception as e:
            print(f"  ❌ Test failed: {e}")

if __name__ == "__main__":
    # Run test when executed directly
    test_lm_studio()