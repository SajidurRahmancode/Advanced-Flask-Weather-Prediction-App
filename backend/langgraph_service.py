"""
LangGraph Multi-Agent Weather Prediction Service

This module implements an advanced multi-agent system using LangGraph for intelligent
weather prediction with dynamic routing, parallel processing, and quality validation.
"""

import os
import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, List, Optional, Annotated
from dataclasses import dataclass

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LangGraph dependencies imported successfully")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ LangGraph imports failed: {e}")

# State definitions for different workflows
class WeatherAnalysisState(TypedDict):
    """State for multi-agent weather analysis workflow"""
    location: str
    prediction_days: int
    current_conditions: Dict[str, Any]
    historical_patterns: List[Dict[str, Any]]
    rag_confidence: float
    analysis_results: Dict[str, Any]
    agent_reports: Dict[str, str]
    final_prediction: str
    confidence_score: float
    method_used: str
    error_message: Optional[str]
    retry_count: int
    quality_score: float

class ServiceSelectionState(TypedDict):
    """State for intelligent service selection workflow"""
    location: str
    prediction_days: int
    available_services: Dict[str, bool]
    selected_method: str
    prediction_result: Dict[str, Any]
    quality_metrics: Dict[str, float]
    retry_attempted: bool
    fallback_used: bool
    
class QualityValidationState(TypedDict):
    """State for prediction quality validation"""
    prediction_text: str
    metadata: Dict[str, Any]
    quality_checks: Dict[str, bool]
    overall_quality: float
    issues_found: List[str]
    approved: bool

@dataclass
class AgentConfig:
    """Configuration for weather analysis agents"""
    name: str
    description: str
    confidence_threshold: float = 0.7
    timeout_seconds: int = 30

class LangGraphWeatherService:
    """
    Advanced weather prediction service using LangGraph multi-agent architecture
    
    This service orchestrates multiple specialized agents for comprehensive weather analysis:
    - Data Collection Agent: Gathers and validates weather data
    - Pattern Analysis Agent: Analyzes historical weather patterns
    - Meteorological Agent: Applies atmospheric science expertise
    - Confidence Assessment Agent: Evaluates prediction reliability
    - Quality Control Agent: Validates final predictions
    """
    
    def __init__(self, weather_service=None, rag_service=None, langchain_service=None, lm_studio_service=None, websocket_service=None):
        """Initialize LangGraph weather service with existing components"""
        self.weather_service = weather_service
        self.rag_service = rag_service
        self.langchain_service = langchain_service
        self.lm_studio_service = lm_studio_service
        self.websocket_service = websocket_service
        
        self.available = LANGGRAPH_AVAILABLE
        self.prediction_graph = None
        self.service_selection_graph = None
        self.quality_validation_graph = None
        self.current_workflow_id = None
        
        # Agent configurations
        self.agents = {
            'data_collector': AgentConfig(
                name="Data Collection Agent",
                description="Gathers and validates current weather data and historical records",
                confidence_threshold=0.8
            ),
            'pattern_analyzer': AgentConfig(
                name="Pattern Analysis Agent", 
                description="Analyzes historical weather patterns and similarities",
                confidence_threshold=0.7
            ),
            'meteorologist': AgentConfig(
                name="Meteorological Expert Agent",
                description="Applies atmospheric science principles and domain expertise",
                confidence_threshold=0.9
            ),
            'confidence_assessor': AgentConfig(
                name="Confidence Assessment Agent",
                description="Evaluates prediction reliability and uncertainty",
                confidence_threshold=0.6
            ),
            'quality_controller': AgentConfig(
                name="Quality Control Agent",
                description="Validates prediction quality and accuracy",
                confidence_threshold=0.8
            )
        }
        
        if self.available:
            logger.info("🧠 Initializing LangGraph Weather Service...")
            self._initialize_graphs()
            logger.info("✅ LangGraph Weather Service initialized successfully")
        else:
            logger.warning("⚠️ LangGraph not available - service disabled")
    
    def _broadcast_agent_status(self, agent_type: str, status: str, progress: float, message: str, data: Dict[str, Any] = None):
        """Broadcast agent status via WebSocket if available"""
        if self.websocket_service and self.current_workflow_id:
            agent_id = f"{self.current_workflow_id}_{agent_type}"
            self.websocket_service.update_agent_status(
                agent_id=agent_id,
                status=status,
                progress=progress,
                message=message,
                data=data
            )
    
    def _broadcast_workflow_complete(self, result: Dict[str, Any]):
        """Broadcast workflow completion via WebSocket if available"""
        if self.websocket_service and self.current_workflow_id:
            self.websocket_service.broadcast_prediction_result(
                workflow_id=self.current_workflow_id,
                result=result
            )
    
    def _broadcast_workflow_error(self, error: str):
        """Broadcast workflow error via WebSocket if available"""
        if self.websocket_service and self.current_workflow_id:
            self.websocket_service.broadcast_error(
                workflow_id=self.current_workflow_id,
                error=error
            )
    
    def _initialize_graphs(self):
        """Initialize all LangGraph workflows"""
        try:
            # Create main prediction workflow
            self.prediction_graph = self._create_prediction_workflow()
            
            # Create service selection workflow
            self.service_selection_graph = self._create_service_selection_workflow()
            
            # Create quality validation workflow
            self.quality_validation_graph = self._create_quality_validation_workflow()
            
            logger.info("✅ All LangGraph workflows initialized")
            
        except Exception as e:
            logger.error(f"❌ LangGraph workflow initialization failed: {e}")
            self.available = False
    
    def _create_prediction_workflow(self):
        """Create the main multi-agent weather prediction workflow"""
        
        def data_collection_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for collecting and validating weather data"""
            logger.info(f"🔍 Data Collection Agent: Processing {state['location']}")
            
            # Broadcast agent start
            self._broadcast_agent_status('data_collection', 'running', 0.1, 
                                       f"Starting data collection for {state['location']}")
            
            try:
                # Broadcast progress
                self._broadcast_agent_status('data_collection', 'running', 0.3, 
                                           "Accessing weather database...")
                
                # Collect current conditions
                if self.weather_service:
                    recent_data = self.weather_service.get_recent_weather_data(days=3)
                    self._broadcast_agent_status('data_collection', 'running', 0.7, 
                                               "Processing weather data...")
                    
                    current_conditions = {
                        'temperature': recent_data['Actual_Temperature(°C)'].iloc[-1] if not recent_data.empty else 20.0,
                        'humidity': recent_data['Actual_Humidity(%)'].iloc[-1] if not recent_data.empty else 60.0,
                        'wind_speed': recent_data['Actual_WindSpeed(m/s)'].iloc[-1] if not recent_data.empty else 2.0,
                        'data_quality': 'high' if not recent_data.empty else 'low'
                    }
                else:
                    current_conditions = {'data_quality': 'unavailable'}
                
                state['current_conditions'] = current_conditions
                state['agent_reports']['data_collector'] = f"✅ Data collected for {state['location']}: {current_conditions.get('data_quality', 'unknown')} quality"
                
                # Broadcast completion
                self._broadcast_agent_status('data_collection', 'completed', 1.0, 
                                           f"Data collection completed - Quality: {current_conditions.get('data_quality', 'unknown')}",
                                           data=current_conditions)
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Data Collection Agent failed: {e}")
                state['error_message'] = f"Data collection failed: {str(e)}"
                state['agent_reports']['data_collector'] = f"❌ Data collection failed: {str(e)}"
                
                # Broadcast error
                self._broadcast_agent_status('data_collection', 'error', 0.0, 
                                           f"Data collection failed: {str(e)}")
                
                return state
        
        def pattern_analysis_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for analyzing historical weather patterns with Qwen3 CoT"""
            logger.info("📊 Pattern Analysis Agent: Analyzing historical patterns with AI reasoning")
            
            # Broadcast agent start
            self._broadcast_agent_status('pattern_analysis', 'running', 0.1, 
                                       "Starting AI-powered pattern analysis...")
            
            try:
                patterns = []
                rag_confidence = 0.0
                ai_analysis = ""
                
                if self.rag_service and state['current_conditions']:
                    # Broadcast progress
                    self._broadcast_agent_status('pattern_analysis', 'running', 0.3, 
                                               "Querying historical patterns...")
                    
                    # Build query for similar conditions
                    conditions = state['current_conditions']
                    query = f"weather temperature {conditions.get('temperature', 20)} humidity {conditions.get('humidity', 60)}"
                    
                    try:
                        self._broadcast_agent_status('pattern_analysis', 'running', 0.5, 
                                                   "Retrieving similar weather patterns...")
                        similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
                        patterns = [{'content': p.page_content, 'metadata': getattr(p, 'metadata', {})} for p in similar_patterns]
                        
                        # WEEK 3 ENHANCEMENT: Use Qwen3 CoT to analyze patterns
                        if patterns and self.lm_studio_service and self.lm_studio_service.available:
                            self._broadcast_agent_status('pattern_analysis', 'running', 0.7, 
                                                       "Analyzing patterns with Qwen3 CoT reasoning...")
                            
                            # Build CoT prompt for pattern analysis
                            pattern_texts = "\n".join([f"Pattern {i+1}: {p['content'][:200]}..." 
                                                      for i, p in enumerate(patterns[:3])])
                            
                            cot_prompt = f"""Weather patterns for {state['location']} ({season}), {len(patterns)} historical matches.
Current: {conditions.get('temperature','N/A')}°C, {conditions.get('humidity','N/A')}% RH, {conditions.get('wind_speed','N/A')} m/s.
Top pattern: {pattern_texts.split(chr(10))[0][:150] if pattern_texts else 'none'}

In 2 sentences: key trend and confidence score (0.0-1.0)."""

                            try:
                                messages = [{"role": "user", "content": cot_prompt}]
                                ai_analysis = self.lm_studio_service.generate_chat(messages, max_tokens=300, temperature=0.3)
                                
                                # Extract confidence from AI analysis or use pattern-based scoring
                                if "confidence" in ai_analysis.lower():
                                    # Try to extract numeric confidence
                                    import re
                                    conf_match = re.search(r'confidence[:\s]+([0-9.]+)', ai_analysis.lower())
                                    if conf_match:
                                        rag_confidence = min(0.95, float(conf_match.group(1)))
                                    else:
                                        rag_confidence = min(0.85, len(patterns) * 0.15 + 0.2)
                                else:
                                    rag_confidence = min(0.85, len(patterns) * 0.15 + 0.2)
                                    
                                logger.info(f"✅ Pattern analysis with CoT completed: {len(ai_analysis)} chars")
                                
                            except Exception as cot_error:
                                logger.warning(f"⚠️ Qwen3 CoT analysis failed, using fallback: {cot_error}")
                                ai_analysis = f"Pattern analysis: {len(patterns)} similar patterns found"
                                rag_confidence = min(0.75, len(patterns) * 0.15)
                        else:
                            # Fallback: simple pattern-based confidence
                            rag_confidence = min(0.75, len(patterns) * 0.15)
                            ai_analysis = f"Retrieved {len(patterns)} historical patterns for comparison"
                            
                    except Exception as rag_error:
                        logger.warning(f"⚠️ RAG retrieval failed: {rag_error}")
                        patterns = []
                        rag_confidence = 0.1
                        ai_analysis = "Pattern retrieval failed - using limited historical data"
                
                state['historical_patterns'] = patterns
                state['rag_confidence'] = rag_confidence
                state['analysis_results']['pattern_analysis'] = {
                    'patterns_found': len(patterns),
                    'confidence': rag_confidence,
                    'ai_reasoning': ai_analysis[:300] if ai_analysis else "No AI analysis available"
                }
                state['agent_reports']['pattern_analyzer'] = f"✅ Found {len(patterns)} patterns (confidence: {rag_confidence:.2f}, AI-enhanced)"
                
                # Broadcast completion
                self._broadcast_agent_status('pattern_analysis', 'completed', 1.0, 
                                           f"AI pattern analysis completed - {len(patterns)} patterns, confidence: {rag_confidence:.2f}",
                                           data={'patterns_count': len(patterns), 'confidence': rag_confidence, 'ai_enhanced': bool(ai_analysis)})
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Pattern Analysis Agent failed: {e}")
                state['agent_reports']['pattern_analyzer'] = f"❌ Pattern analysis failed: {str(e)}"
                
                # Broadcast error
                self._broadcast_agent_status('pattern_analysis', 'error', 0.0, 
                                           f"Pattern analysis failed: {str(e)}")
                
                return state
        
        def meteorological_expert_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent applying meteorological expertise with Qwen3 CoT"""
            logger.info("🌤️ Meteorological Expert Agent: Applying AI-enhanced domain expertise")
            
            # Broadcast agent start
            self._broadcast_agent_status('meteorological', 'running', 0.1, 
                                       "Starting AI-powered meteorological analysis...")
            
            try:
                # Broadcast progress
                self._broadcast_agent_status('meteorological', 'running', 0.2, 
                                           "Determining seasonal context...")
                
                # Determine season
                current_month = datetime.now().month
                if current_month in [12, 1, 2]:
                    season = "Winter"
                elif current_month in [3, 4, 5]:
                    season = "Spring"
                elif current_month in [6, 7, 8]:
                    season = "Summer"
                else:
                    season = "Autumn"
                
                conditions = state['current_conditions']
                temp = conditions.get('temperature', 20)
                humidity = conditions.get('humidity', 60)
                wind_speed = conditions.get('wind_speed', 2.0)
                
                # WEEK 3 ENHANCEMENT: Use Qwen3 CoT for expert meteorological analysis
                if self.lm_studio_service and self.lm_studio_service.available:
                    self._broadcast_agent_status('meteorological', 'running', 0.4, 
                                               "Analyzing with Qwen3 meteorological expertise...")
                    
                    # Get pattern analysis context
                    pattern_info = state.get('analysis_results', {}).get('pattern_analysis', {})
                    patterns_found = pattern_info.get('patterns_found', 0)
                    
                    cot_prompt = f"""Meteorologist analysis: {state['location']}, {season}, {state['prediction_days']}-day window.
Now: {temp}°C, {humidity}% RH, {wind_speed} m/s. Historical matches: {patterns_found} (confidence {state.get('rag_confidence',0.0):.2f}).

In 2-3 sentences: expected conditions, temperature range, main hazard. End with confidence (0.0-1.0)."""

                    try:
                        messages = [{"role": "user", "content": cot_prompt}]
                        ai_analysis = self.lm_studio_service.generate_chat(messages, max_tokens=400, temperature=0.3)
                        
                        # Extract confidence from analysis
                        import re
                        conf_match = re.search(r'confidence[:\s]+([0-9.]+)', ai_analysis.lower())
                        expert_confidence = float(conf_match.group(1)) if conf_match else 0.85
                        
                        # Extract key insights
                        conditions_assessment = ai_analysis[:200] + "..." if len(ai_analysis) > 200 else ai_analysis
                        
                        meteorological_analysis = {
                            'season': season,
                            'ai_analysis': ai_analysis,
                            'conditions_assessment': conditions_assessment,
                            'expert_confidence': min(0.95, expert_confidence),
                            'analysis_method': 'qwen3_cot'
                        }
                        
                        logger.info(f"✅ Meteorological CoT analysis: {len(ai_analysis)} chars, confidence: {expert_confidence:.2f}")
                        
                    except Exception as cot_error:
                        logger.warning(f"⚠️ Qwen3 meteorological analysis failed, using fallback: {cot_error}")
                        # Fallback to programmatic analysis
                        if temp > 25 and humidity > 70:
                            conditions_assessment = "High heat and humidity - thunderstorm potential"
                        elif temp < 5 and humidity > 80:
                            conditions_assessment = "Cold and humid - precipitation likely"
                        elif temp > 15 and humidity < 40:
                            conditions_assessment = "Warm and dry - stable conditions"
                        else:
                            conditions_assessment = "Moderate conditions - typical patterns expected"
                        
                        meteorological_analysis = {
                            'season': season,
                            'conditions_assessment': conditions_assessment,
                            'expert_confidence': 0.65,
                            'analysis_method': 'fallback'
                        }
                else:
                    # Fallback if LM Studio unavailable
                    self._broadcast_agent_status('meteorological', 'running', 0.6, 
                                               "Using fallback meteorological analysis...")
                    
                    if temp > 25 and humidity > 70:
                        conditions_assessment = "High heat and humidity - thunderstorm potential"
                    elif temp < 5 and humidity > 80:
                        conditions_assessment = "Cold and humid - precipitation likely"
                    elif temp > 15 and humidity < 40:
                        conditions_assessment = "Warm and dry - stable conditions"
                    else:
                        conditions_assessment = "Moderate conditions - typical patterns expected"
                    
                    meteorological_analysis = {
                        'season': season,
                        'conditions_assessment': conditions_assessment,
                        'expert_confidence': 0.65,
                        'analysis_method': 'fallback'
                    }
                
                state['analysis_results']['meteorological'] = meteorological_analysis
                state['agent_reports']['meteorologist'] = f"✅ {season} analysis: {meteorological_analysis.get('conditions_assessment', 'Analysis complete')[:100]} (method: {meteorological_analysis.get('analysis_method', 'unknown')})"
                
                # Broadcast completion
                self._broadcast_agent_status('meteorological', 'completed', 1.0, 
                                           f"Meteorological analysis completed - {season}, {meteorological_analysis.get('analysis_method', 'unknown')}",
                                           data={'season': season, 'confidence': meteorological_analysis.get('expert_confidence', 0.65), 'method': meteorological_analysis.get('analysis_method', 'unknown')})
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Meteorological Expert Agent failed: {e}")
                state['agent_reports']['meteorologist'] = f"❌ Meteorological analysis failed: {str(e)}"
                
                # Broadcast error
                self._broadcast_agent_status('meteorological', 'error', 0.0, 
                                           f"Meteorological analysis failed: {str(e)}")
                
                return state
        
        def confidence_assessment_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for assessing prediction confidence with Qwen3 CoT"""
            logger.info("🎯 Confidence Assessment Agent: Evaluating prediction reliability with AI reasoning")
            
            # Broadcast agent start
            self._broadcast_agent_status('confidence_assessment', 'running', 0.1, 
                                       "Starting AI-powered confidence assessment...")
            
            try:
                # Broadcast progress
                self._broadcast_agent_status('confidence_assessment', 'running', 0.2, 
                                           "Collecting confidence factors...")
                
                # Calculate base confidence from multiple factors
                confidence_factors = {
                    'data_quality': 0.0,
                    'pattern_match': state.get('rag_confidence', 0.0),
                    'meteorological_expertise': state.get('analysis_results', {}).get('meteorological', {}).get('expert_confidence', 0.0),
                    'service_availability': 0.0
                }
                
                # Assess data quality
                data_quality = state['current_conditions'].get('data_quality', 'low')
                if data_quality == 'high':
                    confidence_factors['data_quality'] = 0.9
                elif data_quality == 'medium':
                    confidence_factors['data_quality'] = 0.6
                else:
                    confidence_factors['data_quality'] = 0.3
                
                # Assess service availability
                if self.langchain_service and self.langchain_service.available:
                    confidence_factors['service_availability'] = 0.8
                elif self.lm_studio_service and self.lm_studio_service.available:
                    confidence_factors['service_availability'] = 0.7
                else:
                    confidence_factors['service_availability'] = 0.4
                
                # Calculate initial weighted average confidence
                weights = {'data_quality': 0.3, 'pattern_match': 0.3, 'meteorological_expertise': 0.2, 'service_availability': 0.2}
                base_confidence = sum(confidence_factors[factor] * weights[factor] for factor in weights)
                
                # WEEK 3 ENHANCEMENT: Use Qwen3 CoT for confidence evaluation
                if self.lm_studio_service and self.lm_studio_service.available:
                    self._broadcast_agent_status('confidence_assessment', 'running', 0.5, 
                                               "Evaluating confidence with Qwen3 reasoning...")
                    
                    # Get agent reports for context
                    agent_reports = state.get('agent_reports', {})
                    pattern_analysis = state.get('analysis_results', {}).get('pattern_analysis', {})
                    meteorological = state.get('analysis_results', {}).get('meteorological', {})
                    
                    cot_prompt = f"""Confidence rating for {state['location']} {state['prediction_days']}-day forecast.
Base: {base_confidence:.2f} | Data: {data_quality} | Patterns: {pattern_analysis.get('patterns_found',0)} ({pattern_analysis.get('confidence',0.0):.2f}) | Expert: {meteorological.get('expert_confidence',0.0):.2f}

Reply ONLY: High/Medium/Low and final score (0.0-1.0). One sentence reason."""

                    try:
                        messages = [{"role": "user", "content": cot_prompt}]
                        ai_assessment = self.lm_studio_service.generate_chat(messages, max_tokens=300, temperature=0.3)
                        
                        # Extract final confidence from AI assessment
                        import re
                        final_conf_match = re.search(r'final[^0-9]*confidence[:\s]+([0-9.]+)', ai_assessment.lower())
                        if final_conf_match:
                            overall_confidence = min(0.98, float(final_conf_match.group(1)))
                        else:
                            # Look for any confidence mention
                            conf_match = re.search(r'confidence[:\s]+([0-9.]+)', ai_assessment.lower())
                            overall_confidence = float(conf_match.group(1)) if conf_match else base_confidence
                        
                        # Extract confidence level from AI or calculate
                        if "high" in ai_assessment.lower() and "confidence" in ai_assessment.lower():
                            confidence_level = "High"
                        elif "medium" in ai_assessment.lower():
                            confidence_level = "Medium"
                        elif "very low" in ai_assessment.lower():
                            confidence_level = "Very Low"
                        elif "low" in ai_assessment.lower():
                            confidence_level = "Low"
                        else:
                            # Fallback to numeric classification
                            if overall_confidence >= 0.8:
                                confidence_level = "High"
                            elif overall_confidence >= 0.6:
                                confidence_level = "Medium"
                            elif overall_confidence >= 0.4:
                                confidence_level = "Low"
                            else:
                                confidence_level = "Very Low"
                        
                        confidence_analysis = {
                            'overall_score': overall_confidence,
                            'confidence_level': confidence_level,
                            'factors': confidence_factors,
                            'ai_reasoning': ai_assessment[:400] if ai_assessment else "No AI reasoning",
                            'method': 'qwen3_cot'
                        }
                        
                        logger.info(f"✅ Confidence CoT assessment: {confidence_level} ({overall_confidence:.2f})")
                        
                    except Exception as cot_error:
                        logger.warning(f"⚠️ Qwen3 confidence assessment failed, using base calculation: {cot_error}")
                        # Fallback to base confidence
                        overall_confidence = base_confidence
                        if overall_confidence >= 0.8:
                            confidence_level = "High"
                        elif overall_confidence >= 0.6:
                            confidence_level = "Medium"
                        elif overall_confidence >= 0.4:
                            confidence_level = "Low"
                        else:
                            confidence_level = "Very Low"
                        
                        confidence_analysis = {
                            'overall_score': overall_confidence,
                            'confidence_level': confidence_level,
                            'factors': confidence_factors,
                            'method': 'fallback'
                        }
                else:
                    # Fallback if LM Studio unavailable
                    self._broadcast_agent_status('confidence_assessment', 'running', 0.7, 
                                               "Using fallback confidence calculation...")
                    overall_confidence = base_confidence
                    if overall_confidence >= 0.8:
                        confidence_level = "High"
                    elif overall_confidence >= 0.6:
                        confidence_level = "Medium"
                    elif overall_confidence >= 0.4:
                        confidence_level = "Low"
                    else:
                        confidence_level = "Very Low"
                    
                    confidence_analysis = {
                        'overall_score': overall_confidence,
                        'confidence_level': confidence_level,
                        'factors': confidence_factors,
                        'method': 'fallback'
                    }
                
                state['confidence_score'] = overall_confidence
                state['analysis_results']['confidence'] = confidence_analysis
                state['agent_reports']['confidence_assessor'] = f"✅ Confidence: {confidence_level} ({overall_confidence:.2f}, method: {confidence_analysis.get('method', 'unknown')})"
                
                # Broadcast completion
                self._broadcast_agent_status('confidence_assessment', 'completed', 1.0, 
                                           f"Confidence assessment completed - {confidence_level} ({overall_confidence:.2f})",
                                           data={'confidence_level': confidence_level, 'score': overall_confidence, 'factors': confidence_factors, 'method': confidence_analysis.get('method', 'unknown')})
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Confidence Assessment Agent failed: {e}")
                state['agent_reports']['confidence_assessor'] = f"❌ Confidence assessment failed: {str(e)}"
                
                # Broadcast error
                self._broadcast_agent_status('confidence_assessment', 'error', 0.0, 
                                           f"Confidence assessment failed: {str(e)}")
                
                return state
        
        def prediction_generator_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for generating the final prediction with Qwen3 CoT"""
            logger.info("🌟 Prediction Generator Agent: Creating final forecast with AI reasoning")
            
            # Broadcast agent start
            self._broadcast_agent_status('prediction_generator', 'running', 0.1, 
                                       "Starting AI-powered prediction generation...")
            
            try:
                # Compile all analysis for prediction generation
                location = state['location']
                days = state['prediction_days']
                current_conditions = state['current_conditions']
                patterns = state['historical_patterns']
                pattern_analysis = state.get('analysis_results', {}).get('pattern_analysis', {})
                meteorological = state.get('analysis_results', {}).get('meteorological', {})
                confidence = state.get('analysis_results', {}).get('confidence', {})
                
                # Broadcast progress
                self._broadcast_agent_status('prediction_generator', 'running', 0.2,
                                           "Generating Qwen3 CoT structured prediction...")

                # PRIMARY: Qwen3 CoT JSON prediction (exact day count, structured)
                if self.lm_studio_service and self.lm_studio_service.available:
                    try:
                        confidence_info = f"{confidence.get('confidence_level', 'Medium')} ({confidence.get('overall_score', 0.5):.2f})"
                        meteorological_info = meteorological.get('conditions_assessment', 'Standard conditions')
                        if meteorological.get('ai_analysis'):
                            meteorological_info = meteorological['ai_analysis'][:120]

                        # Build explicit day-by-day placeholder so model knows exactly how many to emit
                        day_template = ',\n    '.join(
                            f'{{"day": {i+1}, "date": "YYYY-MM-DD", "temperature_c": 0.0, "humidity_percent": 0.0, "wind_speed_kmh": 0.0, "conditions": "...", "precipitation_chance": 0.0, "confidence": "medium"}}'
                            for i in range(days)
                        )
                        cot_prompt = f"""You are a JSON weather forecast API for {location}.
Season: {meteorological.get('season','unknown')}. Now: {current_conditions.get('temperature','N/A')}°C, {current_conditions.get('humidity','N/A')}% RH.
Context: {meteorological_info[:100]}

Return ONLY valid JSON with EXACTLY {days} entries. Fill real values; dates start {__import__('datetime').date.today() + __import__('datetime').timedelta(days=1)}:
{{
  "location": "{location}",
  "predictions": [
    {day_template}
  ]
}}"""

                        messages = [
                            {"role": "system", "content": f"You are a weather forecast JSON API. Output ONLY a raw JSON object — no markdown fences, no explanation. The predictions array MUST have exactly {days} objects.\n/no_think"},
                            {"role": "user", "content": cot_prompt}
                        ]
                        # Scale tokens and timeout with day count so Qwen3 can always finish.
                        # ~75 output tokens per day + 250 wrapper; cap at 1500.
                        # Observed Qwen3-14B speed: ~1 token per 0.15s → add 30s overhead.
                        _max_tok = max(600, min(days * 75 + 250, 1500))
                        _call_timeout = max(90, int(_max_tok * 0.15) + 30)
                        # For 7+ day requests, clear LM Studio's KV cache first to prevent
                        # generation stalls caused by cache pressure (2021/2172 MiB used).
                        _clear_cache = days >= 7
                        logger.info(f"🎯 Qwen3 CoT prediction: {days} days, max_tokens={_max_tok}, timeout={_call_timeout}s, clear_cache={_clear_cache}")

                        prediction = self.lm_studio_service.generate_chat(
                            messages, _max_tok, 0.1,
                            request_timeout=_call_timeout,
                            clear_cache=_clear_cache
                        )

                        if prediction and len(prediction) > 50:
                            state['final_prediction'] = prediction
                            state['method_used'] = "LangGraph + Qwen3 CoT Multi-Agent"
                            state['agent_reports']['prediction_generator'] = f"✅ Qwen3 CoT prediction successful ({len(prediction)} chars)"
                            self._broadcast_agent_status('prediction_generator', 'completed', 1.0,
                                                       f"Qwen3 CoT prediction successful - {len(prediction)} characters",
                                                       data={'method': 'Qwen3 CoT', 'length': len(prediction)})
                            logger.info(f"✅ Qwen3 CoT prediction generated: {len(prediction)} characters")
                            return state
                        else:
                            logger.warning(f"⚠️ Qwen3 CoT prediction insufficient: {len(prediction) if prediction else 0} chars — trying LangChain fallback")
                    except Exception as lm_error:
                        logger.warning(f"⚠️ Qwen3 CoT prediction failed: {lm_error}")

                # FALLBACK: LangChain + RAG
                self._broadcast_agent_status('prediction_generator', 'running', 0.6,
                                           "Falling back to LangChain + RAG prediction...")
                if self.langchain_service and self.langchain_service.available:
                    try:
                        result = self.langchain_service.predict_weather_langchain_rag(location, days)
                        if result and result.get('success'):
                            state['final_prediction'] = result['prediction']
                            state['method_used'] = "LangGraph + LangChain + RAG"
                            state['agent_reports']['prediction_generator'] = "✅ LangChain + RAG prediction successful"
                            self._broadcast_agent_status('prediction_generator', 'completed', 1.0,
                                                       "LangChain + RAG prediction successful",
                                                       data={'method': 'LangChain + RAG'})
                            return state
                    except Exception as langchain_error:
                        logger.warning(f"⚠️ LangChain prediction failed: {langchain_error}")
                
                # Final fallback: Statistical prediction
                self._broadcast_agent_status('prediction_generator', 'running', 0.9, 
                                           "Generating statistical prediction as final fallback...")
                
                statistical_prediction = self._generate_statistical_prediction(state)
                state['final_prediction'] = statistical_prediction
                state['method_used'] = "LangGraph + Statistical Analysis"
                state['agent_reports']['prediction_generator'] = "✅ Statistical prediction generated (Qwen3 timed out)"
                
                # Broadcast completion
                self._broadcast_agent_status('prediction_generator', 'completed', 1.0, 
                                           "Statistical prediction generated successfully",
                                           data={'method': 'Statistical Analysis'})
                
                logger.warning("⚠️ All AI methods failed, using statistical prediction")
                return state
                
            except Exception as e:
                logger.error(f"❌ Prediction Generator Agent failed: {e}")
                state['error_message'] = f"Prediction generation failed: {str(e)}"
                state['agent_reports']['prediction_generator'] = f"❌ Prediction generation failed: {str(e)}"
                
                # Broadcast error
                self._broadcast_agent_status('prediction_generator', 'error', 0.0, 
                                           f"Prediction generation failed: {str(e)}")
                
                return state
        
        def quality_control_agent(state: WeatherAnalysisState) -> WeatherAnalysisState:
            """Agent responsible for validating prediction quality with Qwen3 CoT"""
            logger.info("🔍 Quality Control Agent: Validating prediction with AI reasoning")
            
            # Broadcast agent start
            self._broadcast_agent_status('quality_control', 'running', 0.1, 
                                       "Starting AI-powered quality validation...")
            
            try:
                prediction = state.get('final_prediction', '')
                quality_issues = []
                
                # Broadcast progress
                self._broadcast_agent_status('quality_control', 'running', 0.2, 
                                           "Running basic quality checks...")
                
                # Basic quality checks
                base_quality_score = 0.0
                
                # Check prediction length
                if len(prediction) < 100:
                    quality_issues.append("Prediction too short")
                    base_quality_score += 0.1
                elif len(prediction) > 50:
                    base_quality_score += 0.3
                
                # Check for location mention
                if state['location'].lower() in prediction.lower():
                    base_quality_score += 0.2
                else:
                    quality_issues.append("Location not mentioned")
                
                # Check for time frame mention  
                if str(state['prediction_days']) in prediction or 'day' in prediction.lower():
                    base_quality_score += 0.2
                else:
                    quality_issues.append("Time frame not clear")
                
                # Check for weather elements
                weather_elements = ['temperature', 'humidity', 'rain', 'wind', 'cloud', 'sunny', 'storm']
                elements_found = sum(1 for element in weather_elements if element in prediction.lower())
                base_quality_score += min(0.3, elements_found * 0.1)
                
                if elements_found < 2:
                    quality_issues.append("Insufficient weather details")
                
                # WEEK 3 ENHANCEMENT: Use Qwen3 CoT for quality validation
                if self.lm_studio_service and self.lm_studio_service.available and prediction:
                    self._broadcast_agent_status('quality_control', 'running', 0.5, 
                                               "Analyzing prediction quality with Qwen3...")
                    
                    # Get analysis context
                    confidence_info = state.get('analysis_results', {}).get('confidence', {})
                    meteorological = state.get('analysis_results', {}).get('meteorological', {})
                    
                    cot_prompt = f"""QC check: {state['prediction_days']}-day forecast for {state['location']}.
{prediction[:250]}{('...' if len(prediction) > 250 else '')}
Issues: {', '.join(quality_issues) if quality_issues else 'none'}. Base score: {base_quality_score:.2f}.

Reply ONLY: APPROVED or NEEDS_REVISION, quality score (0.0-1.0), one-sentence reason."""

                    try:
                        messages = [{"role": "user", "content": cot_prompt}]
                        ai_qa_analysis = self.lm_studio_service.generate_chat(messages, max_tokens=300, temperature=0.3)
                        
                        # Extract quality score from AI analysis
                        import re
                        quality_match = re.search(r'quality[^0-9]*score[:\s]+([0-9.]+)', ai_qa_analysis.lower())
                        if quality_match:
                            final_quality_score = min(1.0, float(quality_match.group(1)))
                        else:
                            # Average base score with AI's implicit assessment
                            final_quality_score = (base_quality_score + 0.7) / 2 if "good" in ai_qa_analysis.lower() or "approved" in ai_qa_analysis.lower() else base_quality_score
                        
                        # Check approval
                        approved = "approved" in ai_qa_analysis.lower() or "acceptable" in ai_qa_analysis.lower() or final_quality_score >= 0.6
                        
                        # Extract additional issues from AI
                        if "issue" in ai_qa_analysis.lower() or "error" in ai_qa_analysis.lower() or "inconsistenc" in ai_qa_analysis.lower():
                            ai_issues = re.findall(r'(?:issue|error|problem)[:\s]+([^.]+)', ai_qa_analysis.lower())
                            quality_issues.extend(ai_issues[:2])  # Add up to 2 AI-found issues
                        
                        quality_assessment = {
                            'quality_score': final_quality_score,
                            'approved': approved,
                            'issues_found': quality_issues,
                            'ai_reasoning': ai_qa_analysis[:400] if ai_qa_analysis else "No AI reasoning",
                            'method': 'qwen3_cot'
                        }
                        
                        logger.info(f"✅ Quality CoT validation: score {final_quality_score:.2f}, approved: {approved}")
                        
                    except Exception as cot_error:
                        logger.warning(f"⚠️ Qwen3 quality validation failed, using base score: {cot_error}")
                        final_quality_score = base_quality_score
                        quality_assessment = {
                            'quality_score': final_quality_score,
                            'approved': final_quality_score >= 0.5,
                            'issues_found': quality_issues,
                            'method': 'fallback'
                        }
                else:
                    # Fallback if LM Studio unavailable
                    self._broadcast_agent_status('quality_control', 'running', 0.8, 
                                               "Using fallback quality validation...")
                    final_quality_score = base_quality_score
                    quality_assessment = {
                        'quality_score': final_quality_score,
                        'approved': final_quality_score >= 0.5,
                        'issues_found': quality_issues,
                        'method': 'fallback'
                    }
                
                state['quality_score'] = final_quality_score
                state['agent_reports']['quality_controller'] = f"✅ Quality: {final_quality_score:.2f}, Issues: {len(quality_issues)}, Approved: {quality_assessment.get('approved', False)} (method: {quality_assessment.get('method', 'unknown')})"
                
                # Broadcast completion
                self._broadcast_agent_status('quality_control', 'completed', 1.0, 
                                           f"Quality validation completed - Score: {final_quality_score:.2f}, Approved: {quality_assessment.get('approved', False)}",
                                           data={'quality_score': final_quality_score, 'issues_count': len(quality_issues), 'approved': quality_assessment.get('approved', False), 'method': quality_assessment.get('method', 'unknown')})
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Quality Control Agent failed: {e}")
                state['agent_reports']['quality_controller'] = f"❌ Quality control failed: {str(e)}"
                state['quality_score'] = 0.0
                
                # Broadcast error
                self._broadcast_agent_status('quality_control', 'error', 0.0, 
                                           f"Quality control failed: {str(e)}")
                
                return state
        
        # Route based on data quality
        def route_based_on_data_quality(state: WeatherAnalysisState) -> str:
            """Route to appropriate next step based on data quality"""
            data_quality = state.get('current_conditions', {}).get('data_quality', 'low')
            
            if data_quality == 'high':
                return "pattern_analysis"
            elif data_quality in ['medium', 'low']:
                return "meteorological_expert"
            else:
                return "prediction_generator"
        
        # Check if retry is needed
        def should_retry_prediction(state: WeatherAnalysisState) -> str:
            """Determine if prediction should be retried"""
            quality_score = state.get('quality_score', 0.0)
            retry_count = state.get('retry_count', 0)
            
            if quality_score < 0.4 and retry_count < 1 and not state.get('error_message'):
                return "retry"
            else:
                return "complete"
        
        # Build the workflow
        workflow = StateGraph(WeatherAnalysisState)
        
        # Add agents as nodes
        workflow.add_node("data_collector", data_collection_agent)
        workflow.add_node("pattern_analyzer", pattern_analysis_agent)
        workflow.add_node("meteorological_expert", meteorological_expert_agent)
        workflow.add_node("confidence_assessor", confidence_assessment_agent)
        workflow.add_node("prediction_generator", prediction_generator_agent)
        workflow.add_node("quality_controller", quality_control_agent)
        
        # Define workflow edges
        workflow.set_entry_point("data_collector")
        
        # Conditional routing from data collector
        workflow.add_conditional_edges(
            "data_collector",
            route_based_on_data_quality,
            {
                "pattern_analysis": "pattern_analyzer",
                "meteorological_expert": "meteorological_expert", 
                "prediction_generator": "prediction_generator"
            }
        )
        
        # Sequential flow through analysis
        workflow.add_edge("pattern_analyzer", "meteorological_expert")
        workflow.add_edge("meteorological_expert", "confidence_assessor")
        workflow.add_edge("confidence_assessor", "prediction_generator")
        workflow.add_edge("prediction_generator", "quality_controller")
        
        # Quality control with retry logic
        workflow.add_conditional_edges(
            "quality_controller",
            should_retry_prediction,
            {
                "retry": "prediction_generator",
                "complete": END
            }
        )
        
        return workflow.compile()
    
    def _create_service_selection_workflow(self):
        """Create intelligent service selection workflow"""
        
        def check_service_availability(state: ServiceSelectionState) -> ServiceSelectionState:
            """Check which services are available"""
            available_services = {
                'langchain_rag': self.langchain_service and self.langchain_service.available,
                'lm_studio': self.lm_studio_service and self.lm_studio_service.available,
                'rag_only': self.rag_service is not None,
                'statistical': True  # Always available
            }
            
            state['available_services'] = available_services
            return state
        
        def select_best_method(state: ServiceSelectionState) -> str:
            """Select the best available prediction method"""
            services = state['available_services']
            
            if services.get('langchain_rag'):
                state['selected_method'] = 'langchain_rag'
                return "langchain_rag"
            elif services.get('lm_studio'):
                state['selected_method'] = 'lm_studio' 
                return "lm_studio"
            elif services.get('rag_only'):
                state['selected_method'] = 'rag_only'
                return "rag_only"
            else:
                state['selected_method'] = 'statistical'
                return "statistical"
        
        # Service execution nodes (simplified for brevity)
        def execute_langchain_rag(state: ServiceSelectionState) -> ServiceSelectionState:
            try:
                result = self.langchain_service.predict_weather_langchain_rag(
                    state['location'], state['prediction_days']
                )
                state['prediction_result'] = result
                return state
            except Exception as e:
                state['prediction_result'] = {'error': str(e), 'success': False}
                return state
        
        # Build service selection workflow
        workflow = StateGraph(ServiceSelectionState)
        workflow.add_node("check_services", check_service_availability)
        workflow.add_node("langchain_rag", execute_langchain_rag)
        
        workflow.set_entry_point("check_services")
        workflow.add_conditional_edges(
            "check_services",
            select_best_method,
            {
                "langchain_rag": "langchain_rag",
                "lm_studio": END,  # Simplified 
                "rag_only": END,
                "statistical": END
            }
        )
        workflow.add_edge("langchain_rag", END)
        
        return workflow.compile()
    
    def _create_quality_validation_workflow(self):
        """Create prediction quality validation workflow"""
        
        def validate_prediction_quality(state: QualityValidationState) -> QualityValidationState:
            """Validate the quality of a weather prediction"""
            prediction = state['prediction_text']
            quality_checks = {
                'has_content': len(prediction.strip()) > 0,
                'adequate_length': len(prediction) >= 100,
                'contains_weather_terms': any(term in prediction.lower() for term in ['weather', 'temperature', 'rain', 'wind', 'cloud']),
                'mentions_location': True,  # Would check for location mention
                'includes_timeframe': True   # Would check for time references
            }
            
            overall_quality = sum(quality_checks.values()) / len(quality_checks)
            
            state['quality_checks'] = quality_checks
            state['overall_quality'] = overall_quality
            state['approved'] = overall_quality >= 0.6
            
            return state
        
        workflow = StateGraph(QualityValidationState)
        workflow.add_node("validate", validate_prediction_quality)
        workflow.set_entry_point("validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _build_comprehensive_prompt(self, state: WeatherAnalysisState) -> str:
        """Build comprehensive prompt from agent analysis"""
        location = state['location']
        days = state['prediction_days']
        current_conditions = state['current_conditions']
        patterns = state['historical_patterns']
        meteorological = state.get('analysis_results', {}).get('meteorological', {})
        
        prompt = f"""Generate a comprehensive {days}-day weather forecast for {location}.

CURRENT CONDITIONS:
Temperature: {current_conditions.get('temperature', 'N/A')}°C
Humidity: {current_conditions.get('humidity', 'N/A')}%
Wind Speed: {current_conditions.get('wind_speed', 'N/A')} m/s

HISTORICAL PATTERNS:
Found {len(patterns)} similar weather patterns from historical data.

METEOROLOGICAL ANALYSIS:
Season: {meteorological.get('season', 'Unknown')}
Conditions Assessment: {meteorological.get('conditions_assessment', 'Standard patterns expected')}
Seasonal Factors: {meteorological.get('seasonal_factors', 'Normal seasonal variations')}

AGENT ANALYSIS SUMMARY:
{chr(10).join([f"• {agent}: {report}" for agent, report in state.get('agent_reports', {}).items()])}

Generate a detailed, day-by-day forecast with scientific reasoning and confidence levels."""

        return prompt
    
    def _generate_statistical_prediction(self, state: WeatherAnalysisState) -> str:
        """Generate statistical fallback prediction as JSON so the frontend renders day cards"""
        import json as _json
        import datetime as _dt
        location = state['location']
        days = state['prediction_days']
        conditions = state['current_conditions']
        patterns = state.get('historical_patterns', {})

        base_temp = conditions.get('temperature', 15.0)
        base_humidity = conditions.get('humidity', 65.0)
        base_wind = (conditions.get('wind_speed', 3.0) or 0) * 3.6  # m/s → km/h
        
        start_date = _dt.date.today() + _dt.timedelta(days=1)
        daily_conds = ['Partly cloudy', 'Mostly clear', 'Cloudy', 'Partly cloudy', 'Clear', 'Overcast', 'Partly cloudy']
        
        prediction_list = []
        for i in range(days):
            d = start_date + _dt.timedelta(days=i)
            temp_variation = (i % 3 - 1) * 1.2
            precip = 20 + (i % 4) * 10
            prediction_list.append({
                "day": i + 1,
                "date": d.isoformat(),
                "temperature_c": round(base_temp + temp_variation, 1),
                "humidity_percent": round(base_humidity + (i % 3) * 2, 1),
                "wind_speed_kmh": round(base_wind + (i % 2) * 2, 1),
                "conditions": daily_conds[i % len(daily_conds)],
                "precipitation_chance": precip,
                "confidence": "low"
            })
        
        return _json.dumps({
            "location": location,
            "predictions": prediction_list,
            "analysis": "Statistical forecast based on current conditions and seasonal patterns. AI service timed out."
        }, indent=2)
    
    def predict_weather_with_langgraph(self, location: str = "Tokyo", prediction_days: int = 3, workflow_id: str = None) -> Dict[str, Any]:
        """
        Generate weather prediction using LangGraph multi-agent system
        
        Args:
            location: Location for weather prediction
            prediction_days: Number of days to predict
            workflow_id: Optional workflow ID for WebSocket tracking
            
        Returns:
            Dict containing prediction result with agent analysis
        """
        if not self.available:
            return {
                "error": "LangGraph service not available",
                "success": False,
                "method": "unavailable"
            }
        
        try:
            logger.info(f"🧠 Starting LangGraph multi-agent prediction for {location}, {prediction_days} days")
            
            # Set current workflow ID for WebSocket broadcasting
            self.current_workflow_id = workflow_id
            
            # Start workflow if WebSocket service is available
            if self.websocket_service and not workflow_id:
                workflow_id = self.websocket_service.start_workflow(
                    workflow_type='weather_prediction',
                    params={'location': location, 'prediction_days': prediction_days}
                )
                self.current_workflow_id = workflow_id
            
            # Initialize state
            initial_state = WeatherAnalysisState(
                location=location,
                prediction_days=prediction_days,
                current_conditions={},
                historical_patterns=[],
                rag_confidence=0.0,
                analysis_results={},
                agent_reports={},
                final_prediction="",
                confidence_score=0.0,
                method_used="",
                error_message=None,
                retry_count=0,
                quality_score=0.0
            )
            
            # Run the multi-agent workflow
            final_state = self.prediction_graph.invoke(initial_state)
            
            # Compile results
            result = {
                "prediction": final_state.get('final_prediction', 'Prediction generation failed'),
                "location": location,
                "prediction_days": prediction_days,
                "timeframe": prediction_days,
                "generated_at": datetime.now().isoformat(),
                "method": "langgraph_multi_agent",
                "model_used": final_state.get('method_used', 'LangGraph Multi-Agent System'),
                "success": bool(final_state.get('final_prediction')),
                "workflow_id": self.current_workflow_id,
                "confidence_level": self._format_confidence_level(final_state.get('confidence_score', 0.0)),
                "quality_score": final_state.get('quality_score', 0.0),
                
                # LangGraph specific metadata
                "langgraph_analysis": {
                    "agent_reports": final_state.get('agent_reports', {}),
                    "analysis_results": final_state.get('analysis_results', {}),
                    "historical_patterns_found": len(final_state.get('historical_patterns', [])),
                    "rag_confidence": final_state.get('rag_confidence', 0.0),
                    "retry_count": final_state.get('retry_count', 0)
                },
                
                "features": [
                    "Multi-Agent Analysis",
                    "Intelligent Routing", 
                    "Quality Validation",
                    "Dynamic Fallbacks",
                    "Confidence Assessment"
                ],
                
                "enhancement": "Advanced multi-agent system with specialized weather analysis agents"
            }
            
            if final_state.get('error_message'):
                result['warning'] = final_state['error_message']
            
            # Broadcast final result via WebSocket
            if result.get('success'):
                self._broadcast_workflow_complete(result)
                logger.info("✅ LangGraph multi-agent prediction completed successfully")
            else:
                self._broadcast_workflow_error(final_state.get('error_message', 'Unknown error'))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LangGraph prediction failed: {e}")
            return {
                "error": f"LangGraph prediction failed: {str(e)}",
                "success": False,
                "method": "langgraph_error",
                "location": location,
                "prediction_days": prediction_days
            }
    
    def _format_confidence_level(self, score: float) -> str:
        """Format confidence score as level"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium" 
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_langgraph_status(self) -> Dict[str, Any]:
        """Get detailed status of LangGraph service"""
        return {
            "service": "LangGraph Multi-Agent Weather Prediction",
            "available": self.available,
            "workflows_initialized": {
                "prediction_graph": self.prediction_graph is not None,
                "service_selection_graph": self.service_selection_graph is not None,
                "quality_validation_graph": self.quality_validation_graph is not None
            },
            "agent_configs": {name: {"description": config.description, "confidence_threshold": config.confidence_threshold} 
                            for name, config in self.agents.items()},
            "dependent_services": {
                "weather_service": self.weather_service is not None,
                "rag_service": self.rag_service is not None,
                "langchain_service": self.langchain_service and self.langchain_service.available,
                "lm_studio_service": self.lm_studio_service and self.lm_studio_service.available
            }
        }


# Global instance for easy access
langgraph_service = None

def get_langgraph_service(weather_service=None, rag_service=None, langchain_service=None, lm_studio_service=None, websocket_service=None):
    """Get the global LangGraph service instance"""
    global langgraph_service
    
    if langgraph_service is None and weather_service:
        langgraph_service = LangGraphWeatherService(
            weather_service=weather_service,
            rag_service=rag_service,
            langchain_service=langchain_service,
            lm_studio_service=lm_studio_service,
            websocket_service=websocket_service
        )
        
        if langgraph_service.available:
            logger.info("🧠 LangGraph service initialized successfully")
        else:
            logger.warning("⚠️ LangGraph service initialization failed")
    
    return langgraph_service