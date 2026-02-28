import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import google.generativeai as genai
import logging

# Import LangChain dependencies with graceful fallbacks
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ LangChain dependencies not available: {e}")

# Import RAG service
try:
    from backend.rag_service import WeatherRAGService
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ RAG service not available: {e}")

# Import LM Studio service
try:
    from backend.lmstudio_service import LMStudioService
    LM_STUDIO_AVAILABLE = True
except ImportError as e:
    LM_STUDIO_AVAILABLE = False
    print(f"⚠️ LM Studio service not available: {e}")

# Import LangChain + RAG service
try:
    from backend.langchain_rag_service import LangChainRAGService
    LANGCHAIN_RAG_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_RAG_AVAILABLE = False
    print(f"⚠️ LangChain + RAG service not available: {e}")

# Import LangGraph service
try:
    from backend.langgraph_service import LangGraphWeatherService
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"⚠️ LangGraph service not available: {e}")

# Import Ensemble service
try:
    from backend.ensemble_service import EnsemblePredictionService
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    ENSEMBLE_AVAILABLE = False
    print(f"⚠️ Ensemble service not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class WeatherPredictionService:
    def __init__(self):
        try:
            logger.info("🔄 Initializing Weather Prediction Service with LangChain...")
            
            # Initialize variables
            self.rag_service = None
            self.lm_studio_service = None
            self.langchain_rag_service = None
            self.langgraph_service = None
            self.ensemble_service = None
            self.llm = None
            self.gemini_available = False
            
            # Load CSV data first (this doesn't require API keys)
            self.csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Generated_electricity_load_japan_past365days_much_more_reduced.csv')
            self.data = None
            self.load_data()
            
            # Try to load environment variables
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            if self.gemini_api_key:
                try:
                    # Configure the Gemini AI
                    genai.configure(api_key=self.gemini_api_key)
                    
                    # Initialize the LangChain Gemini model (if available)
                    if LANGCHAIN_AVAILABLE:
                        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash-exp",
                            google_api_key=self.gemini_api_key,
                            temperature=0.7,
                            max_tokens=2000,
                            convert_system_message_to_human=True
                        )
                        logger.info("✅ Gemini AI with LangChain initialized successfully")
                    else:
                        self.llm = None
                        logger.warning("⚠️ LangChain not available, using basic Gemini API")
                    
                    self.gemini_available = True
                    
                    # Initialize RAG service with Gemini
                    if RAG_AVAILABLE:
                        try:
                            self.rag_service = WeatherRAGService(
                                weather_data_path=self.csv_path,
                                gemini_api_key=self.gemini_api_key
                            )
                            logger.info("✅ RAG service initialized successfully")
                        except Exception as rag_error:
                            logger.warning(f"⚠️ RAG service initialization failed: {rag_error}")
                            
                except Exception as gemini_error:
                    logger.warning(f"⚠️ Gemini AI initialization failed: {gemini_error}")
                    self.gemini_available = False
            else:
                logger.warning("⚠️ GEMINI_API_KEY not found - Gemini AI features disabled")
                
            # Try to initialize RAG service with local embeddings if Gemini failed
            if not self.rag_service and RAG_AVAILABLE:
                try:
                    logger.info("🔄 Attempting RAG service initialization with local embeddings fallback")
                    # Create a dummy API key for local-only operation
                    self.rag_service = WeatherRAGService(
                        weather_data_path=self.csv_path,
                        gemini_api_key="dummy_key_for_local"  # Will trigger local fallback
                    )
                    logger.info("✅ RAG service initialized with local embeddings fallback")
                except Exception as local_rag_error:
                    logger.warning(f"⚠️ Local RAG service initialization also failed: {local_rag_error}")
            
            # Initialize LM Studio service for local LLM (independent of Gemini)
            if LM_STUDIO_AVAILABLE:
                try:
                    self.lm_studio_service = LMStudioService()
                    if self.lm_studio_service.available:
                        logger.info("🏠 LM Studio service initialized and available")
                    else:
                        logger.warning("⚠️ LM Studio service initialized but not available")
                except Exception as e:
                    logger.error(f"⚠️ Could not initialize LM Studio service: {e}")
                    self.lm_studio_service = None
            else:
                logger.warning("⚠️ LM Studio service not available")
                self.lm_studio_service = None
            
            # Initialize LangChain + RAG service for advanced orchestration
            if LANGCHAIN_RAG_AVAILABLE:
                try:
                    self.langchain_rag_service = LangChainRAGService(
                        weather_service=self,
                        lm_studio_service=self.lm_studio_service,
                        rag_service=self.rag_service
                    )
                    if self.langchain_rag_service.available:
                        logger.info("🧠 LangChain + RAG service initialized successfully")
                    else:
                        logger.warning("⚠️ LangChain + RAG service initialized but not available")
                except Exception as e:
                    logger.error(f"⚠️ Could not initialize LangChain + RAG service: {e}")
                    self.langchain_rag_service = None
            else:
                logger.warning("⚠️ LangChain + RAG service not available")
                self.langchain_rag_service = None
            
            # Initialize LangGraph multi-agent service for advanced orchestration
            if LANGGRAPH_AVAILABLE:
                try:
                    self.langgraph_service = LangGraphWeatherService(
                        weather_service=self,
                        rag_service=self.rag_service,
                        langchain_service=self.langchain_rag_service,
                        lm_studio_service=self.lm_studio_service
                    )
                    if self.langgraph_service.available:
                        logger.info("🧠 LangGraph multi-agent service initialized successfully")
                    else:
                        logger.warning("⚠️ LangGraph service initialized but not available")
                except Exception as e:
                    logger.error(f"⚠️ Could not initialize LangGraph service: {e}")
                    self.langgraph_service = None
            else:
                logger.warning("⚠️ LangGraph service not available")
                self.langgraph_service = None

            # Initialize Ensemble service (Week 4)
            if ENSEMBLE_AVAILABLE:
                try:
                    self.ensemble_service = EnsemblePredictionService(
                        lm_studio_service=self.lm_studio_service,
                        rag_service=self.rag_service,
                        langgraph_service=self.langgraph_service,
                        langchain_service=self.langchain_rag_service,
                        weather_service=self,
                    )
                    logger.info("🎯 Ensemble prediction service initialized successfully")
                except Exception as e:
                    logger.error(f"⚠️ Could not initialize Ensemble service: {e}")
                    self.ensemble_service = None
            else:
                logger.warning("⚠️ Ensemble service not available")
                self.ensemble_service = None

            logger.info(f"✅ Weather service initialized successfully with {len(self.data) if self.data is not None else 0} records")
            
        except Exception as e:
            logger.error(f"❌ Error initializing weather service: {str(e)}")
            # Don't raise the exception - allow partial initialization
            # Initialize minimal required attributes
            if not hasattr(self, 'data'):
                self.data = None
            if not hasattr(self, 'rag_service'):
                self.rag_service = None
            if not hasattr(self, 'lm_studio_service'):
                self.lm_studio_service = None
            if not hasattr(self, 'langchain_rag_service'):
                self.langchain_rag_service = None
            if not hasattr(self, 'gemini_available'):
                self.gemini_available = False
            elif not LANGCHAIN_RAG_AVAILABLE:
                logger.info("⚠️ LangChain + RAG not available - install langchain for advanced features")
            else:
                logger.info("⚠️ LangChain + RAG requires both LM Studio and RAG services")
            
            logger.info(f"✅ Weather service initialized successfully with {len(self.data) if self.data is not None else 0} records")
            
        except Exception as e:
            logger.error(f"❌ Error initializing weather service: {str(e)}")
            raise
    
    def load_data(self):
        """Load and prepare the CSV data"""
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
                
            self.data = pd.read_csv(self.csv_path)
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
            self.data = self.data.sort_values('Datetime')
            print(f"Loaded {len(self.data)} records from CSV")
            logger.info(f"📊 CSV data loaded: {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV data: {e}")
            print(f"Error loading CSV data: {e}")
            self.data = None
            raise
    
    def get_recent_weather_data(self, days=7):
        """Get recent weather data for analysis"""
        try:
            if self.data is None:
                logger.error("❌ No weather data available")
                return None
            
            # Get the last N days of data
            end_date = self.data['Datetime'].max()
            start_date = end_date - timedelta(days=days)
            
            recent_data = self.data[self.data['Datetime'] >= start_date].copy()
            
            # Group by day and calculate averages
            recent_data['Date'] = recent_data['Datetime'].dt.date
            daily_avg = recent_data.groupby('Date').agg({
                'Forecast_Temperature(°C)': 'mean',
                'Forecast_Humidity(%)': 'mean',
                'Forecast_Solar(kWh/m²/day)': 'mean',
                'Forecast_WindSpeed(m/s)': 'mean',
                'Forecast_Rainfall(mm)': 'mean',
                'Forecast_CloudCover(0-10)': 'mean',
                'Actual_Temperature(°C)': 'mean',
                'Actual_Humidity(%)': 'mean',
                'Actual_Solar(kWh/m²/day)': 'mean',
                'Actual_WindSpeed(m/s)': 'mean',
                'Actual_Rainfall(mm)': 'mean',
                'Actual_CloudCover(0-10)': 'mean',
                'Weather_Variability_Index': 'mean'
            }).round(2)
            
            logger.info(f"📈 Retrieved {len(daily_avg)} days of recent weather data")
            return daily_avg
            
        except Exception as e:
            logger.error(f"❌ Error getting recent weather data: {str(e)}")
            return None
    
    def get_weather_statistics(self):
        """Get statistical summary of weather data"""
        try:
            if self.data is None:
                return None
            
            weather_cols = [
                'Actual_Temperature(°C)', 'Actual_Humidity(%)', 
                'Actual_Solar(kWh/m²/day)', 'Actual_WindSpeed(m/s)',
                'Actual_Rainfall(mm)', 'Actual_CloudCover(0-10)'
            ]
            
            stats = self.data[weather_cols].describe().round(2)
            logger.info("📊 Weather statistics calculated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating weather statistics: {str(e)}")
            return None
    
    def create_weather_prompt(self, location="Tokyo", prediction_days=3):
        """Create a comprehensive prompt for weather prediction"""
        try:
            recent_data = self.get_recent_weather_data(7)
            stats = self.get_weather_statistics()
            
            if recent_data is None or stats is None:
                logger.error("❌ Unable to load weather data for prompt creation")
                return None
            
            # Build weather context
            recent_summary = self._build_weather_context(recent_data)
            stats_summary = self._build_stats_context(stats)
            
            prompt_template = PromptTemplate(
                input_variables=["location", "prediction_days", "recent_summary", "stats_summary"],
                template="""You are an advanced weather prediction AI specializing in Japanese weather patterns. You have access to historical electricity load data from Japan which contains comprehensive weather information.

Location: {location}, Japan
Prediction Period: Next {prediction_days} days
Current Date: November 2025 (Late Autumn)

RECENT WEATHER DATA (Last 7 days):
{recent_summary}

HISTORICAL STATISTICS:
{stats_summary}

Please provide a detailed weather forecast with the following structure:

**WEATHER FORECAST for {location} - Next {prediction_days} days:**

**Day-by-Day Predictions:**
[For each day, provide:]
- Day X: [Date estimate]
- Temperature: High XX°C / Low XX°C
- Humidity: XX%
- Precipitation: XX% chance, XX mm expected
- Wind: XX km/h from [direction]
- Cloud Cover: XX% (Clear/Partly Cloudy/Cloudy/Overcast)
- Conditions: [Brief description]

**WEATHER ANALYSIS:**
- Current Trend: [Describe recent weather patterns]
- Seasonal Context: [November weather expectations for Japan]
- Confidence Level: [High/Medium/Low] and reasoning

**ACTIONABLE INSIGHTS:**
- Best times for outdoor activities
- Clothing recommendations
- Energy consumption expectations
- Any weather warnings or advisories

**DATA SOURCE:** Japan Electricity Load Historical Weather Data

Please provide realistic, detailed predictions based on typical Japanese seasonal patterns and the historical data trends."""
            )
            
            logger.info("🔮 Weather prompt created successfully")
            return prompt_template, {
                "location": location,
                "prediction_days": prediction_days,
                "recent_summary": recent_summary,
                "stats_summary": stats_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating weather prompt: {str(e)}")
            return None
    
    def _build_weather_context(self, recent_data):
        """Build readable context from recent weather data"""
        try:
            if recent_data.empty:
                return "No recent weather data available"
            
            # Get latest values
            latest = recent_data.iloc[-1]
            context = f"""
Temperature: {latest['Actual_Temperature(°C)']:.1f}°C
Humidity: {latest['Actual_Humidity(%)']:.1f}%
Wind Speed: {latest['Actual_WindSpeed(m/s)']:.1f} m/s
Rainfall: {latest['Actual_Rainfall(mm)']:.1f} mm
Cloud Cover: {latest['Actual_CloudCover(0-10)']:.1f}/10
Solar Radiation: {latest['Actual_Solar(kWh/m²/day)']:.1f} kWh/m²/day
Weather Variability: {latest['Weather_Variability_Index']:.2f}

Recent 7-day averages:
- Temperature: {recent_data['Actual_Temperature(°C)'].mean():.1f}°C
- Humidity: {recent_data['Actual_Humidity(%)'].mean():.1f}%
- Wind: {recent_data['Actual_WindSpeed(m/s)'].mean():.1f} m/s
"""
            return context.strip()
            
        except Exception as e:
            logger.error(f"❌ Error building weather context: {str(e)}")
            return f"Weather context error: {str(e)}"
    
    def _build_stats_context(self, stats):
        """Build readable context from weather statistics"""
        try:
            temp_stats = stats['Actual_Temperature(°C)']
            humidity_stats = stats['Actual_Humidity(%)']
            
            context = f"""
Temperature Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C (avg: {temp_stats['mean']:.1f}°C)
Humidity Range: {humidity_stats['min']:.1f}% to {humidity_stats['max']:.1f}% (avg: {humidity_stats['mean']:.1f}%)
Data Quality: Based on {len(self.data)} historical records from Japanese power grid data
"""
            return context.strip()
            
        except Exception as e:
            logger.error(f"❌ Error building stats context: {str(e)}")
            return f"Statistics context error: {str(e)}"
    
    def predict_weather(self, location="Tokyo", prediction_days=3):
        """Generate weather prediction using LangChain and Gemini AI"""
        try:
            logger.info(f"🔮 Starting weather prediction for {location}, {prediction_days} days...")
            
            # Create the weather prompt
            prompt_result = self.create_weather_prompt(location, prediction_days)
            if prompt_result is None:
                logger.error("❌ Failed to create weather prompt")
                return {
                    "error": "Unable to load weather data for prediction",
                    "success": False
                }
            
            prompt_template, prompt_vars = prompt_result
            
            # Format the prompt with variables
            try:
                formatted_prompt = prompt_template.format(**prompt_vars)
                logger.info("✅ Prompt formatted successfully")
            except Exception as e:
                logger.error(f"❌ Error formatting prompt: {str(e)}")
                return {
                    "error": f"Prompt formatting failed: {str(e)}",
                    "success": False
                }
            
            # Generate prediction using LangChain with proper 2024 syntax
            try:
                logger.info("🤖 Calling Gemini AI through LangChain...")
                
                # Call the LLM directly with the formatted prompt string
                response = self.llm.invoke(formatted_prompt)
                
                # Extract the prediction text
                if hasattr(response, 'content'):
                    prediction_text = response.content
                else:
                    prediction_text = str(response)
                
                logger.info("✅ AI prediction generated successfully")
                
                return {
                    "prediction": prediction_text,
                    "location": location,
                    "prediction_days": prediction_days,
                    "generated_at": datetime.now().isoformat(),
                    "data_source": "Japan Electricity Load Data (Past 365 days)",
                    "model_used": "LangChain + Gemini Pro",
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"❌ LangChain/Gemini API error: {str(e)}")
                
                # Check if it's a quota/rate limit error
                if "quota" in str(e).lower() or "429" in str(e) or "rate" in str(e).lower():
                    logger.info("🔄 Quota exceeded - using statistical fallback prediction...")
                    return self.generate_statistical_prediction(location, prediction_days, prompt_vars)
                else:
                    return {
                        "error": f"AI prediction failed: {str(e)}",
                        "success": False,
                        "fallback_available": True
                    }
            
        except Exception as e:
            logger.error(f"❌ Overall prediction error: {str(e)}")
            return {
                "error": f"Weather prediction service error: {str(e)}",
                "success": False
            }
    
    def get_data_summary(self):
        """Get summary of the loaded data"""
        try:
            if self.data is None:
                logger.error("❌ No data available for summary")
                return None
            
            summary = {
                "total_records": len(self.data),
                "date_range": {
                    "start": self.data['Datetime'].min().isoformat(),
                    "end": self.data['Datetime'].max().isoformat()
                },
                "locations": self.data['Plant_Area'].unique().tolist(),
                "weather_parameters": [
                    "Temperature", "Humidity", "Solar Radiation", 
                    "Wind Speed", "Rainfall", "Cloud Cover"
                ],
                "data_quality": {
                    "missing_values": int(self.data.isnull().sum().sum()),
                    "completeness": f"{((1 - self.data.isnull().sum().sum() / self.data.size) * 100):.1f}%"
                }
            }
            
            logger.info("📊 Data summary generated successfully")
            return summary
        
        except Exception as e:
            logger.error(f"❌ Error generating data summary: {str(e)}")
            return {"error": str(e)}
    
    def generate_statistical_prediction(self, location="Tokyo", prediction_days=3, prompt_vars=None):
        """Generate weather prediction using statistical analysis when AI is unavailable"""
        try:
            logger.info(f"📊 Generating statistical prediction for {location}, {prediction_days} days")
            
            # Get recent weather data and statistics
            recent_data = self.get_recent_weather_data(days=7)
            stats = self.get_weather_statistics()
            
            if recent_data is None or len(recent_data) == 0:
                return {
                    "error": "No weather data available for statistical analysis",
                    "success": False
                }
            
            # Create statistical forecast
            prediction_text = self.create_statistical_forecast(location, prediction_days, recent_data, stats)
            
            logger.info("✅ Statistical prediction generated successfully")
            
            return {
                "prediction": prediction_text,
                "location": location,
                "prediction_days": prediction_days,
                "generated_at": datetime.now().isoformat(),
                "data_source": "Statistical Analysis of Japan Electricity Load Data",
                "model_used": "Statistical Pattern Analysis (AI Fallback)",
                "success": True,
                "fallback_used": True,
                "note": "AI service temporarily unavailable - using statistical analysis"
            }
            
        except Exception as e:
            logger.error(f"❌ Statistical prediction failed: {str(e)}")
            return {
                "error": f"Statistical prediction failed: {str(e)}",
                "success": False,
                "fallback_available": False
            }
    
    def create_statistical_forecast(self, location, prediction_days, recent_data, stats):
        """Create a weather forecast based on statistical analysis"""
        try:
            # Analyze recent trends
            if len(recent_data) > 1:
                temp_trend = recent_data['Actual_Temperature(°C)'].iloc[-1] - recent_data['Actual_Temperature(°C)'].iloc[0]
                humidity_trend = recent_data['Actual_Humidity(%)'].iloc[-1] - recent_data['Actual_Humidity(%)'].iloc[0]
                wind_trend = recent_data['Actual_WindSpeed(m/s)'].iloc[-1] - recent_data['Actual_WindSpeed(m/s)'].iloc[0]
            else:
                temp_trend = humidity_trend = wind_trend = 0
            
            # Get current season
            current_month = datetime.now().month
            if current_month in [12, 1, 2]:
                season = "Winter"
                season_context = "cold temperatures, lower humidity, possible snow"
            elif current_month in [3, 4, 5]:
                season = "Spring"
                season_context = "mild temperatures, variable conditions, rain possible"
            elif current_month in [6, 7, 8]:
                season = "Summer"
                season_context = "warm temperatures, higher humidity, thunderstorms possible"
            else:
                season = "Autumn"
                season_context = "cooling temperatures, moderate humidity, changing weather"
            
            # Latest readings
            latest_temp = recent_data['Actual_Temperature(°C)'].iloc[-1]
            latest_humidity = recent_data['Actual_Humidity(%)'].iloc[-1]
            latest_wind = recent_data['Actual_WindSpeed(m/s)'].iloc[-1]
            latest_rainfall = recent_data['Actual_Rainfall(mm)'].iloc[-1]
            
            # Generate forecast
            forecast = f"""🌤️ **{prediction_days}-Day Weather Forecast for {location}**

📊 **Current Conditions ({season}):**
• Temperature: {latest_temp:.1f}°C
• Humidity: {latest_humidity:.1f}%
• Wind Speed: {latest_wind:.1f} m/s
• Recent Rainfall: {latest_rainfall:.1f}mm

📈 **Forecast Analysis:**
Based on statistical analysis of recent weather patterns and seasonal trends:

"""
            
            for day in range(1, prediction_days + 1):
                # Project trends with some realistic variation
                temp_change = temp_trend * (day / len(recent_data)) + (-2 + 4 * (day % 2))  # Add some variation
                projected_temp = latest_temp + temp_change
                
                humidity_change = humidity_trend * (day / len(recent_data)) + (-5 + 10 * ((day + 1) % 2))
                projected_humidity = max(30, min(95, latest_humidity + humidity_change))
                
                wind_change = wind_trend * (day / len(recent_data)) + (-1 + 2 * (day % 2))
                projected_wind = max(0, latest_wind + wind_change)
                
                # Simple rain probability based on humidity and season
                rain_probability = "Low"
                if projected_humidity > 70:
                    rain_probability = "Moderate" if season in ["Spring", "Autumn"] else "High"
                elif projected_humidity > 85:
                    rain_probability = "High"
                
                forecast += f"""**Day {day}:**
• Temperature: {projected_temp:.1f}°C ({"↗" if temp_change > 0 else "↘" if temp_change < 0 else "→"})
• Humidity: {projected_humidity:.1f}%
• Wind: {projected_wind:.1f} m/s
• Rain Probability: {rain_probability}
• Conditions: {season_context}

"""
            
            forecast += f"""📋 **Statistical Summary:**
• Historical Average Temperature: {stats['Actual_Temperature(°C)']['mean']:.1f}°C
• Temperature Range: {stats['Actual_Temperature(°C)']['min']:.1f}°C to {stats['Actual_Temperature(°C)']['max']:.1f}°C
• Average Humidity: {stats['Actual_Humidity(%)']['mean']:.1f}%
• Average Wind Speed: {stats['Actual_WindSpeed(m/s)']['mean']:.1f} m/s

⚠️ **Note:** This forecast is generated using statistical analysis of historical weather data. For official weather forecasts, please consult meteorological services.
🤖 **AI Status:** Gemini API quota exceeded - using statistical analysis fallback."""

            return forecast
            
        except Exception as e:
            logger.error(f"❌ Error creating statistical forecast: {str(e)}")
            return f"Statistical analysis unavailable: {str(e)}"

    # ===================== RAG-ENHANCED METHODS =====================
    
    def predict_weather_with_rag(self, location="Japan", timeframe=7):
        """Enhanced weather prediction using RAG + historical patterns"""
        try:
            logger.info(f"🔍 Starting RAG-enhanced weather prediction for {location} ({timeframe} days)")
            
            if not self.rag_service:
                logger.warning("⚠️ RAG service not available, falling back to standard prediction")
                return self.predict_weather(location, timeframe)
            
            # Get current weather context
            current_date = datetime.now()
            current_season = self._get_season(current_date.month)
            
            # Get latest weather data for similarity search
            if self.data is not None and not self.data.empty:
                latest_data = self.data.tail(1).iloc[0]
                current_temp = float(latest_data['Actual_Temperature(°C)'])
                current_humidity = float(latest_data['Actual_Humidity(%)'])
                
                logger.info(f"📊 Current conditions: {current_temp}°C, {current_humidity}% humidity, {current_season}")
                
                # Retrieve similar weather patterns using RAG
                query = f"temperature {current_temp}°C humidity {current_humidity}% {current_season} season Tokyo Japan"
                similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
                
                if similar_patterns:
                    # Combine current data with similar patterns for enhanced prediction
                    rag_enhanced_forecast = self._generate_rag_enhanced_forecast(
                        current_temp, current_humidity, current_season, 
                        similar_patterns, timeframe
                    )
                    
                    # Try AI prediction with historical context
                    try:
                        ai_prediction = self._get_ai_prediction_with_rag_context(
                            location, timeframe, similar_patterns, 
                            current_temp, current_humidity, current_season
                        )
                        
                        return f"""🧠 **RAG-Enhanced Weather Prediction for {location}**
📅 **Prediction Period:** {timeframe} days from {current_date.strftime('%Y-%m-%d')}
🌡️ **Current Conditions:** {current_temp}°C, {current_humidity}% humidity, {current_season}

{ai_prediction}

📈 **Historical Pattern Analysis:**
{rag_enhanced_forecast}

🔬 **RAG Intelligence:** Found {len(similar_patterns)} similar weather patterns from historical data to enhance prediction accuracy."""
                        
                    except Exception as e:
                        logger.error(f"❌ AI prediction failed: {str(e)}")
                        return f"""🧠 **RAG-Enhanced Statistical Forecast for {location}**
📅 **Prediction Period:** {timeframe} days from {current_date.strftime('%Y-%m-%d')}

{rag_enhanced_forecast}

⚠️ **Note:** AI prediction unavailable (quota/error). Using RAG-enhanced statistical analysis."""
                
                else:
                    logger.warning("⚠️ No similar patterns found, using standard prediction")
                    return self.predict_weather(location, timeframe)
            
            else:
                logger.error("❌ No historical data available for RAG enhancement")
                return self.predict_weather(location, timeframe)
                
        except Exception as e:
            logger.error(f"❌ RAG-enhanced prediction failed: {str(e)}")
            return self.predict_weather(location, timeframe)
    
    def _generate_rag_enhanced_forecast(self, current_temp, current_humidity, current_season, similar_patterns, days):
        """Generate forecast based on similar historical patterns"""
        try:
            forecast = "🔮 **RAG-Enhanced Forecast Based on Similar Historical Patterns:**\n\n"
            
            # Analyze similar patterns
            temp_trends = []
            humidity_trends = []
            wind_trends = []
            
            for pattern in similar_patterns:
                # Extract metadata from pattern
                content = pattern.page_content
                metadata = pattern.metadata
                
                temp_trends.append(metadata.get('temperature', current_temp))
                humidity_trends.append(metadata.get('humidity', current_humidity))
                wind_trends.append(metadata.get('wind_speed', 10.0))
            
            # Calculate weighted averages based on similarity
            avg_temp = np.mean(temp_trends) if temp_trends else current_temp
            avg_humidity = np.mean(humidity_trends) if humidity_trends else current_humidity
            avg_wind = np.mean(wind_trends) if wind_trends else 10.0
            
            # Generate day-by-day forecast with pattern-based adjustments
            for day in range(1, days + 1):
                # Apply seasonal and temporal adjustments
                temp_adjustment = np.random.normal(0, 2.0)  # Natural variation
                humidity_adjustment = np.random.normal(0, 5.0)
                wind_adjustment = np.random.normal(0, 2.0)
                
                predicted_temp = avg_temp + temp_adjustment
                predicted_humidity = max(20, min(100, avg_humidity + humidity_adjustment))
                predicted_wind = max(0, avg_wind + wind_adjustment)
                
                # Weather condition based on patterns
                condition = self._determine_condition_from_patterns(
                    predicted_temp, predicted_humidity, current_season
                )
                
                forecast += f"""**Day {day}:**
• Temperature: {predicted_temp:.1f}°C
• Humidity: {predicted_humidity:.1f}%
• Wind Speed: {predicted_wind:.1f} m/s
• Conditions: {condition}
• Pattern Confidence: {"High" if len(similar_patterns) >= 5 else "Moderate"}

"""
            
            return forecast
            
        except Exception as e:
            logger.error(f"❌ Error generating RAG-enhanced forecast: {str(e)}")
            return "RAG forecast generation failed"
    
    def _get_ai_prediction_with_rag_context(self, location, timeframe, similar_patterns, current_temp, current_humidity, current_season):
        """Get AI prediction enhanced with RAG context"""
        try:
            # Prepare context from similar patterns
            pattern_context = "Historical Similar Weather Patterns:\n"
            for i, pattern in enumerate(similar_patterns[:5]):  # Use top 5 patterns
                metadata = pattern.metadata
                pattern_context += f"Pattern {i+1}: {metadata.get('temperature', 'N/A')}°C, {metadata.get('humidity', 'N/A')}% humidity, {metadata.get('season', 'N/A')}\n"
            
            # Enhanced prompt with historical context
            enhanced_prompt = f"""You are a weather prediction expert with access to historical weather patterns.

CURRENT CONDITIONS:
- Location: {location}
- Current Temperature: {current_temp}°C
- Current Humidity: {current_humidity}%
- Season: {current_season}
- Prediction Period: {timeframe} days

{pattern_context}

Based on these similar historical patterns and current conditions, provide a detailed {timeframe}-day weather forecast for {location}. 

Include:
1. Daily temperature, humidity, and wind predictions
2. Weather conditions and precipitation probability
3. How historical patterns influence your predictions
4. Confidence levels based on pattern similarity

Format your response in a clear, structured manner with daily breakdowns."""

            # Get AI prediction
            response = self.llm.invoke(enhanced_prompt)
            return str(response.content) if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ AI prediction with RAG context failed: {str(e)}")
            
            # Check if it's a quota/rate limit error
            if "quota" in str(e).lower() or "429" in str(e) or "rate" in str(e).lower():
                logger.info("🔄 Quota exceeded in RAG prediction - returning fallback message...")
                return f"""**Statistical Forecast for {location} ({timeframe} days)**

⚠️ AI service temporarily unavailable due to quota limits. 
The prediction below is based on historical pattern analysis.

This forecast uses RAG-enhanced statistical modeling based on similar weather patterns 
from historical data. While not AI-generated, it leverages the same historical patterns 
that would inform an AI prediction."""
            else:
                raise e
    
    def _determine_condition_from_patterns(self, temp, humidity, season):
        """Determine weather condition based on temperature, humidity, and season"""
        if humidity > 85:
            return "Rainy" if temp > 10 else "Snow/Rain"
        elif humidity > 70:
            return "Cloudy" if season in ["Winter"] else "Partly Cloudy"
        elif temp > 25 and season in ["Summer"]:
            return "Hot and Sunny"
        elif temp < 5:
            return "Cold and Clear" if humidity < 50 else "Cold and Cloudy"
        else:
            return "Clear" if humidity < 60 else "Partly Cloudy"
    
    def _get_season(self, month):
        """Get season based on month (Japanese seasons)"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Unknown"
    
    # ===== LOCAL LLM PREDICTION METHODS =====
    
    def predict_weather_with_local_llm(self, location="Tokyo", prediction_days=3):
        """Generate weather prediction using local LM Studio LLM"""
        try:
            logger.info(f"🏠 Starting local LLM prediction for {location}, {prediction_days} days...")
            
            if not self.lm_studio or not self.lm_studio.available:
                logger.warning("⚠️ LM Studio not available - using statistical fallback")
                result = self.generate_statistical_prediction(location, prediction_days)
                if result and result.get('success'):
                    result['fallback_used'] = False  # Not a quota issue
                    result['model_used'] = "Statistical Analysis (LM Studio Not Available)"
                    result['note'] = "LM Studio not available - ensure LM Studio is running and model is loaded"
                return result
            
            # Create optimized prompt for local LLM
            prompt_result = self.create_local_llm_prompt(location, prediction_days)
            if not prompt_result:
                logger.error("❌ Failed to create local LLM prompt")
                # Use statistical prediction but mark it as local fallback
                result = self.generate_statistical_prediction(location, prediction_days)
                if result and result.get('success'):
                    result['fallback_used'] = False  # Not a quota issue
                    result['model_used'] = "Statistical Analysis (Local LLM Setup Issue)"
                    result['note'] = "Local LLM setup issue - using statistical analysis"
                return result
            
            # Generate prediction using LM Studio
            prediction = self.lm_studio.generate_weather_prediction(prompt_result)
            
            if prediction:
                logger.info("✅ Local LLM prediction generated successfully")
                return {
                    "prediction": prediction,
                    "location": location,
                    "prediction_days": prediction_days,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": f"Local LLM ({self.lm_studio.model_info.get('id', 'LM Studio')})",
                    "method": "local_llm",
                    "source": "local",
                    "enhancement": "Local AI processing - unlimited usage",
                    "success": True
                }
            else:
                logger.warning("⚠️ Local LLM prediction failed - using statistical fallback")
                result = self.generate_statistical_prediction(location, prediction_days)
                if result and result.get('success'):
                    result['fallback_used'] = False  # Not a quota issue
                    result['model_used'] = "Statistical Analysis (Local LLM Failed)"
                    result['note'] = "Local LLM prediction failed - using statistical analysis"
                return result
                
        except Exception as e:
            logger.error(f"❌ Local LLM prediction error: {str(e)}")
            result = self.generate_statistical_prediction(location, prediction_days)
            if result and result.get('success'):
                result['fallback_used'] = False  # Not a quota issue
                result['model_used'] = "Statistical Analysis (Local LLM Error)"
                result['note'] = f"Local LLM error: {str(e)} - using statistical analysis"
            return result
    
    def predict_weather_with_rag_local_llm(self, location="Japan", timeframe=7):
        """Enhanced weather prediction using RAG + Local LLM"""
        try:
            logger.info(f"🧠 Starting RAG + Local LLM prediction for {location} ({timeframe} days)")
            
            # Check services availability
            if not self.rag_service:
                logger.warning("⚠️ RAG service not available - falling back to local LLM")
                return self.predict_weather_with_local_llm(location, timeframe)
            
            if not self.lm_studio or not self.lm_studio.available:
                logger.warning("⚠️ LM Studio not available - falling back to Gemini RAG")
                return self.predict_weather_with_rag(location, timeframe)
            
            # Get current weather context
            current_date = datetime.now()
            current_season = self._get_season(current_date.month)
            
            # Get latest weather data for similarity search
            if self.data is not None and not self.data.empty:
                latest_data = self.data.tail(1).iloc[0]
                current_temp = float(latest_data['Actual_Temperature(°C)'])
                current_humidity = float(latest_data['Actual_Humidity(%)'])
                
                logger.info(f"📊 Current conditions: {current_temp}°C, {current_humidity}% humidity, {current_season}")
                
                # Retrieve similar weather patterns using RAG
                query = f"temperature {current_temp}°C humidity {current_humidity}% {current_season} season Tokyo Japan"
                similar_patterns = self.rag_service.retrieve_similar_weather(query, k=5)
                
                if similar_patterns:
                    # Create enhanced prompt with RAG context
                    rag_enhanced_prompt = self._create_rag_local_llm_prompt(
                        location, timeframe, current_temp, current_humidity, 
                        current_season, similar_patterns
                    )
                    
                    # Generate prediction using local LLM with RAG context
                    prediction = self.lm_studio.generate_weather_prediction(rag_enhanced_prompt)
                    
                    if prediction:
                        # Add pattern analysis
                        pattern_analysis = self._generate_rag_enhanced_forecast(
                            current_temp, current_humidity, current_season, 
                            similar_patterns, timeframe
                        )
                        
                        combined_prediction = f"""🧠 **RAG-Enhanced Local AI Prediction for {location}**
📅 **Prediction Period:** {timeframe} days from {current_date.strftime('%Y-%m-%d')}
🌡️ **Current Conditions:** {current_temp}°C, {current_humidity}% humidity, {current_season}

**🤖 Local AI Forecast:**
{prediction}

**📈 Historical Pattern Analysis:**
{pattern_analysis}

**🔬 Enhancement Details:**
- Found {len(similar_patterns)} similar weather patterns from historical data
- Processed by local LLM (no quota limitations)
- Combined AI reasoning with historical pattern matching"""

                        return {
                            "prediction": combined_prediction,
                            "location": location,
                            "timeframe": timeframe,
                            "generated_at": datetime.now().isoformat(),
                            "model_used": f"RAG + Local LLM ({self.lm_studio.model_info.get('id', 'LM Studio')})",
                            "method": "rag_local_llm",
                            "source": "local_rag",
                            "enhancement": "RAG + Local AI - best of both worlds",
                            "pattern_count": len(similar_patterns),
                            "success": True
                        }
                    else:
                        logger.warning("⚠️ Local LLM prediction failed - using RAG statistical analysis")
                        return self.predict_weather_with_rag(location, timeframe)
                
                else:
                    logger.warning("⚠️ No similar patterns found - using standard local LLM")
                    return self.predict_weather_with_local_llm(location, timeframe)
            
            else:
                logger.error("❌ No historical data available for RAG enhancement")
                return self.predict_weather_with_local_llm(location, timeframe)
                
        except Exception as e:
            logger.error(f"❌ RAG + Local LLM prediction failed: {str(e)}")
            return self.predict_weather_with_local_llm(location, timeframe)
    
    def predict_weather_hybrid(self, location="Tokyo", prediction_days=3, prefer_local=True):
        """Hybrid prediction with multiple fallback options"""
        try:
            logger.info(f"🔄 Starting hybrid prediction for {location} (prefer_local={prefer_local})")
            
            # Option 1: RAG + Local LLM (best option)
            if prefer_local and self.lm_studio and self.lm_studio.available and self.rag_service:
                logger.info("🧠 Using RAG + Local LLM (optimal)")
                return self.predict_weather_with_rag_local_llm(location, prediction_days)
            
            # Option 2: Local LLM only
            if prefer_local and self.lm_studio and self.lm_studio.available:
                logger.info("🏠 Using Local LLM only")
                return self.predict_weather_with_local_llm(location, prediction_days)
            
            # Option 3: RAG + Gemini (cloud)
            if self.rag_service:
                logger.info("☁️ Using RAG + Gemini (cloud)")
                try:
                    return self.predict_weather_with_rag(location, prediction_days)
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        logger.warning("⚠️ Gemini quota exceeded")
                    else:
                        logger.warning(f"⚠️ RAG + Gemini failed: {e}")
            
            # Option 4: Standard Gemini
            if hasattr(self, 'llm') and self.llm:
                logger.info("🤖 Using standard Gemini")
                try:
                    return self.predict_weather(location, prediction_days)
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        logger.warning("⚠️ Gemini quota exceeded")
                    else:
                        logger.warning(f"⚠️ Standard Gemini failed: {e}")
            
            # Option 5: Statistical fallback (always works)
            logger.info("📊 Using statistical analysis fallback")
            result = self.generate_statistical_prediction(location, prediction_days)
            if result:
                result['method'] = 'hybrid_fallback'
                result['enhancement'] = 'Statistical analysis - all AI services unavailable'
            return result
            
        except Exception as e:
            logger.error(f"❌ Hybrid prediction failed: {str(e)}")
            return self.generate_statistical_prediction(location, prediction_days)
    
    def create_local_llm_prompt(self, location, prediction_days):
        """Create optimized prompt for local LLM"""
        try:
            # Get recent data and statistics (smaller context for local LLM)
            recent_data = self.get_recent_weather_data(days=2)  # Less data for efficiency
            stats_df = self.get_weather_statistics()
            
            if recent_data is None or stats_df is None:
                return None
            
            # Extract statistics from DataFrame
            temp_col = 'Actual_Temperature(°C)'
            humidity_col = 'Actual_Humidity(%)'
            wind_col = 'Actual_WindSpeed(m/s)'
            
            avg_temp = stats_df.loc['mean', temp_col] if temp_col in stats_df.columns else 15.0
            min_temp = stats_df.loc['min', temp_col] if temp_col in stats_df.columns else 5.0
            max_temp = stats_df.loc['max', temp_col] if temp_col in stats_df.columns else 30.0
            avg_humidity = stats_df.loc['mean', humidity_col] if humidity_col in stats_df.columns else 60.0
            
            # Get current season
            current_month = datetime.now().month
            current_season = self._get_season(current_month)
            
            # Create concise prompt optimized for local models
            prompt = f"""**WEATHER FORECAST REQUEST**

Location: {location}
Forecast Period: {prediction_days} days
Current Date: {datetime.now().strftime('%Y-%m-%d')}

**CURRENT CONDITIONS:**"""
            
            if not recent_data.empty:
                latest = recent_data.iloc[-1]
                prompt += f"""
Temperature: {latest['Actual_Temperature(°C)']:.1f}°C
Humidity: {latest['Actual_Humidity(%)']:.1f}%
Wind Speed: {latest['Actual_WindSpeed(m/s)']:.1f} m/s
Rainfall: {latest['Actual_Rainfall(mm)']:.1f} mm
Cloud Cover: {latest['Actual_CloudCover(0-10)']:.1f}/10
"""
            
            # Add statistical context (condensed)
            prompt += f"""
**STATISTICAL CONTEXT:**
Average Temperature: {avg_temp:.1f}°C
Temperature Range: {min_temp:.1f}°C to {max_temp:.1f}°C
Average Humidity: {avg_humidity:.1f}%
Season: {current_season}

**TASK:**
Provide a detailed {prediction_days}-day weather forecast for {location}, including:

1. **Daily Breakdown:** High/low temperatures, conditions, precipitation chance
2. **Weather Patterns:** Expected conditions (sunny, cloudy, rainy, etc.)
3. **Wind & Humidity:** Daily predictions with specific values
4. **Confidence Levels:** How certain you are about each prediction
5. **Notable Changes:** Any significant weather pattern shifts

Format as a clear, structured forecast with daily sections."""

            return prompt
            
        except Exception as e:
            logger.error(f"❌ Error creating local LLM prompt: {str(e)}")
            return None
    
    def _create_rag_local_llm_prompt(self, location, timeframe, current_temp, current_humidity, current_season, similar_patterns):
        """Create enhanced prompt combining RAG context with local LLM optimization"""
        
        # Base prompt
        prompt = f"""**RAG-ENHANCED WEATHER FORECAST**

Location: {location}
Forecast Period: {timeframe} days
Current Conditions: {current_temp}°C, {current_humidity}% humidity, {current_season}

**HISTORICAL SIMILAR PATTERNS:**"""
        
        # Add top 3 similar patterns for context
        for i, pattern in enumerate(similar_patterns[:3]):
            metadata = pattern.metadata
            prompt += f"""
Pattern {i+1}: {metadata.get('temperature', 'N/A')}°C, {metadata.get('humidity', 'N/A')}% humidity
Season: {metadata.get('season', 'N/A')}, Wind: {metadata.get('wind_speed', 'N/A')} m/s"""
        
        prompt += f"""

**TASK:**
Based on current conditions and these {len(similar_patterns)} similar historical patterns, provide a detailed {timeframe}-day forecast including:

1. **Pattern Analysis:** How current conditions relate to historical data
2. **Daily Forecasts:** Temperature, humidity, wind, precipitation for each day
3. **Confidence Assessment:** Reliability based on pattern similarity
4. **Weather Evolution:** How conditions will change over the forecast period

Use the historical patterns to inform your predictions while accounting for seasonal and temporal variations."""

        return prompt
    
    def predict_weather_langchain_rag(self, location="Tokyo", prediction_days=3):
        """Ultimate weather prediction using LangChain + RAG orchestration"""
        try:
            logger.info(f"🧠 Starting LangChain + RAG prediction for {location}, {prediction_days} days...")
            
            if not self.langchain_rag_service or not self.langchain_rag_service.available:
                logger.warning("⚠️ LangChain + RAG not available - falling back to RAG + Local LLM")
                if self.rag_service and self.lm_studio_service and self.lm_studio_service.available:
                    return self.predict_weather_with_rag_local_llm(location, prediction_days)
                elif self.lm_studio_service and self.lm_studio_service.available:
                    return self.predict_weather_with_local_llm(location, prediction_days)
                else:
                    result = self.generate_statistical_prediction(location, prediction_days)
                    if result and result.get('success'):
                        result['fallback_used'] = True
                        result['method'] = "statistical_fallback"
                        result['model_used'] = "Statistical Analysis (LangChain + RAG Unavailable)"
                        result['note'] = "LangChain + RAG service unavailable - using statistical analysis"
                        result['alternatives'] = [
                            "🏠 Local LLM Only (requires LM Studio)",
                            "🧠 RAG + Local LLM (requires LM Studio + API key)",
                            "🔄 Hybrid Smart Fallback"
                        ]
                    return result
            
            # Use LangChain + RAG for the most advanced prediction
            result = self.langchain_rag_service.predict_weather_langchain_rag(location, prediction_days)
            
            if result and result.get('success'):
                logger.info("✅ LangChain + RAG prediction completed successfully")
                # Add additional metadata
                result['timeframe'] = prediction_days  # Ensure consistent naming
                result['advanced_features'] = [
                    "LangChain Orchestration",
                    "RAG Pattern Retrieval", 
                    "Conversation Memory",
                    "Multi-step Reasoning",
                    "Confidence Assessment"
                ]
                return result
            else:
                logger.warning("⚠️ LangChain + RAG prediction failed - using fallback")
                return self.predict_weather_with_rag_local_llm(location, prediction_days)
                
        except Exception as e:
            logger.error(f"❌ LangChain + RAG prediction error: {str(e)}")
            # Intelligent fallback chain
            try:
                if self.rag_service and self.lm_studio:
                    return self.predict_weather_with_rag_local_llm(location, prediction_days)
                elif self.lm_studio:
                    return self.predict_weather_with_local_llm(location, prediction_days)
                else:
                    result = self.generate_statistical_prediction(location, prediction_days)
                    if result and result.get('success'):
                        result['fallback_used'] = False
                        result['model_used'] = "Statistical Analysis (LangChain Error)"
                        result['note'] = f"LangChain + RAG error: {str(e)} - using statistical analysis"
                    return result
            except Exception as fallback_error:
                logger.error(f"❌ All prediction methods failed: {str(fallback_error)}")
                return {
                    "error": "All prediction methods failed",
                    "success": False,
                    "details": f"LangChain error: {str(e)}, Fallback error: {str(fallback_error)}"
                }
    
    def predict_weather_with_langgraph(self, location="Tokyo", prediction_days=3):
        """Advanced weather prediction using LangGraph multi-agent system"""
        try:
            logger.info(f"🧠 Starting LangGraph multi-agent prediction for {location}, {prediction_days} days...")
            
            if not self.langgraph_service or not self.langgraph_service.available:
                logger.warning("⚠️ LangGraph multi-agent service not available - falling back to LangChain + RAG")
                if self.langchain_rag_service and self.langchain_rag_service.available:
                    return self.predict_weather_langchain_rag(location, prediction_days)
                elif self.rag_service and self.lm_studio_service and self.lm_studio_service.available:
                    return self.predict_weather_with_rag_local_llm(location, prediction_days)
                elif self.lm_studio_service and self.lm_studio_service.available:
                    return self.predict_weather_with_local_llm(location, prediction_days)
                else:
                    result = self.generate_statistical_prediction(location, prediction_days)
                    if result and result.get('success'):
                        result['fallback_used'] = True
                        result['method'] = "statistical_fallback"
                        result['model_used'] = "Statistical Analysis (LangGraph Unavailable)"
                        result['note'] = "LangGraph multi-agent service unavailable - using statistical analysis"
                        result['alternatives'] = [
                            "🧠 LangChain + RAG (requires services)",
                            "🏠 Local LLM Only (requires LM Studio)",
                            "📚 RAG + Local LLM (requires LM Studio + API key)"
                        ]
                    return result
            
            # Use LangGraph multi-agent system for the most advanced prediction
            result = self.langgraph_service.predict_weather_with_langgraph(location, prediction_days)
            
            if result and result.get('success'):
                logger.info("✅ LangGraph multi-agent prediction completed successfully")
                # Add additional metadata for LangGraph predictions
                result['timeframe'] = prediction_days  # Ensure consistent naming
                result['advanced_features'] = [
                    "Multi-Agent Architecture",
                    "Intelligent Routing",
                    "Quality Validation", 
                    "Dynamic Fallbacks",
                    "Confidence Assessment",
                    "Specialized Weather Agents"
                ]
                return result
            else:
                logger.warning("⚠️ LangGraph multi-agent prediction failed - using fallback")
                return self.predict_weather_langchain_rag(location, prediction_days)
                
        except Exception as e:
            logger.error(f"❌ LangGraph multi-agent prediction error: {str(e)}")
            # Intelligent fallback chain
            try:
                if self.langchain_rag_service and self.langchain_rag_service.available:
                    return self.predict_weather_langchain_rag(location, prediction_days)
                elif self.rag_service and self.lm_studio:
                    return self.predict_weather_with_rag_local_llm(location, prediction_days)
                elif self.lm_studio:
                    return self.predict_weather_with_local_llm(location, prediction_days)
                else:
                    result = self.generate_statistical_prediction(location, prediction_days)
                    if result and result.get('success'):
                        result['fallback_used'] = True
                        result['model_used'] = "Statistical Analysis (LangGraph Error)"
                        result['note'] = f"LangGraph multi-agent error: {str(e)} - using statistical analysis"
                    return result
            except Exception as fallback_error:
                logger.error(f"❌ All prediction methods failed: {str(fallback_error)}")
                return {
                    "error": "All prediction methods failed",
                    "success": False,
                    "details": f"LangGraph error: {str(e)}, Fallback error: {str(fallback_error)}"
                }
    
    def get_lm_studio_status(self):
        """Get detailed status of LM Studio service"""
        try:
            if not self.lm_studio:
                return {"status": "❌ Not initialized", "available": False}
            
            return self.lm_studio.test_connection()
            
        except Exception as e:
            return {"status": f"❌ Error: {str(e)}", "available": False}
    
    def get_langchain_rag_status(self):
        """Get detailed status of LangChain + RAG service"""
        try:
            if not self.langchain_rag_service:
                return {
                    "status": "❌ Not initialized", 
                    "available": False,
                    "reason": "LangChain + RAG service not initialized",
                    "requirements": [
                        "LM Studio service must be available",
                        "RAG service must be initialized", 
                        "LangChain packages must be installed"
                    ],
                    "alternatives": [
                        "🏠 Local LLM Only",
                        "🧠 RAG + Local LLM",
                        "🔄 Hybrid Smart Fallback"
                    ]
                }
            
            return self.langchain_rag_service.get_service_status()
            
        except Exception as e:
            return {"status": f"❌ Error: {str(e)}", "available": False}
    
    # ------------------------------------------------------------------
    # Week 4: Ensemble prediction
    # ------------------------------------------------------------------

    def predict_weather_ensemble(
        self, location: str = "Tokyo", prediction_days: int = 3,
        season: str = "auto", enable_multiquery: bool = True
    ):
        """Confidence-weighted ensemble prediction combining all available methods."""
        try:
            logger.info(f"🎯 Starting ensemble prediction for {location}, {prediction_days} days")

            if not self.ensemble_service:
                # Lazy initialise if missed during __init__
                if ENSEMBLE_AVAILABLE:
                    self.ensemble_service = EnsemblePredictionService(
                        lm_studio_service=self.lm_studio_service,
                        rag_service=self.rag_service,
                        langgraph_service=self.langgraph_service,
                        langchain_service=self.langchain_rag_service,
                        weather_service=self,
                    )
                else:
                    # Hard fallback: delegate to LangGraph
                    logger.warning("Ensemble service unavailable – falling back to LangGraph")
                    return self.predict_weather_with_langgraph(location, prediction_days)

            result = self.ensemble_service.predict_ensemble(
                location=location,
                prediction_days=prediction_days,
                season=season,
                enable_multiquery=enable_multiquery,
            )

            if result and result.get("success"):
                logger.info("✅ Ensemble prediction completed successfully")
                result["timeframe"] = prediction_days
                result["advanced_features"] = [
                    "Multi-Method Ensemble",
                    "Confidence-Weighted Aggregation",
                    "Qwen3 CoT Meta-Synthesis",
                    "Dynamic Method Selection",
                    "Quality Filtering",
                ]
                return result
            else:
                logger.warning("Ensemble prediction returned failure – falling back to LangGraph")
                return self.predict_weather_with_langgraph(location, prediction_days)

        except Exception as e:
            logger.error(f"❌ Ensemble prediction error: {e}")
            try:
                return self.predict_weather_with_langgraph(location, prediction_days)
            except Exception:
                return self.generate_statistical_prediction(location, prediction_days)

    def get_ensemble_status(self):
        """Get status of the ensemble prediction service."""
        try:
            if not self.ensemble_service:
                return {"status": "Not initialized", "available": False}
            return self.ensemble_service.get_status()
        except Exception as e:
            return {"status": f"Error: {str(e)}", "available": False}

    # ------------------------------------------------------------------

    def get_langgraph_status(self):
        """Get detailed status of LangGraph multi-agent service"""
        try:
            if not self.langgraph_service:
                return {
                    "status": "❌ Not initialized", 
                    "available": False,
                    "reason": "LangGraph multi-agent service not initialized",
                    "requirements": [
                        "LangGraph package must be installed",
                        "Weather service must be available",
                        "At least one AI service (LangChain/LM Studio) recommended"
                    ],
                    "capabilities": [
                        "🤖 Multi-Agent Analysis",
                        "🔀 Intelligent Routing",
                        "✅ Quality Validation",
                        "🔄 Dynamic Fallbacks",
                        "📊 Confidence Assessment"
                    ],
                    "alternatives": [
                        "🧠 LangChain + RAG",
                        "🏠 Local LLM Only",
                        "📚 RAG + Local LLM"
                    ]
                }
            
            return self.langgraph_service.get_langgraph_status()
            
        except Exception as e:
            return {"status": f"❌ Error: {str(e)}", "available": False}
    
    def search_weather_patterns(self, query, k=5):
        """Search for specific weather patterns using natural language"""
        try:
            if not self.rag_service:
                return {"error": "RAG service not available"}
            
            # Use RAG service to search patterns
            results = self.rag_service.search_by_conditions(query, k=k)
            
            formatted_results = []
            for result in results:
                metadata = result.metadata
                formatted_results.append({
                    "content": result.page_content,
                    "temperature": metadata.get('temperature'),
                    "humidity": metadata.get('humidity'),
                    "season": metadata.get('season'),
                    "date": metadata.get('date'),
                    "wind_speed": metadata.get('wind_speed')
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Pattern search failed: {str(e)}")
            return {"error": str(e)}
            
            
        except Exception as e:
            logger.error(f"❌ Data summary error: {str(e)}")
            return None