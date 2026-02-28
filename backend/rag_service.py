import os
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime

# Import RAG dependencies
try:
    from langchain.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    RAG_IMPORTS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ RAG dependencies imported successfully")
except ImportError as e:
    RAG_IMPORTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ RAG imports failed: {e}")
    
    # Create dummy classes to avoid errors
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class Chroma:
        def __init__(self, *args, **kwargs):
            pass
            
    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

# Import local embeddings fallback
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    LOCAL_EMBEDDINGS_AVAILABLE = True
    logger.info("✅ Local embeddings (SentenceTransformers) available")
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    logger.warning("⚠️ SentenceTransformers not available - no local embeddings fallback")

class LocalEmbeddings:
    """Local embeddings fallback using SentenceTransformers"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize local embeddings model"""
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")
        
        try:
            logger.info(f"🔄 Loading local embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("✅ Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load local embedding model: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Local embedding failed for query: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"❌ Local embedding failed for documents: {e}")
            raise

class WeatherRAGService:
    def __init__(self, weather_data_path: str, gemini_api_key: str):
        """Initialize RAG service for weather data"""
        self.weather_data_path = weather_data_path
        self.gemini_api_key = gemini_api_key
        self.embeddings = None
        self.local_embeddings = None
        self.use_local_fallback = False
        self.vector_store = None
        self.retriever = None
        self.vector_db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')
        
        if not RAG_IMPORTS_AVAILABLE:
            logger.warning("⚠️ RAG dependencies not available - service will be disabled")
            return
        
        logger.info("🎯 Initializing Weather RAG Service...")
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG components"""
        try:
            # Try to initialize Google embeddings first
            self._setup_google_embeddings()
            
            # Setup local embeddings as fallback
            self._setup_local_embeddings_fallback()
            
            # Load or create vector store
            self._setup_vector_store()
            
            logger.info("✅ RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
    
    def _setup_google_embeddings(self):
        """Setup Google Generative AI embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key
            )
            logger.info("✅ Google Embeddings initialized")
        except Exception as e:
            logger.warning(f"⚠️ Google embeddings failed: {e}")
            self.embeddings = None
    
    def _setup_local_embeddings_fallback(self):
        """Setup local embeddings as fallback"""
        try:
            if LOCAL_EMBEDDINGS_AVAILABLE:
                self.local_embeddings = LocalEmbeddings()
                logger.info("✅ Local embeddings fallback ready")
            else:
                logger.warning("⚠️ Local embeddings not available")
        except Exception as e:
            logger.warning(f"⚠️ Local embeddings setup failed: {e}")
            self.local_embeddings = None
    
    def _setup_vector_store(self):
        """Setup vector store with fallback handling"""
        try:
            # Use Google embeddings if available
            active_embeddings = self.embeddings
            
            # Fallback to local embeddings if Google fails
            if not active_embeddings and self.local_embeddings:
                active_embeddings = self.local_embeddings
                self.use_local_fallback = True
                logger.info("🔄 Using local embeddings fallback")
            
            if not active_embeddings:
                raise Exception("No embeddings available (Google or local)")

            # Helper: count embeddings stored in the ChromaDB SQLite file.
            def _count_stored_embeddings():
                try:
                    import sqlite3 as _sqlite3
                    db_file = os.path.join(self.vector_db_path, "chroma.sqlite3")
                    if not os.path.exists(db_file):
                        return 0
                    conn = _sqlite3.connect(db_file)
                    count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                    conn.close()
                    return count
                except Exception:
                    return -1  # unknown – don't rebuild

            needs_rebuild = False

            # Check if vector store directory already exists and is non-empty
            if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
                stored_count = _count_stored_embeddings()
                if stored_count == 0:
                    logger.warning("⚠️ Existing vector store is empty – rebuilding with local embeddings")
                    needs_rebuild = True
                    # Delete the empty SQLite file so Chroma creates a fresh one
                    try:
                        import shutil
                        shutil.rmtree(self.vector_db_path)
                    except Exception as _del_err:
                        logger.warning(f"⚠️ Could not remove old vector_db: {_del_err}")
                else:
                    logger.info(f"📂 Loading existing vector store ({stored_count} embeddings)...")
                    self.vector_store = Chroma(
                        persist_directory=self.vector_db_path,
                        embedding_function=active_embeddings
                    )
            else:
                needs_rebuild = True

            if needs_rebuild:
                # When Google embeddings are unavailable, prefer local embeddings
                build_embeddings = active_embeddings
                if active_embeddings is self.embeddings and self.local_embeddings:
                    # Google embeddings may fail during document ingestion; use
                    # local embeddings so the rebuild is guaranteed to succeed.
                    build_embeddings = self.local_embeddings
                    self.use_local_fallback = True
                    logger.info("🔄 Rebuilding vector store using local embeddings")

                logger.info("🔄 Creating new vector store from weather data...")
                documents = self._process_weather_data()

                if not documents:
                    raise Exception("No documents created from weather data")

                os.makedirs(self.vector_db_path, exist_ok=True)
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=build_embeddings,
                    persist_directory=self.vector_db_path
                )

                # Persist the vector store
                try:
                    self.vector_store.persist()
                except Exception:
                    pass  # newer Chroma versions auto-persist
                logger.info(f"💾 Vector store persisted with {len(documents)} documents")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
        except Exception as e:
            logger.error(f"❌ Vector store setup failed: {str(e)}")
            # Allow service to continue without RAG rather than crashing
            self.vector_store = None
            self.retriever = None
    
    def _process_weather_data(self) -> List[Document]:
        """Convert weather CSV data into documents for RAG"""
        try:
            logger.info(f"📊 Processing weather data from: {self.weather_data_path}")
            
            if not os.path.exists(self.weather_data_path):
                logger.error(f"❌ Weather data file not found: {self.weather_data_path}")
                return []
            
            df = pd.read_csv(self.weather_data_path)
            logger.info(f"📈 Loaded {len(df)} records from CSV")
            
            documents = []
            
            # Convert datetime column
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df['Date'] = df['Datetime'].dt.date
            df['Month'] = df['Datetime'].dt.month
            df['Hour'] = df['Datetime'].dt.hour
            
            # Group by date for daily summaries
            daily_groups = df.groupby('Date')
            
            for date, group in daily_groups:
                # Create comprehensive weather summary for each day
                daily_stats = self._create_daily_weather_document(date, group)
                documents.append(daily_stats)
            
            # Also create hourly patterns for more granular retrieval
            hourly_samples = df.sample(min(500, len(df)//4))  # Sample hourly data
            for _, row in hourly_samples.iterrows():
                hourly_doc = self._create_hourly_weather_document(row)
                documents.append(hourly_doc)
            
            # Create seasonal summaries
            seasonal_docs = self._create_seasonal_documents(df)
            documents.extend(seasonal_docs)
            
            logger.info(f"📚 Created {len(documents)} weather documents for RAG")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error processing weather data: {str(e)}")
            return []
    
    def _create_daily_weather_document(self, date, group_data) -> Document:
        """Create a daily weather document"""
        # Calculate daily statistics
        daily_stats = {
            'date': str(date),
            'avg_temp': group_data['Actual_Temperature(°C)'].mean(),
            'min_temp': group_data['Actual_Temperature(°C)'].min(),
            'max_temp': group_data['Actual_Temperature(°C)'].max(),
            'avg_humidity': group_data['Actual_Humidity(%)'].mean(),
            'min_humidity': group_data['Actual_Humidity(%)'].min(),
            'max_humidity': group_data['Actual_Humidity(%)'].max(),
            'avg_wind': group_data['Actual_WindSpeed(m/s)'].mean(),
            'max_wind': group_data['Actual_WindSpeed(m/s)'].max(),
            'total_rainfall': group_data['Actual_Rainfall(mm)'].sum(),
            'avg_solar': group_data['Actual_Solar(kWh/m²/day)'].mean(),
            'avg_cloud_cover': group_data['Actual_CloudCover(0-10)'].mean(),
            'weather_variability': group_data['Weather_Variability_Index'].mean(),
            'season': self._get_season(date.month),
            'day_of_week': group_data['Day_of_Week'].iloc[0],
            'is_weekend': group_data['Is_Weekend'].iloc[0] == 1,
            'records_count': len(group_data)
        }
        
        # Create detailed content
        content = f"""Daily Weather Summary - {daily_stats['date']}

Location: Tokyo, Japan
Season: {daily_stats['season']}
Day: {daily_stats['day_of_week']} {'(Weekend)' if daily_stats['is_weekend'] else '(Weekday)'}

Temperature Analysis:
- Average: {daily_stats['avg_temp']:.1f}°C
- Range: {daily_stats['min_temp']:.1f}°C to {daily_stats['max_temp']:.1f}°C
- Daily variation: {daily_stats['max_temp'] - daily_stats['min_temp']:.1f}°C

Atmospheric Conditions:
- Humidity: {daily_stats['avg_humidity']:.1f}% (range: {daily_stats['min_humidity']:.0f}%-{daily_stats['max_humidity']:.0f}%)
- Wind Speed: {daily_stats['avg_wind']:.1f} m/s (max: {daily_stats['max_wind']:.1f} m/s)
- Cloud Cover: {daily_stats['avg_cloud_cover']:.1f}/10
- Solar Radiation: {daily_stats['avg_solar']:.2f} kWh/m²/day

Precipitation:
- Total Rainfall: {daily_stats['total_rainfall']:.2f} mm
- Weather Stability: {daily_stats['weather_variability']:.3f} (variability index)

Weather Pattern: {'Stable' if daily_stats['weather_variability'] < 5 else 'Variable' if daily_stats['weather_variability'] < 10 else 'Highly Variable'}
Data Quality: {daily_stats['records_count']} hourly measurements"""

        # Enhanced metadata for better retrieval
        metadata = {
            **daily_stats,
            'doc_type': 'daily_summary',
            'temp_category': self._categorize_temperature(daily_stats['avg_temp']),
            'humidity_category': self._categorize_humidity(daily_stats['avg_humidity']),
            'wind_category': self._categorize_wind(daily_stats['avg_wind']),
            'rainfall_category': self._categorize_rainfall(daily_stats['total_rainfall']),
            'weather_pattern': 'stable' if daily_stats['weather_variability'] < 5 else 'variable'
        }

        # Convert numpy/pandas types to plain Python types so Chroma accepts them
        def _to_py(v):
            import numpy as np
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.bool_,)):    return bool(v)
            return v
        metadata = {k: _to_py(v) for k, v in metadata.items()}
        
        return Document(page_content=content, metadata=metadata)
    
    def _create_hourly_weather_document(self, row) -> Document:
        """Create an hourly weather document"""
        content = f"""Hourly Weather Data - {row['Datetime']}

Location: Tokyo, Japan
Time: {pd.to_datetime(row['Datetime']).strftime('%H:%M')} 
Season: {self._get_season(pd.to_datetime(row['Datetime']).month)}

Current Conditions:
- Temperature: {row['Actual_Temperature(°C)']:.1f}°C
- Humidity: {row['Actual_Humidity(%)']:.1f}%
- Wind Speed: {row['Actual_WindSpeed(m/s)']:.1f} m/s
- Rainfall: {row['Actual_Rainfall(mm)']:.2f} mm/hour
- Solar Radiation: {row['Actual_Solar(kWh/m²/day)']:.2f} kWh/m²/day
- Cloud Cover: {row['Actual_CloudCover(0-10)']:.1f}/10
- Weather Variability: {row['Weather_Variability_Index']:.3f}

Plant Data Context:
- Generation Capacity: {row['Gen_Capacity(MW)']} MW
- Plant Volume: {row['Plant_Volume']}
- User Demand: {row['User_Amount']}"""

        metadata = {
            'date': pd.to_datetime(row['Datetime']).date().isoformat(),
            'hour': pd.to_datetime(row['Datetime']).hour,
            'temp': row['Actual_Temperature(°C)'],
            'humidity': row['Actual_Humidity(%)'],
            'wind': row['Actual_WindSpeed(m/s)'],
            'rainfall': row['Actual_Rainfall(mm)'],
            'season': self._get_season(pd.to_datetime(row['Datetime']).month),
            'doc_type': 'hourly_data',
            'temp_category': self._categorize_temperature(row['Actual_Temperature(°C)']),
            'humidity_category': self._categorize_humidity(row['Actual_Humidity(%)']),
            'wind_category': self._categorize_wind(row['Actual_WindSpeed(m/s)']),
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _create_seasonal_documents(self, df) -> List[Document]:
        """Create seasonal weather pattern documents"""
        documents = []
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        df['Season'] = df['Datetime'].dt.month.apply(
            lambda x: seasons[(x % 12) // 3]
        )
        
        for season in seasons:
            season_data = df[df['Season'] == season]
            if len(season_data) == 0:
                continue
            
            # Calculate seasonal statistics
            seasonal_stats = {
                'season': season,
                'avg_temp': season_data['Actual_Temperature(°C)'].mean(),
                'min_temp': season_data['Actual_Temperature(°C)'].min(),
                'max_temp': season_data['Actual_Temperature(°C)'].max(),
                'avg_humidity': season_data['Actual_Humidity(%)'].mean(),
                'avg_wind': season_data['Actual_WindSpeed(m/s)'].mean(),
                'total_rainfall': season_data['Actual_Rainfall(mm)'].sum(),
                'avg_solar': season_data['Actual_Solar(kWh/m²/day)'].mean(),
                'avg_cloud_cover': season_data['Actual_CloudCover(0-10)'].mean(),
                'records_count': len(season_data)
            }
            
            content = f"""Seasonal Weather Pattern - {season}

Location: Tokyo, Japan
Data Period: {seasonal_stats['records_count']} records

Typical {season} Conditions:
- Average Temperature: {seasonal_stats['avg_temp']:.1f}°C
- Temperature Range: {seasonal_stats['min_temp']:.1f}°C to {seasonal_stats['max_temp']:.1f}°C
- Typical Humidity: {seasonal_stats['avg_humidity']:.1f}%
- Average Wind: {seasonal_stats['avg_wind']:.1f} m/s
- Total Precipitation: {seasonal_stats['total_rainfall']:.1f} mm
- Solar Radiation: {seasonal_stats['avg_solar']:.2f} kWh/m²/day
- Cloud Cover: {seasonal_stats['avg_cloud_cover']:.1f}/10

Season Characteristics: {self._get_seasonal_characteristics(season)}"""

            metadata = {
                **seasonal_stats,
                'doc_type': 'seasonal_pattern',
                'temp_category': self._categorize_temperature(seasonal_stats['avg_temp'])
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _get_season(self, month):
        """Determine season based on month"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    
    def _get_seasonal_characteristics(self, season):
        """Get typical characteristics for each season"""
        characteristics = {
            'Winter': 'Cold temperatures, lower humidity, variable wind patterns, minimal rainfall',
            'Spring': 'Mild temperatures, moderate humidity, variable weather conditions, occasional rain',
            'Summer': 'Warm to hot temperatures, higher humidity, potential for thunderstorms, rainy season',
            'Autumn': 'Cooling temperatures, moderate humidity, stable conditions, clear skies'
        }
        return characteristics.get(season, 'Variable seasonal patterns')
    
    def _categorize_temperature(self, temp):
        """Categorize temperature"""
        if temp < 5:
            return 'very_cold'
        elif temp < 15:
            return 'cold'
        elif temp < 25:
            return 'mild'
        elif temp < 30:
            return 'warm'
        else:
            return 'hot'
    
    def _categorize_humidity(self, humidity):
        """Categorize humidity"""
        if humidity < 30:
            return 'very_dry'
        elif humidity < 50:
            return 'dry'
        elif humidity < 70:
            return 'moderate'
        elif humidity < 85:
            return 'humid'
        else:
            return 'very_humid'
    
    def _categorize_wind(self, wind):
        """Categorize wind speed"""
        if wind < 2:
            return 'calm'
        elif wind < 5:
            return 'light'
        elif wind < 10:
            return 'moderate'
        else:
            return 'strong'
    
    def _categorize_rainfall(self, rainfall):
        """Categorize rainfall"""
        if rainfall == 0:
            return 'none'
        elif rainfall < 1:
            return 'light'
        elif rainfall < 5:
            return 'moderate'
        else:
            return 'heavy'
    
    def is_available(self):
        """Check if RAG service is available"""
        return RAG_IMPORTS_AVAILABLE and self.retriever is not None
    
    def retrieve_similar_weather(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve similar weather patterns based on query with fallback handling"""
        try:
            if not self.is_available():
                logger.warning("⚠️ RAG service not available")
                return []
            
            # Enhance query with weather-specific keywords
            enhanced_query = f"weather conditions {query} Tokyo Japan temperature humidity wind rainfall"
            
            # Try primary embeddings first, then fallback
            docs = []
            embedding_error = None
            
            try:
                docs = self.retriever.get_relevant_documents(enhanced_query)
                logger.info(f"✅ Retrieved documents using {'local' if self.use_local_fallback else 'Google'} embeddings")
            except Exception as e:
                embedding_error = str(e)
                logger.warning(f"⚠️ Primary embedding retrieval failed: {e}")
                
                # Try switching to local embeddings if Google failed
                if not self.use_local_fallback and self.local_embeddings:
                    try:
                        logger.info("🔄 Attempting local embeddings fallback...")
                        
                        # Create a new retriever with local embeddings instead of
                        # trying to swap the embedding_function attribute (not
                        # available in newer langchain-community Chroma versions).
                        local_vector_store = Chroma(
                            persist_directory=self.vector_db_path,
                            embedding_function=self.local_embeddings
                        )
                        local_retriever = local_vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": k * 2}
                        )
                        docs = local_retriever.invoke(enhanced_query)
                        
                        logger.info("✅ Local embeddings fallback successful")
                        
                    except Exception as fallback_error:
                        logger.error(f"❌ Local embeddings fallback also failed: {fallback_error}")
                        # Return empty result if both fail
                        raise Exception(f"Both Google and local embeddings failed. Google: {embedding_error}, Local: {fallback_error}")
                else:
                    raise e
            
            results = []
            for doc in docs[:k]:  # Limit results
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "doc_type": doc.metadata.get('doc_type', 'unknown'),
                    "relevance_score": getattr(doc, 'score', 0.0)
                })
            
            logger.info(f"🔍 Retrieved {len(results)} similar weather patterns for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"❌ RAG retrieval failed: {str(e)}")
            return []
    
    def multi_query_retrieval(self, location: str, days: int, season: str = None, k: int = 10, lm_studio_service=None) -> Dict[str, Any]:
        """Enhanced retrieval using multi-query approach with LM Studio
        
        Args:
            location: Location for weather prediction
            days: Number of days to predict  
            season: Optional season filter
            k: Number of documents to retrieve
            lm_studio_service: LM Studio service for query generation
            
        Returns:
            Dict with retrieved documents and query variations
        """
        try:
            if not self.is_available():
                logger.warning("⚠️ RAG service not available")
                return {"documents": [], "query_variations": [], "total_retrieved": 0}
            
            logger.info(f"🔍 Multi-query retrieval for {location} ({days} days)")
            
            # Generate query variations
            query_variations = self._generate_query_variations_sync(location, days, season, lm_studio_service)
            logger.info(f"📝 Generated {len(query_variations)} query variations")
            
            # Retrieve documents for each query variation
            all_results = []
            seen_content = set()
            
            for i, query in enumerate(query_variations):
                try:
                    # Use existing retrieve_similar_weather method
                    results = self.retrieve_similar_weather(query, k=max(3, k // len(query_variations)))
                    
                    # Deduplicate based on content hash
                    for result in results:
                        content_hash = hash(result['content'][:200])  # Hash first 200 chars
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            result['source_query'] = query
                            result['query_index'] = i
                            all_results.append(result)
                            
                    logger.info(f"✅ Query {i+1}/{len(query_variations)}: Retrieved {len(results)} docs")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Query {i+1} failed: {e}")
                    continue
            
            # Sort by relevance score (if available)
            all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit final results
            final_results = all_results[:k]
            
            logger.info(f"📊 Multi-query retrieval complete: {len(final_results)} unique documents from {len(all_results)} total")
            
            return {
                "documents": final_results,
                "query_variations": query_variations,
                "total_retrieved": len(all_results),
                "final_count": len(final_results),
                "deduplication": len(all_results) - len(final_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-query retrieval failed: {e}")
            return {"documents": [], "query_variations": [], "total_retrieved": 0}
    
    def _generate_query_variations_sync(self, location: str, days: int, season: str = None, lm_studio_service=None) -> List[str]:
        """Generate query variations for better retrieval coverage
        
        Args:
            location: Location name
            days: Number of days
            season: Optional season
            lm_studio_service: LM Studio service for AI-powered generation
            
        Returns:
            List of query strings
        """
        # Base query
        base_query = f"weather prediction {location} {days} days"
        
        # Try using LM Studio for intelligent query generation
        if lm_studio_service and lm_studio_service.available:
            try:
                logger.info("🤖 Using LM Studio for query generation")
                
                prompt = f"""Generate 5 different search queries to find relevant historical weather patterns for:
- Location: {location}
- Forecast period: {days} days
- Season: {season or 'any'}

Each query should focus on different aspects:
1. General weather patterns
2. Temperature and humidity trends
3. Seasonal characteristics
4. Extreme weather events
5. Atmospheric conditions

Return ONLY the queries, one per line, without numbering."""

                response = lm_studio_service.generate_text(prompt, max_tokens=300)
                
                # Extract queries from response  
                if response:
                    generated_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
                    if len(generated_queries) >= 3:  # Use if we got enough queries
                        logger.info(f"✅ Generated {len(generated_queries)} queries using LM Studio")
                        return generated_queries[:6]  # Limit to 6
                        
            except Exception as e:
                logger.warning(f"⚠️ LM Studio query generation failed: {e}, falling back to template-based")
        
        # Fallback: Template-based query variations
        logger.info("📋 Using template-based query generation")
        
        season_part = f"{season} season " if season else ""
        
        queries = [
            f"{base_query}",
            f"{season_part}weather patterns {location} temperature humidity",
            f"historical weather {location} {season_part}conditions",
            f"{location} weather forecast {days} days {season_part}trends",
            f"meteorological data {location} {season_part}precipitation wind"
        ]
        
        # Add season-specific queries if provided
        if season:
            queries.append(f"{season} weather characteristics {location}")
        
        return queries
    
    def search_by_conditions(self, temp_range=None, humidity_range=None, season=None, k=5):
        """Search for weather patterns by specific conditions"""
        try:
            if not self.vector_store:
                return []
            
            # Build filter conditions
            filter_conditions = {}
            
            if season:
                filter_conditions['season'] = season
            
            # Create query based on conditions
            query_parts = []
            if temp_range:
                min_temp, max_temp = temp_range
                query_parts.append(f"temperature between {min_temp}°C and {max_temp}°C")
            
            if humidity_range:
                min_hum, max_hum = humidity_range
                query_parts.append(f"humidity between {min_hum}% and {max_hum}%")
            
            if season:
                query_parts.append(f"{season} season")
            
            query = " ".join(query_parts) if query_parts else "weather conditions"
            
            # Perform search
            docs = self.retriever.get_relevant_documents(query)
            
            # Filter results based on conditions
            filtered_results = []
            for doc in docs:
                metadata = doc.metadata
                
                # Apply temperature filter
                if temp_range and 'avg_temp' in metadata:
                    if not (temp_range[0] <= metadata['avg_temp'] <= temp_range[1]):
                        continue
                
                # Apply humidity filter  
                if humidity_range and 'avg_humidity' in metadata:
                    if not (humidity_range[0] <= metadata['avg_humidity'] <= humidity_range[1]):
                        continue
                
                filtered_results.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "doc_type": metadata.get('doc_type', 'unknown')
                })
                
                if len(filtered_results) >= k:
                    break
            
            logger.info(f"🎯 Found {len(filtered_results)} matching weather patterns")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Conditional search failed: {str(e)}")
            return []
    
    def is_available(self):
        """Check if RAG service is available"""
        return self.retriever is not None