from flask import Blueprint, request, jsonify, render_template, session
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_wtf.csrf import CSRFProtect
from backend.models import db, User, Image
import json
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Production AI/ML infrastructure
# ---------------------------------------------------------------------------
try:
    from backend.auth_guard     import require_auth, get_session_user_id
    from backend.rate_limiter   import rate_limiter
    from backend.circuit_breaker import circuit_breakers
    from backend.prompt_security import security_guard
    from backend.ml_observability import observability
    try:
        from backend.validators import WeatherPredictionRequest, ValidationError, PYDANTIC_AVAILABLE
    except Exception:
        PYDANTIC_AVAILABLE = False
        WeatherPredictionRequest = None
        ValidationError = Exception
    PRODUCTION_GUARDS = True
except Exception as _prod_err:
    PRODUCTION_GUARDS = False
    logger_boot = logging.getLogger(__name__)
    logger_boot.warning("Production guards not loaded: %s", _prod_err)
    # Stub require_auth so the app still boots
    def require_auth(f):
        from functools import wraps
        @wraps(f)
        def _inner(*a, **kw):
            from flask import session, jsonify
            if 'user_id' not in session:
                return jsonify({'error': 'Authentication required'}), 401
            return f(*a, **kw)
        return _inner
    def get_session_user_id():
        from flask import session
        return session.get('user_id')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__)

# Initialize CSRF protection
csrf = CSRFProtect()

# Initialize weather service and WebSocket integration
weather_service = None
websocket_service = None

try:
    from backend.weather_service import WeatherPredictionService
    from backend.websocket_service import websocket_service as ws_service
    
    weather_service = WeatherPredictionService()
    websocket_service = ws_service
    print("✅ Weather prediction service initialized successfully")
    print("✅ WebSocket service initialized successfully")
    
    # Initialize weather service with WebSocket support if LangGraph is available
    if hasattr(weather_service, 'langgraph_service') and weather_service.langgraph_service:
        weather_service.langgraph_service.websocket_service = websocket_service
        print("✅ LangGraph service connected to WebSocket")
        
except Exception as e:
    print(f"❌ Failed to initialize services: {e}")
    print(f"   Error details: {type(e).__name__}: {str(e)}")
    logger.error(f"Service initialization failed: {str(e)}")
    weather_service = None
    websocket_service = None

@api.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users (protected route)"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@api.route('/users/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@api.route('/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    """Update a user"""
    current_user_id = get_jwt_identity()
    user = User.query.get_or_404(user_id)
    
    # Only allow users to update their own profile or admin functionality
    # For now, allow any authenticated user to update any user
    data = request.get_json()
    
    if 'username' in data:
        # Check if username is already taken by another user
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Username already exists'}), 400
        user.username = data['username']
    
    if 'email' in data:
        # Check if email is already taken by another user
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Email already exists'}), 400
        user.email = data['email']
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error updating user'}), 500

@api.route('/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """Delete a user and all associated images"""
    user = User.query.get_or_404(user_id)
    
    try:
        # Associated images will be deleted automatically due to cascade
        db.session.delete(user)
        db.session.commit()
        return jsonify({'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting user'}), 500

@api.route('/images', methods=['GET'])
@jwt_required()
def get_images():
    """Get all images (protected route)"""
    images = Image.query.all()
    return jsonify([image.to_dict() for image in images])

@api.route('/images', methods=['POST'])
@jwt_required()
def create_image():
    """Create a new image"""
    data = request.get_json()
    
    if not data or 'title' not in data or 'url' not in data or 'user_id' not in data:
        return jsonify({'message': 'Title, URL, and user_id are required'}), 400
    
    # Check if user exists
    if not User.query.get(data['user_id']):
        return jsonify({'message': 'User does not exist'}), 400
    
    new_image = Image(
        title=data['title'],
        url=data['url'],
        user_id=data['user_id']
    )
    
    try:
        db.session.add(new_image)
        db.session.commit()
        return jsonify({
            'id': new_image.id,
            'title': new_image.title,
            'url': new_image.url,
            'user_id': new_image.user_id,
            'message': 'Image created successfully'
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating image'}), 500

@api.route('/images/<int:image_id>', methods=['DELETE'])
@jwt_required()
def delete_image(image_id):
    """Delete an image"""
    image = Image.query.get_or_404(image_id)
    
    try:
        db.session.delete(image)
        db.session.commit()
        return jsonify({'message': 'Image deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting image'}), 500

@api.route('/users/<int:user_id>/images', methods=['GET'])
@jwt_required()
def get_user_images(user_id):
    """Get all images for a specific user"""
    user = User.query.get_or_404(user_id)
    images = Image.query.filter_by(user_id=user_id).all()
    return jsonify([image.to_dict() for image in images])

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (public)"""
    return jsonify({'status': 'healthy', 'message': 'Flask API is running'})

@api.route('/weather/test', methods=['GET', 'POST'])
def test_weather_api():
    """Test endpoint to verify API connectivity"""
    try:
        logger.info("🧪 Weather API test endpoint called")
        
        # Check session authentication
        from flask import session
        if 'user_id' not in session:
            logger.warning("❌ Unauthorized test request")
            return jsonify({'error': 'Authentication required', 'endpoint': 'test'}), 401
        
        # Check if service is available
        service_status = "available" if weather_service is not None else "unavailable"
        
        # Get request info
        method = request.method
        data = None
        
        if method == 'POST':
            if request.is_json:
                data = request.get_json()
                logger.info(f"📊 Received JSON: {data}")
            else:
                data = request.form.to_dict()
                logger.info(f"📊 Received form: {data}")
        else:
            data = request.args.to_dict()
            logger.info(f"📊 Received query: {data}")
        
        logger.info("✅ Weather API test completed successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'Weather API endpoint is working',
            'method': method,
            'data_received': data,
            'weather_service': service_status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Weather API test failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'API test failed: {str(e)}',
            'endpoint': 'test'
        }), 500

@api.route('/weather/test', methods=['GET'])
def test_weather_service():
    """Test weather service functionality"""
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available',
            'status': 'failed'
        })
    
    try:
        # Test basic functionality
        summary = weather_service.get_data_summary()
        recent_data = weather_service.get_recent_weather_data(3)
        
        return jsonify({
            'status': 'success',
            'message': 'Weather service is working',
            'data_available': summary is not None,
            'recent_data_available': recent_data is not None,
            'records_count': len(weather_service.data) if weather_service.data is not None else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Weather service test failed: {str(e)}'
        }), 500

@api.route('/weather/predict', methods=['GET', 'POST'])
@require_auth
def predict_weather():
    """Weather prediction endpoint using LangChain and Gemini AI"""
    try:
        logger.info("🔮 Weather prediction request received")
        
        if weather_service is None:
            logger.error("❌ Weather service not available")
            return jsonify({
                'error': 'Weather service not available',
                'message': 'Please check Gemini API key configuration'
            }), 500
        
        # Get parameters from request
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json() or {}
                logger.info(f"📊 Received JSON data: {data}")
            else:
                data = request.form.to_dict()
                logger.info(f"📊 Received form data: {data}")
        else:
            data = request.args.to_dict()
            logger.info(f"📊 Received query params: {data}")
        
        location = data.get('location', 'Tokyo')
        prediction_days = int(data.get('prediction_days', 3))
        
        logger.info(f"🌍 Predicting weather for {location}, {prediction_days} days")
        
        # Validate input
        if prediction_days < 1 or prediction_days > 10:
            logger.warning(f"❌ Invalid prediction days: {prediction_days}")
            return jsonify({
                'error': 'Invalid prediction days',
                'message': 'Prediction days must be between 1 and 10'
            }), 400
        
        # Generate prediction
        logger.info("🤖 Calling weather service for prediction...")
        result = weather_service.predict_weather(location, prediction_days)
        
        logger.info(f"📈 Prediction result status: {result.get('success', False)}")
        
        if result.get('success'):
            logger.info("✅ Weather prediction completed successfully")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Weather prediction failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 500
            
    except ValueError as e:
        logger.error(f"❌ ValueError in predict_weather: {e}")
        return jsonify({
            'error': 'Invalid input data',
            'message': str(e),
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"❌ Exception in predict_weather: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'success': False
        }), 500

@api.route('/weather/data-summary', methods=['GET'])
def get_weather_data_summary():
    """Get summary of the weather dataset"""
    # Check session-based authentication
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available'
        }), 500
    
    try:
        print("Getting weather data summary...")  # Debug log
        summary = weather_service.get_data_summary()
        print(f"Summary result: {summary}")  # Debug log
        
        if summary:
            return jsonify(summary)
        else:
            return jsonify({
                'error': 'Unable to load data summary'
            }), 500
    except Exception as e:
        print(f"Exception in get_weather_data_summary: {e}")
        return jsonify({
            'error': 'Failed to get data summary',
            'message': str(e)
        }), 500

@api.route('/weather/recent-data', methods=['GET'])
def get_recent_weather_data():
    """Get recent weather data for visualization"""
    # Check session-based authentication
    from flask import session
    if 'user_id' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    if weather_service is None:
        return jsonify({
            'error': 'Weather service not available'
        }), 500
    
    try:
        days = int(request.args.get('days', 7))
        recent_data = weather_service.get_recent_weather_data(days)
        
        if recent_data is not None:
            # Convert to JSON-serializable format
            result = {
                'data': recent_data.to_dict('records'),
                'dates': [str(date) for date in recent_data.index],
                'days_requested': days
            }
            return jsonify(result)
        else:
            return jsonify({
                'error': 'Unable to load recent data'
            }), 500
    except Exception as e:
        return jsonify({
            'error': 'Failed to get recent data',
            'message': str(e)
        }), 500

# ===================== RAG-ENHANCED WEATHER ENDPOINTS =====================

@api.route('/weather/predict-rag', methods=['POST'])
@csrf.exempt
@require_auth
def predict_weather_rag():
    """Enhanced weather prediction using RAG + historical patterns"""
    try:
        logger.info("🧠 RAG-enhanced weather prediction request received")
        
        if not weather_service:
            return jsonify({'error': 'Weather service not available'}), 500
        
        # Get request data
        data = request.get_json() if request.is_json else {}
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 7))
        
        # Validate timeframe
        if timeframe < 1 or timeframe > 14:
            return jsonify({'error': 'Timeframe must be between 1 and 14 days'}), 400
        
        logger.info(f"🔍 Starting RAG prediction: {location}, {timeframe} days")
        
        # Generate RAG-enhanced prediction
        prediction = weather_service.predict_weather_with_rag(location, timeframe)
        
        return jsonify({
            'prediction': prediction,
            'location': location,
            'timeframe': timeframe,
            'method': 'RAG-Enhanced',
            'timestamp': datetime.now().isoformat(),
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG prediction error: {str(e)}")
        return jsonify({'error': f'RAG prediction failed: {str(e)}'}), 500

@api.route('/weather/rag-search', methods=['POST'])
@csrf.exempt
def rag_search_patterns():
    """Search historical weather patterns using RAG"""
    try:
        logger.info("🔍 RAG weather pattern search request received")
        
        # Check authentication
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not weather_service or not weather_service.rag_service:
            return jsonify({'error': 'RAG service not available'}), 500
        
        # Get request data
        data = request.get_json() if request.is_json else {}
        query = data.get('query', '')
        limit = int(data.get('limit', 5))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        if limit > 20:
            limit = 20  # Cap results
        
        logger.info(f"🔍 RAG searching patterns: '{query}', limit: {limit}")
        
        # Search using RAG service
        results = weather_service.rag_service.retrieve_similar_weather(query, k=limit)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.get('content', ''),
                'metadata': result.get('metadata', {}),
                'doc_type': result.get('doc_type', 'unknown'),
                'relevance_score': result.get('relevance_score', 0.0)
            })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG search error: {str(e)}")
        return jsonify({'error': f'RAG search failed: {str(e)}'}), 500

@api.route('/weather/rag-stats', methods=['GET'])
def get_rag_stats():
    """Get RAG service statistics and status"""
    try:
        logger.info("📊 RAG statistics request received")
        
        # Check authentication
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not weather_service:
            return jsonify({'error': 'Weather service not available'}), 500
        
        # Get RAG service stats
        if weather_service.rag_service:
            rag_stats = weather_service.rag_service.get_stats()
            rag_available = weather_service.rag_service.is_available()
            
            return jsonify({
                'success': True,
                'rag_available': rag_available,
                'stats': rag_stats,
                'service_status': 'active' if rag_available else 'inactive',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'rag_available': False,
                'stats': {},
                'service_status': 'not_initialized',
                'error': 'RAG service not initialized',
                'timestamp': datetime.now().isoformat()
            }), 200
        
    except Exception as e:
        logger.error(f"❌ RAG stats error: {str(e)}")
        return jsonify({'error': f'RAG stats failed: {str(e)}'}), 500

# ===== LOCAL LLM ENDPOINTS =====

@api.route('/weather/predict-local', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
@require_auth
def predict_weather_local():
    """Local LLM weather prediction endpoint"""
    try:
        logger.info("🏠 Local LLM weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        logger.info(f"📊 Received JSON data: {data}")
        
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🏠 Predicting weather for {location}, {timeframe} days using local LLM")
        logger.info("🤖 Calling weather service for local LLM prediction...")
        
        # Get local LLM prediction
        result = weather_service.predict_weather_with_local_llm(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ Local LLM weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Local LLM weather prediction failed: {result}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Local LLM prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/predict-rag-local', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
@require_auth
def predict_weather_rag_local():
    """RAG + Local LLM weather prediction endpoint"""
    try:
        logger.info("🧠 RAG + Local LLM weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 7))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🧠 Predicting weather for {location}, {timeframe} days using RAG + Local LLM")
        
        # Get RAG + Local LLM prediction
        result = weather_service.predict_weather_with_rag_local_llm(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ RAG + Local LLM weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ RAG + Local LLM prediction failed")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ RAG + Local LLM prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/predict-hybrid', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
@require_auth
def predict_weather_hybrid():
    """Hybrid prediction endpoint with intelligent fallback"""
    try:
        logger.info("🔄 Hybrid weather prediction request received")
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        prefer_local = data.get('prefer_local', True)
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🔄 Hybrid prediction for {location}, {timeframe} days (prefer_local={prefer_local})")
        
        # Get hybrid prediction
        result = weather_service.predict_weather_hybrid(location, timeframe, prefer_local)
        
        if result and result.get('success'):
            logger.info("✅ Hybrid weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ Hybrid prediction failed")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"❌ Hybrid prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/lm-studio-status', methods=['GET'])
def get_lm_studio_status():
    """Get LM Studio service status and connection info"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📊 LM Studio status request received")
        
        # Get LM Studio status
        status = weather_service.get_lm_studio_status()
        
        return jsonify({
            'success': True,
            'lm_studio_status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ LM Studio status error: {str(e)}")
        return jsonify({
            'error': f'LM Studio status check failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/predict-multiquery-rag', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
@require_auth
def predict_weather_multiquery_rag():
    """Weather prediction using enhanced Multi-Query RAG with AI-powered query generation"""
    try:
        logger.info("🔍 Multi-Query RAG weather prediction request received")
        
        # Check if services are available
        if weather_service is None:
            logger.error("❌ Weather service not initialized")
            return jsonify({
                "error": "Weather service not available",
                "success": False
            }), 500
        
        if not weather_service.rag_service or not weather_service.rag_service.is_available():
            logger.error("❌ RAG service not available")
            return jsonify({
                "error": "RAG service not available",
                "success": False
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        location = data.get('location', 'Tokyo')
        days = int(data.get('days', 3))
        season = data.get('season', None)
        k = int(data.get('k', 10))
        
        # Validate inputs
        if days < 1 or days > 14:
            return jsonify({"error": "Days must be between 1 and 14"}), 400
        
        if k < 5 or k > 20:
            k = 10  # Default to 10
        
        logger.info(f"🔍 Multi-Query RAG for {location}, {days} days, season={season}, k={k}")
        
        # Get LM Studio service for AI query generation
        lm_studio = weather_service.lm_studio_service if weather_service.lm_studio_service and weather_service.lm_studio_service.available else None
        
        if lm_studio:
            logger.info(f"✅ Using LM Studio ({lm_studio.model_name}) for AI query generation")
        else:
            logger.info("⚠️ LM Studio not available - using template-based query generation")
        
        # Perform multi-query retrieval
        rag_result = weather_service.rag_service.multi_query_retrieval(
            location=location,
            days=days,
            season=season,
            k=k,
            lm_studio_service=lm_studio
        )
        
        # Check if we got results
        if not rag_result['documents']:
            logger.warning("⚠️ No documents retrieved from multi-query RAG")
            return jsonify({
                "error": "No relevant historical data found",
                "query_variations": rag_result['query_variations'],
                "success": False
            }), 404
        
        # Generate prediction using LM Studio if available
        prediction_text = ""
        method_used = "template-based"
        
        if lm_studio:
            try:
                # Build context from top documents
                context = "\n\n".join([
                    f"Document {i+1}:\n{doc['content']}" 
                    for i, doc in enumerate(rag_result['documents'][:5])
                ])
                
                # Create prediction prompt
                prompt = f"""Based on historical weather data for {location}, predict the weather for the next {days} days.

Historical Context from Multiple Retrieval Queries:
{context}

Season: {season or 'current season'}

Provide a detailed {days}-day forecast including:
1. Overall weather pattern and trends
2. Temperature range (high/low)
3. Precipitation probability
4. Humidity levels
5. Wind conditions
6. Any notable weather patterns or warnings

Format your response as a clear, day-by-day forecast."""

                logger.info(f"🤖 Generating prediction with {lm_studio.model_name}")
                prediction_response = lm_studio.generate_chat(
                    messages=[
                        {"role": "system", "content": "You are an expert meteorologist providing accurate, detailed weather forecasts based on historical patterns."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                if prediction_response:
                    prediction_text = prediction_response
                    method_used = f"ai-powered ({lm_studio.model_name})"
                    logger.info(f"✅ AI prediction generated: {len(prediction_text)} characters")
                else:
                    logger.warning("⚠️ AI prediction returned empty")
                    prediction_text = "AI prediction unavailable - see retrieved context below"
                    method_used = "context-only"
                    
            except Exception as e:
                logger.error(f"❌ AI prediction generation failed: {e}")
                prediction_text = f"AI prediction error: {str(e)}"
                method_used = "error-fallback"
        else:
            # No LM Studio - provide summary of retrieved context
            prediction_text = f"Retrieved {len(rag_result['documents'])} relevant historical weather patterns for {location}. See context below for details."
            method_used = "context-only"
        
        # Build response
        response = {
            'success': True,
            'location': location,
            'days': days,
            'season': season,
            'prediction': prediction_text,
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
                    'source_query': doc.get('source_query', 'N/A')[:60] + '...' if doc.get('source_query', '') else 'N/A'
                }
                for doc in rag_result['documents'][:3]
            ],
            'method': f'multi-query-rag-{method_used}',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Multi-Query RAG prediction successful: {method_used}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Multi-Query RAG prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Multi-Query RAG prediction failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/predict-langchain-rag', methods=['POST'])
@csrf.exempt  # Exempt from CSRF for API
@require_auth
def predict_weather_langchain_rag():
    """Ultimate weather prediction using LangChain + RAG orchestration"""
    try:
        logger.info("🧠 LangChain + RAG weather prediction request received")
        
        # Check if weather service is available
        if weather_service is None:
            logger.error("❌ Weather service not initialized")
            return jsonify({
                "error": "Weather service not available. Please check server configuration.",
                "suggestion": "The service requires proper API keys. Try using local-only methods.",
                "alternatives": [
                    "Local LLM Only (if LM Studio is running)",
                    "Statistical Analysis (always available)"
                ],
                "success": False
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        location = data.get('location', 'Tokyo')
        timeframe = int(data.get('timeframe', 3))
        
        # Validate inputs
        if timeframe < 1 or timeframe > 10:
            return jsonify({"error": "Timeframe must be between 1 and 10 days"}), 400
            
        logger.info(f"🧠 LangChain + RAG prediction for {location}, {timeframe} days")
        
        # Check if the method exists on the weather service
        if not hasattr(weather_service, 'predict_weather_langchain_rag'):
            logger.error("❌ LangChain + RAG method not available on weather service")
            return jsonify({
                "error": "LangChain + RAG method not available",
                "suggestion": "Service may not be fully initialized. Try another prediction method.",
                "alternatives": [
                    "🏠 Local LLM Only",
                    "🧠 RAG + Local LLM", 
                    "🔄 Hybrid Smart Fallback"
                ],
                "success": False
            }), 503
        
        # Get LangChain + RAG prediction
        result = weather_service.predict_weather_langchain_rag(location, timeframe)
        
        if result and result.get('success'):
            logger.info("✅ LangChain + RAG weather prediction successful")
            return jsonify(result), 200
        else:
            logger.error(f"❌ LangChain + RAG prediction failed")
            
            # Determine error type for better frontend handling
            error_response = result or {"error": "Prediction failed", "success": False}
            
            # Check for timeout conditions
            if result and (
                result.get('timeout_occurred') or
                (result.get('error') and 'timeout' in str(result.get('error')).lower()) or
                (result.get('note') and 'taking longer than expected' in str(result.get('note')).lower())
            ):
                error_response['error_type'] = 'timeout'
                if not error_response.get('error'):
                    error_response['error'] = 'Prediction is taking longer than expected'
            
            # Check for service unavailable conditions
            elif result and (
                'not available' in str(result.get('error', '')).lower() or
                'service unavailable' in str(result.get('error', '')).lower() or
                'not properly initialized' in str(result.get('error', '')).lower()
            ):
                error_response['error_type'] = 'service_unavailable'
                if not error_response.get('error'):
                    error_response['error'] = 'AI service is not currently available'
            
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"❌ LangChain + RAG prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/langchain-rag-status', methods=['GET'])
def get_langchain_rag_status():
    """Get LangChain + RAG service status and capabilities"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📊 LangChain + RAG status request received")
        
        # Get LangChain + RAG status
        status = weather_service.get_langchain_rag_status()
        
        return jsonify({
            'success': True,
            'langchain_rag_status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ LangChain + RAG status error: {str(e)}")
        return jsonify({
            'error': f'LangChain + RAG status check failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/service-overview', methods=['GET'])
def get_service_overview():
    """Get comprehensive overview of all weather prediction services"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📈 Service overview request received")
        
        overview = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        # Check Gemini service
        overview['services']['gemini'] = {
            'name': 'Google Gemini AI',
            'available': hasattr(weather_service, 'llm') and weather_service.llm is not None,
            'type': 'cloud',
            'quota_limited': True,
            'endpoints': ['/weather/predict']
        }
        
        # Check RAG service
        rag_available = weather_service.rag_service is not None
        overview['services']['rag'] = {
            'name': 'RAG (Retrieval Augmented Generation)',
            'available': rag_available,
            'type': 'enhancement',
            'quota_limited': True,  # Because it uses Gemini embeddings
            'endpoints': ['/weather/predict-rag', '/weather/rag-search']
        }
        
        # Check LM Studio service
        lm_studio_available = (
            hasattr(weather_service, 'lm_studio') and 
            weather_service.lm_studio is not None and 
            weather_service.lm_studio.available
        )
        overview['services']['lm_studio'] = {
            'name': 'LM Studio Local LLM',
            'available': lm_studio_available,
            'type': 'local',
            'quota_limited': False,
            'endpoints': ['/weather/predict-local', '/weather/predict-rag-local']
        }
        
        # Check hybrid service
        overview['services']['hybrid'] = {
            'name': 'Hybrid Intelligent Fallback',
            'available': True,  # Always available with statistical fallback
            'type': 'intelligent',
            'quota_limited': False,
            'endpoints': ['/weather/predict-hybrid'],
            'fallback_chain': ['RAG + Local LLM', 'Local LLM', 'RAG + Gemini', 'Standard Gemini', 'Statistical Analysis']
        }
        
        # Check statistical fallback
        overview['services']['statistical'] = {
            'name': 'Statistical Analysis',
            'available': True,
            'type': 'fallback',
            'quota_limited': False,
            'endpoints': ['Embedded in all prediction methods']
        }
        
        # Summary
        available_services = sum(1 for service in overview['services'].values() if service['available'])
        quota_free_services = sum(1 for service in overview['services'].values() if service['available'] and not service['quota_limited'])
        
        overview['summary'] = {
            'total_services': len(overview['services']),
            'available_services': available_services,
            'quota_free_services': quota_free_services,
            'recommended_endpoint': '/weather/predict-hybrid',
            'status': '✅ Operational' if available_services >= 2 else '⚠️ Limited'
        }
        
        return jsonify({
            'success': True,
            'overview': overview
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Service overview error: {str(e)}")
        return jsonify({
            'error': f'Service overview failed: {str(e)}',
            'success': False
        }), 500

@api.route('/weather/predict-langgraph', methods=['POST'])
@require_auth
def predict_weather_langgraph():
    """Ultimate weather prediction using LangGraph multi-agent system"""
    try:
        logger.info("🧠 LangGraph multi-agent weather prediction request received")
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "success": False
            }), 400
        
        location = data.get('location', 'Tokyo')
        timeframe = data.get('timeframe', 3)
        
        # Validate inputs
        if not isinstance(timeframe, int) or timeframe < 1 or timeframe > 14:
            return jsonify({
                "error": "Invalid timeframe. Must be between 1 and 14 days",
                "success": False
            }), 400
        
        logger.info(f"🧠 LangGraph multi-agent prediction for {location}, {timeframe} days")
        
        # Check if LangGraph method is available
        if not hasattr(weather_service, 'predict_weather_with_langgraph'):
            logger.error("❌ LangGraph multi-agent method not available on weather service")
            return jsonify({
                "error": "LangGraph multi-agent method not available",
                "success": False,
                "alternatives": [
                    "🧠 LangChain + RAG: /weather/predict-langchain-rag",
                    "🏠 Local LLM: /weather/predict-local",
                    "📚 RAG + Local LLM: /weather/predict-rag-local"
                ]
            }), 503
        
        # Get LangGraph multi-agent prediction with route-level timeout
        import concurrent.futures as _cf
        _timeout_secs = 420  # Route-level cap: scaled for up to 14-day forecasts
        _pool = _cf.ThreadPoolExecutor(max_workers=1)
        _future = _pool.submit(weather_service.predict_weather_with_langgraph, location, timeframe)
        _pool.shutdown(wait=False)  # Don't block on cleanup
        try:
            result = _future.result(timeout=_timeout_secs)
        except _cf.TimeoutError:
            logger.warning(f"⏱️ LangGraph route timeout after {_timeout_secs}s – returning fallback")
            return jsonify({
                "success": False,
                "error": "LangGraph prediction timed out. The model took too long to respond.",
                "error_type": "timeout",
                "location": location,
                "timeframe": timeframe
            }), 504
        
        if result and result.get('success'):
            logger.info("✅ LangGraph multi-agent weather prediction successful")
            # --- Parse the LLM's prediction string server-side so the frontend
            #     always receives a clean structured array, even when the raw
            #     string is truncated or wrapped in ```json fences. ---
            raw_pred = result.get('prediction', '')
            if isinstance(raw_pred, str):
                import re as _re
                stripped = _re.sub(r'^```json\s*', '', raw_pred.strip(), flags=_re.IGNORECASE)
                stripped = _re.sub(r'^```\s*', '', stripped, flags=_re.IGNORECASE)
                stripped = _re.sub(r'```\s*$', '', stripped, flags=_re.IGNORECASE).strip()
                parsed_forecast = None

                # Strategy 1: direct parse (works when JSON is complete)
                try:
                    parsed_forecast = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

                # Strategy 2: extract every complete {…} day-object via regex.
                # Robust regardless of where truncation occurs — no bracket-counting needed.
                if not (parsed_forecast and isinstance(parsed_forecast.get('predictions'), list)):
                    day_matches = _re.findall(r'\{[^{}]+\}', stripped)
                    complete_days = []
                    for m in day_matches:
                        try:
                            obj = json.loads(m)
                            # Accept objects that look like a forecast day
                            if 'day' in obj or 'conditions' in obj or 'temperature_c' in obj:
                                complete_days.append(obj)
                        except Exception:
                            pass
                    if complete_days:
                        # Salvage analysis text if present
                        analysis_match = _re.search(
                            r'"analysis"\s*:\s*"(.*?)(?:"|$)', stripped, _re.DOTALL)
                        analysis_text = analysis_match.group(1) if analysis_match else ''
                        parsed_forecast = {'predictions': complete_days, 'analysis': analysis_text}
                        logger.info(f"✅ Forecast repaired via day-object extraction: {len(complete_days)} days")

                if parsed_forecast and isinstance(parsed_forecast.get('predictions'), list):
                    result['forecast_days']     = parsed_forecast['predictions']
                    result['forecast_analysis'] = parsed_forecast.get('analysis', '')
                    logger.info(f"✅ Forecast parsed server-side: {len(result['forecast_days'])} days")
            return jsonify(result), 200
        else:
            logger.error(f"❌ LangGraph multi-agent prediction failed")
            
            # Determine error type for better frontend handling
            error_response = result or {"error": "Prediction failed", "success": False}
            
            # Check for timeout conditions
            if result and (
                result.get('timeout_occurred') or
                (result.get('error') and 'timeout' in str(result.get('error')).lower()) or
                (result.get('note') and 'taking longer than expected' in str(result.get('note')).lower())
            ):
                error_response['error_type'] = 'timeout'
                if not error_response.get('error'):
                    error_response['error'] = 'Prediction is taking longer than expected'
            
            # Check for service unavailable conditions
            elif result and (
                'not available' in str(result.get('error', '')).lower() or
                'service unavailable' in str(result.get('error', '')).lower() or
                'not properly initialized' in str(result.get('error', '')).lower()
            ):
                error_response['error_type'] = 'service_unavailable'
                if not error_response.get('error'):
                    error_response['error'] = 'LangGraph multi-agent service is not currently available'
            
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"❌ LangGraph multi-agent prediction endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

@api.route('/weather/predict-langgraph-websocket', methods=['POST'])
def predict_weather_langgraph_websocket():
    """Ultimate weather prediction using LangGraph multi-agent system with real-time WebSocket updates"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        
        logger.info("🧠🔗 LangGraph WebSocket multi-agent weather prediction request received")
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "success": False
            }), 400
        
        location = data.get('location', 'Tokyo')
        timeframe = data.get('timeframe', 3)
        
        # Validate inputs
        if not isinstance(timeframe, int) or timeframe < 1 or timeframe > 14:
            return jsonify({
                "error": "Timeframe must be an integer between 1 and 14 days",
                "success": False
            }), 400
            
        if not location or len(location.strip()) == 0:
            return jsonify({
                "error": "Location cannot be empty",
                "success": False
            }), 400
        
        logger.info(f"🧠🔗 LangGraph WebSocket prediction for {location}, {timeframe} days")
        
        # Check if weather service supports LangGraph
        if not weather_service:
            return jsonify({
                "error": "Weather service not available",
                "success": False,
                "error_type": "service_unavailable"
            }), 503
        
        if not hasattr(weather_service, 'predict_weather_with_langgraph'):
            return jsonify({
                "error": "LangGraph multi-agent system not available in weather service",
                "success": False,
                "error_type": "feature_unavailable",
                "available_methods": ["basic", "rag", "local_llm"] 
            }), 501
        
        # Start WebSocket workflow if available
        workflow_id = None
        if websocket_service:
            workflow_id = websocket_service.start_workflow(
                workflow_type='weather_prediction',
                params={'location': location, 'prediction_days': timeframe}
            )
            logger.info(f"🔗 Started WebSocket workflow: {workflow_id}")
        
        # Get LangGraph multi-agent prediction with WebSocket support
        result = weather_service.predict_weather_with_langgraph(
            location=location, 
            prediction_days=timeframe,
            workflow_id=workflow_id
        )
        
        if result and result.get('success'):
            logger.info("✅ LangGraph WebSocket multi-agent weather prediction successful")
            
            # Add WebSocket workflow ID to response
            if workflow_id:
                result['workflow_id'] = workflow_id
                result['websocket_enabled'] = True
            
            return jsonify(result), 200
        else:
            logger.error(f"❌ LangGraph WebSocket multi-agent prediction failed")
            error_response = {
                "error": result.get('error', 'LangGraph prediction failed - unknown error'),
                "success": False,
                "method": "langgraph_websocket_failed",
                "location": location,
                "timeframe": timeframe,
                "error_type": "prediction_failed"
            }
            
            # Broadcast error if WebSocket is available
            if websocket_service and workflow_id:
                websocket_service.broadcast_error(workflow_id, error_response['error'])
            
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"❌ LangGraph WebSocket prediction endpoint error: {str(e)}")
        
        # Broadcast error if WebSocket is available
        if 'workflow_id' in locals() and websocket_service and workflow_id:
            websocket_service.broadcast_error(workflow_id, f"Server error: {str(e)}")
        
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False,
            "error_type": "server_error"
        }), 500

@api.route('/weather/langgraph-status', methods=['GET'])
def get_langgraph_status():
    """Get LangGraph multi-agent service status and capabilities"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
            
        logger.info("📊 LangGraph multi-agent status request received")
        
        # Get LangGraph status
        status = weather_service.get_langgraph_status()
        
        return jsonify({
            'success': True,
            'langgraph_status': status,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ LangGraph status check error: {str(e)}")
        return jsonify({
            'error': f'LangGraph status check failed: {str(e)}',
            'success': False
        }), 500


# ---------------------------------------------------------------------------
# Week 4: Ensemble prediction endpoint
# ---------------------------------------------------------------------------

@api.route('/weather/predict-ensemble', methods=['POST'])
@require_auth
def predict_weather_ensemble():
    """
    Confidence-weighted ensemble prediction with full production guards.

    Combines LangGraph, Multi-query RAG, Standard RAG, and Statistical
    predictions via Qwen3 CoT meta-synthesis.

    Body JSON fields:
      location         str   default "Tokyo"
      days             int   1-14, default 3
      season           str   "auto"|"Winter"|"Spring"|"Summer"|"Autumn"
      enable_multiquery bool  default true
    """
    try:
        user_id = str(get_session_user_id())

        # --- Rate limiting ---
        if PRODUCTION_GUARDS:
            allowed, rl_info = rate_limiter.check(user_id, 'authenticated')
            if not allowed:
                return jsonify({"error": "Rate limit exceeded", **rl_info}), 429

        data = request.get_json() or {}

        # --- Input schema validation (Pydantic) ---
        if PRODUCTION_GUARDS and PYDANTIC_AVAILABLE and WeatherPredictionRequest:
            try:
                req = WeatherPredictionRequest(**data)
                location         = req.location
                days             = req.days
                season           = req.season or 'auto'
                enable_multiquery = req.enable_multiquery
            except ValidationError as ve:
                if PRODUCTION_GUARDS: rate_limiter.release(user_id)
                return jsonify({"error": "Validation failed", "details": str(ve)}), 400
        else:
            location          = str(data.get('location', 'Tokyo')).strip() or 'Tokyo'
            days              = int(data.get('days', data.get('prediction_days', 3)))
            season            = str(data.get('season', 'auto'))
            enable_multiquery = bool(data.get('enable_multiquery', True))

        # --- Prompt injection guard ---
        if PRODUCTION_GUARDS:
            sec = security_guard.validate_location(location)
            if not sec.is_safe:
                rate_limiter.release(user_id)
                return jsonify({"error": sec.rejection_reason, "success": False}), 400
            location = sec.sanitized_input

        # --- Basic validation ---
        if not (1 <= days <= 14):
            if PRODUCTION_GUARDS: rate_limiter.release(user_id)
            return jsonify({"error": "days must be between 1 and 14"}), 400
        if season not in ('auto', 'Winter', 'Spring', 'Summer', 'Autumn'):
            season = 'auto'

        logger.info(
            "🎯 Ensemble prediction: location=%s days=%d season=%s multiquery=%s",
            location, days, season, enable_multiquery
        )

        if not weather_service:
            return jsonify({"error": "Weather service not available", "success": False}), 503

        import concurrent.futures as _cf
        _ens_timeout = 300
        _pool2 = _cf.ThreadPoolExecutor(max_workers=1)
        _future2 = _pool2.submit(
            weather_service.predict_weather_ensemble,
            location=location,
            prediction_days=days,
            season=season,
            enable_multiquery=enable_multiquery,
        )
        _pool2.shutdown(wait=False)  # Don't block on cleanup
        try:
            result = _future2.result(timeout=_ens_timeout)
        except _cf.TimeoutError:
            logger.warning(f"⏱️ Ensemble route timeout after {_ens_timeout}s")
            if PRODUCTION_GUARDS: rate_limiter.release(user_id)
            return jsonify({
                "success": False,
                "error": "Ensemble prediction timed out. The model took too long to respond.",
                "error_type": "timeout",
                "location": location,
                "days": days
            }), 504

        if result and result.get('success'):
            logger.info("✅ Ensemble prediction successful")
            return jsonify(result), 200
        else:
            logger.error("❌ Ensemble prediction failed: %s", result.get('error', 'unknown'))
            return jsonify({
                "error": result.get('error', 'Ensemble prediction failed'),
                "success": False,
                "method": "ensemble_failed",
                "location": location,
                "prediction_days": days,
            }), 500

    except (TypeError, ValueError) as e:
        if PRODUCTION_GUARDS: rate_limiter.release(str(get_session_user_id()) if 'user_id' in session else '__none__')
        return jsonify({"error": f"Invalid parameters: {str(e)}", "success": False}), 400

    except Exception as e:
        logger.error("❌ Ensemble endpoint error: %s", str(e))
        if PRODUCTION_GUARDS: rate_limiter.release(str(get_session_user_id()) if 'user_id' in session else '__none__')
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500

    finally:
        if PRODUCTION_GUARDS:
            try:
                rate_limiter.release(str(session.get('user_id', '__none__')))
            except Exception:
                pass


@api.route('/weather/ensemble-status', methods=['GET'])
@require_auth
def get_ensemble_status():
    """Return status of the ensemble prediction service."""
    try:
        status = weather_service.get_ensemble_status() if weather_service else {"available": False}
        return jsonify({"success": True, "ensemble_status": status,
                        "timestamp": datetime.now().isoformat()}), 200

    except Exception as e:
        logger.error("❌ Ensemble status error: %s", str(e))
        return jsonify({"error": str(e), "success": False}), 500


# ---------------------------------------------------------------------------
# Production monitoring endpoints
# ---------------------------------------------------------------------------

@api.route('/monitoring/dashboard', methods=['GET'])
@require_auth
def monitoring_dashboard():
    """
    Real-time MLOps monitoring dashboard.

    Returns:
      - prediction latency percentiles (p50 / p95 / p99)
      - success / cache-hit rates
      - method breakdown
      - circuit breaker states
      - rate limiter stats
    """
    try:
        payload = {
            "predictions": observability.get_dashboard_metrics() if PRODUCTION_GUARDS else {},
            "circuit_breakers": circuit_breakers.get_all_status() if PRODUCTION_GUARDS else {},
            "rate_limiter": rate_limiter.get_stats() if PRODUCTION_GUARDS else {},
            "guards_active": PRODUCTION_GUARDS,
            "timestamp": datetime.now().isoformat(),
        }
        return jsonify({"success": True, **payload}), 200

    except Exception as e:
        logger.error("❌ Monitoring dashboard error: %s", e)
        return jsonify({"error": str(e), "success": False}), 500


@api.route('/monitoring/circuit-breakers/<string:breaker_name>/reset', methods=['POST'])
@require_auth
def reset_circuit_breaker(breaker_name: str):
    """
    Manually reset a circuit breaker to CLOSED state.

    Path param:
      breaker_name — one of: lm_studio, rag_service, langgraph, ensemble
    """
    if not PRODUCTION_GUARDS:
        return jsonify({"error": "Production guards not active"}), 503

    cb = circuit_breakers.get(breaker_name)
    if cb is None:
        return jsonify({
            "error": f"Unknown breaker '{breaker_name}'",
            "valid_names": ["lm_studio", "rag_service", "langgraph", "ensemble"],
        }), 404

    cb.reset()
    logger.info("🔄 Circuit breaker '%s' manually reset by user %s",
                breaker_name, session.get('user_id'))
    return jsonify({
        "success": True,
        "breaker": breaker_name,
        "new_state": cb.get_status()["state"],
        "timestamp": datetime.now().isoformat(),
    }), 200


@api.route('/monitoring/rate-limiter/reset/<string:user_id>', methods=['POST'])
@require_auth
def reset_rate_limit_user(user_id: str):
    """Admin: clear rate-limit bucket for a specific user_id."""
    if not PRODUCTION_GUARDS:
        return jsonify({"error": "Production guards not active"}), 503

    rate_limiter.reset_user(user_id)
    logger.info("🔄 Rate limit reset for user '%s' by admin %s",
                user_id, session.get('user_id'))
    return jsonify({
        "success": True,
        "cleared_user": user_id,
        "timestamp": datetime.now().isoformat(),
    }), 200


# ---------------------------------------------------------------------------
# Electricity load model (pkl-based, no CSV at runtime)
# ---------------------------------------------------------------------------

def _get_electricity_model():
    """Lazy import to avoid circular deps at module load."""
    try:
        from backend.electricity_model_service import electricity_model
        return electricity_model
    except Exception as exc:
        logger.error("⚡ Could not load electricity_model_service: %s", exc)
        return None


@api.route('/weather/predict-electricity', methods=['POST'])
@require_auth
def predict_electricity_load():
    """
    Predict electricity load using the trained pkl model (no CSV needed).

    Request JSON:
        location          str   (informational, not used by model)
        days              int   1-14  default 7
        forecast_temp     float °C
        forecast_humidity float %
        forecast_solar    float kWh/m²/day
        forecast_wind     float m/s
        forecast_rain     float mm
        forecast_cloud    float 0-10
        season            str   Spring|Summer|Autumn|Winter
        belnder_forecast  float demand forecast MW  (optional)
        is_holiday        int   0|1
        datetime_str      str   ISO format (default: now)

    Response JSON:
        success           bool
        method            "electricity_model_pkl"
        predicted_load_mw float
        confidence_band   [lower, upper]
        confidence_level  High|Medium|Low
        prediction        str  (human-readable summary)
        model_info        dict
    """
    data = request.get_json(silent=True) or {}

    location = data.get("location", "Tokyo")
    days     = min(int(data.get("days", 7)), 14)

    em = _get_electricity_model()
    if em is None or not em.is_available():
        return jsonify({
            "success": False,
            "error": "Electricity model not available. Run: python train_electricity_model.py",
        }), 503

    try:
        result = em.predict_for_weather(
            location=location,
            days=days,
            forecast_temp=float(data.get("forecast_temp", 20.0)),
            forecast_humidity=float(data.get("forecast_humidity", 65.0)),
            forecast_solar=float(data.get("forecast_solar", 4.5)),
            forecast_wind=float(data.get("forecast_wind", 6.0)),
            forecast_rain=float(data.get("forecast_rain", 4.8)),
            forecast_cloud=float(data.get("forecast_cloud", 3.3)),
            season=data.get("season", "Autumn"),
            is_holiday=int(data.get("is_holiday", 0)),
            belnder_forecast=float(data.get("belnder_forecast", 1076.5)),
        )
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.error("⚡ Electricity load prediction error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/weather/electricity-model-status', methods=['GET'])
@require_auth
def electricity_model_status():
    """Return metadata and performance metrics of the trained electricity pkl model."""
    em = _get_electricity_model()
    if em is None:
        return jsonify({"success": False, "error": "Service unavailable"}), 503
    info = em.get_model_info()
    return jsonify({"success": True, **info}), 200
