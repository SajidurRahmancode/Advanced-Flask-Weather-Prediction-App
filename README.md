# 🌤️ Advanced Flask Weather Prediction App

A comprehensive Flask application featuring cutting-edge AI-powered weather prediction with a **LangGraph 5-agent multi-agent system**, LangChain + RAG orchestration, Qwen3-14B local LLM via LM Studio, circuit breakers, rate limiting, prompt-injection protection, ML observability, and structured day-card forecasts for 3–14 days.

![Weather Prediction Dashboard](Weather.png)

## 🚀 Overview

This application combines traditional web development with modern AI technologies to provide intelligent weather predictions. The system features:

- **🕸️ LangGraph Multi-Agent System**: 5 specialized AI agents (Data Collector → Pattern Analyzer → Meteorologist → Confidence Assessor → Prediction Generator) collaborate for maximum forecast accuracy
- **🧠 Qwen3-14B Local LLM**: Chain-of-thought JSON forecasting via LM Studio — 3–14 day structured day cards, all AI-generated
- **📚 RAG (Retrieval-Augmented Generation)**: Historical weather pattern retrieval from ChromaDB with 263 local embeddings
- **🏠 LangChain + RAG Orchestration**: Advanced multi-step reasoning with conversation memory
- **🔐 Complete Authentication System**: Session-based auth, CSRF protection, token-bucket rate limiting, prompt-injection protection
- **🛡️ Circuit Breakers**: Fault isolation for LM Studio, RAG, LangGraph, and Ensemble services
- **📊 ML Observability**: Per-request JSONL tracing, rolling p50/p95/p99 latency, success rate, cache-hit rate
- **⚡ Real-time Predictions**: WebSocket agent monitoring + multi-layer fallbacks ensuring reliability
- **🎯 Structured Day Cards**: Server-side JSON parsing renders emoji day cards for every forecast length

## 🌟 Key Features

### Weather Prediction Capabilities
- **🕸️ LangGraph 5-Agent Pipeline**: Data Collector → Pattern Analyzer → Meteorologist → Confidence Assessor → Prediction Generator, all coordinated by LangGraph state machines
- **🧠 Qwen3-14B CoT Forecasting**: Structured JSON output with scaled `max_tokens` and timeout per day count (3d=90s, 7d=146s, 10d=180s, 14d=225s)
- **🗂️ KV Cache Management**: `clear_prompt_cache()` erases LM Studio slots before large requests, preventing generation stalls from cache pressure
- **📚 LangChain + RAG Orchestration**: Advanced chain-of-thought reasoning with conversation memory and historical pattern retrieval
- **🔄 Graceful Multi-Layer Fallbacks**: Qwen3 CoT → LangChain + RAG → Statistical JSON (always returns structured day cards)
- **🔍 Multi-Query RAG**: Multiple query expansion for richer historical pattern retrieval
- **📊 Electricity Load ML Model**: GradientBoosting PKL model (R²=0.9227, MAPE=2.74%) for energy demand prediction

### Security & Reliability
- **🔐 `@require_auth` Decorator**: Single decorator replaces 21 copy-pasted session checks
- **⏱️ Token Bucket Rate Limiter**: Anonymous 10 req/hr, authenticated 60 req/hr, RFC-compliant headers
- **🛡️ Prompt Injection Protection**: 18 block patterns, location/query/output sanitization
- **🔌 Circuit Breakers**: 4 named breakers (LM Studio, RAG, LangGraph, Ensemble) — open circuits return immediately instead of blocking
- **🔑 Password Hash Fix**: `String(256)` column prevents scrypt hash truncation that silently broke all logins
- **🛡️ CSRF**: Correct instance exemption applied to all API blueprints

### Observability & Infrastructure
- **📈 ML Observability**: Per-request JSONL tracing with `trace_id`, latency, method, cache-hit, fallback flags
- **📊 Live Metrics Endpoint**: Rolling p50/p95/p99 latency + success rate at `GET /api/monitoring/dashboard`
- **🌐 WebSocket Agent Monitor**: Real-time LangGraph agent progress streamed to frontend via Flask-SocketIO
- **🗂️ Pydantic v2 Schemas**: Typed request/response validation with automatic fallback

### User Experience
- **🃏 Structured Day Cards**: Every forecast (3–14 days) renders emoji + temperature + humidity + wind + confidence badge — never raw JSON
- **Modern UI**: Bootstrap 5 responsive design with agent insight panels
- **Method Selection**: Interactive dropdown covering all 9 prediction endpoints
- **Real-time Feedback**: WebSocket-powered agent status updates during LangGraph predictions
- **Detailed Results**: Per-day confidence levels, precipitation bars, analysis blocks

## 📋 Prerequisites

### XAMPP Setup (MySQL Database)
1. **Download and install XAMPP** from [https://www.apachefriends.org/](https://www.apachefriends.org/)
2. **Start XAMPP Control Panel** as Administrator
3. **Start Apache and MySQL services** in XAMPP Control Panel
4. **Verify MySQL is running** - you should see "Running" status in green

### LM Studio Setup (Optional - for Local LLM)
1. **Download LM Studio** from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Install Qwen3-14B** (`qwen/qwen3-14b`) — the model the app is optimised for. Alternatives: Qwen2.5-14B, Mistral-7B
3. **Start the Local Server** in LM Studio
4. **Configure API endpoint** at `http://127.0.0.1:1234` (default)
5. **Recommended RAM**: 16 GB+ for Qwen3-14B; 8 GB minimum for Qwen2.5-7B

### Google AI API (Optional - for Gemini AI)
1. **Get API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Create .env file** and add: `GEMINI_API_KEY=your_api_key_here`
3. **Enhanced predictions** will use Gemini for advanced reasoning

### Database Setup
**Automatic Setup** (Recommended):
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run automatic database setup
python setup_database.py
```

**Manual Setup** (Alternative):
1. Open phpMyAdmin (click "Admin" button next to MySQL in XAMPP)
2. Create database `flask_react_app`
3. Tables will be created automatically on first run

The setup script will:
- Create the database `flask_react_app` automatically
- Initialize all required tables
- Test MySQL connection
- Load weather data (274+ historical records)
- Initialize vector store for RAG
- Provide troubleshooting tips if issues occur

## 🏗️ Architecture & Project Structure

```
📁 Flask-Weather-Prediction-App/
├── 🐍 app.py                          # Main Flask application
├── ⚙️ .env                           # Configuration (create from template)
├── 🗃️ setup_database.py              # Automated MySQL setup
├── 📋 requirements.txt                # Python dependencies
├── 📊 data/                          # Weather datasets & ML models
│   ├── Generated_electricity_load_japan_past365days.csv
│   ├── electricity_load_model.pkl     # 🆕 GradientBoosting model (R²=0.9227)
│   └── vector_db/                    # ChromaDB with 263 local embeddings
├── 🔧 backend/                       # Backend services
│   ├── 🌐 routes.py                  # API endpoints (30+ endpoints)
│   ├── 🔐 auth.py                    # Authentication system
│   ├── 🛡️ auth_guard.py             # 🆕 @require_auth decorator
│   ├── ⏱️ rate_limiter.py           # 🆕 Token bucket rate limiter
│   ├── 🔒 prompt_security.py        # 🆕 Prompt injection protection
│   ├── 🔌 circuit_breaker.py        # 🆕 4-service circuit breakers
│   ├── 📈 ml_observability.py       # 🆕 JSONL tracing + p99 metrics
│   ├── 🗂️ models.py                 # Database models
│   ├── 🌤️ weather_service.py         # Core weather prediction logic
│   ├── 🕸️ langgraph_service.py      # 🆕 LangGraph 5-agent pipeline
│   ├── 🧠 langchain_rag_service.py   # LangChain + RAG orchestration
│   ├── 🏠 lmstudio_service.py        # Local LLM + cache management
│   ├── 📚 rag_service.py             # Vector database RAG
│   ├── ⚡ websocket_service.py       # Real-time agent monitoring
│   ├── 🎯 ensemble_service.py        # Ensemble prediction service
│   └── ⚡ electricity_model_service.py # 🆕 PKL model service
├── 🧪 test_eval_pipeline.py          # 🆕 107-test security/ML suite
├── 🧪 test_electricity_model.py      # 🆕 26-test PKL model suite
├── 🎨 frontend/                      # Web interface
│   ├── 📄 templates/
│   │   ├── 🏠 base.html
│   │   ├── 🔑 login.html
│   │   ├── 📊 dashboard.html
│   │   └── 🌤️ weather_dashboard.html  # Day-card rendering + WebSocket UI
│   └── 🎯 static/
│       ├── css/main.css
│       └── js/main.js
└── 🗄️ flasking_py311/               # Python 3.11 virtual environment
```

## 🛠️ Tech Stack

### Backend Technologies
- **🐍 Flask 3.1.1**: Modern Python web framework
- **🗄️ Flask-SQLAlchemy**: Object-Relational Mapping
- **🔐 Flask-JWT-Extended**: Token-based authentication
- **🛡️ Flask-WTF**: CSRF protection and forms
- **🌐 Flask-CORS**: Cross-origin resource sharing
- **💾 MySQL + PyMySQL**: Database with XAMPP support

### AI & Machine Learning
- **🕸️ LangGraph**: Multi-agent state machine orchestration
- **🧠 LangChain**: Advanced LLM orchestration with conversation memory
- **🤖 Qwen3-14B via LM Studio**: Local CoT JSON forecasting, `/no_think` optimised
- **📚 ChromaDB**: Vector database with 263 local embeddings
- **🔍 Sentence Transformers** (`all-MiniLM-L6-v2`): Local embeddings, no API key required
- **📊 scikit-learn + joblib**: GradientBoosting electricity load model (R²=0.9227)
- **🤖 Google Generative AI**: Cloud-based intelligence (optional)
- **📊 Pandas + NumPy**: Data processing and analysis

### Frontend Technologies
- **🎨 Bootstrap 5**: Modern responsive framework
- **⚡ Jinja2**: Server-side templating
- **📱 FontAwesome**: Professional icons
- **🎯 JavaScript ES6+**: Interactive functionality

### Infrastructure
- **⚡ Flask-SocketIO**: WebSocket support for real-time agent monitoring
- **🔌 Pydantic v2**: Request/response schema validation
- **🐳 ChromaDB**: Vector storage for pattern matching
- **🏠 XAMPP**: Local MySQL development
- **🔧 Python-dotenv**: Environment configuration
- **📝 Logging + JSONL Tracing**: Per-request traces with `trace_id`, latency, method, fallback flags

## 🚀 Quick Start Guide

### 1️⃣ Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd Flask-Weather-Prediction-App

# Create virtual environment (optional but recommended)
python -m venv flasking
source flasking/bin/activate  # On Windows: flasking\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2️⃣ Database Initialization
```bash
# Automatic setup (recommended)
python setup_database.py

# This will:
# ✅ Create MySQL database 'flask_react_app'
# ✅ Initialize user and weather tables
# ✅ Load 274+ historical weather records
# ✅ Setup ChromaDB vector store
# ✅ Test all connections
```

### 3️⃣ Configuration (Optional)
Create `.env` file for enhanced features:
```env
# Database (auto-configured for XAMPP)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=flask_react_app
DB_USER=root
DB_PASSWORD=

# API Keys (optional)
GEMINI_API_KEY=your_google_api_key_here

# Local LLM — default port for LM Studio
LM_STUDIO_API_URL=http://127.0.0.1:1234
```

### 4️⃣ Launch Application
```bash
# Start Flask server
python app.py

# Server will be available at:
# 🌐 http://localhost:5000
# 🔑 Auth: http://localhost:5000/auth/login
# 🌤️ Weather: http://localhost:5000/auth/weather
```

### 5️⃣ Access Weather Dashboard
1. **Register/Login** at `http://localhost:5000/auth/signup`
2. **Navigate to Weather Dashboard** (automatically available after login)
3. **Select Prediction Method**:
   - **LangChain + RAG**: Ultimate AI with historical patterns
   - **RAG + Local LLM**: Local AI with pattern matching
   - **Local LLM Only**: Privacy-focused predictions
   - **Statistical Analysis**: Mathematical baseline

## 🌤️ Weather Prediction Methods

### 🕸️ LangGraph Multi-Agent Pipeline (Recommended)
- **5 Specialised Agents**: Data Collector → Pattern Analyzer → Meteorologist → Confidence Assessor → Prediction Generator
- **Qwen3-14B CoT**: Structured JSON output with exactly N day objects, rendered as emoji day cards
- **Scaled Resources**: Token budget and HTTP timeout scale with day count (3d=600 tok/90s → 14d=1300 tok/225s)
- **KV Cache Clearing**: Frees LM Studio cache before 7+ day requests, preventing mid-generation stalls
- **WebSocket Monitoring**: Real-time agent progress streamed to the dashboard
- **Robust Fallbacks**: Qwen3 CoT → LangChain + RAG → Statistical JSON (always returns structured cards)

**Best for**: All forecast lengths (3–14 days), maximum AI quality

### 🧠 LangChain + RAG
- **Advanced Orchestration**: Multi-step reasoning with conversation memory
- **Historical Patterns**: Retrieves similar conditions from 263 local embeddings
- **Confidence Assessment**: Built-in prediction reliability scoring
- **Multi-Query Expansion**: Multiple query variants for richer retrieval

**Best for**: Historical-pattern-informed predictions

### 🏠 Local LLM via LM Studio
- **Complete Privacy**: All processing happens locally, no data leaves the machine
- **No API Limits**: Unlimited predictions without quotas
- **Qwen3 Optimised**: `/no_think` suppression, `strip_thinking()` post-processing
- **Circuit Breaker**: Open circuit returns immediately on repeated failures

**Best for**: Privacy-conscious users

### 📊 Statistical Analysis (Always-Available Fallback)
- **JSON Day Objects**: Returns same structured format as AI methods — renders day cards
- **Current-Conditions Seeded**: Uses live temperature/humidity/wind as baseline
- **Seasonal Variation**: Applies daily variation patterns
- **Zero Dependencies**: No LM Studio, no API keys required

**Best for**: Reliable baseline when AI services are unavailable

## 🛡️ Security & Authentication

### Session-Based Authentication
- **`@require_auth` Decorator**: Single decorator protects all authenticated routes — replaces 21 copy-pasted session checks
- **Secure Login/Logout**: Session management with Flask-Login
- **CSRF Protection**: Correct instance exemption applied per blueprint
- **Password Security**: scrypt hashing stored in `String(256)` column (previously truncated at 128, silently breaking all logins)
- **Route Protection**: Authentication required for all weather features and the monitoring dashboard

### API Security
- **Token Bucket Rate Limiter**: Anonymous 10 req/hr, authenticated 60 req/hr. Returns `429 Too Many Requests` with RFC-compliant `Retry-After` and `X-RateLimit-*` headers
- **Prompt Injection Protection**: 18 block patterns catch `ignore all previous instructions`, jailbreak attempts, system-prompt leaks, and role-play attacks. Location and output fields are independently sanitised
- **Pydantic v2 Schemas**: All prediction requests validated with typed schemas before reaching the AI layer
- **Error Handling**: Secure error messages without information leakage
- **CORS Configuration**: Controlled cross-origin access

### Fault Isolation
- **Circuit Breakers** (`backend/circuit_breaker.py`): 4 named breakers — `lm_studio`, `rag`, `langgraph`, `ensemble`. Trips after repeated failures, returns immediately while open, self-resets after cooldown
- **Multi-Layer Fallbacks**: Every prediction path degrades gracefully — AI failure never surfaces as a 500 error to the user

## 📊 API Endpoints

### Authentication Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/login` | Login page |
| POST | `/auth/login` | Process login |
| GET | `/auth/signup` | Registration page |
| POST | `/auth/signup` | Process registration |
| GET | `/auth/logout` | Logout user |
| GET | `/auth/dashboard` | User dashboard |
| GET | `/auth/weather` | Weather prediction interface |

### Weather Prediction API
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/weather/predict` | Standard AI prediction |
| POST | `/api/weather/predict-rag` | RAG-enhanced prediction |
| POST | `/api/weather/predict-local` | Local LLM prediction |
| POST | `/api/weather/predict-rag-local` | RAG + Local LLM |
| POST | `/api/weather/predict-hybrid` | Hybrid smart fallback |
| POST | `/api/weather/predict-langchain-rag` | LangChain + RAG orchestration |
| POST | `/api/weather/predict-langgraph` | **🆕 LangGraph 5-agent pipeline (3–14 day cards)** |
| POST | `/api/weather/predict-multiquery-rag` | **🆕 Multi-Query RAG prediction** |
| POST | `/api/weather/predict-ensemble` | **🆕 Ensemble of all methods** |

All prediction endpoints accept:
```json
{ "location": "Tokyo", "timeframe": 7 }
```
And return `forecast_days[]` — a pre-parsed array of day objects for direct card rendering.

### Data & Analysis API
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/weather/rag-search` | Search weather patterns |
| GET | `/api/weather/rag-stats` | RAG service statistics |
| GET | `/api/weather/data-summary` | Weather data overview |
| GET | `/api/weather/recent-data` | Recent weather observations |
| GET | `/api/monitoring/dashboard` | **🆕 Live p50/p95/p99 latency + success metrics** |

### Service Status API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/weather/lm-studio-status` | LM Studio availability |
| GET | `/api/weather/langchain-rag-status` | LangChain service status |
| GET | `/api/weather/langgraph-status` | **🆕 LangGraph agent status** |
| GET | `/api/health` | Overall system health |

## 🔧 Advanced Configuration

### LM Studio Setup
```bash
# 1. Download LM Studio from https://lmstudio.ai/
# 2. Download recommended model:
#    - qwen/qwen3-14b (14GB, optimised — best results)
#    - qwen2.5-14b-instruct (14GB, fast alternative)
#    - mistral-7b-instruct (7GB, lightweight)
# 3. Start Local Server in LM Studio
# 4. Verify endpoint: http://127.0.0.1:1234
# 5. App auto-detects the model and applies Qwen3 optimisations
#    (strips <think> tags, injects /no_think, clears KV cache for 7+ day forecasts)
```

### Google AI Integration
```bash
# 1. Get API key from Google AI Studio
# 2. Add to .env file: GEMINI_API_KEY=your_key
# 3. Restart application
# 4. Enhanced predictions with Gemini Pro will be available
```

### Vector Database Optimization
```bash
# Reset vector store (if needed)
python -c "
import shutil
shutil.rmtree('./chromadb_store', ignore_errors=True)
python setup_database.py  # Rebuilds vector store
"

# The ChromaDB store contains:
# ✅ 274+ weather observations
# ✅ Semantic embeddings for pattern matching
# ✅ Metadata for filtering (season, location, etc.)
```

## 🚨 Troubleshooting Guide

### MySQL Connection Issues
```bash
# Check XAMPP MySQL status
# ❌ XAMPP not running: Start MySQL service in XAMPP Control Panel
# ❌ Port 3306 busy: Check if another MySQL service is running
# ❌ Access denied: Verify credentials in .env file
# ❌ Database doesn't exist: Run python setup_database.py

# Test connection manually
python -c "
import pymysql
try:
    conn = pymysql.connect(host='localhost', user='root', password='')
    print('✅ MySQL connection successful')
    conn.close()
except Exception as e:
    print(f'❌ MySQL connection failed: {e}')
"
```

### LM Studio Issues
```bash
# ❌ LM Studio not detected
# 1. Ensure LM Studio is running
# 2. Start Local Server in LM Studio
# 3. Check endpoint: http://127.0.0.1:1234/v1/models
# 4. Load Qwen3-14B (or compatible model)

# Test LM Studio connectivity
curl http://127.0.0.1:1234/v1/models
# Should return JSON with available models

# ❌ 7/10/14-day forecast times out or returns wrong day count
# App automatically clears LM Studio KV cache (POST /slots/{id}/action)
# before large requests — requires LM Studio >= 0.3.5
# If on older version, reduce forecast days or restart LM Studio manually

# ❌ <think> tags leaking into output
# App injects /no_think in every system prompt and sets
# chat_template_kwargs: {enable_thinking: false} in the payload
# If still appearing: ensure Qwen3 model is loaded (not Qwen2.5)
```

### AI Service Issues
```bash
# ❌ "Service not available" errors
# 1. Check Google AI API key in .env file
# 2. Verify internet connection for cloud AI
# 3. Ensure LM Studio is running for local AI
# 4. Application will fallback to statistical methods

# ❌ ChromaDB/RAG errors
python setup_database.py  # Rebuilds vector store
```

### Common Fixes
```bash
# Reset everything (nuclear option)
# 1. Stop Flask application
# 2. Delete chromadb_store/ directory
# 3. Drop flask_react_app database in phpMyAdmin
# 4. Run: python setup_database.py
# 5. Run: python app.py

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall --no-cache-dir

# Check logs for detailed error information
# Logs are printed to console with detailed service status
```

## 🎯 Development & Testing

### Running Tests
```bash
# Full evaluation pipeline (107 tests)
python test_eval_pipeline.py      # auth, rate limiter, circuit breaker,
                                  # prompt security, validators, observability

# Electricity ML model (26 tests)
python test_electricity_model.py  # artifact structure, metric gates,
                                  # batch + single-row predictions

# Individual service tests
python test_weather_service.py    # Weather prediction service
python test_lm_studio.py          # LM Studio connectivity
python test_langgraph_week3.py    # LangGraph 5-agent pipeline
python test_week4_ensemble.py     # Ensemble service

# Manual API testing (all day counts — requires login cookie)
curl -X POST http://localhost:5000/api/weather/predict-langgraph \
  -H "Content-Type: application/json" \
  -d '{"location": "Tokyo", "timeframe": 7}' \
  -b cookies.txt
```

### Development Mode
```bash
# Enable Flask debug mode
export FLASK_ENV=development  # Linux/Mac
set FLASK_ENV=development     # Windows

# Debug logging
export FLASK_DEBUG=1
python app.py  # Will show detailed debug information
```

### Adding New Features
```bash
# Backend development
# 1. Add routes in backend/routes.py
# 2. Add services in backend/weather_service.py
# 3. Update models in backend/models.py if needed

# Frontend development
# 1. Update templates in frontend/templates/
# 2. Add styling in frontend/static/css/
# 3. Add JavaScript in frontend/static/js/
```

## 🔮 Future Roadmap

### ✅ Completed (Feb 2026)
- **✅ LangGraph 5-Agent Pipeline**: Full multi-agent weather prediction
- **✅ Extended Forecasts**: 3–14 day structured day cards, all Qwen3-powered
- **✅ Ensemble Methods**: Ensemble service combining multiple AI approaches
- **✅ Rate Limiting + Prompt Security**: Token bucket + 18-pattern injection protection
- **✅ Circuit Breakers**: Fault isolation for all AI services
- **✅ ML Observability**: JSONL tracing + rolling p50/p95/p99 metrics
- **✅ Electricity Load Model**: PKL GradientBoosting (R²=0.9227)

### Planned Features
- **🌍 Multi-location Support**: Global weather predictions with location validation
- **🎨 Visualization**: Interactive temperature/precipitation charts
- **📱 Mobile App**: React Native companion app
- **🔔 Alerts**: Custom weather threshold alerts
- **📈 Analytics**: User prediction accuracy tracking dashboard
- **🌐 API Gateway**: Public API with API-key authentication

### AI Enhancements
- **🧠 Model Fine-tuning**: Domain-specific weather prediction LoRA adapters
- **🔍 Real-time Data**: Live weather API integration (JMA, OpenWeatherMap)
- **🎯 Location-specific Models**: Regional seasonal pattern learning
- **📝 Natural Language**: Conversational forecast queries via LangGraph

## 📜 License & Contributing

### License
This project is licensed under the MIT License. See LICENSE file for details.

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Add type hints for new functions
- Include comprehensive logging
- Write tests for new features
- Update documentation for API changes

## 🤝 Support & Community

### Getting Help
- **📚 Documentation**: Check this README for detailed information
- **🐛 Issues**: Report bugs via GitHub Issues
- **💬 Discussions**: Join GitHub Discussions for questions
- **📧 Contact**: Reach out for collaboration opportunities

### Acknowledgments
- **LangChain**: For powerful LLM orchestration framework
- **LM Studio**: For local LLM hosting capabilities
- **ChromaDB**: For efficient vector storage
- **Google AI**: For Gemini API integration
- **Flask Community**: For excellent web framework
- **Bootstrap**: For responsive UI components

---

**Made with ❤️ using Flask, LangChain, and AI technologies**

*Experience the future of weather prediction with multiple AI approaches, local privacy, and intelligent fallbacks.*
