# 🎬 YouTube RAG System

An intelligent system that processes YouTube videos through transcription, summarization, and retrieval-augmented generation (RAG) to enable conversational querying of video content.

## 🎯 **Project Overview**

This application allows users to:

1. **Process YouTube Videos**: Automatically download, transcribe, and summarize YouTube video content
2. **Store Knowledge**: Create embeddings and store processed content in a vector database (Milvus)
3. **Query Intelligently**: Ask questions about video content using RAG (Retrieval-Augmented Generation)
4. **Chat Interface**: Interact with the system through a modern web-based chat interface

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Vector DB     │
│   (Vue.js)      │◄──►│   (FastAPI)     │◄──►│   (Milvus)      │
│   Port 3000     │    │   Port 8000     │    │   Port 19530    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   LLM Server    │    │   Document DB   │
                       │  (LM Studio)    │    │   (MongoDB)     │
                       │  Tailscale URL  │    │   Port 27017    │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ **Technology Stack**

### Backend
- **FastAPI**: REST API framework
- **LangChain**: Document processing and LLM integration
- **Whisper**: Speech-to-text transcription
- **Milvus**: Vector database for embeddings
- **MongoDB**: Document storage
- **PyTube**: YouTube video downloading
- **BGE-M3**: Sparse embeddings for hybrid search

### Frontend
- **Vue.js 3**: Reactive frontend framework
- **Vite**: Fast build tool
- **Axios**: HTTP client for API calls

### AI/ML
- **Local LLM**: Via LM Studio (Qwen models)
- **Custom Embeddings**: Dense and sparse vector embeddings
- **Hybrid Search**: Combines semantic and keyword search

## 🚀 **Getting Started**

### Quick Start (FastAPI Only)

If you just want to run the FastAPI backend:

```bash
# 1. Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt

# 3. Start external services (Milvus, MongoDB, LM Studio)
# See detailed setup below

# 4. Start FastAPI server
python app.py
# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Prerequisites

1. **Python 3.8+**
2. **Node.js 16+** and npm
3. **Docker** (for Milvus)
4. **MongoDB** (local or remote instance)
5. **LM Studio** (running locally with specified models)

### Step 1: Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd digital_training

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
```

### Step 2: Setup External Services

#### **Start Milvus (Vector Database)**
```bash
# Using Docker Compose
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

#### **Start MongoDB**
```bash
# Using Docker
docker run --name mongodb -p 27017:27017 -d mongo:latest

# Or install locally and start service
# Ubuntu/Debian: sudo systemctl start mongod
# macOS: brew services start mongodb-community
```

#### **Setup LM Studio**
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load the following models:
   - `qwen/qwen3-30b-a3b` (Chat model)
   - `text-embedding-qwen3-embedding-8b@q5_0` (Sparse embeddings)
   - `text-embedding-nomic-embed-text-v1.5` (Dense embeddings)
3. Start the local server in LM Studio
4. Update `Config.TAILSCALE_SERVER` in `main.py` with your LM Studio URL

### Step 3: Setup Configuration

#### **Create Prompt Catalog**
```bash
# Create prompt_catalog.json with your system prompts
cat > prompt_catalog.json << EOF
{
  "system": {
    "transcript_summarizer": "You are an expert at summarizing YouTube video transcripts. Provide concise, informative summaries that capture the key points and main ideas."
  }
}
EOF
```

#### **Create URL List**
```bash
# Create directory and URL file
mkdir -p media/it
echo "https://www.youtube.com/watch?v=example" > media/it/videos.txt
```

### Step 4: Process YouTube Videos

```bash
# Activate virtual environment
source venv/bin/activate

# Process videos from URL file
python main.py download --url-file media/it/videos.txt --batch-size 2 --save-data processed_results.json

# Or process specific URLs
python main.py download --urls "https://youtube.com/watch?v=..." --batch-size 1
```

### Step 5: Start FastAPI Backend Server

The backend uses FastAPI to provide a REST API for the YouTube RAG system. Here are the different ways to start it:

**Option 1: Direct execution (Recommended)**
```bash
# Activate virtual environment
source venv/bin/activate

# Start the FastAPI server directly
python app.py
```

**Option 2: Using uvicorn command**
```bash
# Start with uvicorn (development mode with auto-reload)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Or for production (without auto-reload)
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Option 3: Using uvicorn with custom settings**
```bash
# Start with custom log level and workers
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info --workers 1
```

The API will be available at `http://localhost:8000`

#### FastAPI Features
- **Auto-reload**: The development server automatically reloads when code changes
- **Interactive API docs**: Visit `http://localhost:8000/docs` for Swagger UI
- **Alternative docs**: Visit `http://localhost:8000/redoc` for ReDoc interface
- **Background processing**: Video processing runs in background tasks
- **CORS enabled**: Frontend can communicate with the API

### Step 6: Start Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📖 **Usage**

### Command Line Interface

```bash
# Download and process videos
python main.py download --url-file videos.txt --batch-size 2

# Query processed data
python main.py query --question "What is the main topic?" --save-data processed_results.json
```

### Web Interface

1. Open `http://localhost:3000` in your browser
2. Type questions about your processed videos
3. Toggle RAG mode on/off to compare responses
4. View source references from the vector database

### API Endpoints

The FastAPI backend provides the following endpoints:

**System Status**
```bash
# Check system status and configuration
curl http://localhost:8000/status

# Get processed data information
curl http://localhost:8000/data

# Get API information
curl http://localhost:8000/api
```

**Video Processing**
```bash
# Process YouTube videos (background task)
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://youtube.com/watch?v=..."], "batch_size": 1}'
```

**Querying System**
```bash
# Query the RAG system (creates new session)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is calisthenics?", "use_rag": true}'

# Follow-up query with conversation context
curl -X POST "http://localhost:8000/follow-up-query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you explain more?", "use_rag": true, "session_id": "your-session-id"}'
```

**Advanced Query Features**
```bash
# Generate sub-queries for better retrieval
curl -X POST "http://localhost:8000/generate-sub-queries" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the benefits of calisthenics?"}'

# Generate step-back query for broader context
curl -X POST "http://localhost:8000/generate-step-back-query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are advanced calisthenics moves?"}'
```

**Interactive API Documentation**
- **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation format

### Testing the API

**1. Check if the server is running:**
```bash
curl http://localhost:8000/api
```

**2. Test video processing:**
```bash
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"], "batch_size": 1}'
```

**3. Test querying (after processing videos):**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this video about?", "use_rag": true}'
```

**4. Check system status:**
```bash
curl http://localhost:8000/status
```

## 🔧 **Configuration**

Key configuration options in `main.py`:

```python
class Config:
    MODEL = "qwen/qwen3-30b-a3b"                    # Chat model
    TAILSCALE_SERVER = "https://your-lm-studio-url" # LM Studio URL
    DENSE_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
    MILVUS_HOST = "127.0.0.1"                       # Milvus connection
    MILVUS_PORT = 19530
    MONGO_URI = 'mongodb://localhost:27017/'        # MongoDB connection
    CHUNK_SIZE = 3200                               # Text chunking size
    BATCH_SIZE = 8                                  # Processing batch size
```

## 📁 **Project Structure**

```
digital_training/
├── main.py                 # Core RAG system implementation
├── app.py                  # FastAPI backend server
├── test_api.py            # API testing script
├── requirements.txt       # Python dependencies
├── requirements_api.txt   # FastAPI dependencies
├── processed_results.json # Processed video data
├── prompt_catalog.json    # System prompts
├── frontend/              # Vue.js frontend
│   ├── src/
│   │   ├── App.vue       # Main chat component
│   │   └── main.js       # Vue initialization
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Build configuration
├── media/it/             # YouTube URL lists
├── temp/                 # Temporary audio files
└── res/                  # Processed transcripts
```

## 🔍 **Features**

### Core Functionality
- **🎵 Audio Processing**: Automatic YouTube audio extraction using faster-whisper
- **📝 Smart Summarization**: AI-powered summarization of video content chunks
- **🔍 Hybrid Search**: Combines semantic similarity and keyword matching (BGE-M3 + dense embeddings)
- **💬 Chat Interface**: Modern, responsive web-based chat UI with session management
- **⚡ Background Processing**: Non-blocking video processing via FastAPI background tasks
- **📊 Source Attribution**: Shows which video segments answer your questions

### Advanced Features
- **🔄 Incremental Processing**: Skips already processed videos to avoid duplicates
- **🎛️ Flexible Querying**: Toggle between RAG and direct LLM responses
- **🗨️ Conversation Context**: Follow-up queries with conversation history
- **🎯 Query Enhancement**: Sub-query generation and step-back query techniques
- **📈 Memory Optimization**: Chunked video processing for large files
- **🔐 Session Management**: Persistent chat sessions with MongoDB storage

### Technical Features
- **🚀 FastAPI Backend**: RESTful API with auto-generated documentation
- **🎨 Interactive API Docs**: Swagger UI and ReDoc for easy testing
- **📱 CORS Support**: Cross-origin requests enabled for frontend integration
- **🗃️ MongoDB Integration**: Persistent storage for videos and chat sessions
- **⚡ Vector Database**: Milvus for efficient similarity search
- **🔧 Configurable**: Flexible configuration for models and parameters

## 🚨 **Troubleshooting**

### Common Issues

1. **Milvus Connection Failed**
   ```bash
   # Check if Milvus is running
   docker ps | grep milvus
   # Restart if needed
   bash standalone_embed.sh restart
   ```

2. **LM Studio Connection Error**
   - Verify LM Studio is running and models are loaded
   - Check the TAILSCALE_SERVER URL in configuration
   - Ensure the embedding models are available

3. **MongoDB Connection Issues**
   ```bash
   # Check MongoDB status
   docker ps | grep mongo
   # Or for local installation
   sudo systemctl status mongod
   ```

4. **Frontend API Connection**
   - Ensure backend is running on port 8000
   - Check CORS configuration in FastAPI
   - Verify proxy settings in `vite.config.js`

### Performance Tips

- Use batch processing for multiple videos
- Adjust `CHUNK_SIZE` for better performance vs. accuracy trade-off
- Monitor Milvus memory usage for large datasets
- Use SSD storage for better vector database performance

## 📄 **License**

This project is intended for educational and research purposes. Please ensure compliance with YouTube's Terms of Service when processing videos.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For questions or issues, please create an issue in the repository.