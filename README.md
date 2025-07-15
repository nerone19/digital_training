# 🎬 YouTube RAG System

An intelligent system that processes YouTube videos through transcription, summarization, and retrieval-augmented generation (RAG) to enable conversational querying of video content.

## 🚀 **Quick Start**

### Prerequisites
- Docker and Docker Compose
- ~20GB free disk space (for AI models)

### 1. Clone Repository
```bash
git clone <repository-url>
cd digital_training
```

### 2. Start Milvus Vector Database
```bash
make milvus/up
```

### 3. Download AI Models
```bash
make download-models
```

### 4. Start All Services
```bash
make up
```

### 5. Access Application
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:5000/docs
- **API Status**: http://localhost:5000/status

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Vector DB     │
│   (Vue.js)      │◄──►│   (FastAPI)     │◄──►│   (Milvus)      │
│   Port 3000     │    │   Port 5000     │    │   Port 19530    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   AI Models     │    │   Document DB   │
                       │  (Local Cache)  │    │   (MongoDB)     │
                       │  BGE-M3/Whisper │    │   Port 27017    │
                       └─────────────────┘    └─────────────────┘
```

## 🛠️ **Technology Stack**

- **FastAPI**: REST API backend
- **Vue.js 3**: Frontend chat interface
- **Milvus**: Vector database for embeddings
- **MongoDB**: Document storage
- **Faster-Whisper**: Audio transcription
- **BGE-M3**: Embedding model for hybrid search
- **Docker**: Containerized deployment

## 📖 **Usage**

### Demo Video
See the system in action:

![Demo Video](media/videos/Kooha-2025-07-15-19-35-57.webm)

> **Note**: If the video doesn't display properly, you can download it from `media/videos/Kooha-2025-07-15-19-35-57.webm`

### Web Interface
1. Open http://localhost:3000
2. Process YouTube videos via API
3. Ask questions about video content
4. View source references and citations

### API Endpoints

**Process Videos**
```bash
curl -X POST "http://localhost:5000/process" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://youtube.com/watch?v=..."], "batch_size": 1}'
```

**Query System**
```bash
curl -X POST "http://localhost:5000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this video about?", "use_rag": true}'
```

## 🔧 **Makefile Commands**

```bash
make milvus/up        # Start Milvus vector database
make download-models  # Download AI models (BGE-M3, Whisper)
make up              # Start all services with Docker Compose
make api/up          # Start API service only
make milvus/down     # Stop Milvus
make milvus/delete   # Delete Milvus data
```

## 📁 **Project Structure**

```
digital_training/
├── app/                    # FastAPI backend
│   ├── app.py             # Main API server
│   ├── requirements.txt   # Python dependencies
│   └── src/               # Source modules
│       ├── audio/         # Audio processing
│       ├── chat/          # Chat session management
│       ├── db/            # Database connections
│       ├── embeddings/    # Embedding models
│       ├── text/          # Text processing
│       └── youtube/       # YouTube processing
├── fe/                    # Vue.js frontend
│   ├── src/App.vue       # Main chat component
│   ├── package.json      # Frontend dependencies
│   └── vite.config.js    # Build configuration
├── scripts/              # Setup scripts
│   ├── initialize_models.sh
│   └── download_models.py
├── docker-compose.yml    # Service orchestration
├── Dockerfile           # Multi-stage container build
└── Makefile            # Automation commands
```

## 🔍 **Features**

- **Audio Processing**: Extracts audio from YouTube videos using faster-whisper
- **Hybrid Search**: Combines semantic and keyword search with BGE-M3 embeddings
- **Chat Interface**: Modern Vue.js frontend with real-time responses
- **Background Processing**: Non-blocking video processing with FastAPI
- **Session Management**: Persistent chat sessions with MongoDB
- **Docker Deployment**: Containerized for easy deployment

## 🚨 **Troubleshooting**

**Milvus Issues**
```bash
make milvus/down && make milvus/up
```

**Missing Models**
```bash
make download-models
```

**Service Status**
```bash
docker-compose ps
curl http://localhost:5000/status
```

## 📄 **License**

Educational and research purposes. Ensure compliance with YouTube's Terms of Service.