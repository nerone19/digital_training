# Configuration
class Config:
    """Configuration settings for the YouTube RAG system."""
    
    MODEL = "qwen/qwen3-30b-a3b"
    TAILSCALE_SERVER = "https://desktop-3oeimac.tail3b962f.ts.net"
    CHAT_API = TAILSCALE_SERVER + "/api/v0/chat/completions"
    EMBEDDING_MODEL = "text-embedding-qwen3-embedding-8b@q5_0"
    DENSE_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
    
    MEDIA_DIR = "./media/it"
    TEMP_DIR = "./temp"
    RES_DIR = './res'
    CHUNK_SIZE = 3200
    CHUNK_OVERLAP = 40
    BATCH_SIZE = 8
    CHUNK_LENGTH = 30
    
    # Milvus settings
    MILVUS_HOST = "127.0.0.1"
    MILVUS_PORT = 19530
    DB_NAME = "milvus_demo"
    COLLECTION_NAME = "hybrid_demo"
    MONGO_URI = 'mongodb://localhost:27017/'
    MONGO_DB = 'db'
    # todo: this is not meaningful yet. redo
    MONGO_COLLECTIONS = { "youtube_videos": {"name": "videos", "description": "collection containing youtube videos"}}
