import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class Config:
    """Configuration class for YouTube RAG system."""
    
    # Tailscale Server Configuration
    TAILSCALE_SERVER = "https://desktop-3oeimac.tail3b962f.ts.net" # the url where your lmstudio server is running.
    CHAT_API = TAILSCALE_SERVER + "/api/v0/chat/completions" 

    # Model Configuration
    MODEL: str = "qwen/qwen3-30b-a3b" # the model you downloaded in lmstudio for 
    DENSE_EMBEDDING_MODEL: str = "text-embedding-nomic-embed-text-v1.5"
    
    # Database Configuration
    MILVUS_HOST: str = "host.docker.internal"
    MILVUS_PORT: int = 19530
    DB_NAME = "milvus_demo"
    COLLECTION_NAME = "hybrid_demo"
    # DB_NAME: str = "youtube_rag"
    # COLLECTION_NAME: str = "video_summaries"
    MONGO_URI: str = "mongodb://localhost:27017/"
    MONGO_DB: str = "youtube_rag"
    
    # Processing Configuration
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 20
    BATCH_SIZE: int = 1
    
    # Directory Configuration
    MEDIA_DIR: str = field(default="media")
    TEMP_DIR: str = field(default="temp")
    RES_DIR: str = field(default="res")
    
    # Audio Processing Configuration
    WHISPER_MODEL_SIZE: str = "turbo"
    WHISPER_DEVICE: str = "cpu"
    CHUNK_DURATION: int = 30 # in seconds
    OVERLAP_DURATION: int = 2 # in seconds
    
    # API Configuration
    REQUEST_TIMEOUT: int = 30

    # Mongo Configuration
    MONGO_URI = 'mongodb://mongo:27017/'
    MONGO_DB = 'db'
    MONGO_COLLECTIONS = { "youtube_videos": {"name": "videos", "description": "collection containing youtube videos"}}
    
    # Prompt catalog configuration.
    PROMPT_CATALOG_PATH = './app/src/prompt/catalog.json'
