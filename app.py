"""
FastAPI Backend for YouTube Video Transcription and RAG System

This module provides a REST API for:
1. Processing YouTube videos (transcription, summarization)
2. Querying the RAG system
3. Managing the vector database

Author: Generated from main.py POC
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import json
import logging
from contextlib import asynccontextmanager

# Import existing classes from main.py
from main import Config, RAGSystem, logger, MongoDB

# Pydantic models for API
class VideoProcessRequest(BaseModel):
    urls: List[str]
    batch_size: Optional[int] = 1

class QueryRequest(BaseModel):
    question: str
    use_rag: Optional[bool] = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    search_results: Optional[List[Dict]] = None

class ProcessStatusResponse(BaseModel):
    status: str
    message: str
    processed_count: Optional[int] = None
    total_count: Optional[int] = None

# Global variables for the RAG system
rag_system: Optional[RAGSystem] = None
config: Optional[Config] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system, config
    config = Config()
    rag_system = RAGSystem(config)
    logger.info("FastAPI application started - RAG system initialized")
    yield
    # Shutdown
    logger.info("FastAPI application shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="YouTube RAG System API",
    description="REST API for YouTube video processing and RAG queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YouTube RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process - Process YouTube videos",
            "query": "/query - Query the RAG system",
            "status": "/status - Get system status",
            "data": "/data - Get processed data info"
        }
    }

@app.post("/process", response_model=ProcessStatusResponse)
async def process_videos(
    request: VideoProcessRequest,
    background_tasks: BackgroundTasks,
    save_file: str = Query(default="processed_results.json", description="File to save results")
):
    """
    Process YouTube videos in the background.
    Returns immediately with a status message.
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    
    # Validate URLs (basic check)
    for url in request.urls:
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {url}")
    
    # Add background task
    background_tasks.add_task(
        process_videos_background,
        request.urls,
        request.batch_size,
        save_file
    )
    
    return ProcessStatusResponse(
        status="accepted",
        message=f"Processing {len(request.urls)} videos in background",
        total_count=len(request.urls)
    )

async def process_videos_background(urls: List[str], batch_size: int, save_file: str):
    """Background task to process videos."""
    try:
        logger.info(f"Starting background processing of {len(urls)} videos")
        
        # Process videos
        file_chunks = rag_system.process_videos(urls, batch_size, save_file)
        
        # Setup vector store
        rag_system.setup_vector_store(file_chunks)
        
        logger.info(f"Completed processing {len(urls)} videos")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system with a question."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Load existing data (this should be optimized in production)
        file_chunks = rag_system.load_existing_data("processed_results.json")
        
        if not file_chunks:
            raise HTTPException(
                status_code=404, 
                detail="No processed data found. Please process some videos first."
            )
        
        # Setup vector store if not already done
        try:
            rag_system.setup_vector_store(file_chunks)
        except Exception as e:
            logger.warning(f"Vector store setup warning: {e}")
        
        # Perform query
        answer = rag_system.query(request.question, request.use_rag)
        
        # Get search results if using RAG
        search_results = None
        if request.use_rag:
            try:
                search_results = rag_system.vector_store.sparse_search(request.question, limit=5)
                # Convert search results to serializable format
                search_results = [
                    {
                        "text": hit.entity.get("text", ""),
                        "score": hit.score,
                        "chunk_id": hit.entity.get("chunk_id", 0),
                        "url": hit.entity.get("url", "")
                    }
                    for hit in search_results
                ]
            except Exception as e:
                logger.warning(f"Could not retrieve search results: {e}")
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            search_results=search_results
        )
        
    except Exception as e:
        logger.error(f"Error in query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status and configuration."""
    if not rag_system or not config:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Try to get data file info
    data_info = {}
    try:
        file_chunks = rag_system.load_existing_data("processed_results.json")
        data_info = {
            "processed_videos": len(file_chunks),
            "data_file_exists": True
        }
    except:
        data_info = {
            "processed_videos": 0,
            "data_file_exists": False
        }
    
    return {
        "status": "running",
        "config": {
            "model": config.MODEL,
            "embedding_model": config.DENSE_EMBEDDING_MODEL,
            "milvus_host": config.MILVUS_HOST,
            "milvus_port": config.MILVUS_PORT,
            "collection_name": config.COLLECTION_NAME
        },
        "data": data_info
    }

@app.get("/data")
async def get_data_info():
    """Get information about processed data."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:

        db = MongoDB()
        return db.list_all(colletion='videos')
        file_chunks = rag_system.load_existing_data("processed_results.json")
        
        if not file_chunks:
            return {
                "videos_count": 0,
                "videos": []
            }
        
        videos_info = []
        for video_id, data in file_chunks.items():
            videos_info.append({
                "video_id": video_id,
                "url": data.get("url", ""),
                "chunks_count": len(data.get("chunks", [])),
                "summaries_count": len(data.get("summaries", []))
            })
        
        return {
            "videos_count": len(file_chunks),
            "videos": videos_info
        }
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get data info: {str(e)}")

@app.post("/generate-sub-queries")
async def generate_sub_queries(request: QueryRequest):
    """Generate sub-queries for better retrieval."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        sub_queries = rag_system.generate_sub_queries(request.question)
        return {
            "original_query": request.question,
            "sub_queries": sub_queries
        }
    except Exception as e:
        logger.error(f"Error generating sub-queries: {e}")
        raise HTTPException(status_code=500, detail=f"Sub-query generation failed: {str(e)}")

@app.post("/generate-step-back-query")
async def generate_step_back_query(request: QueryRequest):
    """Generate a step-back query for better context."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        step_back_query = rag_system.generate_step_back_query(request.question)
        return {
            "original_query": request.question,
            "step_back_query": step_back_query
        }
    except Exception as e:
        logger.error(f"Error generating step-back query: {e}")
        raise HTTPException(status_code=500, detail=f"Step-back query generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)