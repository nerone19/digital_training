"""
YouTube Video Transcription and RAG System

This module provides functionality to:
1. Load and transcribe YouTube videos
2. Generate summaries using local LLM
3. Create embeddings and store in Milvus vector database
4. Perform hybrid search and RAG queries

Author: Refactored from original POC
"""

import os
import json
import glob
import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pytubefix import YouTube
from pytubefix.cli import on_progress
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, 
    DataType, Collection, AnnSearchRequest, RRFRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from langchain_community.document_loaders import YoutubeLoader, YoutubeAudioLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
import json
from pymongo import MongoClient

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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




mongo_collections = { "youtube_videos": {"name": "videos", "description": "collection containing youtube videos"}}

class MongoDB():

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MongoDB, cls).__new__(cls)
            cls.instance._client = MongoClient(Config.MONGO_URI)
            cls.instance._db = cls.instance._client[Config.MONGO_DB]  # Changed this line
        return cls.instance
     
    @property
    def client(self):
        return self._client
    
    @property
    def db(self):
        return self._db
    
    def list_all(self,colletion):
        cursor = self._db[colletion].find({})
        docs = []
        for doc in cursor:
            # Convert ObjectId to string to avoid serialization issues
            doc['_id'] = str(doc['_id'])
            docs.append(doc)
        return list(docs)
    
    def load_existing_data(self, collection):
        videos = self.list_all(collection)
        videos_data = []
        for video_chunk in videos:
            keys = video_chunk.keys()
            _, video_id = keys
            video_info = video_chunk[video_id]
            videos_data.append(video_info)
        return videos_data
    

    def populate_db_from_json(self, json_file, collection_name):
        col = self.db[collection_name]
        with open(json_file, "r") as file:
            json_data = json.load(file) 
        
        for k,v in json_data.items():
            col.insert_one({k:v})


class CustomWhisperParser(OpenAIWhisperParserLocal):
    """Custom Whisper parser for local speech recognition."""
    
    def __init__(self, batch_size: int = 8, chunk_length: int = 30):
        """Initialize the custom Whisper parser."""
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise ImportError(f"Required packages not found: {e}")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3"
        self.batch_size = batch_size
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            chunk_length_s=chunk_length
        )


class CustomEmbeddingModel(Embeddings):
    """Custom embedding model using LM Studio API."""
    
    def __init__(self, model_name: str, base_url: str = Config.TAILSCALE_SERVER + "/v1", 
                 api_key: str = "lm-studio", timeout: int = 30):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._test_connection()
    
    def _test_connection(self):
        """Test if LM Studio is accessible."""
        try:
            response = requests.get(f"{self.base_url}/models", 
                                  headers=self.headers, timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to LM Studio")
        except Exception as e:
            logger.warning(f"Could not connect to LM Studio: {e}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from LM Studio API."""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model_name, "input": texts}
        
        try:
            response = requests.post(url, headers=self.headers, 
                                   json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.exceptions.RequestException as e:
            raise Exception(f"LM Studio API error: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self._get_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_embeddings([text])[0]


class YouTubeProcessor:
    """Handles YouTube video processing and transcription."""
    
    def __init__(self, config: Config):
        self.config = config
        self.whisper_parser = CustomWhisperParser()
        
    def load_existing_transcripts(self) -> List[Dict[str, Any]]:
        """Load existing video transcripts from text files."""
        loaded_videos = []
        
        for doc_path in glob.glob(f"{self.config.MEDIA_DIR}/*.txt"):
            with open(doc_path, "r") as file:
                for line in file:
                    url = line.strip()
                    if not url:
                        continue
                        
                    try:
                        yt = YouTube(url)
                        title = yt.title
                        
                        loader = YoutubeLoader.from_youtube_url(
                            url, language=['en', 'it'], continue_on_failure=True
                        )
                        transcript = loader.load()
                        
                        loaded_videos.append({
                            "url": url, 
                            "title": title, 
                            "transcript": transcript
                        })
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        loaded_videos.append({
                            "url": url, 
                            "title": None, 
                            "transcript": None
                        })
        
        return loaded_videos
    
    def transcribe_videos(self, urls: List[str]) -> List[Any]:
        """Transcribe videos using Whisper."""
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        
        loader = GenericLoader(
            YoutubeAudioLoader(urls, self.config.TEMP_DIR),
            self.whisper_parser
        )
        docs = loader.load()
        
        # Clean up temporary files
        for filename in os.listdir(self.config.TEMP_DIR):
            file_path = os.path.join(self.config.TEMP_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return docs
    
    def save_transcripts(self, docs: List[Any]) -> List[str]:
        """Save transcripts to text files."""

        os.makedirs(self.config.RES_DIR, exist_ok=True)

        saved_files = []
        
        for doc in docs:
            filename = doc.metadata.get("source").split("/")[-1]
            file_path = os.path.join(self.config.RES_DIR, filename)
            
            with open(file_path, "w") as file:
                file.write(doc.page_content)
            
            saved_files.append(filename)
        
        return saved_files


class TextProcessor:
    """Handles text processing, chunking, and summarization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            base_url=f"{config.TAILSCALE_SERVER}/v1",
            api_key="lm-studio",
            model=config.MODEL,
        )
        
        # Load prompt catalog
        with open("./prompt_catalog.json") as f:
            self.prompt_catalog = json.load(f)
    
    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()
    
    @staticmethod
    def format_reasoning_prompt(prompt: str, no_think: bool = True) -> str:
        """Format prompt for reasoning models."""
        return prompt + " /no_think" if no_think else prompt
    
    @staticmethod
    def create_hash_id(text: str) -> str:
        """Create hash ID from text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def chunk_documents(self, docs: List[Any], urls: List[str], ids: List[str]) -> Dict[str, Dict]:
        """Split documents into chunks."""
        file_chunks = {}
        text_splitter = TokenTextSplitter(
            chunk_size=self.config.CHUNK_SIZE, 
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        
        for doc, url,id in zip(docs, urls, ids):
            chunks = text_splitter.split_text(doc.page_content)
            
            file_chunks[id] = {
                "chunks": chunks,
                "url": url,
                "summaries": []
            }
        
        return file_chunks
    
    def generate_summaries(self, file_chunks: Dict[str, Dict]) -> None:
        """Generate summaries for each chunk."""
        system_prompt = self.prompt_catalog.get('system', {}).get('transcript_summarizer', '')
        
        for file_data in file_chunks.values():
            for chunk in file_data['chunks']:
                messages = [
                    SystemMessage(system_prompt),
                    HumanMessage(self.format_reasoning_prompt(chunk))
                ]
                
                response = self.llm.invoke(messages)
                summary = self.remove_think_tags(response.content)
                file_data['summaries'].append(summary)


class VectorStore:
    """Handles vector database operations with Milvus."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = CustomEmbeddingModel(config.DENSE_EMBEDDING_MODEL)
        self.bge_model = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Milvus connection and database."""
        connections.connect(host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT)
        
        # Setup database
        from pymilvus import db
        existing_dbs = db.list_database()
        
        if self.config.DB_NAME in existing_dbs:
            db.using_database(self.config.DB_NAME)
            collections = utility.list_collections()
            for col_name in collections:
                Collection(col_name).drop()
            db.drop_database(self.config.DB_NAME)
        
        db.create_database(self.config.DB_NAME)
        db.using_database(self.config.DB_NAME)
    
    def create_collection(self):
        """Create Milvus collection with schema."""
        # Get embedding dimension
        sample_embedding = self.embedding_model.embed_query("test")
        embedding_dim = len(sample_embedding)
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, 
                       auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="hashed_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_id", dtype=DataType.INT8),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="summary", dtype=DataType.BOOL)
        ]
        
        schema = CollectionSchema(fields)
        
        if utility.has_collection(self.config.COLLECTION_NAME):
            Collection(self.config.COLLECTION_NAME).drop()
        
        self.collection = Collection(self.config.COLLECTION_NAME, schema, 
                                   consistency_level="Strong")
        
        # Create indices
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index("sparse_vector", sparse_index)
        
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.collection.create_index("dense_vector", dense_index)
        
        self.collection.load()
    
    def insert_documents(self, file_chunks: Dict[str, Dict]):
        """Insert documents into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=20
        )
        
        for file_id, data in file_chunks.items():
            for idx, (chunk, summary) in enumerate(zip(data['chunks'], data['summaries'])):
                # Insert summary
                sparse_emb = self.bge_model([summary])
                dense_emb = self.embedding_model.embed_query(summary)
                
                self.collection.insert({
                    "text": summary,
                    "hashed_title": file_id,
                    "summary": True,
                    "chunk_id": idx,
                    "sparse_vector": sparse_emb.get('sparse'),
                    "dense_vector": dense_emb,
                    "url": data.get('url')
                })
                
                # Insert chunk pieces
                documents = text_splitter.split_text(chunk)
                for doc in documents:
                    dense_chunk_emb = self.embedding_model.embed_query(doc)
                    sparse_chunk_emb = self.bge_model([doc])
                    
                    self.collection.insert({
                        "text": doc,
                        "hashed_title": file_id,
                        "summary": False,
                        "chunk_id": idx,
                        "sparse_vector": sparse_chunk_emb.get('sparse'),
                        "dense_vector": dense_chunk_emb,
                        "url": data.get('url')
                    })
    
    def hybrid_search(self, query: str, limit: int = 5) -> List[Any]:
        """Perform hybrid search using both sparse and dense vectors."""
        query_sparse = self.bge_model([query])['sparse']
        query_dense = self.embedding_model.embed_query(query)
        
        # Sparse search request
        sparse_request = AnnSearchRequest(
            data=query_sparse,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
        
        # Dense search request
        dense_request = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
        
        # Perform hybrid search
        reranker = RRFRanker()
        results = self.collection.hybrid_search(
            [sparse_request, dense_request],
            reranker,
            limit=limit
        )
        
        return results
    
    def sparse_search(self, query: str, limit: int = 10) -> List[Any]:
        """Perform sparse search on summaries only."""
        query_sparse = self.bge_model([query])['sparse']
        
        results = self.collection.search(
            [query_sparse],
            anns_field="sparse_vector",
            limit=limit,
            expr='summary==True',
            output_fields=["text", "summary", "chunk_id"],
            param={"metric_type": "IP", "params": {}}
        )[0]
        
        return [hit for hit in results]


class RAGSystem:
    """Main RAG system orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.youtube_processor = YouTubeProcessor(config)
        self.text_processor = TextProcessor(config)
        self.vector_store = VectorStore(config)
        
        # Ensure directories exist
        os.makedirs(config.MEDIA_DIR, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)

    def load_existing_data(self, data_file: str) -> Dict[str, Dict]:
        """Load existing processed data from file."""
        try:
            db = MongoDB()

            with open(data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file {data_file} not found")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data file {data_file}")
            return {}
    
    def save_batch_data(self, file_chunks: Dict[str, Dict], save_file: str):
        """Save or append batch data to JSON file."""
        existing_data = {}
        
        # Load existing data if file exists
        if os.path.exists(save_file):
            try:
                with open(save_file, 'r') as f:
                    existing_data = json.load(f)
                logger.info(f"Loaded existing data from {save_file} with {len(existing_data)} entries")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {save_file}, starting fresh")
                existing_data = {}
        
        # Merge new data with existing data
        existing_data.update(file_chunks)
        
        # Save the merged data
        with open(save_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        logger.info(f"Saved {len(file_chunks)} new entries to {save_file} (total: {len(existing_data)})")
    
    def check_already_processed(self, urls: List[str], existing_data: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
        """
        Check which URLs have already been processed by comparing hash IDs from filenames and URLs.
        
        Args:
            urls: List of YouTube URLs to check
            existing_data: Dictionary of existing processed data with hash IDs as keys
        
        Returns:
            Tuple of (new_urls, skipped_urls)
        """
        new_urls = []
        skipped_urls = []
        
        # Create a set of existing URLs for faster lookup
        existing_urls = set()
        for data in existing_data.values():
            if 'url' in data:
                existing_urls.add(data['url'])
        
        for url in urls:
            try:

                # Second check: Generate hash from video ID/filename
                yt = YouTube(url)
                video_id = yt.video_id

                # First check: URL already exists in data
                if url in existing_urls:
                    logger.info(f"Skipping already processed video (URL match): {url}")
                    skipped_urls.append(url)
                    continue
                
                
                if video_id in existing_data:
                    logger.info(f"Skipping already processed video (hash match): {url} (id: {video_id}...)")
                    skipped_urls.append(url)
                else:
                    new_urls.append({"url":url, "id" :video_id})
                    
            except Exception as e:
                logger.error(f"Error checking {url}: {e}")
                # If we can't check, assume it's new to be safe
                new_urls.append(url)
        
        logger.info(new_urls)
        return new_urls, skipped_urls

    def process_videos(self, urls: List[str], batch_size: int = 1, save_file: str = None) -> Dict[str, Dict]:
        """Process videos end-to-end with batching support and incremental saving."""
        logger.info(f"Processing {len(urls)} videos in batches of {batch_size}...")
        
        # Load existing data if save_file exists
        existing_data = {}
        if save_file and os.path.exists(save_file):
            existing_data = self.load_existing_data(save_file)
            logger.info(f"Loaded {len(existing_data)} existing entries from {save_file}")
        
        # Check which URLs are already processed
        new_urls, skipped_urls = self.check_already_processed(urls, existing_data)
        
        if skipped_urls:
            logger.info(f"Skipped {len(skipped_urls)} already processed videos")
        
        if not new_urls:
            logger.info("All videos already processed!")
            return existing_data
        
        logger.info(f"Processing {len(new_urls)} new videos")
        logger.info(new_urls)
        all_file_chunks = existing_data.copy()  # Start with existing data
        
        # Process only new videos in batches
        for i in range(0, len(new_urls), batch_size):
            batch_items = new_urls[i:i + batch_size]
            # Convert tuples to lists explicitly
            batch_urls = [item["url"] for item in batch_items]
            batch_ids = [item["id"] for item in batch_items]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_urls)} videos")
            try:
                # Transcribe batch
                docs = self.youtube_processor.transcribe_videos(batch_urls)
                
                if not docs:
                    logger.warning(f"No documents processed in batch {i//batch_size + 1}")
                    continue
                
                # Save transcripts
                self.youtube_processor.save_transcripts(docs)
                
                # Chunk documents
                file_chunks = self.text_processor.chunk_documents(docs, batch_urls, batch_ids)
                
                # Generate summaries
                self.text_processor.generate_summaries(file_chunks)
                
                # Save batch results immediately if save_file is provided
                if save_file:
                    self.save_batch_data(file_chunks, save_file)
                
                # Merge with existing chunks
                all_file_chunks.update(file_chunks)
                
                logger.info(f"Completed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch even if this one fails
                continue
        
        return all_file_chunks
    def setup_vector_store(self, file_chunks: Dict[str, Dict]):
        """Setup vector store with processed documents."""
        logger.info("Setting up vector store...")
        
        self.vector_store.create_collection()
        self.vector_store.insert_documents(file_chunks)
        
        logger.info("Vector store setup complete")
    
    def query(self, question: str, use_rag: bool = True) -> str:
        """Query the RAG system."""
        if not use_rag:
            # Direct LLM query
            messages = [HumanMessage(TextProcessor.format_reasoning_prompt(question))]
            logger.info(messages)
            print('mes', messages, flush=True)
            response = self.text_processor.llm.invoke(messages)
            return response.content
        
        # RAG query
        search_results = self.vector_store.sparse_search(question)
        context = "\n".join([str(hit) for hit in search_results])
        
        system_prompt = """You are an AI assistant. You are able to find answers to questions from the contextual passage snippets provided."""
        
        user_prompt = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """
        
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(self.text_processor.format_reasoning_prompt(user_prompt))
        ]
        
        response = self.text_processor.llm.invoke(messages)
        return self.text_processor.remove_think_tags(response.content)
    
    def generate_sub_queries(self, query: str) -> str:
        """Generate sub-queries for better retrieval."""
        system_prompt = """You are a helpful assistant that generates search queries based on an input query.
        
        Perform query decomposition. Given a user question, break it down into distinct sub questions that 
        you need to answer the original question.
        
        If there are acronyms or words you are not familiar with, do not try to rephrase them."""
        
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(self.text_processor.format_reasoning_prompt(query))
        ]
        
        response = self.text_processor.llm.invoke(messages)
        return self.text_processor.remove_think_tags(response.content)
    
    def generate_step_back_query(self, query: str) -> str:
        """Generate a step-back query for better context."""
        system_prompt = """You are an expert at taking specific questions and extracting more generic questions that get the 
        underlying principles needed to answer the specific question.
        
        Given a specific user question, write a more generic question that needs to be answered in order to answer
        the specific question.
        
        If you don't recognize a word or acronym, do not try to rewrite it."""
        
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(self.text_processor.format_reasoning_prompt(query))
        ]
        
        response = self.text_processor.llm.invoke(messages)
        return self.text_processor.remove_think_tags(response.content)




import argparse

def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a text file."""
    urls = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip empty lines and comments
                    urls.append(url)
    except FileNotFoundError:
        logger.error(f"URL file {file_path} not found")
        sys.exit(1)
    return urls


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration from command line arguments."""
    if args.model:
        config.MODEL = args.model
    if args.embedding_model:
        config.DENSE_EMBEDDING_MODEL = args.embedding_model
    if args.milvus_host:
        config.MILVUS_HOST = args.milvus_host
    if args.milvus_port:
        config.MILVUS_PORT = args.milvus_port
    
    return config


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """Update configuration from command line arguments."""
    if args.model:
        config.MODEL = args.model
    if args.embedding_model:
        config.DENSE_EMBEDDING_MODEL = args.embedding_model
    if args.milvus_host:
        config.MILVUS_HOST = args.milvus_host
    if args.milvus_port:
        config.MILVUS_PORT = args.milvus_port
    
    return config


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Transcription and RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Download and process videos
    python script.py download --urls "https://youtube.com/watch?v=..." --batch-size 2

    # Load from URL file
    python script.py download --url-file urls.txt --save-data processed_data.json

    # Query with existing data
    python script.py query --data-file processed_data.json --question "What is the main topic?"

    # Interactive mode
    python script.py query --data-file processed_data.json --interactive

    # Direct LLM query (no RAG)
    python script.py query --question "Explain calisthenics" --no-rag
            """
        )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Download/Process mode
    download_parser = subparsers.add_parser('download', help='Download and process videos')
    download_parser.add_argument('--url-file', help='File containing YouTube URLs (one per line)', default='media/videos.txt')
    download_parser.add_argument('--batch-size', type=int, default=1, 
                                help='Batch size for processing videos (default: 1)')
    download_parser.add_argument('--skip-vector-store', action='store_true',
                                help='Skip vector store setup (only process and save)', default=True)
    download_parser.add_argument('--save-data', default='processed_results.json',
                                help='File to save processed data (default: processed_results.json)')
    
    # Query mode
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('--question', help='Question to ask')
    query_parser.add_argument('--data-file', help='File containing processed data')
    query_parser.add_argument('--save-data', default='processed_results.json',
                                help='File to save processed data (default: processed_results.json)')
    
    # Global options
    parser.add_argument('--config', help='Configuration file (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging', default=False)
    parser.add_argument('--model', help='Override LLM model')
    parser.add_argument('--embedding-model', help='Override embedding model')
    parser.add_argument('--milvus-host', default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus-port', type=int, default=19530, help='Milvus port')
    
    return parser

import sys

def main():
    """Main execution function with argument parsing."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = Config()
    
    

    # Load custom config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except FileNotFoundError:
            logger.error(f"Configuration file {args.config} not found")
            sys.exit(1)

    config = update_config_from_args(config, args)
    # Initialize RAG system
    rag_system = RAGSystem(config)

    if args.mode == 'download':
        # Download and process mode
        urls = []
        
        if args.url_file:
            urls.extend(load_urls_from_file(args.url_file))
        
        if not urls:
            logger.error("No URLs provided. Use --url-file")
            sys.exit(1)
        
        logger.info(f"Processing {len(urls)} URLs")
        
        # Process videos
        file_chunks = rag_system.process_videos(urls, args.batch_size, args.save_data)
        
        # Save results
        with open(args.save_data, 'w') as f:
            json.dump(file_chunks, f, indent=2)
        logger.info(f"Processed data saved to {args.save_data}")
        
        # Setup vector store unless skipped
        if not args.skip_vector_store:
            rag_system.setup_vector_store(file_chunks)
            logger.info("Vector store setup complete")

    elif args.mode == 'query':            
        # Load existing data
        file_chunks = rag_system.load_existing_data(args.save_data)
        if not file_chunks:
            logger.error("No data loaded. Check your data file.")
            sys.exit(1)
        
        # Setup vector store
        rag_system.setup_vector_store(file_chunks)
        
        # Query the system
        if args.question:
            answer = rag_system.query(args.question)
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
        else:
            logger.error("No question provided. Use --question")
            sys.exit(1)


if __name__ == "__main__":
    main()