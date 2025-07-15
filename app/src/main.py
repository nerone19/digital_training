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
import logging
import os
from typing import List, Dict, Any, Tuple
from pytubefix import YouTube
from pytubefix.cli import on_progress

from langchain_core.messages import HumanMessage, SystemMessage
import json

from src.db.mongo import MongoDB, ChatSessionRepository, VideoRepository
from src.chat.classes import ChatSessionManager
from src.chat.builders import ConversationContextBuilder
from src.db.milvus import VectorStore
from src.youtube.processor import YouTubeProcessor
from src.text.processor import TextProcessor
from src.config import Config


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RAGSystem:
    """Main RAG system orchestrator."""
    def __init__(self, config):
        self.config = config

        # Initialize your existing components
        self.youtube_processor = YouTubeProcessor(config)
        self.text_processor = TextProcessor(config)
        self.vector_store = VectorStore(config)
        
        # Initialize database layer
        MongoDB.configure(self.config.MONGO_URI, self.config.MONGO_DB)
        self.db = MongoDB()
        # Initialize repositories (data access layer)
        self.chat_repository = ChatSessionRepository(self.db)
        self.video_repository = VideoRepository(self.db)
        
        # Initialize chat manager (business logic layer)
        self.chat_manager = ChatSessionManager(self.chat_repository)
        # Ensure directories exist
        os.makedirs(config.MEDIA_DIR, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)

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
    
    def check_already_processed(self, urls: List[str], existing_data: List[Dict[str, Dict]]) -> Tuple[List[str], List[str]]:
        new_urls = []
        skipped_urls = []
        
        # Create sets for faster lookup
        existing_urls = set()
        existing_video_ids = set()
        
        for data in existing_data:
            video_keys = [k for k in data.keys() if k != '_id']
            
            # Handle empty video_keys
            if not video_keys:
                continue
                
            if len(video_keys) > 1:
                raise ValueError('Document structure changed. Expected only one key.')
            
            video_id = video_keys[0]
            existing_video_ids.add(video_id)
            
            # Safely access the nested data
            if video_id in data and 'url' in data[video_id]:
                existing_urls.add(data[video_id]['url'])
        
        for url in urls:
            try:
                yt = YouTube(url)
                video_id = yt.video_id
                
                # Check both URL and video ID
                if url in existing_urls or video_id in existing_video_ids:
                    logger.info(f"Skipping already processed video: {url}")
                    skipped_urls.append(url)
                else:
                    new_urls.append({"url": url, "id": video_id})
                    
            except Exception as e:
                logger.error(f"Error checking {url}: {e}")
                new_urls.append(url)  # Assume new if can't check
        
        return new_urls, skipped_urls

    def process_videos(self, urls: List[str], batch_size: int = 1) -> Dict[str, Dict]:
        """Process videos end-to-end with batching support and incremental saving."""
        logger.info(f"Processing {len(urls)} videos in batches of {batch_size}...")
        
        existing_videos = self.video_repository.list_all_videos()
        # Check which URLs are already processed
        new_urls, skipped_urls = self.check_already_processed(urls, existing_videos)
        
        
        if skipped_urls:
            logger.info(f"Skipped {len(skipped_urls)} already processed videos")
        
        if not new_urls:
            logger.info("All videos already processed!")
            return existing_videos
        
        logger.info(f"Processing {len(new_urls)} new videos")
        logger.info(new_urls)
        
        # Process only new videos in batches
        for i in range(0, len(new_urls), batch_size):
            batch_items = new_urls[i:i + batch_size]
            # Convert tuples to lists explicitly
            batch_urls = [item["url"] for item in batch_items]
            batch_ids = [item["id"] for item in batch_items]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_urls)} videos")
            try:
                # Transcribe batch
                docs_path = self.youtube_processor.transcribe_videos(batch_urls, self.text_processor.save_current_transcript)
                
                if not docs_path:
                    logger.warning(f"No documents processed in batch {i//batch_size + 1}")
                    continue
                
                # Chunk documents
                file_chunks = self.text_processor.chunk_documents(docs_path, batch_urls, batch_ids)
                
                # Generate summaries
                self.text_processor.generate_summaries(file_chunks)
                
                self.video_repository.save_video_data(file_chunks)
                
                logger.info(f"Completed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch even if this one fails
                continue
        

        return self.video_repository.list_all_videos()

    def setup_vector_store(self, documents: Dict[str, Dict]):
        """Setup vector store with processed documents."""
        logger.info("Setting up vector store...")
        
        self.vector_store.create_collection()
        self.vector_store.insert_documents(documents)
        
        logger.info("Vector store setup complete")

    def query(self, question: str, use_rag: bool = True, use_summary: bool = True) -> str:
        """Query the RAG system - your original method unchanged."""
        if not use_rag:
            messages = [HumanMessage(TextProcessor.format_reasoning_prompt(question))]
            # todo: make this more neat, less convoluted.
            response = self.text_processor.summarizer.llm.invoke(messages)
            return response.content, None
        
        if use_summary:
            custom_expr='summary==True'
        else: 
            custom_expr='summary==False'

        search_results = self.vector_store.sparse_search(question, expr = custom_expr)
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
        
        response = self.text_processor.summarizer.llm.invoke(messages)

        try:
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
            search_results = None


        return self.text_processor.remove_think_tags(response.content), search_results
    
    def chat_with_context(self, user_input: str, use_rag: bool = True, 
                         include_context: bool = True, 
                         max_context_messages: int = 10) -> Tuple[str, Dict[str, Any]]:
        """Send a message with conversation context and get response."""
        
        # Ensure we have an active session
        if not self.chat_manager.current_session_id:
            self.chat_manager.start_new_session()
        
        # Add user message to session
        self.chat_manager.add_user_message(user_input)
        
        # Build query with context if requested
        if include_context:
            conversation_context = self.chat_manager.get_conversation_context(
                max_context_messages, exclude_last=True  # Exclude the message we just added
            )
            enhanced_query = ConversationContextBuilder.build_rag_context(
                conversation_context, user_input
            )
        else:
            enhanced_query = user_input
        
        # Get response from RAG system
        response_content, search_results = self.query(enhanced_query, use_rag=use_rag)
        
        # Prepare metadata
        metadata = {
            "use_rag": use_rag,
            "include_context": include_context,
            "context_length": len(conversation_context) if include_context else 0,
            "session_id": self.chat_manager.current_session_id,
            "search_results": search_results
        }
        
        # Add assistant message to session
        self.chat_manager.add_assistant_message(response_content, metadata)
        
        return response_content, metadata
    
    
    def create_new_session(self, question, answer, metadata):

        if not self.chat_manager.current_session_id:
            self.chat_manager.start_new_session()

        self.chat_manager.add_user_message(question)
        # Add assistant message to session
        self.chat_manager.add_assistant_message(answer, metadata)
        return self.chat_manager.current_session_id

    
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
        
        response = self.text_processor.summarizer.invoke(messages)
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
        
        response = self.text_processor.summarizer.invoke(messages)
        return self.text_processor.remove_think_tags(response.content)


import sys

def main():
    """Main execution function with argument parsing."""


    # Load configuration
    config = Config()
    
    # Initialize RAG system
    rag_system = RAGSystem(config)


if __name__ == "__main__":
    main()