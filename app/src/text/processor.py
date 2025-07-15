import os
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from .chunkers import TextChunker, TokenBasedChunker
from .summarizer import Summarizer, LLMSummarizer
from src.config import Config
from src.utils import get_formatted_file_name

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text processing, chunking, and summarization."""
    
    def __init__(self, config: Config, chunker: Optional[TextChunker] = None, 
                 summarizer: Optional[Summarizer] = None):
        self.config = config
        self.chunker = chunker or TokenBasedChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.summarizer = summarizer or LLMSummarizer(config)
    
    def chunk_documents(self, docs: List[str], urls: List[str], ids: List[str]) -> Dict[str, Dict]:
        """Split documents into chunks."""
        file_chunks = {}
        
        for doc, url, doc_id in zip(docs, urls, ids):
            file_path = os.path.join(self.config.RES_DIR, doc)
            
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                chunks = self.chunker.chunk_text(content)
                
                file_chunks[doc_id] = {
                    "chunks": chunks,
                    "url": url,
                    "chunk_size": self.config.CHUNK_SIZE,
                    "chunk_overlap": self.config.CHUNK_OVERLAP,
                    "summaries": [],
                    "document_path": file_path
                }
                
                logger.info(f"Chunked document {doc} into {len(chunks)} chunks")
                
            except FileNotFoundError:
                logger.error(f"Document file not found: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error processing document {doc}: {e}")
                continue
        
        return file_chunks
    

    def generate_summaries(self, file_chunks: Dict[str, Dict]) -> None:
        """Generate summaries for each chunk."""
        for file_id, file_data in file_chunks.items():
            chunks = file_data['chunks']
            logger.info(f"Generating summaries for {len(chunks)} chunks in {file_id}")
            
            try:
                summaries = self.summarizer.summarize_batch(chunks)
                file_data['summaries'] = summaries
                logger.info(f"Generated {len(summaries)} summaries for {file_id}")
            except Exception as e:
                logger.error(f"Failed to generate summaries for {file_id}: {e}")
                file_data['summaries'] = [""] * len(chunks)  # Empty summaries as fallback
    
    def process_single_document(self, file_path: str, url: str, doc_id: str) -> Dict[str, Any]:
        """Process a single document (chunk and summarize)."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            chunks = self.chunker.chunk_text(content)
            summaries = self.summarizer.summarize_batch(chunks)
            
            return {
                "chunks": chunks,
                "summaries": summaries,
                "url": url,
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "document_path": file_path
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def get_document_stats(self, file_chunks: Dict[str, Dict]) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        total_chunks = sum(len(data['chunks']) for data in file_chunks.values())
        total_summaries = sum(len(data['summaries']) for data in file_chunks.values())
        
        return {
            "total_documents": len(file_chunks),
            "total_chunks": total_chunks,
            "total_summaries": total_summaries,
            "average_chunks_per_doc": total_chunks / len(file_chunks) if file_chunks else 0
        }
    
    def save_current_transcript(self, doc: Document):
        os.makedirs(self.config.RES_DIR, exist_ok=True)
        if not doc:
            raise(f"doc cannot be {doc}" )
        filename = get_formatted_file_name(doc)
        file_path = os.path.join(self.config.RES_DIR, filename)
        print(file_path)
        if not os.path.exists(file_path):

            with open(file_path, "w") as out_file:
                out_file.write(doc.page_content)
        else:
            with open(file_path, "a") as out_file:
                out_file.write(doc.page_content)

    @staticmethod
    def format_reasoning_prompt(prompt: str, no_think: bool = True) -> str:
        """Format prompt for reasoning models."""
        return prompt + " /no_think" if no_think else prompt
    
    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()
    