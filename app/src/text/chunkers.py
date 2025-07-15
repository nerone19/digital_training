import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter

from src.config import Config

logger = logging.getLogger(__name__)

class TextChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

class TokenBasedChunker(TextChunker):
    """Token-based text chunking implementation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text using token-based chunking."""
        return self.splitter.split_text(text)

class RecursiveChunker(TextChunker):
    """Recursive character-based text chunking implementation."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text using recursive character chunking."""
        return self.splitter.split_text(text)
