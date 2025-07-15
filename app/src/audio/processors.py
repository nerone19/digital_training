import os
import logging
from typing import List, Any, Callable, Optional
from pathlib import Path

from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import YoutubeAudioLoader

from .whisper_parser import WhisperParser
from src.config import Config
from src.utils import cleanup_directory,get_formatted_file_name

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.whisper_parser = WhisperParser(config)
    
    def transcribe_from_urls(self, urls: List[str], callback: Optional[Callable] = None) -> List[Any]:
        """Transcribe audio from YouTube URLs."""
        docs = []
        
        try:
            os.makedirs(self.config.TEMP_DIR, exist_ok=True)
            
            loader = GenericLoader(
                YoutubeAudioLoader(urls, self.config.TEMP_DIR),
                self.whisper_parser
            )
            
            for doc in loader.lazy_load():
                if callback:
                    callback(doc)
                
                doc_name = get_formatted_file_name(doc)
                if doc_name not in docs:
                    docs.append(doc_name)
                    
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise (f"Transcription failed: {e}")
        
        finally:
            # Clean up temporary files
            cleanup_directory(self.config.TEMP_DIR)
        
        return docs
    
    def save_transcript(self, doc: Any, output_dir: Optional[str] = None) -> None:
        """Save transcript document to file."""
        if output_dir is None:
            output_dir = self.config.RES_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not doc:
            raise ("Document cannot be None")
        
        filename = get_formatted_file_name(doc)
        file_path = os.path.join(output_dir, filename)
        
        logger.debug(f"Saving transcript to: {file_path}")
        
        mode = "a" if os.path.exists(file_path) else "w"
        try:
            with open(file_path, mode) as out_file:
                out_file.write(doc.page_content)
        except Exception as e:
            raise (f"Failed to save transcript: {e}")
    
    