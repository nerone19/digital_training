import os
import logging
from typing import List
from pathlib import Path
from langchain_core.documents import Document



logger = logging.getLogger(__name__)

def cleanup_directory(directory_path: str) -> None:
    """Clean up all files in a directory."""
    try:
        if not os.path.exists(directory_path):
            return
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        logger.debug(f"Cleaned up directory: {directory_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup directory {directory_path}: {e}")

# todo :move this somewhere else.
def get_formatted_file_name(doc: Document) -> str:
    return doc.metadata.get("source").split("/")[-1]
