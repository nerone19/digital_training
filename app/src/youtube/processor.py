import os
import glob
import logging
from typing import List, Dict, Any, Optional, Callable

from langchain_community.document_loaders import YoutubeLoader

from .downloader import YouTubeDownloader
from src.audio.processors import AudioProcessor
from src.core.exceptions import YouTubeDownloadError
from src.config import Config

logger = logging.getLogger(__name__)

class YouTubeProcessor:
    """Handles YouTube video processing and transcription."""
    
    def __init__(self, config: Config):
        self.config = config
        self.downloader = YouTubeDownloader(config)
        self.audio_processor = AudioProcessor(config)
    
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
                        metadata = self.downloader.get_video_metadata(url)
                        loader = YoutubeLoader.from_youtube_url(
                            url, 
                            language=['en', 'it'], 
                            continue_on_failure=True
                        )
                        transcript = loader.load()
                        
                        loaded_videos.append({
                            "url": url,
                            "title": metadata["title"],
                            "transcript": transcript,
                            "metadata": metadata
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        loaded_videos.append({
                            "url": url,
                            "title": None,
                            "transcript": None,
                            "metadata": None,
                            "error": str(e)
                        })
        
        return loaded_videos
    
    def transcribe_videos(self, urls: List[str], callback: Optional[Callable] = None) -> List[Any]:
        """Transcribe videos using Whisper."""
        # Validate URLs first
        valid_videos = self.downloader.validate_urls(urls)
        valid_urls = [video["url"] for video in valid_videos]
        
        if not valid_urls:
            raise YouTubeDownloadError("No valid YouTube URLs provided")
        
        logger.info(f"Transcribing {len(valid_urls)} videos")
        
        # Process transcription
        def transcript_callback(doc):
            if callback:
                callback(doc)
            self.audio_processor.save_transcript(doc)
        
        return self.audio_processor.transcribe_from_urls(valid_urls, transcript_callback)
    
    def get_video_info_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for multiple videos."""
        video_info = []
        
        for url in urls:
            try:
                metadata = self.downloader.get_video_metadata(url)
                video_info.append(metadata)
            except YouTubeDownloadError as e:
                logger.error(f"Failed to get info for {url}: {e}")
                video_info.append({
                    "url": url,
                    "error": str(e),
                    "accessible": False
                })
        
        return video_info
    
    def filter_accessible_videos(self, urls: List[str]) -> List[str]:
        """Filter out inaccessible or private videos."""
        accessible_urls = []
        
        for url in urls:
            if self.downloader.check_video_accessibility(url):
                accessible_urls.append(url)
            else:
                logger.warning(f"Skipping inaccessible video: {url}")
        
        return accessible_urls
    