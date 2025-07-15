import logging
from typing import List, Dict, Any, Optional
from pytubefix import YouTube
from pytubefix.cli import on_progress

from src.config import Config
from src.core.exceptions import YouTubeDownloadError

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """Handles YouTube video downloading and metadata extraction."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_video_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from YouTube video."""
        try:
            yt = YouTube(url, on_progress_callback=on_progress)
            return {
                "url": url,
                "video_id": yt.video_id,
                "title": yt.title,
                "author": yt.author,
                "length": yt.length,
                "views": yt.views,
                "description": yt.description,
                "publish_date": yt.publish_date,
                "thumbnail_url": yt.thumbnail_url
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {url}: {e}")
            raise YouTubeDownloadError(f"Failed to extract metadata: {e}")
    
    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        try:
            yt = YouTube(url)
            return yt.video_id
        except Exception as e:
            logger.error(f"Failed to extract video ID from {url}: {e}")
            raise YouTubeDownloadError(f"Failed to extract video ID: {e}")
    
    def validate_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """Validate YouTube URLs and return video info."""
        valid_videos = []
        
        for url in urls:
            try:
                video_id = self.get_video_id(url)
                valid_videos.append({"url": url, "id": video_id})
            except :
                logger.warning(f"Skipping invalid URL: {url}")
                continue
        
        return valid_videos
    
    def check_video_accessibility(self, url: str) -> bool:
        """Check if video is accessible and not private/deleted."""
        try:
            yt = YouTube(url)
            # Try to access basic properties
            _ = yt.title
            _ = yt.length
            return True
        except Exception as e:
            logger.warning(f"Video not accessible: {url} - {e}")
            return False