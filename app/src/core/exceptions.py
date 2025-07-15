class YouTubeRAGException(Exception):
    """Base exception for YouTube RAG system."""
    pass

class AudioProcessingError(YouTubeRAGException):
    """Exception raised during audio processing."""
    pass

class EmbeddingError(YouTubeRAGException):
    """Exception raised during embedding generation."""
    pass

class VectorStoreError(YouTubeRAGException):
    """Exception raised during vector store operations."""
    pass

class YouTubeDownloadError(YouTubeRAGException):
    """Exception raised during YouTube video download."""
    pass

class ConfigurationError(YouTubeRAGException):
    """Exception raised for configuration issues."""
    pass