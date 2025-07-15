import tempfile
import logging
import numpy as np
from faster_whisper import WhisperModel
from typing import Iterator, Optional, Callable
from langchain_core.documents import Document
from langchain_core.document_loaders.blob_loaders import Blob
import librosa
from pydub import AudioSegment
import tempfile
import os

from src.core.exceptions import AudioProcessingError
from src.config import Config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperParser:
    def __init__(self, config: Config):
        """Initialize the Whisper processor."""
        self.config = config
        self.model = config.WHISPER_MODEL_SIZE
        self.chunk_duration = config.CHUNK_DURATION
        self.chunk_overlap = config.CHUNK_OVERLAP
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize Whisper model."""
        try:
            self.model = WhisperModel(
                self.model,
                device=self.config.WHISPER_DEVICE,
                compute_type="float32"
            )
            logger.info(f"Initialized Whisper model: {self.config.WHISPER_MODEL_SIZE}")
        except Exception as e:
            raise AudioProcessingError(f"Failed to initialize Whisper model: {e}")
        
    def process_audio_chunk(self, audio_data, sample_rate=16000):
        """
        Process a single audio chunk
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Convert to the format Whisper expects
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1
                )
                audio_segment.export(temp_file.name, format="wav")
                
                # Transcribe
                segments, info = self.model.transcribe(temp_file.name, beam_size=5)
                text = " ".join([segment.text for segment in segments])
                
                # Clean up
                os.unlink(temp_file.name)
                
                return text.strip()
                
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return ""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Lazily parse the blob and yield Document objects for each chunk
        Args:
            blob: The audio blob to parse
        Yields:
            Document objects containing transcribed text chunks
        """
        # Save blob to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(blob.as_bytes())
            temp_file_path = temp_file.name

        try:
            # Load audio file
            audio, sr = librosa.load(temp_file_path, sr=16000)
            chunk_samples = int(self.chunk_duration * sr)
            overlap_samples = int(self.chunk_overlap * sr)
            
            chunk_index = 0
            for i in range(0, len(audio), chunk_samples - overlap_samples):
                chunk = audio[i:i + chunk_samples]
                
                # Skip very short chunks
                if len(chunk) < sr:  # Less than 1 second
                    continue
                
                text = self.process_audio_chunk(chunk, sr)
                if text:
                    # Calculate time metadata
                    start_time = i / sr
                    end_time = min((i + len(chunk)) / sr, len(audio) / sr)
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": blob.source if hasattr(blob, 'source') else "audio_file",
                            "chunk_index": chunk_index,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": end_time - start_time,
                            "model_size": self.model,
                            "language": getattr(self.model, 'detected_language', 'unknown')
                        }
                    )
                    yield doc
                    chunk_index += 1
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
