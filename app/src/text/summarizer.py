import json
import logging
import re
import os 
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Config
from src.core.exceptions import YouTubeRAGException

logger = logging.getLogger(__name__)

class Summarizer(ABC):
    """Abstract base class for text summarization."""
    
    @abstractmethod
    def summarize(self, text: str) -> str:
        """Generate summary for given text."""
        pass
    
    @abstractmethod
    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Generate summaries for multiple texts."""
        pass

class LLMSummarizer(Summarizer):
    """LLM-based text summarization implementation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            base_url=f"{config.TAILSCALE_SERVER}/v1",
            api_key="lm-studio",
            model=config.MODEL,
        )
        self.prompt_catalog = self._load_prompt_catalog()
    
    def _load_prompt_catalog(self) -> Dict[str, Any]:
        """Load prompt catalog from JSON file."""
        try:
            with open(Config.PROMPT_CATALOG_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Prompt catalog not found at {Config.PROMPT_CATALOG_PATH}, using default prompts")
            return {"system": {"transcript_summarizer": "You are a helpful assistant that summarizes text."}}
    
    
    def summarize(self, text: str) -> str:
        """Generate summary for given text."""
        system_prompt = self.prompt_catalog.get('system', {}).get('transcript_summarizer', '')
        
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(self.format_reasoning_prompt(text))
        ]
        
        try:
            response = self.llm.invoke(messages)
            return self.remove_think_tags(response.content)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise YouTubeRAGException(f"Summarization failed: {e}")
    
    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Generate summaries for multiple texts."""
        summaries = []
        
        for i, text in enumerate(texts):
            try:
                summary = self.summarize(text)
                summaries.append(summary)
                logger.debug(f"Generated summary {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Failed to summarize text {i+1}: {e}")
                summaries.append("")  # Add empty summary for failed attempts
        
        return summaries
    
    @staticmethod
    def format_reasoning_prompt(prompt: str, no_think: bool = True) -> str:
        """Format prompt for reasoning models."""
        return prompt + " /no_think" if no_think else prompt
    
    @staticmethod
    def remove_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        pattern = r'<think>.*?</think>'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()