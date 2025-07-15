from typing import List
import logging 

import requests
from src.config import Config
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomEmbeddingModel(Embeddings):
    """Custom embedding model using LM Studio API."""
    
    def __init__(self, model_name: str, base_url: str = Config.TAILSCALE_SERVER + "/v1", 
                 api_key: str = "lm-studio", timeout: int = 30):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self._test_connection()
    
    def _test_connection(self):
        """Test if LM Studio is accessible."""
        try:
            response = requests.get(f"{self.base_url}/models", 
                                  headers=self.headers, timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to LM Studio")
        except Exception as e:
            logger.warning(f"Could not connect to LM Studio: {e}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from LM Studio API."""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model_name, "input": texts}
        
        try:
            response = requests.post(url, headers=self.headers, 
                                   json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.exceptions.RequestException as e:
            raise Exception(f"LM Studio API error: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self._get_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_embeddings([text])[0]
