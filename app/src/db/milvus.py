from typing import Any, List, Optional
import logging
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, 
    DataType, Collection, AnnSearchRequest, RRFRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from langchain_community.document_loaders import YoutubeLoader, YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from src.config import Config
from src.embeddings.custom import CustomEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector database operations with Milvus."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = CustomEmbeddingModel(config.DENSE_EMBEDDING_MODEL)
        self.bge_model = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection = None
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Milvus connection and database."""
        connections.connect(host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT)
        
        # Setup database
        from pymilvus import db
        existing_dbs = db.list_database()
        logger.info(existing_dbs)
        if self.config.DB_NAME in existing_dbs:
            db.using_database(self.config.DB_NAME)
            collections = utility.list_collections()
            logger.info(existing_dbs)
            for col_name in collections:
                Collection(col_name).drop()
            db.drop_database(self.config.DB_NAME)
        
        db.create_database(self.config.DB_NAME)
        db.using_database(self.config.DB_NAME)
    
    def create_collection(self):
        """Create Milvus collection with schema."""
        # Get embedding dimension
        sample_embedding = self.embedding_model.embed_query("test")
        embedding_dim = len(sample_embedding)
        
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, 
                       auto_id=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="hashed_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_id", dtype=DataType.INT8),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="summary", dtype=DataType.BOOL)
        ]
        
        schema = CollectionSchema(fields)
        
        if utility.has_collection(self.config.COLLECTION_NAME):
            Collection(self.config.COLLECTION_NAME).drop()
        
        self.collection = Collection(self.config.COLLECTION_NAME, schema, 
                                   consistency_level="Strong")
        
        # Create indices
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.collection.create_index("sparse_vector", sparse_index)
        
        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        self.collection.create_index("dense_vector", dense_index)
        
        self.collection.load()
    
    def insert_documents(self, documents):
        """Insert documents into vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=20
        )
        
        # Iterate through the MongoDB cursor
        for document in documents:
            video_key = [k for k in document.keys() if k!='_id']
            video_id = video_key[0]
            if len(video_key) > 1:
                raise('document structure is changed. we only expect one key.')
            data = document[video_id]  # The document itself contains the data
            
            for idx, (chunk, summary) in enumerate(zip(data['chunks'], data['summaries'])):
                # Insert summary (split if too long)
                if len(summary) > 512:
                    summary_chunks = text_splitter.split_text(summary)
                    for sub_idx, summary_chunk in enumerate(summary_chunks):
                        sparse_emb = self.bge_model([summary_chunk])
                        dense_emb = self.embedding_model.embed_query(summary_chunk)
                        self.collection.insert({
                            "text": summary_chunk,
                            "hashed_title": video_id,
                            "summary": True,
                            "chunk_id": idx,
                            "sparse_vector": sparse_emb.get('sparse'),
                            "dense_vector": dense_emb,
                            "url": data.get('url')
                        })
                else:
                    sparse_emb = self.bge_model([summary])
                    dense_emb = self.embedding_model.embed_query(summary)
                    self.collection.insert({
                        "text": summary,
                        "hashed_title": video_id,
                        "summary": True,
                        "chunk_id": idx,
                        "sparse_vector": sparse_emb.get('sparse'),
                        "dense_vector": dense_emb,
                        "url": data.get('url')
                    })
                
                # Insert chunk pieces
                documents = text_splitter.split_text(chunk)
                for doc in documents:
                    dense_chunk_emb = self.embedding_model.embed_query(doc)
                    sparse_chunk_emb = self.bge_model([doc])
                    self.collection.insert({
                        "text": doc,
                        "hashed_title": video_id,
                        "summary": False,
                        "chunk_id": idx,
                        "sparse_vector": sparse_chunk_emb.get('sparse'),
                        "dense_vector": dense_chunk_emb,
                        "url": data.get('url')
                    })
        
    def hybrid_search(self, query: str, limit: int = 5) -> List[Any]:
        """Perform hybrid search using both sparse and dense vectors."""
        query_sparse = self.bge_model([query])['sparse']
        query_dense = self.embedding_model.embed_query(query)
        
        # Sparse search request
        sparse_request = AnnSearchRequest(
            data=query_sparse,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
        
        # Dense search request
        dense_request = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
        
        # Perform hybrid search
        reranker = RRFRanker()
        results = self.collection.hybrid_search(
            [sparse_request, dense_request],
            reranker,
            limit=limit
        )
        
        return results
    
    def sparse_search(self, query: str, limit: int = 10, expr: Optional[str] = None) -> List[Any]:
        """Perform sparse search on summaries only."""
        query_sparse = self.bge_model([query])['sparse']
        
        if expr:
            results = self.collection.search(
                [query_sparse],
                anns_field="sparse_vector",
                limit=limit,
                expr=expr,
                output_fields=["text", "summary", "chunk_id", "url"],
                param={"metric_type": "IP", "params": {}}
            )[0]
        else: 
            results = self.collection.search(
            [query_sparse],
            anns_field="sparse_vector",
            limit=limit,
            output_fields=["text", "summary", "chunk_id", "url"],
            param={"metric_type": "IP", "params": {}}
        )[0]
        
        return [hit for hit in results]
