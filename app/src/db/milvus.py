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
        self._initialized = False
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Milvus connection and database."""
        connections.connect(host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT, db_name=self.config.DB_NAME)
        
        from pymilvus import db
        existing_dbs = db.list_database()
        
        if self.config.DB_NAME not in existing_dbs:
            db.create_database(self.config.DB_NAME)
        
        db.using_database(self.config.DB_NAME)
    
    def initialized(self, value=None):
        """Get or set the initialized state."""
        if value is not None:
            self._initialized = value
        return self._initialized
    
    def create_collection(self):
        """Create Milvus collection with schema only if it doesn't exist."""
        # Only recreate if collection doesn't exist or schema changed
        if not utility.has_collection(self.config.COLLECTION_NAME):
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
            self.collection = Collection(self.config.COLLECTION_NAME, schema, 
                                    consistency_level="Strong")
            
            # Create indices
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.collection.create_index("sparse_vector", sparse_index)
            
            dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
            self.collection.create_index("dense_vector", dense_index)
        else:
            # Collection exists, just connect to it
            self.collection = Collection(self.config.COLLECTION_NAME)
        
        self.collection.load()
        self._initialized = True
        
    def get_existing_video_ids(self):
        """Get all existing video IDs from the collection."""
        try:
            # Query all unique hashed_title values
            results = self.collection.query(
                expr="pk != ''",  # Get all records
                output_fields=["hashed_title"]
            )
            return set(result["hashed_title"] for result in results)
        except Exception as e:
            logger.warning(f"Could not fetch existing video IDs: {e}")
            return set()

    def insert_documents(self, documents):
        """Insert only new documents into vector store."""
        existing_video_ids = self.get_existing_video_ids()
        new_documents = []
        
        # Filter out already processed documents
        for idx, document in enumerate(documents):
            video_key = [k for k in document.keys() if k != '_id']
            video_id = video_key[0]
            
            if video_id not in existing_video_ids:
                new_documents.append(document)
        
        if not new_documents:
            logger.info("No new documents to insert")
            return
        
        logger.info(f"Inserting {len(new_documents)} new documents")
        
        # Your existing insertion logic here, but only for new_documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=20
        )
        
        # Iterate through the new documents only
        for document in new_documents:
            video_key = [k for k in document.keys() if k!='_id']
            video_id = video_key[0]
            if len(video_key) > 1:
                raise('document structure is changed. we only expect one key.')
            data = document[video_id]  # The document itself contains the data
            logger.info(f"Processing document for video_id: {video_id}")
            logger.info(f"Document has {len(data.get('chunks', []))} chunks and {len(data.get('summaries', []))} summaries")
            
            for idx, (chunk, summary) in enumerate(zip(data['chunks'], data['summaries'])):
                # Insert summary (split if too long)
                if len(summary) > 512:
                    summary_chunks = text_splitter.split_text(summary)
                    for summary_chunk in summary_chunks:
                        try:
                            sparse_emb = self.bge_model([summary_chunk])
                            dense_emb = self.embedding_model.embed_query(summary_chunk)
                            sparse_vector = sparse_emb.get('sparse') if isinstance(sparse_emb, dict) else sparse_emb
                        except Exception as e:
                            logger.error(f"Error generating embeddings for summary chunk: {e}")
                            continue
                        insert_result = self.collection.insert({
                            "text": summary_chunk,
                            "hashed_title": video_id,
                            "summary": True,
                            "chunk_id": idx,
                            "sparse_vector": sparse_vector,
                            "dense_vector": dense_emb,
                            "url": data.get('url')
                        })
                        logger.info(insert_result)
                        logger.info(f"Inserted summary chunk {idx} for video {video_id}")
                else:
                    try:
                        sparse_emb = self.bge_model([summary])
                        dense_emb = self.embedding_model.embed_query(summary)
                        sparse_vector = sparse_emb.get('sparse') if isinstance(sparse_emb, dict) else sparse_emb
                    except Exception as e:
                        logger.error(f"Error generating embeddings for summary: {e}")
                        continue
                    insert_result = self.collection.insert({
                        "text": summary,
                        "hashed_title": video_id,
                        "summary": True,
                        "chunk_id": idx,
                        "sparse_vector": sparse_vector,
                        "dense_vector": dense_emb,
                        "url": data.get('url')
                    })
                    logger.info(f"Inserted summary {idx} for video {video_id}")
                
                # Insert chunk pieces
                documents = text_splitter.split_text(chunk)
                for doc in documents:
                    logger.info('chunk of chunk', doc)
                    try:
                        dense_chunk_emb = self.embedding_model.embed_query(doc)
                        sparse_chunk_emb = self.bge_model([doc])
                        sparse_vector = sparse_chunk_emb.get('sparse') if isinstance(sparse_chunk_emb, dict) else sparse_chunk_emb
                    except Exception as e:
                        logger.error(f"Error generating embeddings for chunk: {e}")
                        continue
                    self.collection.insert({
                        "text": doc,
                        "hashed_title": video_id,
                        "summary": False,
                        "chunk_id": idx,
                        "sparse_vector": sparse_vector,
                        "dense_vector": dense_chunk_emb,
                        "url": data.get('url')
                    })
            self.collection.flush()
        
    def hybrid_search(self, query: str, limit: int = 5) -> List[Any]:
        """Perform hybrid search using both sparse and dense vectors."""
        sparse_emb = self.bge_model([query])
        query_sparse = sparse_emb.get('sparse') if isinstance(sparse_emb, dict) else sparse_emb
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
        # Debug: Check if collection has data
        try:
            count = self.collection.num_entities
            logger.info(f"Collection has {count} entities")
            if count == 0:
                logger.warning("No entities in collection - returning empty results")
                return []
        except Exception as e:
            logger.error(f"Error checking collection count: {e}")
            
        try:
            logger.info("Calling BGE model...")
            sparse_emb = self.bge_model([query])
            logger.info(f"BGE model call successful")
            logger.info(f"BGE model output type: {type(sparse_emb)}")
            
            if hasattr(sparse_emb, 'keys'):
                logger.info(f"BGE model output keys: {list(sparse_emb.keys())}")
                query_sparse = sparse_emb['sparse']
            else:
                logger.info("BGE model output has no keys attribute")
                query_sparse = sparse_emb
                
            logger.info(f"Query sparse type: {type(query_sparse)}")
            
            # Convert to list if it's a numpy array to avoid boolean context issues
            if hasattr(query_sparse, 'tolist'):
                query_sparse = query_sparse.tolist()
                
            logger.info(f"Sparse query vector prepared successfully")
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
        try:
            if expr:
                logger.info(f"Searching with expression: {expr}")
                results = self.collection.search(
                    [query_sparse],
                    anns_field="sparse_vector",
                    limit=limit,
                    expr=expr,
                    output_fields=["text", "summary", "chunk_id", "url"],
                    param={"metric_type": "IP", "params": {}}
                )[0]
            else: 
                logger.info("Searching without expression")
                results = self.collection.search(
                    [query_sparse],
                    anns_field="sparse_vector",
                    limit=limit,
                    output_fields=["text", "summary", "chunk_id", "url"],
                    param={"metric_type": "IP", "params": {}}
                )[0]
        except Exception as e:
            logger.error(f"Error during search: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
        logger.info(f"Search returned {len(results)} results")
        return [hit for hit in results]
