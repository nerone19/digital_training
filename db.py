"""
db.py - Database layer with no knowledge of chat-specific logic
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, DESCENDING, cursor
from bson import ObjectId
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDB:
    _instance = None
    _config = None
    
    @classmethod
    def configure(cls, mongo_uri: str, db_name: str):
        cls._config = {'uri': mongo_uri, 'db_name': db_name}
    
    def __new__(cls):
        if cls._instance is None:
            if cls._config is None:
                raise ValueError("Must call MongoDB.configure() first")
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        config = self.__class__._config
        self._client = MongoClient(config['uri'])
        self._db = self._client[config['db_name']]
        self._setup_collections()
        self._initialized = True
    
    def _setup_collections(self):
        """Setup collections and indexes."""
        # Create indexes for chat sessions
        self._db.chat_sessions.create_index("session_id", unique=True)
        self._db.chat_sessions.create_index([("updated_at", DESCENDING)])
        self._db.chat_sessions.create_index("created_at")
        
        # Text search index for chat content
        try:
            self._db.chat_sessions.create_index([
                ("title", "text"),
                ("messages.content", "text")
            ])
        except Exception:
            pass  # Index might already exist
        
        # Create indexes for videos collection
        if "videos" not in self._db.list_collection_names():
            self._db.videos.create_index("url", unique=True)
            self._db.videos.create_index("video_id", unique=True)
    
    @property
    def client(self):
        return self._client
    
    @property
    def db(self):
        return self._db
    
    # Generic document operations
    def insert_document(self, collection: str, document: Dict[str, Any]) -> Any:
        """Insert a document into a collection."""
        return self._db[collection].insert_one(document)
    
    def find_document(self, collection: str, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document."""
        doc = self._db[collection].find_one(filter_dict)
        if doc:
            doc['_id'] = str(doc['_id'])
        return doc
    
    def find_documents(self, collection: str, filter_dict: Dict[str, Any] = None, 
                      sort_by: Optional[List[tuple]] = None, limit: Optional[int] = None, 
                      skip: Optional[int] = None, projection: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Find multiple documents with optional sorting, limiting, and projection."""
        filter_dict = filter_dict or {}
        cursor = self._db[collection].find(filter_dict, projection)
        
        if sort_by:
            cursor = cursor.sort(sort_by)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        
        docs = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            docs.append(doc)
        return docs
  
    def list_documents(self, collection: str, filter_dict: Dict[str, Any] = None, 
                        sort_by: Optional[List[tuple]] = None, limit: Optional[int] = None, 
                        skip: Optional[int] = None, projection: Optional[Dict] = None) -> cursor:
            """Find multiple documents with optional sorting, limiting, and projection."""
            filter_dict = filter_dict or {}
            cursor = self._db[collection].find(filter_dict, projection)
            
            if sort_by:
                cursor = cursor.sort(sort_by)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            return cursor
        

    def update_document(self, collection: str, filter_dict: Dict[str, Any], 
                       update_dict: Dict[str, Any]) -> bool:
        """Update a document."""
        result = self._db[collection].update_one(filter_dict, update_dict)
        return result.modified_count > 0
    
    def delete_document(self, collection: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete a document."""
        result = self._db[collection].delete_one(filter_dict)
        return result.deleted_count > 0
    
    def aggregate(self, collection: str, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """Perform aggregation query."""
        return list(self._db[collection].aggregate(pipeline))
    
    def text_search(self, collection: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Perform text search on a collection."""
        cursor = self._db[collection].find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        results = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)
        
        return results
    
    def count_documents(self, collection: str, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents in a collection."""
        filter_dict = filter_dict or {}
        return self._db[collection].count_documents(filter_dict)


class ChatSessionRepository:
    """Repository pattern for chat session data access - knows about chat structure but not business logic."""
    
    def __init__(self, db: MongoDB):
        self.db = db
        self.collection = "chat_sessions"
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new chat session."""
        result = self.db.insert_document(self.collection, session_data)
        return session_data["session_id"]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a chat session by ID."""
        return self.db.find_document(self.collection, {"session_id": session_id})
    
    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a chat session."""
        return self.db.update_document(
            self.collection, 
            {"session_id": session_id}, 
            {"$set": update_data}
        )
    
    def add_message_to_session(self, session_id: str, message_data: Dict[str, Any]) -> bool:
        """Add a message to a session."""
        return self.db.update_document(
            self.collection,
            {"session_id": session_id},
            {
                "$push": {"messages": message_data},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            }
        )
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return self.db.delete_document(self.collection, {"session_id": session_id})
    
    def list_sessions(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """List chat sessions with pagination."""
        projection = {
            "session_id": 1, 
            "title": 1, 
            "created_at": 1, 
            "updated_at": 1,
            "metadata": 1, 
            "messages": {"$slice": -1}  # Include last message
        }
        
        sessions = self.db.find_documents(
            self.collection,
            projection=projection,
            sort_by=[("updated_at", DESCENDING)],
            limit=limit,
            skip=skip
        )
        
        # Enhance with message count and last message preview
        for session in sessions:
            if 'messages' in session and session['messages']:
                session['last_message'] = session['messages'][-1]['content'][:100] + "..."
                # Get actual message count
                count_result = self.db.aggregate(self.collection, [
                    {"$match": {"session_id": session["session_id"]}},
                    {"$project": {"message_count": {"$size": "$messages"}}}
                ])
                session['message_count'] = count_result[0]["message_count"] if count_result else 0
            else:
                session['last_message'] = "No messages"
                session['message_count'] = 0
        
        return sessions
    
    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search chat sessions by content."""
        return self.db.text_search(self.collection, query, limit)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about chat sessions."""
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_sessions": {"$sum": 1},
                    "total_messages": {"$sum": {"$size": "$messages"}},
                    "avg_messages_per_session": {"$avg": {"$size": "$messages"}},
                    "oldest_session": {"$min": "$created_at"},
                    "newest_session": {"$max": "$created_at"}
                }
            }
        ]
        
        result = self.db.aggregate(self.collection, pipeline)
        if result:
            stats = result[0]
            stats.pop('_id')
            return stats
        
        return {
            "total_sessions": 0,
            "total_messages": 0,
            "avg_messages_per_session": 0,
            "oldest_session": None,
            "newest_session": None
        }


class VideoRepository:
    """Repository pattern for video data access."""
    
    def __init__(self, db: MongoDB):
        self.db = db
        self.collection = "videos"
    
    def save_video_data(self, video_data: Dict[str, Any]) -> Any:
        """Save processed video data."""
        return self.db.insert_document(self.collection, video_data)
    
    def get_video_by_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video data by video ID."""
        return self.db.find_document(self.collection, {"video_id": video_id})
    
    def get_video_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video data by URL."""
        return self.db.find_document(self.collection, {"url": url})
    
    # todo: not efficient. We need to use a cursor to not keep all objects in memory.
    def list_all_videos(self) -> Dict[str, Dict]:
        return self.db.list_documents(self.collection)

    def load_existing_video_data(self) -> Dict[str, Dict]:
        """Load existing video data in the expected format for the RAG system."""
        videos = self.list_all_videos()
        videos_data = {}
        
        for video_doc in videos:
            # Skip the _id field and get the actual video data
            for key, value in video_doc.items():
                if key != '_id':  # Skip the MongoDB _id field
                    video_id = key
                    videos_data[video_id] = value
                    break  # Each document should only have one video ID key
        
        return videos_data
    
    def populate_from_json(self, json_file: str):
        """Populate database from JSON file."""
        with open(json_file, "r") as file:
            json_data = json.load(file)
        
        for k, v in json_data.items():
            self.db.insert_document(self.collection, {k: v})